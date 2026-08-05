"""Benchmarks for the MHC pre/post ops.

Production shapes follow DeepSeek V4 (arXiv:2606.19348) and the mHC paper
(arXiv:2512.24880):

  - DeepSeek-V4-Pro   hidden_size=7168, hc_mult=4 (config.json)
  - DeepSeek-V4-Flash hidden_size=4096, hc_mult=4 (config.json)
  - mHC paper Table 5: Dimension 1280/1920/2560 (3B/9B/27B); sinkhorn 20 iters.
  - batch = num_tokens in production: Decode 32/128/512, Prefill 1024/4096/8192.
Kernel constraint: c_x must be a multiple of 64 (block_C tiles of 64/128);
7168 and 4096 both satisfy this (7168 = 56*128, 4096 = 32*128).
"""

import math
from typing import Optional

import pytest
import torch

from benchmarks.benchmark_base import BenchmarkBase, BenchmarkReport
from tileops.ops import MHCPostOp, MHCPreOp
from workloads.mhc import MHCPostTest, MHCPreTest


class _MHCPreTestBaseline(MHCPreTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, phi: torch.Tensor, x: torch.Tensor, b: torch.Tensor,
                    alpha_pre, alpha_post, alpha_res,
                    sinkhorn_repeat: int, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        xsqr = x * x
        norm_eps = 0.0001
        r_ref = torch.sqrt(xsqr.sum(dim=1)) / math.sqrt(n_expand * c_x) + norm_eps
        H = torch.zeros([batch, n_expand * n_expand + 2 * n_expand],
                        device="cuda", dtype=torch.float)
        for i in range(batch):
            H[i, :] = x[i, :].float() @ phi

        H_pre_ref = H[:, :n_expand]
        H_res_ref = H[:, 2 * n_expand:]
        H_res_ref = H_res_ref.reshape(batch, n_expand, n_expand)

        b_pre_ref = b[:n_expand]
        b_res_ref = b[2 * n_expand:]
        b_res_ref = b_res_ref.reshape([n_expand, n_expand])

        H_pre_ref = torch.sigmoid(alpha_pre * H_pre_ref / r_ref.unsqueeze(-1) + b_pre_ref)
        H_res_ref = alpha_res * H_res_ref / r_ref.unsqueeze(-1).unsqueeze(-1) + b_res_ref

        H_res_ref_tmp = H_res_ref.max(dim=-1, keepdim=True).values

        H_res_ref = torch.exp(H_res_ref - H_res_ref_tmp)
        for _i in range(sinkhorn_repeat):
            H_res_ref = H_res_ref / (H_res_ref.sum(dim=-1, keepdim=True) + eps)
            H_res_ref = H_res_ref / (H_res_ref.sum(dim=-2, keepdim=True) + eps)
        x_in_reshaped = x.reshape([batch, n_expand, c_x])
        x_res_ref = torch.zeros([batch, n_expand, c_x], device="cuda", dtype=torch.bfloat16)
        x_layer_ref = torch.zeros([batch, c_x], device="cuda", dtype=torch.bfloat16)

        h_res_ref = H_res_ref
        h_pre_ref = H_pre_ref
        for i in range(batch):
            h_res_tmp = h_res_ref[i, :, :].float()
            h_pre_tmp = h_pre_ref[i, :].float()
            x_in_reshaped_tmp = x_in_reshaped[i, :, :].float()
            x_res_ref[i, :, :] = h_res_tmp @ x_in_reshaped_tmp
            x_layer_ref[i, :] = h_pre_tmp @ x_in_reshaped_tmp

        x_res_ref = x_res_ref.reshape(batch, n_expand * c_x)

        x_res_ref = x_res_ref.bfloat16()
        x_layer_ref = x_layer_ref.bfloat16()
        return x_res_ref, x_layer_ref


class MHCPreBenchmark(BenchmarkBase[MHCPreTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        flops = 2 * t.batch * (
            (t.n_expand * t.n_expand * t.c_x * t.c_x) *
            (t.n_expand * t.n_expand + 2 * t.n_expand) + t.n_expand * t.c_x)
        return flops

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        return (t.n_expand * 3 + 1) * t.c_x + (t.n_expand * t.c_x) * (
            t.n_expand * t.n_expand + 2 * t.n_expand)


_MHC_PRE_BENCH_PARAMS = [
    pytest.param(1, 4, 1280, torch.bfloat16, True, id="small"),
    pytest.param(2, 4, 1920, torch.bfloat16, True, id="medium"),
    pytest.param(4, 4, 2560, torch.bfloat16, True, id="large"),
]


@pytest.mark.parametrize("batch, n_expand, c_x, dtype, tune", _MHC_PRE_BENCH_PARAMS)
def test_mhc_pre_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                       tune: bool) -> None:
    test = _MHCPreTestBaseline(batch, n_expand, c_x, dtype)
    bm = MHCPreBenchmark(test)
    inputs = test.gen_inputs()

    op = MHCPreOp(tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


class _MHCPostTestBaseline(MHCPostTest):
    """Adds baseline ref_program for benchmark profiling."""

    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


class MHCPostBenchmark(BenchmarkBase[MHCPostTest]):

    def calculate_flops(self) -> Optional[float]:
        t = self.workload
        # Post-operator is element-wise: x_out = h_post @ x_layer_out + x_res,
        # where h_post [n, n] @ x_layer_out [n, c_x] is an n x n x c_x FMA and
        # the broadcast add of x_res is another n x c_x FMA.
        # => 2 * batch * n * c_x (matches perf/formulas.py:mhc_post_roofline).
        return 2 * t.batch * t.n_expand * t.c_x

    def calculate_memory(self) -> Optional[float]:
        t = self.workload
        # x_layer_out [batch, n*c_x] + h_post [batch, n] + x_res [batch, n*c_x]
        # + x_out [batch, n*c_x]. batch was previously missing from this term.
        return (t.n_expand * 2 + 1) * t.c_x * t.batch


_MHC_POST_BENCH_PARAMS = [
    # Same production shape set as MHCPreOp (see _MHC_PRE_BENCH_PARAMS).
    pytest.param(1, 4, 1280, torch.bfloat16, True, id="small"),
    pytest.param(2, 4, 1920, torch.bfloat16, True, id="medium"),
    pytest.param(4, 4, 2560, torch.bfloat16, True, id="large"),
    pytest.param(32, 4, 4096, torch.bfloat16, True, id="v4-flash-decode-s"),
    pytest.param(32, 4, 7168, torch.bfloat16, True, id="v4-pro-decode-s"),
    pytest.param(128, 4, 4096, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-flash-decode-m"),
    pytest.param(128, 4, 7168, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-pro-decode-m"),
    pytest.param(512, 4, 4096, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-flash-decode-l"),
    pytest.param(512, 4, 7168, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-pro-decode-l"),
    pytest.param(1024, 4, 4096, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-flash-prefill-s"),
    pytest.param(1024, 4, 7168, torch.bfloat16, True, marks=pytest.mark.full,
                 id="v4-pro-prefill-s"),
    pytest.param(8192, 4, 4096, torch.bfloat16, True, marks=pytest.mark.nightly,
                 id="v4-flash-prefill-l"),
    pytest.param(4096, 4, 7168, torch.bfloat16, True, marks=pytest.mark.nightly,
                 id="v4-pro-prefill-l"),
    pytest.param(128, 2, 2048, torch.bfloat16, True, marks=pytest.mark.nightly,
                 id="expand2-c2048"),
    pytest.param(128, 8, 1024, torch.bfloat16, True, marks=pytest.mark.nightly,
                 id="expand8-c1024"),
]


@pytest.mark.parametrize("batch, n_expand, c_x, dtype, tune", _MHC_POST_BENCH_PARAMS)
def test_mhc_post_bench(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                         tune: bool) -> None:
    test = _MHCPostTestBaseline(batch, n_expand, c_x, dtype)
    bm = MHCPostBenchmark(test)
    inputs = test.gen_inputs()

    op = MHCPostOp(tune=tune)
    result = bm.profile(op, *inputs)
    BenchmarkReport.record(op, locals(), result, tag="tileops")

    result_bl = bm.profile(test.ref_program, *inputs)
    BenchmarkReport.record(op, locals(), result_bl, tag="torch-ref")


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
