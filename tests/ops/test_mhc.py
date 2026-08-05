"""Tests for the MHC pre/post ops."""

import math

import pytest
import torch
import torch.nn.functional as F

from tests.test_base import FixtureBase, TestBase
from tileops.kernels.mhc import MHCPostKernel
from tileops.kernels.mhc.mhc_post import _mhc_post_all_n4_kernel, _mhc_post_kernel
from tileops.ops import MHCPostOp, MHCPreOp
from workloads.mhc import MHCPostTest as _MHCPostTestWorkload
from workloads.mhc import MHCPreTest as _MHCPreTestWorkload


class MHCPreTest(_MHCPreTestWorkload, TestBase):
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


class MHCPreFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]


def _cosine_compare(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Compare using cosine similarity (MHC ops use bf16 and need looser checks)."""
    cos_sim = F.cosine_similarity(output_ref, output, dim=-1, eps=1e-8)
    assert cos_sim.min() > 0.99, \
        f"cosine similarity too low: {cos_sim.min().item()}"


@MHCPreFixture
def test_mhc_pre_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                    tune: bool) -> None:
    test = MHCPreTest(batch, n_expand, c_x, dtype)
    op = MHCPreOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


class MHCPostTest(_MHCPostTestWorkload, TestBase):
    def ref_program(self, x_layer_out: torch.Tensor, h_post: torch.Tensor,
                    x_res: torch.Tensor) -> torch.Tensor:
        batch = self.batch
        n_expand = self.n_expand
        c_x = self.c_x

        x_out_ref = (h_post.unsqueeze(2).float() @ x_layer_out.unsqueeze(1).float()).reshape(
            batch, n_expand * c_x) + x_res.float()
        x_out_ref = x_out_ref.bfloat16()
        return x_out_ref


class MHCPostFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            pytest.param(1, 4, 1280, torch.bfloat16, False, marks=pytest.mark.smoke),
            pytest.param(2, 4, 1920, torch.bfloat16, False, marks=pytest.mark.full),
            pytest.param(4, 4, 2560, torch.bfloat16, False, marks=pytest.mark.full),
        ]),
    ]





@MHCPostFixture
def test_mhc_post_op(batch: int, n_expand: int, c_x: int, dtype: torch.dtype,
                     tune: bool) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_cosine_compare)


class _UnexpectedMHCPostKernel:
    """Sentinel proving invalid/empty inputs never reach a device kernel."""

    supported_archs = [80, 89, 90]

    def __init__(self, *args, **kwargs) -> None:
        del args, kwargs

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("MHCPostOp contract path unexpectedly reached the kernel")


def _contract_only_op() -> MHCPostOp:
    return MHCPostOp(
        kernel_map={"mhc_post_kernel": _UnexpectedMHCPostKernel},
        tune=False,
    )


def _assert_mhc_post_close(output: torch.Tensor, output_ref: torch.Tensor) -> None:
    """Check the MHC post output contract and every BF16 element."""
    assert output.shape == output_ref.shape
    assert output.dtype == output_ref.dtype
    assert output.device == output_ref.device
    torch.testing.assert_close(
        output,
        output_ref,
        rtol=1.6e-2,
        atol=1.6e-2,
    )


class MHCPostScaleFixture(FixtureBase):
    PARAMS = [
        ("batch, n_expand, c_x, dtype, tune", [
            # Minimum valid shape: guards against a zero-program grid.
            pytest.param(
                1, 1, 1, torch.bfloat16, False,
                marks=pytest.mark.smoke, id="minimum-grid",
            ),
            # Aligned intermediate shape, distinct from benchmark workloads.
            pytest.param(
                2, 4, 1024, torch.bfloat16, False,
                marks=pytest.mark.full, id="aligned-intermediate",
            ),
            # Isolates the manifest-valid N != 4 path without a C tail.
            pytest.param(
                1, 3, 128, torch.bfloat16, False,
                marks=pytest.mark.full, id="n-expand-three",
            ),
        ]),
    ]


@MHCPostScaleFixture
def test_mhc_post_scale_boundaries(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: torch.dtype,
    tune: bool,
) -> None:
    test = MHCPostTest(batch, n_expand, c_x, dtype)
    op = MHCPostOp(tune=tune)
    test.check(op, *test.gen_inputs(), compare=_assert_mhc_post_close)


@pytest.mark.parametrize(
    "c_x",
    [
        pytest.param(65, id="tail-one", marks=pytest.mark.smoke),
        pytest.param(127, id="tail-large", marks=pytest.mark.full),
        pytest.param(129, id="second-block-tail", marks=pytest.mark.full),
    ],
)
def test_mhc_post_non_divisible_tail(c_x: int) -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=c_x, dtype=torch.bfloat16)
    op = MHCPostOp(tune=False)
    test.check(op, *test.gen_inputs(), compare=_assert_mhc_post_close)


@pytest.mark.smoke
def test_mhc_post_fake_output_shape() -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        x_layer_out = torch.empty((2, 65), device="cuda", dtype=torch.bfloat16)
        h_post = torch.empty((2, 3), device="cuda", dtype=torch.float32)
        x_res = torch.empty((2, 195), device="cuda", dtype=torch.bfloat16)
        output = MHCPostOp(tune=False)(x_layer_out, h_post, x_res)

    assert output.shape == x_res.shape
    assert output.dtype == x_res.dtype
    assert output.device == x_res.device
    assert output.stride() == (195, 1)
    assert output.is_contiguous()


@pytest.mark.smoke
@pytest.mark.usefixtures("isolated_dynamo")
def test_mhc_post_torch_compile_fullgraph() -> None:
    """A cold fullgraph trace must preserve the public output contract."""
    test = MHCPostTest(batch=2, n_expand=3, c_x=65, dtype=torch.bfloat16)
    inputs = test.gen_inputs()
    compiled_op = torch.compile(MHCPostOp(tune=False), fullgraph=True)

    output = compiled_op(*inputs)
    output_ref = test.ref_program(*inputs)

    _assert_mhc_post_close(output, output_ref)
    assert output.stride() == (195, 1)
    assert output.is_contiguous()


@pytest.mark.parametrize(
    "input_index, invalid_dtype, input_name",
    [
        pytest.param(
            0, torch.float16, "x_layer_out",
            id="x-layer-out", marks=pytest.mark.smoke,
        ),
        pytest.param(
            1, torch.bfloat16, "h_post",
            id="h-post", marks=pytest.mark.full,
        ),
        pytest.param(
            2, torch.float16, "x_res",
            id="x-res", marks=pytest.mark.full,
        ),
    ],
)
def test_mhc_post_rejects_invalid_dtype(
    input_index: int,
    invalid_dtype: torch.dtype,
    input_name: str,
) -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=64, dtype=torch.bfloat16)
    inputs = list(test.gen_inputs())
    inputs[input_index] = inputs[input_index].to(invalid_dtype)

    with pytest.raises(ValueError, match=rf"input '{input_name}'.*expected"):
        _contract_only_op()(*inputs)


@pytest.mark.smoke
def test_mhc_post_rejects_invalid_rank() -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=64, dtype=torch.bfloat16)
    x_layer_out, h_post, x_res = test.gen_inputs()

    with pytest.raises(ValueError, match="2D tensors"):
        _contract_only_op()(x_layer_out.unsqueeze(0), h_post, x_res)


@pytest.mark.smoke
def test_mhc_post_rejects_batch_mismatch() -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=64, dtype=torch.bfloat16)
    x_layer_out, _, x_res = test.gen_inputs()
    h_post = torch.randn((2, 4), device="cuda", dtype=torch.float32)

    with pytest.raises(ValueError, match="matching batch dimensions"):
        _contract_only_op()(x_layer_out, h_post, x_res)


@pytest.mark.smoke
def test_mhc_post_rejects_residual_width_mismatch() -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=64, dtype=torch.bfloat16)
    x_layer_out, h_post, _ = test.gen_inputs()
    x_res = torch.randn((1, 255), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="n_expand \\* c_x"):
        _contract_only_op()(x_layer_out, h_post, x_res)


@pytest.mark.smoke
def test_mhc_post_rejects_cpu_inputs() -> None:
    x_layer_out = torch.empty((1, 64), dtype=torch.bfloat16)
    h_post = torch.empty((1, 4), dtype=torch.float32)
    x_res = torch.empty((1, 256), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="CUDA/MACA tensors"):
        _contract_only_op()(x_layer_out, h_post, x_res)


@pytest.mark.smoke
def test_mhc_post_rejects_cross_device_inputs() -> None:
    op = _contract_only_op()
    x_layer_out = torch.empty((1, 64), device="cuda", dtype=torch.bfloat16)
    h_post = torch.empty((1, 4), device="cpu", dtype=torch.float32)
    x_res = torch.empty((1, 256), device="cuda", dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same device"):
        op(x_layer_out, h_post, x_res)


@pytest.mark.parametrize(
    "input_index, input_name",
    [
        pytest.param(
            0, "x_layer_out", id="x-layer-out", marks=pytest.mark.smoke,
        ),
        pytest.param(
            1, "h_post", id="h-post", marks=pytest.mark.full,
        ),
        pytest.param(
            2, "x_res", id="x-res", marks=pytest.mark.full,
        ),
    ],
)
def test_mhc_post_rejects_noncontiguous_input(
    input_index: int,
    input_name: str,
) -> None:
    test = MHCPostTest(batch=1, n_expand=4, c_x=64, dtype=torch.bfloat16)
    inputs = list(test.gen_inputs())
    rows = inputs[input_index].shape[0]
    cols = inputs[input_index].shape[1]
    strided = torch.randn(
        (rows, cols * 2),
        device="cuda",
        dtype=inputs[input_index].dtype,
    )[:, ::2]
    assert strided.shape == inputs[input_index].shape
    assert not strided.is_contiguous()
    inputs[input_index] = strided

    with pytest.raises(ValueError, match=rf"{input_name}.*contiguous"):
        _contract_only_op()(*inputs)


@pytest.mark.parametrize(
    "batch, n_expand, c_x",
    [
        pytest.param(0, 4, 64, id="empty-batch", marks=pytest.mark.smoke),
        pytest.param(1, 0, 64, id="empty-expand", marks=pytest.mark.full),
        pytest.param(1, 4, 0, id="empty-channel", marks=pytest.mark.full),
    ],
)
def test_mhc_post_empty_dimensions(
    batch: int,
    n_expand: int,
    c_x: int,
) -> None:
    test = MHCPostTest(batch, n_expand, c_x, torch.bfloat16)
    inputs = test.gen_inputs()
    output_ref = test.ref_program(*inputs)
    output = _contract_only_op()(*inputs)

    assert output.shape == output_ref.shape
    assert output.dtype == output_ref.dtype
    assert output.device == output_ref.device
    assert output.is_contiguous()
    assert output.numel() == 0


@pytest.mark.parametrize(
    "block_c, threads",
    [
        pytest.param(64, 128, id="c64-t128", marks=pytest.mark.smoke),
        pytest.param(64, 256, id="c64-t256", marks=pytest.mark.full),
        pytest.param(128, 128, id="c128-t128", marks=pytest.mark.full),
        pytest.param(128, 256, id="c128-t256", marks=pytest.mark.full),
    ],
)
def test_mhc_post_unique_fixed_configs(block_c: int, threads: int) -> None:
    """Every source-unique config must be correct before autotune can use it."""
    test = MHCPostTest(batch=1, n_expand=3, c_x=129, dtype=torch.bfloat16)
    kernel = MHCPostKernel(
        batch=1,
        n_expand=3,
        c_x=129,
        dtype=torch.bfloat16,
        config={"block_C": block_c, "threads": threads},
        tune=False,
    )
    test.check(kernel, *test.gen_inputs(), compare=_assert_mhc_post_close)


@pytest.mark.parametrize(
    "block_c, threads",
    [
        pytest.param(64, 128, id="c64-t128", marks=pytest.mark.smoke),
        pytest.param(64, 256, id="c64-t256", marks=pytest.mark.full),
        pytest.param(128, 128, id="c128-t128", marks=pytest.mark.full),
        pytest.param(128, 256, id="c128-t256", marks=pytest.mark.full),
    ],
)
def test_mhc_post_all_n4_fixed_configs(block_c: int, threads: int) -> None:
    """All-N4 must cover its tail correctly for every unique config."""
    test = MHCPostTest(batch=1, n_expand=4, c_x=129, dtype=torch.bfloat16)
    # feat 版 all-n4 kernel 签名为 (block_x_b, block_C, threads)
    kernel = _mhc_post_all_n4_kernel(1, 4, 129, "bfloat16")(1, block_c, threads)
    test.check(kernel, *test.gen_inputs(), compare=_assert_mhc_post_close)


@pytest.mark.smoke
def test_mhc_post_all_n4_rejects_other_n() -> None:
    with pytest.raises(ValueError, match="requires n_expand=4"):
        _mhc_post_all_n4_kernel(1, 3, 129, "bfloat16")


@pytest.mark.parametrize(
    "batch, n_expand, expected_all_n",
    [
        # feat 版 dispatch 边界为 _ALL_N4_MIN_BATCH = 8
        pytest.param(7, 4, False, id="below-boundary", marks=pytest.mark.smoke),
        pytest.param(8, 4, True, id="at-boundary", marks=pytest.mark.full),
        pytest.param(32, 3, False, id="wrong-n-fallback", marks=pytest.mark.full),
    ],
)
def test_mhc_post_public_dispatch_boundary(
    batch: int,
    n_expand: int,
    expected_all_n: bool,
) -> None:
    c_x = 129
    test = MHCPostTest(batch, n_expand, c_x, torch.bfloat16)
    kernel = MHCPostKernel(
        batch=batch,
        n_expand=n_expand,
        c_x=c_x,
        dtype=torch.bfloat16,
        config={"block_C": 64, "threads": 128},
        tune=False,
    )
    expected_factory = _mhc_post_all_n4_kernel if expected_all_n else _mhc_post_kernel
    assert kernel.kernel is expected_factory(batch, n_expand, c_x, "bfloat16")
    test.check(kernel, *test.gen_inputs(), compare=_assert_mhc_post_close)


@pytest.mark.smoke
def test_mhc_post_op_reuses_across_dispatch_boundary() -> None:
    op = MHCPostOp(tune=False)
    # feat 版边界为 8：7 走通用 kernel，8 走 all-n4，7 复用通用 kernel 缓存
    for batch in (7, 8, 7):
        test = MHCPostTest(batch, 4, 65, torch.bfloat16)
        test.check(op, *test.gen_inputs(), compare=_assert_mhc_post_close)


if __name__ == "__main__":
    pytest.main([__file__, "-vvs"])
