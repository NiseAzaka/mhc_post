import functools
import itertools
from typing import Optional

import tilelang
import tilelang.language as T
import torch

from tileops.kernels.kernel_base import Kernel

__all__ = ["MHCPostKernel"]

# Below this batch size we keep the generic 3D kernel; at/above it (n_expand==4)
# the denser all-n4 2D layout is beneficial. Kept at 8 so small production
# prefill chunks pick the denser layout earlier.
_ALL_N4_MIN_BATCH = 8
_DEFAULT_BLOCK_X_B = 1
_DEFAULT_BLOCK_C = 64
_DEFAULT_THREADS = 128
# Kept for backward-compatible configs/custom-op signature. The C dimension is
# a grid dimension again (see note in autotune_configs), so there is no
# software-pipelined C loop for num_stages to control; it is fixed to 1.
_DEFAULT_NUM_STAGES = 1


def _build_mhc_post_kernel(
    batch: int,
    n_expand: int,
    c_x: int,
    x_dtype: str,
):

    dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mhc_func(block_x_b, block_C, num_stages=1, threads=128):
        block_x_b = min(block_x_b, batch)
        # num_stages is accepted for ABI/config compatibility only.
        _ = num_stages
        # Vectorized T.copy requires the sliced tile to be fully in bounds.
        # Fall back to masked scalar loads for irregular shapes.
        vectorized = (batch % block_x_b == 0) and (c_x % block_C == 0)

        @T.macro
        def _get_output_x(
                x_layer_out: T.Tensor([batch, c_x], x_dtype),
                h_post: T.Tensor([batch, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(batch, block_x_b),
                    n_expand,
                    T.ceildiv(c_x, block_C),
                    threads=threads,
            ) as (bx, bn, by):

                # Keep C as a grid dimension: each CTA owns one C tile, which
                # preserves the block-level parallelism that the 13:05
                # implementation had. h_post/x_layer_out are shared (reused
                # across the tile); x_res/x_out stream straight to/from HBM
                # since every element is touched exactly once.
                h_post_shared = T.alloc_shared([block_x_b], dtype)
                x_layer_out_shared = T.alloc_shared([block_x_b, block_C], dtype)

                for i in T.Parallel(block_x_b):
                    if bx * block_x_b + i < batch:
                        h_post_shared[i] = h_post[bx * block_x_b + i, bn]

                if vectorized:
                    T.copy(
                        x_layer_out[bx * block_x_b:(bx + 1) * block_x_b,
                                    by * block_C:(by + 1) * block_C],
                        x_layer_out_shared,
                    )
                else:
                    for i, j in T.Parallel(block_x_b, block_C):
                        c_idx = by * block_C + j
                        b_idx = bx * block_x_b + i
                        if b_idx < batch:
                            if c_idx < c_x:
                                x_layer_out_shared[i, j] = x_layer_out[b_idx, c_idx]

                for i, j in T.Parallel(block_x_b, block_C):
                    c_idx = by * block_C + j
                    b_idx = bx * block_x_b + i
                    if b_idx < batch:
                        if c_idx < c_x:
                            x_out[b_idx, bn * c_x + c_idx] = (
                                h_post_shared[i] * x_layer_out_shared[i, j]
                                + T.cast(x_res[b_idx, bn * c_x + c_idx], dtype)
                            )

        @T.prim_func
        def mhc_post(
                x_layer_out: T.Tensor([batch, c_x], x_dtype),
                h_post: T.Tensor([batch, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            _get_output_x(x_layer_out, h_post, x_res, x_out)

        return mhc_post

    return _mhc_func


@functools.lru_cache(maxsize=32)
def _mhc_post_kernel(batch: int, n_expand: int, c_x: int, x_dtype: str = "bfloat16"):
    return _build_mhc_post_kernel(
        batch,
        n_expand,
        c_x,
        x_dtype,
    )


@functools.lru_cache(maxsize=32)
def _mhc_post_all_n4_kernel(
    batch: int,
    n_expand: int,
    c_x: int,
    x_dtype: str = "bfloat16",
):
    if n_expand != 4:
        raise ValueError(f"all-N candidate requires n_expand=4, got {n_expand}")

    dtype = "float32"

    @tilelang.jit(
        out_idx=[-1],
        pass_configs={
            tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        },
        compile_flags=["-O3", "-DENABLE_BF16"])
    def _mhc_func(block_x_b, block_C, num_stages=1, threads=128):
        block_x_b = min(block_x_b, batch)
        # num_stages is accepted for ABI/config compatibility only.
        _ = num_stages
        vectorized = (batch % block_x_b == 0) and (c_x % block_C == 0)

        @T.prim_func
        def mhc_post(
                x_layer_out: T.Tensor([batch, c_x], x_dtype),
                h_post: T.Tensor([batch, n_expand], dtype),
                x_res: T.Tensor([batch, n_expand * c_x], x_dtype),
                x_out: T.Tensor([batch, n_expand * c_x], x_dtype),
        ):
            with T.Kernel(
                    T.ceildiv(batch, block_x_b),
                    T.ceildiv(c_x, block_C),
                    threads=threads,
            ) as (bx, by):
                h_post_shared = T.alloc_shared([block_x_b, n_expand], dtype)
                x_layer_out_shared = T.alloc_shared([block_x_b, block_C], dtype)

                for i, j in T.Parallel(block_x_b, n_expand):
                    if bx * block_x_b + i < batch:
                        h_post_shared[i, j] = h_post[bx * block_x_b + i, j]

                if vectorized:
                    T.copy(
                        x_layer_out[bx * block_x_b:(bx + 1) * block_x_b,
                                    by * block_C:(by + 1) * block_C],
                        x_layer_out_shared,
                    )
                else:
                    for i, j in T.Parallel(block_x_b, block_C):
                        c_idx = by * block_C + j
                        b_idx = bx * block_x_b + i
                        if b_idx < batch:
                            if c_idx < c_x:
                                x_layer_out_shared[i, j] = x_layer_out[b_idx, c_idx]

                for i, j, k in T.Parallel(block_x_b, n_expand, block_C):
                    c_idx = by * block_C + k
                    b_idx = bx * block_x_b + i
                    if b_idx < batch:
                        if c_idx < c_x:
                            x_out[b_idx, j * c_x + c_idx] = (
                                h_post_shared[i, j] * x_layer_out_shared[i, k]
                                + T.cast(x_res[b_idx, j * c_x + c_idx], dtype)
                            )

        return mhc_post

    return _mhc_func


def _select_mhc_post_kernel(
    batch: int,
    n_expand: int,
    c_x: int,
    x_dtype: str,
):
    if n_expand == 4 and batch >= _ALL_N4_MIN_BATCH:
        return _mhc_post_all_n4_kernel(batch, n_expand, c_x, x_dtype)
    return _mhc_post_kernel(batch, n_expand, c_x, x_dtype)


@torch.library.custom_op("top::mhc_post_wrapped_kernel", mutates_args=())
def _mhc_post_wrapped_kernel(batch: int, n_expand: int, c_x: int, dtype: str,
                             block_x_b: int, block_C: int, num_stages: int,
                             threads: int, x_layer_out: torch.Tensor,
                             h_post: torch.Tensor,
                             x_res: torch.Tensor) -> torch.Tensor:
    return _select_mhc_post_kernel(batch, n_expand, c_x, dtype)(
        block_x_b, block_C, num_stages, threads,
    )(x_layer_out, h_post, x_res)


@_mhc_post_wrapped_kernel.register_fake
def _(
    batch: int,
    n_expand: int,
    c_x: int,
    dtype: str,
    block_x_b: int,
    block_C: int,
    num_stages: int,
    threads: int,
    *input,
) -> torch.Tensor:
    return torch.empty(
        (batch, n_expand * c_x),
        dtype=input[0].dtype,
        device=input[0].device,
    )


def _mhc_post_compile_default(
    x_layer_out: torch.Tensor,
    h_post: torch.Tensor,
    x_res: torch.Tensor,
) -> torch.Tensor:
    """Traceable default-config entry that keeps TileLang construction opaque."""
    batch, c_x = x_layer_out.shape
    n_expand = h_post.shape[1]
    return _mhc_post_wrapped_kernel(
        batch,
        n_expand,
        c_x,
        "bfloat16",
        _DEFAULT_BLOCK_X_B,
        _DEFAULT_BLOCK_C,
        _DEFAULT_NUM_STAGES,
        _DEFAULT_THREADS,
        x_layer_out,
        h_post,
        x_res,
    )


class MHCPostKernel(Kernel):
    supported_archs: list[int] = [80, 89, 90]

    def __init__(self,
                 batch,
                 n_expand,
                 c_x,
                 dtype: torch.dtype = torch.float32,
                 config: Optional[dict] = None,
                 tune=False):
        super().__init__()
        self.batch = batch
        self.n_expand = n_expand
        self.c_x = c_x
        self.dtype = dtype
        self.weights_dtype = torch.float32
        self.kernel = _select_mhc_post_kernel(
            self.batch,
            self.n_expand,
            self.c_x,
            self.dtype_str,
        )

        self.init_config(config, tune)

    @property
    def default_config(self) -> dict:
        return {
            "block_x_b": _DEFAULT_BLOCK_X_B,
            "block_C": _DEFAULT_BLOCK_C,
            "num_stages": _DEFAULT_NUM_STAGES,
            "threads": _DEFAULT_THREADS,
        }

    @property
    def autotune_configs(self) -> list[dict]:
        # 15:07 log showed that moving C out of the grid into a pipelined
        # loop collapsed CTA parallelism and regressed latency. Keep C on the
        # grid and tune only real knobs; num_stages is fixed to 1 for
        # signature/config compatibility (no software-pipelined C loop).
        block_x_b = [1, 8, 64]
        block_C = [64, 128]
        num_stages = [1]
        threads = [128, 256]
        _configs = list(itertools.product(block_x_b, block_C, num_stages, threads))

        configs = [{
            "block_x_b": c[0],
            "block_C": c[1],
            "num_stages": c[2],
            "threads": c[3],
        } for c in _configs]
        return configs

    def forward(self, x_layer_out, h_post, x_res):

        result = _mhc_post_wrapped_kernel(self.batch, self.n_expand, self.c_x, self.dtype_str,
                                          self.config["block_x_b"], self.config["block_C"],
                                          self.config.get("num_stages", _DEFAULT_NUM_STAGES),
                                          self.config["threads"],
                                          x_layer_out, h_post, x_res)
        return result
