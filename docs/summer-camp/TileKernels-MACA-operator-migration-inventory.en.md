# TileKernels-MACA-operator-migration-inventory

[简体中文](TileKernels-MACA-待迁移算子盘点.md) | [**English**](TileKernels-MACA-operator-migration-inventory.en.md)

All operators pending migration come from the default `dev` branch of [MetaX-MACA/TileKernels-Metax](https://github.com/MetaX-MACA/TileKernels-Metax). This inventory includes only candidate operators that have undergone a preliminary assessment, offer general reuse value, have room for MetaX C500 adaptation or performance optimization, and can support standalone tests and Benchmarks.

| Category | Operator | Status | Difficulty | Description |
| --- | --- | --- | --- | --- |
| Engram | engram_fused_weight | Pending migration | Low | Elementwise weight fusion, vectorized memory access, and FP32 output. |
| Engram | engram_gate | Pending migration | High | Persistent Kernel, shared-memory pipelining, reduction, RMSNorm, gating, and backward computation. |
| Engram | engram_grad_w_reduce | Pending migration | Low | Weight-gradient reduction, pipelined copies, and in-place accumulation. |
| Engram | engram_hash | Pending migration | Low | int64 hashing, bitwise operations, modulo, and index computation. |
| Manifold HyperConnection | mhc_multilayer_recompute | Pending migration | High | Device pointers, multilayer recomputation, runtime Tensor construction, and double buffering. |
| Manifold HyperConnection | mhc_post | Pending optimization | Medium | Matrix mixing, forward and backward passes, Reducer, and pipelining. |
| Manifold HyperConnection | mhc_pre_big_fuse | Pending migration | High | RMSNorm, Sigmoid, Sinkhorn, Split-K, and fused Kernel. |
| MoE Routing | moe_aux_fi | Pending migration | Low | Shared-memory histogram, atomic addition, and statistical normalization. |
| MoE Routing | moe_expand_to_fused | Pending migration | Medium | Scatter, Expert-Major layout, FP8/FP4, and scale-factor layout. |
| MoE Routing | moe_get_fused_mapping | Pending migration | High | Warp histogram, Grid synchronization, prefix sum, and stable ranking. |
| MoE Routing | moe_inplace_unique_group_indices | Pending migration | Low | Bitmap, bitwise operations, and in-place deduplication. |
| MoE Routing | moe_mask_indices_by_tp | Pending migration | Low | TP mapping, integer indexing, and range filtering. |
| MoE Routing | moe_normalize_weight | Pending migration | Low | Top-K weight normalization, row reduction, and numerical stability. |
| MoE Routing | moe_reduce_fused | Pending migration | Medium | Gather, weighted reduction, FP8 output, and scale factors. |
| MoE Routing | moe_group_count | Pending optimization | Low | Shared-memory histogram, atomic operations, and parallel statistics. |
| MoE Gating | moe_top2_sum_gate | Pending migration | High | Multiple scoring methods, group selection, stable Top-K, Expert mapping, and EP/TP Mask. |
| MoE Gating | moe_topk_gate | Pending migration | Medium | Top-K, maximum reduction, and stable index selection. |
| MoE Gating | moe_topk_sum_group_idx | Pending migration | Medium | Group selection, Lane Shuffle, and Wavefront collaboration. |
| Quantization | quant_cast_back | Pending migration | High | FP8/FP4 dequantization, scale factors, layout handling, and E5M6. |
| Quantization | quant_cast_back_e5m6 | Pending migration | Medium | E5M6, bit packing and unpacking, and custom Fragment. |
| Quantization | quant_per_block_cast | Pending migration | Medium | Blockwise quantization, AMax, FP8/FP4, and custom Layout. |
| Quantization | quant_per_block_cast_lossless | Pending migration | Medium | Lossless requantization, exponent operations, saturation handling, and Device Assert. |
| Quantization | quant_per_channel_cast | Pending migration | Low | Parameter validation, wrapping, and Kernel dispatch. |
| Quantization | quant_per_channel_cast_transpose | Pending migration | Medium | Per-channel quantization, register transpose, Shared-Memory Swizzle, and column reduction. |
| Quantization | quant_per_channel_cast_fused | Pending migration | High | Per-channel quantization, Gather/Expand, FP8 Rescale, and Lane broadcast. |
| Quantization | quant_per_token_cast | Pending migration | High | Per-Token quantization, AMax, custom Layout, multiple numeric formats, and Strided input. |
| Quantization | quant_per_token_cast_e5m6 | Pending migration | High | E5M6 encoding, bit packing, rounding, and low-level floating-point conversion. |
| Fused SwiGLU + Quantization | quant_swiglu_bwd_token_cast | Pending migration | High | Dequantization, SwiGLU backward pass, multi-output quantization, and gradient reduction. |
| Fused SwiGLU + Quantization | quant_swiglu_fwd_channel_cast_transpose | Pending migration | High | SwiGLU forward pass, per-channel quantization, transpose, and Packed BF16. |
| Fused SwiGLU + Quantization | quant_swiglu_fwd_token_cast | Pending migration | High | SwiGLU forward pass, per-Token quantization, Mask, atomic operations, and persistent Kernel. |
| Transpose | batched_transpose | Pending migration | Medium | Batched transpose, register transpose, Shared-Memory Swizzle, and vectorization. |
