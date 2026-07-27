**简体中文** | [English](README.en.md)

<div align="center">
  <img src="https://raw.githubusercontent.com/tile-ai/TileOPs/main/assets/logo.png" width="350"/>
  <h1>TileOPs</h1>
  <p><strong>面向大语言模型、由规范驱动的 GPU 算子库——帮助 AI Agent 构建、评估和优化算子</strong></p>
  <p>基于 <a href="https://github.com/tile-ai/tilelang">TileLang</a> 构建</p>
  <!-- <p>
    <a href="https://pypi.org/project/tileops/"><img src="https://img.shields.io/badge/PyPI-tileops-1E90FF" alt="PyPI version" height="20"></a>
  </p> -->
  <p>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/tileops/manifest"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-implemented.json" alt="Spec coverage"></a>
    <a href="https://github.com/tile-ai/TileOPs/tree/main/benchmarks"><img src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Ftile-ai%2FTileOPs%2Fstats%2Fmanifest-benchmark.json" alt="Bench coverage"></a>
  </p>
  <p>
    <a href="#安装"><b>安装</b></a> |
    <a href="#快速开始"><b>快速开始</b></a> |
    <a href="#文档"><b>文档</b></a>
  </p>
</div>

> **项目状态**：TileOPs 正在积极开发中，API 可能发生变化。

## 首届开源英才夏令营

夏令营学员应使用 `summer-camp-2026` 分支，并遵循[算子迁移指南](docs/summer-camp/README.md)。该指南规定了算子认领、Manifest/实现双 PR流程、MetaX GPU 验证、Benchmark、Roofline 证据和验收要求。

## 概述

TileOPs 是一个基于 [TileLang](https://github.com/tile-ai/tilelang)、面向大语言模型训练和推理的GPU 算子库。除了持续提供可用于生产的算子，TileOPs 还探索一种**规范驱动的开发模式**：AI Agent 可以读取声明式算子规范、生成 Kernel 实现，并依据硬件理论性能上限进行评估，同时尽量减少人工脚手架。

### 架构

每个算子都严格分为两个层次：

- **Op**（L2）——无状态 Python 入口，负责参数校验、dtype 转换和内存布局，并兼容 CUDA Graph 与 `torch.compile`。
- **Kernel**（L1）——TileLang GPU 实现，包含针对具体硬件的优化策略（Ampere、Hopper）。

这种分层使面向用户的行为与 GPU 策略相互独立，AI Agent 和开发者可以修改其中一层，而不对另一层产生意外影响。

### 主要特性

- **规范驱动**——每个算子都由机器可读的 Manifest（`tileops/manifest/`）声明，
  其中定义签名、工作负载和 Roofline 公式，同时作为 AI Agent 生成代码与自动验证的入口。
- **Roofline 评估**——Kernel 性能依据硬件 Speed-of-Light 理论上限评估，而不是只与相对基线比较。
- **自动调优**——内置对 tile 大小、流水线和调度参数的搜索。
- **轻量依赖**——仅依赖 TileLang、PyTorch 和 einops。

## 安装

TileOPs 可以从 PyPI 安装，也可以从源码构建。运行时需要支持 CUDA 的 GPU。

### 前置条件

- Python >= 3.10
- PyTorch >= 2.1
- CUDA Toolkit
- NVIDIA GPU：**Hopper**（SM_90）
- [TileLang](https://github.com/tile-ai/tilelang) == 0.1.9

### 从 PyPI 安装

```bash
pip install tileops
```

### 从源码安装

```bash
git clone https://github.com/tile-ai/TileOPs
cd TileOPs
make install    # 开发依赖 + pre-commit hooks
```

> [!NOTE]
> 如果系统已经安装 CUDA 和 TileLang，但构建时遇到问题，请运行：
> `PIP_NO_BUILD_ISOLATION=1 pip install -e '.[dev]' -v && pre-commit install`

验证安装：

```bash
python -m pytest tests/ -q    # 需要 CUDA GPU
```

## 快速开始

```python
import torch
from tileops.ops import GemmOp

M, N, K = 1024, 1024, 512
dtype = torch.float16

gemm = GemmOp(M, N, K, dtype=dtype)

A = torch.randn(M, K, device="cuda", dtype=dtype)
B = torch.randn(K, N, device="cuda", dtype=dtype)

C = gemm(A, B)
```

## 文档

设计文档和开发指南位于 [`docs/`](docs/) 目录。完整 API 参考和性能表发布在
[TileOPs.github.io](https://github.com/tile-ai/TileOPs.github.io)。

## 参与贡献

设计文档请参阅 [`docs/`](docs/)。分支与提交规范位于[`.claude/conventions/types.sh`](.claude/conventions/types.sh)。

## 许可证

TileOPs 使用 [MIT License](LICENSE) 发布。
