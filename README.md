<div align="center">
  <img src="assets/logo.png" width="300" alt="TileOPs-Metax Logo">

  <h1>TileOPs-Metax</h1>

  <p>
    <strong>面向大语言模型、由规范驱动的 GPU 算子库</strong>
  </p>

  <p>
    <strong>简体中文</strong> |
    <a href="README.en.md">English</a>
  </p>
</div>

> **项目状态**：TileOPs-Metax 正在积极开发中，API 可能发生变化。

## 首届开源英才夏令营

夏令营学员应使用 `summer-camp-2026` 分支，并遵循[算子迁移指南](docs/summer-camp/README.zh-CN.md)，完成算子认领、开发实现、MetaX GPU 验证、Benchmark、Roofline 分析和代码提交。

## 项目简介

TileOPs-Metax 是一个基于[TileLang](https://github.com/tile-ai/tilelang)、面向大语言模型训练和推理的GPU 算子库。项目采用规范驱动的开发模式，帮助开发者和 AI Agent 构建、评估和优化高性能算子。

### 主要特性

- **规范驱动**：使用机器可读的 Manifest 定义算子签名和工作负载。
- **性能评估**：结合 Benchmark 与 Roofline 分析评估 Kernel 性能。
- **分层设计**：用户接口与硬件优化策略相互独立。

## 详细文档

- [查看完整中文说明](README.zh-CN.md)
- [View the full English README](README.en.md)
- [查看夏令营算子迁移指南](docs/summer-camp/README.zh-CN.md)
