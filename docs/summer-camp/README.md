# 首届开源英才夏令营算子迁移指南

本专项面向 2026 年 8 月 3 日至 8 月 6 日线下夏令营。目标是在真实沐曦 MetaX C500 上，将 `MetaX-MACA/TileKernels-Metax` 默认 `dev` 分支中适合开放且尚未迁入的 TileLang Kernel，按照 Manifest → Test → Op/Kernel → Benchmark 信任链迁入本仓库，并留下可复现、可复用的验证证据。

## 时间安排与完成标准

| 时间 | 里程碑 |
|---|---|
| 8 月 3 日 | 完成环境验证、阅读规范，并认领一个尚未迁移的算子 |
| 8 月 4 日 | 提交并通过 Manifest PR 的快速 Review；创建实现 PR，并通过基础正确性测试 |
| 8 月 5 日 18:00 前 | 实现 PR 达到可 Review 状态，测试和 C500 性能验证证据完整 |
| 8 月 5 日晚 | 助教完成初步检查，并在 PR 中列出需要修复的阻塞问题 |
| 8 月 6 日 10:30 前 | 完成阻塞问题修复；11:00 冻结参评版本并确定答辩名单 |
| 8 月 6 日下午 | 进行成果答辩，展示算子实现、正确性、性能优化和开源价值 |

推荐通过[模力方舟算力市场](https://ai.gitee.com/compute)租用在线 MetaX C500 算力，并选择夏令营专属镜像：`PyTorch-Agent / 2.8.0 / Python 3.12 / MACA 3.7.1.5`。具体使用方法和环境自检要求见[详细中文指南](README.zh-CN.md)。

- [详细中文指南](README.zh-CN.md)
- [English](README.en.md)
