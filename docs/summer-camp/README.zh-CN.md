# 首届开源英才夏令营算子迁移指南

[**简体中文**](README.zh-CN.md) |
[English](README.en.md)

本专项面向 2026 年 8 月 3 日至 8 月 6 日线下夏令营。目标是在 MetaX GPU 上，把 `TileKernels-Metax` 中适合开放的 TileLang Kernel 按TileOPs 的 Manifest → Test → Op/Kernel → Benchmark 信任链迁入本仓库，并留下可复用的验证证据。

## 时间与完成定义

| 时间 | 里程碑 |
| --- | --- |
| 8 月 3 日 | 完成环境验证、阅读规范、认领一个未被占用的算子 |
| 8 月 4 日 | 提交并通过 Manifest PR；实现 PR 至少通过正确性测试 |
| 8 月 5 日 | 完成边界/异常测试、Benchmark 和 Roofline 分析 |
| 8 月 6 日中午 | PR 达到可 Review 状态，证据齐全 |
| 8 月 6 日下午 | 演示算子、正确性、性能结果和优化判断 |

一个任务只有在以下条件全部满足后才算完成：Manifest 可校验；Op 与
Kernel 分层；正确性、边界和错误路径测试通过；独立基线 Benchmark
可运行；Roofline 公式和实测结果可解释；PR 模板无空项；Review 阻塞项已关闭。

## 1. 准备环境

需要 Python 3.10+、Git、可用的 MetaX 驱动/运行时和 MetaX GPU。推荐在筹备组提供的容器中工作。

```bash
git clone --recurse-submodules \
  --branch summer-camp-2026 \
  https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax.git 
cd TileOPs-Metax

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
PIP_NO_BUILD_ISOLATION=1 python -m pip install -e '.[dev]' -v

python scripts/validate_manifest.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py tests/test_validate_manifest.py
```

如果基础安装失败，先记录操作系统、Python、驱动、MACA、GPU 型号、失败命令和退出码；不要在同一个 PR 中顺手修改环境与算子。

## 2. 认领算子

1. 查看筹备组发布的候选算子清单，只选择状态为“待迁移”的行。
2. 在认领 Issue 留言：姓名、算子 ID、预计完成时间、是否需要结对。
3. 由维护者把状态改为“已认领”后再开始，避免重复开发。
4. 发现源实现不完整、依赖缺失或范围过大时，立即在 Issue 说明；不要静默换题。

难度是排期参考，不代表质量门槛。首次贡献建议选择“一星/二星”、形状和 dtype
较少、已有 PyTorch 参考实现的算子。

## 3. 使用两个 PR 建立信任链

### PR A：Manifest

从 `summer-camp-2026` 创建 `manifest/<operator-id>`：

```bash
git switch summer-camp-2026
git pull --ff-only
git switch -c manifest/<operator-id>
```

只提交：

- `tileops/manifest/<operator-id>.yaml`；
- Manifest 校验或契约测试（确有必要时）；
- 对工作负载、输入输出和 Roofline 公式的说明。

新 Manifest 初始状态必须是 `spec-only`。Manifest PR 合入后，再创建实现分支。

### PR B：实现

从已合入 Manifest 的目标分支创建 `feat/<operator-id>`，提交：

- `tileops/ops/` 下无状态 Op；
- `tileops/kernels/` 下 TileLang Kernel；
- `tests/` 下正确性、边界和异常测试；
- `benchmarks/ops/` 下独立基线 Benchmark；
- Manifest 中允许随实现更新的状态/来源/工作负载字段。

不要把不相关重构、依赖升级或多个算子塞进同一个 PR。

## 4. 迁移要求

- 先固定参考语义和失败行为，再写实现。
- Op 负责参数校验、dtype/layout 处理和调用 Kernel；Kernel 不承担用户接口职责。
- 不直接复制源仓库测试结论；用当前仓库的测试入口重新建立证据。
- 需要跨文件修改时，保持一个最小闭环：一个 Manifest、一个 Op、一个或少量策略 Kernel、一组测试、一个 Benchmark。
- 所有测试先失败后实现；失败必须由功能缺失造成，而不是路径或语法错误。

## 5. 正确性与测试

最低测试矩阵包括：

- 代表性常规形状；
- 非整块、最小值等边界形状；
- 所有声明支持的 dtype；
- 非连续输入（接口声明支持时）；
- 非法维度、dtype、shape 的明确异常；
- 与独立 PyTorch 参考实现比较，注明 `atol`/`rtol`；
- 在真实 MetaX GPU 上运行。

常用命令：

```bash
python scripts/validate_manifest.py
python -m pytest -q tests/<test_file>.py
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

在 PR 中粘贴命令和简洁结果，不提交巨大的原始日志。

## 6. Benchmark 与 Roofline

Benchmark 必须与正确性测试分开，并使用独立基线（通常为 PyTorch
原语或清晰的参考组合）。至少包含：

- 预热次数、测量次数和同步方式；
- 输入 shape、dtype、布局和设备；
- TileOPs 延迟、基线延迟和加速比；
- Manifest Roofline 公式所需的 FLOPs、读写字节数；
- `achieved / theoretical` 比值及瓶颈判断；
- 原始命令、提交 SHA、软件/驱动/GPU 信息。

不要用被测实现充当基线，不要只报告最快一次，也不要把编译时间混入稳定态延迟。

## 7. 提交 PR

使用仓库统一 PR 模板，标题建议：

```text
[算子名] feat: 本次新增功能简短说明
[算子名] optimize: 本次优化简短说明
```

根据改动性质在 `feat` 和 `optimize` 中二选一。

提交前：

```bash
git diff --check
python scripts/validate_manifest.py
python -m pytest -q <本算子测试>
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

PR 必须链接认领 Issue，完整填写小组课题信息、优化方案、精度验证、优化前后性能、加速比和 mcProfiler 瓶颈分析，并保留源文件、源提交 SHA、测试命令和 MetaX 硬件证据。

## 8. Review 与汇报

技术 Review 依次检查 Manifest、参考语义、Op/Kernel 分层、测试矩阵、Benchmark 公平性和 Roofline 解释。流程 Review 检查认领状态、PR 范围、模板完整度、证据可复现性和阻塞项。

最终汇报建议用五分钟说明：

1. 算子解决什么问题；
2. 从哪里迁移、改了什么；
3. 如何证明正确；
4. 在 MetaX C500 上表现如何、离 Roofline 多远；
5. 下一步最值得优化什么。

遇到阻塞时，在认领 Issue 中给出“命令 + 退出码 + 最小日志 + 已尝试方法”，并 @当值维护者。切勿提交密码、Token、私钥、容器地址或完整环境变量。
