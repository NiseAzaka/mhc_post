# 2026 Summer Camp Operator Migration Guide

[简体中文](README.zh-CN.md) | [**English**](README.en.md)

This project is designed for the in-person summer camp from August 3 to August 6, 2026. Participants select a project from the curated operator list and generalize, adapt, validate, and optimize reusable TileLang kernels for MetaX C500. Each contribution must follow the TileOPs Manifest → Test → Op/Kernel → Benchmark chain of trust and produce reproducible, maintainable open-source results.

> **C500 acceptance baseline:** Code editing, documentation, Manifest validation, and formatting may run elsewhere. Final Kernel compilation and execution, correctness/boundary/error tests, Benchmark, mcProfiler, Roofline measurements, and PR acceptance evidence must come from a real MetaX C500.

## Schedule and Completion Criteria

| Time | Milestone |
|---|---|
| August 3 | Validate the environment, read the contribution guidelines, and claim a primary operator from the curated list |
| August 4 | Submit and pass the fast review for the Manifest PR; open the implementation PR and pass basic correctness tests |
| By 18:00 on August 5 | Bring the implementation PR to a review-ready state with complete test results and C500 performance evidence |
| Evening of August 5 | Teaching assistants complete the initial review and list blocking issues in the PR |
| By 10:30 on August 6 | Resolve all blocking issues; freeze the submitted version and finalize the presentation list at 11:00 |
| Afternoon of August 6 | Present the implementation, correctness evidence, performance optimization, and open-source value |

A task is complete only when its Manifest has been merged and validates, the Op and Kernel layers are clearly separated, correctness/boundary/error tests pass, an independent baseline Benchmark runs, the Roofline formulas and measurements are explainable, the PR template and C500 evidence are complete, and all blocking review comments are resolved.

## 1. Prepare the Environment

You need Python 3.10+, Git, an available MetaX driver/runtime, and a MetaX GPU. Use the container provided by the organizers whenever possible.

> [!TIP]
> **Recommended summer-camp online environment:** Rent MetaX C500 compute through the [Gitee AI Compute Marketplace](https://ai.gitee.com/compute) and select the summer-camp image: `PyTorch-Agent / 2.8.0 / Python 3.12 / MACA 3.7.1.5`. Image availability, pricing, and displayed names are subject to the platform. After creating the instance, complete the environment self-check in Section 1.1 and use `mx-smi` to confirm that the assigned device is a MetaX C500.

> [!WARNING]
> **Do not run `make install`, `pip install tileops`, `pip install -e '.[dev]'` (without
> `--no-deps`), and do not create a venv without `--system-site-packages`.**
>
> The container's TileLang is an in-place source build for MACA (e.g.
> `/opt/tilelang-metax-v0.1.10`), not a pip package — `pip show tilelang` finds nothing.
> pip therefore treats it as "not installed" and pulls the official **CUDA** wheel over it;
> a fresh venv cuts off the MetaX PyTorch build. Either case needs a rebuild or reinstall
> to recover, so both count as destructive.
>
> Likewise, do not pass `-c constraints.txt`: those pins target the CUDA CI runner and would
> downgrade the `apache-tvm-ffi` that `libtilelang.so` is ABI-coupled to. Such a mismatch is
> invisible at `import` time and only fails when the first kernel compiles.

### 1.1 Setup and self-check

`tileops` does not need to be installed. Set `PYTHONPATH` and it imports directly.

Before starting, fork the official repository to your GitLink account, then clone your personal fork. In the commands below, `origin` refers to your personal fork and `upstream` refers to the official repository:

```bash
git clone https://www.gitlink.org.cn/<your-account>/TileOPs-Metax.git
cd TileOPs-Metax
git remote add upstream https://www.gitlink.org.cn/ccf-ai-infra/TileOPs-Metax.git
git fetch upstream
git switch -c summer-camp-2026 --track upstream/summer-camp-2026
git pull --ff-only

# Point at the container's pre-built MACA TileLang (adjust to the actual path), plus this repo
export PYTHONPATH=/opt/tilelang-metax-v0.1.10:$PWD:$PYTHONPATH

# Self-check: TileLang must resolve under /opt/tilelang-metax-*, and the backend must be maca.
# A site-packages path or a cuda backend means pip has overwritten the environment — fix that first
python -c "import tilelang; print(tilelang.__version__); print(tilelang.__file__)"
python -c "from tilelang.utils.target import determine_target; print(determine_target('auto'))"
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
mx-smi

# Verify the repository works
python scripts/validate_manifest.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
```

If you need to run scripts from outside the repository, `--no-deps` is the only safe install form (it stops pip from resolving tilelang):

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

If the environment self-check fails, record the operating system, Python version, driver, MACA version, GPU model, failing command, and exit code. Do not mix environment fixes with an operator migration in the same PR.

### 1.2 Operator availability on MetaX C500

Read this section **before** claiming an operator, or you may pick one that cannot run on C500 at all.

**MACA-specific kernels and dispatch.** This repository ships MACA implementations for some operators, selected at the Op layer through `is_maca()` in `tileops/utils/utils.py`:

```python
# tileops/ops/attention/deepseek_dsa.py
if is_maca():
    kernel_cls = SparseMlaMACAKernel
elif is_hopper():
    kernel_cls = SparseMlaKernel
```

The MACA-specific kernels currently present:

```text
tileops/kernels/gemm_maca.py
tileops/kernels/grouped_gemm/grouped_gemm_persistent_maca.py
tileops/kernels/moe/moe_grouped_gemm_persistent_fused_act_maca.py
tileops/kernels/moe/shared_expert_mlp_maca.py
tileops/kernels/reduction/argreduce_maca.py
tileops/kernels/deltanet/compute_w_u_bwd_maca.py
tileops/kernels/deltanet/deltanet_bwd_maca.py
tileops/kernels/gated_deltanet/gated_deltanet_prefill_maca.py
```

**The arch gate.** On C500, `torch.cuda.get_device_capability()` reports `(8, 0)`, so `get_sm_version()` returns `80`. That number follows NVIDIA's SM encoding and **says nothing about C500's actual architecture** — it only feeds the kernel gate comparison. Do not conclude "C500 behaves like Ampere" from it.

20 kernel declarations exclude `80` (17 with `[90]`, 3 with `[89, 90]`), across these files:

```text
attention/deepseek_dsa_decode.py    attention/deepseek_mla_decode.py
attention/gqa_bwd.py                attention/gqa_decode_bs1.py
attention/gqa_fwd.py                attention/gqa_fwd_fp8.py
attention/gqa_fwd_ws.py             attention/gqa_prefill_fwd_ws.py
attention/gqa_sliding_window_fwd.py attention/gqa_sliding_window_varlen_fwd.py
bmm.py                              gemm.py
deltanet_recurrence.py              gated_deltanet_recurrence.py
grouped_gemm/grouped_gemm_persistent.py
grouped_gemm/grouped_gemm_persistent_3wg.py
moe/moe_grouped_gemm_persistent_3wg_fused_act.py
```

They depend on Hopper-only features such as the warp-specialization barrier intrinsic `ptx_init_barrier_thread_count`. Bypassing the gate does not help — lowering then fails:

```text
tvm.error.InternalError: Unresolved call ir.Op(name="tirx.ptx_init_barrier_thread_count", ...)
```

> [!IMPORTANT]
> **A gated kernel does not mean an unusable Op.** `GemmKernel` in `gemm.py` declares
> `[89, 90]` and is indeed gated on C500 — but `GemmOp` dispatches through `is_maca()` to
> `gemm_maca.py` (`supported_archs = [80, 86, 89, 90]`), so **`GemmOp` works on C500**;
> verified at `M,N,K` of 1024³ and 4096³.
>
> To judge whether an operator is usable on C500, look at **which kernel the Op layer
> actually dispatches to, not the `supported_archs` of one kernel**. The most reliable check
> is to construct the Op and run it.

If an Op still raises the following on C500, it has no MACA dispatch path and is not a suitable migration target:

```text
ValueError: BmmFp8Kernel is not supported on architecture 80
```

Note that this message carries only the number `80` and no device name, which invites the misreading that you are on an NVIDIA Ampere card. When you see `architecture 80` on C500, read it as described above: it is just the return value of `get_sm_version()`.

To survey:

```bash
grep -rn "supported_archs" tileops/kernels/       # gate declarations
grep -rn "is_maca" tileops/ops/                   # Ops that already have MACA dispatch
ls tileops/kernels/**/*maca*.py                   # existing MACA-specific kernels
```

Adding a `*_maca.py` kernel plus `is_maca()` dispatch for an operator that lacks one is a good migration target.

**A usable Op does not mean every shape works.** Reduction operators have a measured shape ceiling on C500. Using `SoftmaxFwdOp` (whose `supported_archs` includes 80, no MACA dispatch needed):

| Input shape | Result |
|---|---|
| `(128, 128)` / `(512, 512)` / `(1024, 1024)` | OK |
| `(4096, 1024)` / `(8192, 1024)` | OK |
| `(1024, 1536)` | fails: `no available layout` (layout inference) |
| `(1024, 2048)` / `(2048, 2048)` / `(4096, 4096)` | fails: `MACALaunch Error: mcErrorInvalidValue` |

The limit is on the **reduction dimension**, not the row count: 8192 rows are fine, while a reduction dimension above 1024 fails. When writing the test matrix and Benchmark workloads, confirm the working range at small sizes before scaling up, and record the measured shape ceiling in your PR evidence.

### 1.3 Known environment issues

**Importing TileLang in both parent and child process triggers SIGKILL.** When a process that has already run `import tilelang` uses `subprocess` to start a child that also imports tilelang, the whole process group is SIGKILLed (`exit 137`, **with no traceback or error output at all**).

As a result, this command aborts with `exit 137` on C500 — it is not a problem with your code:

```bash
python -m pytest -q tests/test_validate_manifest.py     # exit 137 at roughly 59%
```

The validator itself is fine. Run it directly, or deselect the affected test:

```bash
python scripts/validate_manifest.py     # exit 0
python -m pytest -q tests/test_validate_manifest.py --deselect \
  "tests/test_validate_manifest.py::TestIntegration::test_validator_passes_on_current_codebase"
```

Minimal reproduction (for upstream triage):

```bash
# Parent imports tilelang, child imports it too -> SIGKILL
python -c "
import tilelang, subprocess, sys
r = subprocess.run([sys.executable,'-c','import tilelang'], capture_output=True, text=True)
print('rc =', r.returncode)
"
# Fine when either the parent or the child does not import tilelang
```

`benchmarks/benchmark_base.py` and `benchmarks/hardware/memory/hbm_bandwidth.py` also use subprocess, so suspect this issue first if benchmarking dies with a silent `exit 137`.

**The arch gate produces failed, not skipped.** For gated operators, even pure argument validation tests (e.g. `test_bmm_fp8_batch_mismatch_raises`) report `failed` rather than `skipped`, because the `ValueError` is raised during Op construction before the assertion runs. For example `pytest -q -m smoke tests/ops/test_bmm.py` measures `13 failed, 8 passed` on C500. When submitting evidence, state which failures come from the environment gate and which come from your own implementation.

## 2. Claim an Operator

1. Claimable operators are limited to the [curated list](TileKernels-MACA-operator-migration-inventory.en.md) published by the organizing team. Candidates primarily come from the default `dev` branch of [`MetaX-MACA/TileKernels-Metax`](https://github.com/MetaX-MACA/TileKernels-Metax), but operators in the source repository are not automatically open for claiming.
2. Each team should choose one medium- or high-difficulty operator as its primary operator. Before claiming, analyze parent-child, containment, dependency, and core implementation relationships. Projects that share the main implementation, core Kernel, or most tests and Benchmarks should not be assigned to different teams.
3. Each team must follow [Operator Claim Instructions Issue #1](https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax/issues/1) to create a separate operator-claim Issue with its team number, operator name, source file path, and source commit SHA. A claim becomes valid only after the information is complete, no conflict exists, and a teaching assistant confirms it; if multiple teams claim the same operator, the first complete Issue confirmed by a teaching assistant takes precedence.
4. If the source implementation is incomplete, required dependencies are missing, or the migration scope is too large, explain the problem in the Issue immediately. Do not switch operators without notice.

**After the primary operator's Manifest PR has been merged, its implementation PR has passed the core correctness tests with no blocking issue, and a teaching assistant has confirmed the status, a team may claim up to two additional operators, for a maximum of three operators in total. Each operator must be claimed and submitted separately. Teams must not reserve operators they have not started.**

## 3. Build the Trust Chain with Two PRs

This section defines the responsibilities and order of the two PRs: PR A establishes the specification, while PR B supplies the implementation, tests, and performance evidence. PR A and PR B use the same `feat/<operator-id>` branch, and both must link the team's operator-claim Issue. See Section 7 for submission formats and checks.

Create the development branch from the latest `summer-camp-2026`:

```bash
git switch summer-camp-2026
git pull --ff-only
git switch -c feat/<operator-id>
```

### PR A: Manifest

PR A defines the operator interface, dtypes, shape rules, workloads, and Roofline formulas before implementation. It is the shared contract for the implementation, tests, and Benchmark.

In the first stage, submit only:

- the new operator entry in `tileops/manifest/<family>.yaml`; create a new Manifest file only when no existing family is appropriate;
- Manifest validation or necessary contract tests;
- an explanation of inputs/outputs, shapes, dtypes, workloads, and Roofline formulas.

A new Manifest must start with `status: spec-only`. PR A requires a fast review by a teaching assistant or maintainer and may be merged after Manifest validation passes. PR A does not review the Op, Kernel, performance, or C500 data.

Do not commit implementation code to this branch before PR A is merged.

### PR B: Implementation

After PR A is merged, continue using the original `feat/<operator-id>` branch. Because merging PR A may produce a new commit SHA, skip the local PR A commit and rebase the subsequent implementation onto the latest target branch:

```bash
git switch feat/<operator-id>
git fetch upstream
git rebase --onto upstream/summer-camp-2026 <local-pr-a-sha> feat/<operator-id>
git push --force-with-lease origin feat/<operator-id>
```

`<local-pr-a-sha>` is the commit SHA used to submit PR A from this branch. After the rebase, only the PR B implementation commits should remain on top of the target branch.

After synchronization, submit:

- a stateless Op under `tileops/ops/`;
- a TileLang Kernel under `tileops/kernels/`;
- correctness, boundary, and error tests under `tests/`;
- an independent baseline Benchmark under `benchmarks/ops/`;
- only the Manifest status, provenance, and workload fields that may be updated with the implementation.

After completing the implementation, tests, and C500 performance validation, create PR B from the same branch.

Do not include unrelated refactoring, dependency upgrades, or multiple operators in one PR.

## 4. Migration Requirements

- Fix the reference semantics and failure behavior before writing the implementation.
- The Op owns argument validation, dtype/layout handling, and Kernel dispatch. The Kernel owns device computation and is not the user-facing interface.
- Do not copy test conclusions from the source repository. Rebuild the evidence through this repository's test entry points.
- Keep each cross-file change as one minimal closed loop: one Manifest, one Op, one or a small number of strategy Kernels, one test group, and one Benchmark.
- Tests should first fail because the target behavior is missing, then pass after the implementation is added. Path and syntax errors are not valid failures.
- Do not use PyTorch or another high-level framework on the host to replace device computation that belongs in the TileLang Kernel.

## 5. Correctness and Testing

The minimum test matrix includes:

- representative regular shapes;
- non-tile-aligned and minimum boundary shapes;
- every declared supported dtype;
- non-contiguous input when supported by the interface;
- explicit exceptions for invalid dimensions, dtypes, and shapes;
- comparison with an independent PyTorch reference, including `atol`/`rtol`;
- final GPU tests on a real MetaX C500.

Common commands (make sure `PYTHONPATH` is set as in Section 1.1 first):

```bash
python scripts/validate_manifest.py
python -m pytest -q tests/<test_file>.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py
pre-commit run --all-files
```

`tests/test_validate_manifest.py` aborts with `exit 137` on C500 due to a known environment issue; see Section 1.3 for how to handle it.

Record the tested commit SHA, complete commands, exit codes, and concise results in the PR. Do not commit large raw logs.

## 6. Benchmark, mcProfiler, and Roofline

The Benchmark must be separate from correctness tests and use an independent baseline, normally a PyTorch primitive or a clear reference composition. Benchmark, mcProfiler, and Roofline measurements must run on a real MetaX C500. Record at least:

- warmup count, measurement count, synchronization method, and statistic;
- input shape, dtype, layout, and device;
- TileOPs latency, baseline latency, and speedup;
- the main bottleneck observed in mcProfiler and how the optimization addresses it;
- FLOPs and bytes required by the Manifest Roofline formulas;
- the `achieved / theoretical` ratio and bottleneck assessment;
- the tested commit SHA, complete commands, software versions, driver version, and GPU details;
- the sGPU slice quota (see below).

> [!IMPORTANT]
> **Mind the sGPU slice.** Your container may hold a GPU slice rather than the whole card. When
> reading `mx-smi`, do not stop at the whole-card memory in the first section (e.g. 65536 MiB) —
> check `Vram Quota` and the `Compute` percentage in the Sliced GPU section, for example a
> 16000 MiB quota at 25% compute. `torch.cuda.get_device_properties(0).total_memory` reports the
> slice value too.
>
> Roofline evidence must record the slice quota and state whether `P_peak` / `BW_peak` are
> whole-card figures or scaled to the slice. Dividing a slice measurement by a whole-card
> theoretical peak yields an `achieved / theoretical` ratio too low to explain. Large Manifest
> workloads may also OOM within a slice's memory.

Do not use the tested implementation as its own baseline, report only the fastest sample, or include compilation time in steady-state latency.

## 7. Submit PRs

This section defines the title, description, template, and pre-submission checks for each PR. See Section 3 for their scope and order. Both PRs must link the operator-claim Issue.

### Project Naming

The project title is used for the presentation and awards, while each PR title identifies a specific operator.

- For a single-operator project, use: `Migration and Optimization of <operator-name> on MetaX C500`.
- For a multi-operator project, name the project after the operator family or shared function, and identify the primary and additional operators.
- Each PR title must still name its corresponding operator, following the one-operator-per-PR rule.

Example:

```text
Project title: Migration and Optimization of Multiple MoE Routing Operators on MetaX C500
Primary operator: moe_group_count
Additional operators: moe_normalize_weight, moe_reduce_fused

PR titles:
[moe_group_count] feat: add routing group-count operator
[moe_normalize_weight] feat: add weight-normalization operator
```

### PR A: Manifest PR

PR A must use the repository's [Manifest PR Template](../../.github/PULL_REQUEST_TEMPLATE/operator-manifest.en.md).

Recommended title:

```text
[operator-name] feat: add spec-only Manifest
```

PR A must describe the operator name, source file path, source commit SHA, interface, workloads, and Roofline formulas. It does not require correctness, performance, mcProfiler, or C500 evidence.

Before submission:

```bash
git diff --check
python scripts/validate_manifest.py
```

### PR B: Implementation PR

PR B must use the repository's [Operator Migration PR Template](../../.github/PULL_REQUEST_TEMPLATE/operator-migration.en.md) in full. Do not remove required sections.

Recommended titles:

```text
[operator-name] feat: brief description of the new feature
[operator-name] optimize: brief description of the optimization
```

Use `feat` for a newly added operator and `optimize` for an existing implementation.

Before submission:

```bash
git diff --check
python scripts/validate_manifest.py
python -m pytest -q <operator-test>
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

PR B must completely describe the group project, optimization approach, correctness validation, before/after performance, speedup, and mcProfiler bottleneck analysis. Preserve the operator name, source file path, source commit SHA, tested commit SHA, complete test commands, and MetaX C500 evidence.

### Pre-submission Self-check

Before marking PR B as ready for review, each team must complete the template in [PR Pre-submission Check Issue #4](https://gitlink.org.cn/ccf-ai-infra/TileOPs-Metax/issues/4). Incomplete items must be reported honestly and must not be checked prematurely.

## 8. Review and Presentation

PR A receives a fast review focused on the Manifest interface, shapes/dtypes, workloads, Roofline formulas, and validation result.

PR B receives a full review:

- Technical Review checks reference semantics, Op/Kernel separation, the test matrix, Benchmark fairness, mcProfiler analysis, and Roofline interpretation.
- Process Review checks the claim status, PR scope, template completeness, reproducibility of C500 evidence, and blocking items.

For the final five-minute presentation, explain:

1. what problem the operator solves;
2. where it came from and what changed during migration;
3. how correctness was established;
4. how it performs on MetaX C500 and how far it is from Roofline;
5. which optimization is most valuable next and how other contributors can reuse the result.

When blocked, post the command, exit code, minimal log, and attempted fixes in the claim Issue, then mention the maintainer on duty. Never commit passwords, tokens, private keys, container addresses, or complete environment variables.
