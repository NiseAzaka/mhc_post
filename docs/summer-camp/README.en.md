# 2026 Summer Camp Operator Migration Guide

[简体中文](README.zh-CN.md) |
[**English**](README.en.md)

This project is designed for the in-person summer camp from August 3 to August 6,
2026. Its goal is to migrate suitable TileLang kernels from
`TileKernels-Metax` into this repository on MetaX GPUs, following the TileOPs
Manifest → Test → Op/Kernel → Benchmark chain of trust and preserving reusable
validation evidence.

## Schedule and Definition of Done

| Date | Milestone |
| --- | --- |
| August 3 | Validate the environment, read the contribution guide, and claim an available operator |
| August 4 | Submit and pass the Manifest PR; make the implementation PR pass correctness tests |
| August 5 | Complete boundary/error tests, Benchmark, and Roofline analysis |
| Noon, August 6 | Make the PR ready for Review with complete evidence |
| Afternoon, August 6 | Present the operator, correctness evidence, performance results, and optimization assessment |

A task is complete only when all of the following conditions are met: the Manifest
passes validation; the Op and Kernel layers are separated; correctness, boundary,
and error-path tests pass; an independent baseline Benchmark runs successfully;
the Roofline formulas and measurements are explainable; the PR template has no
empty required sections; and all blocking Review comments are resolved.

## 1. Prepare the Environment

You need Python 3.10+, Git, an available MetaX driver/runtime, and a MetaX GPU.
Use the container provided by the organizers whenever possible.

```bash
git clone --recurse-submodules \
  --branch summer-camp-2026 \
  https://gitlink.org.cn/Beckylu/TileOPs-Metax.git
cd TileOPs-Metax

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
PIP_NO_BUILD_ISOLATION=1 python -m pip install -e '.[dev]' -v

python scripts/validate_manifest.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py tests/test_validate_manifest.py
```

If the base installation fails, record the operating system, Python version,
driver, MACA version, GPU model, failing command, and exit code. Do not mix
environment fixes and operator changes in the same PR.

## 2. Claim an Operator

1. Open the candidate operator list published by the organizers and select only
   an operator marked “待迁移” (ready for migration).
2. Comment on the claim Issue with your name, operator ID, expected completion
   time, and whether you need a partner.
3. Wait for a maintainer to mark the operator as claimed before starting, to
   prevent duplicated work.
4. If the source implementation is incomplete, dependencies are missing, or the
   scope is too large, report it in the Issue immediately. Do not silently switch
   tasks.

Difficulty is a scheduling reference, not a lower quality bar. First-time contributors should prefer one- or two-star operators with fewer shapes and dtypes and an existing PyTorch reference implementation.

## 3. Build the Trust Chain with Two PRs

### PR A: Manifest

Create `manifest/<operator-id>` from `summer-camp-2026`:

```bash
git switch summer-camp-2026
git pull --ff-only
git switch -c manifest/<operator-id>
```

Submit only:

- `tileops/manifest/<operator-id>.yaml`;
- Manifest validation or contract tests when necessary;
- an explanation of the workload, inputs/outputs, and Roofline formulas.

A new Manifest must start with the `spec-only` status. Create the implementation
branch only after the Manifest PR is merged.

### PR B: Implementation

Create `feat/<operator-id>` from the target branch that contains the merged
Manifest. Submit:

- a stateless Op under `tileops/ops/`;
- a TileLang Kernel under `tileops/kernels/`;
- correctness, boundary, and error tests under `tests/`;
- an independent baseline Benchmark under `benchmarks/ops/`;
- only the Manifest status, provenance, and workload fields that may be updated
  with the implementation.

Do not include unrelated refactoring, dependency upgrades, or multiple operators
in one PR.

## 4. Migration Requirements

- Fix the reference semantics and failure behavior before writing the
  implementation.
- The Op owns argument validation, dtype/layout handling, and Kernel dispatch.
  The Kernel owns device computation and is not the user-facing interface.
- Do not copy test conclusions from the source repository. Rebuild evidence
  through this repository's test entry points.
- When changing multiple files, keep one minimal closed loop: one Manifest, one
  Op, one or a small number of strategy Kernels, one test group, and one
  Benchmark.
- Make tests fail for the missing behavior before implementing it. A path or
  syntax error is not a valid failing test.

## 5. Correctness and Testing

The minimum test matrix includes:

- representative regular shapes;
- non-tile-aligned and minimum boundary shapes;
- every declared supported dtype;
- non-contiguous input when supported by the interface;
- explicit exceptions for invalid dimensions, dtypes, and shapes;
- comparison with an independent PyTorch reference, including `atol`/`rtol`;
- execution on a real MetaX GPU.

Common commands:

```bash
python scripts/validate_manifest.py
python -m pytest -q tests/<test_file>.py
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

Paste commands and concise results into the PR. Do not commit large raw logs.

## 6. Benchmark and Roofline

The Benchmark must be separate from correctness tests and must use an independent baseline, normally a PyTorch primitive or a clear reference composition. Include at least:

- warmup count, measurement count, and synchronization method;
- input shape, dtype, layout, and device;
- TileOPs latency, baseline latency, and speedup;
- FLOPs and bytes required by the Manifest Roofline formulas;
- the `achieved / theoretical` ratio and bottleneck assessment;
- the original command, commit SHA, software versions, driver, and GPU details.

Do not use the tested implementation as its own baseline, report only the fastest sample, or include compilation time in steady-state latency.

## 7. Submit the PR

Use the repository's standard PR template. Recommended titles:

```text
[operator-name] feat: brief description of this new feature
[operator-name] optimize: brief description of this optimization
```

Choose either `feat` or `optimize` according to the type of change.

Before submission:

```bash
git diff --check
python scripts/validate_manifest.py
python -m pytest -q <operator-test>
python -m pytest -q benchmarks/tests
pre-commit run --all-files
```

The PR must link the claim Issue and completely describe the group project, optimization approach, correctness validation, before/after performance, speedup, and mcProfiler bottleneck analysis. It must also preserve the source file, source
commit SHA, test commands, and MetaX hardware evidence.

## 8. Review and Presentation

Technical Review checks the Manifest, reference semantics, Op/Kernel separation, test matrix, Benchmark fairness, and Roofline interpretation. Process Review checks the claim status, PR scope, template completeness, evidence reproducibility, and blocking items.

For the final five-minute presentation, explain:

1. what problem the operator solves;
2. where it came from and what changed during migration;
3. how correctness was established;
4. how it performs on MetaX C500 and how far it is from Roofline;
5. which optimization is most valuable next.

When blocked, post the command, exit code, minimal log, and attempted fixes in the claim Issue, then mention the maintainer on duty. Never commit passwords, Tokens, private keys, container addresses, or complete environment variables.
