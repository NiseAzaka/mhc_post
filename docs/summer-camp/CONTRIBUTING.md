# Summer Camp Operator Contribution Workflow

Use the trust chain `Manifest → Test → Op/Kernel → Benchmark`.

1. Claim exactly one `待迁移` item in the published task list.
2. Submit a Manifest-only PR with status `spec-only`.
3. After the specification is accepted, create an implementation PR.
4. Add a failing behavioral test before each implementation increment.
5. Keep the Op stateless and responsible for validation/layout/dtype/dispatch;
   keep device computation in the TileLang Kernel.
6. Verify correctness and error paths on a real MetaX GPU.
7. Add an independent performance baseline and explain the Manifest Roofline.
8. Complete the repository PR template and obtain technical plus process review.

Minimum commands:

```bash
python scripts/validate_manifest.py
python -m pytest -q tests/<operator-test>.py
python -m pytest -q benchmarks/tests
python -m pytest -q tests/test_ops_manifest.py tests/test_validate_manifest.py
pre-commit run --all-files
```

A valid evidence block names the tested commit, GPU, driver/MACA,
Python/PyTorch/TileLang, exact command, exit code, and concise result. Never
post credentials, container addresses, private environment variables, or raw
large logs.