# Summer Camp Review Gate

A contribution is merge-ready only when all applicable items pass:

- claim Issue, source file/SHA, and License are traceable;
- a new specification entered as `spec-only`;
- Manifest signature, constraints, workloads, FLOPs, and bytes validate;
- Op/Kernel responsibilities and supported range match the Manifest;
- independent-reference tests cover representative, boundary, dtype, and
  invalid-input cases;
- real MetaX GPU evidence names the tested commit and environment;
- performance uses an independent baseline, correct synchronization, absolute
  latency, and reproducible Roofline calculations;
- manifest, related tests, benchmark contracts, pre-commit, and diff checks pass;
- the PR contains no credentials, host addresses, caches, or raw large logs;
- risks, limitations, and rollback are stated.

Technical reviewers approve semantics, TileLang, tests, and performance.
Process reviewers approve claim state, scope, evidence completeness, and
follow-up tracking. Any blocking comment must state a concrete risk and a
verifiable completion condition.