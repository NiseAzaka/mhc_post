<!--
Language: English
Chinese template: .github/PULL_REQUEST_TEMPLATE/operator-migration.zh-CN.md

PR title format:
  [operator-name] optimize: brief description of this optimization
  [operator-name] feat: brief description of this new feature

Choose either optimize or feat; do not keep both.
-->

## Group Project Information

Project title:

Full project overview: Summarize the group's overall research goal, target
operator, main optimization direction, and expected deliverables.

### PR Summary

Operator name:

Group: Group XX

Members: XXX, XXX, XXX

Change type:

- [ ] `feat`: add a new operator or capability
- [ ] `optimize`: optimize an existing implementation

### 1. Optimization Approach in This PR

Briefly describe the hardware optimization techniques used, such as memory access
optimization, pipelining, tiling, or compute-unit communication optimization.

Primary issue before optimization:

Optimization approach:

Key code or configuration changes:

### 2. Correctness Validation

Reference implementation:

Test command:

Test shape/dtype:

Error tolerance (atol/rtol):

Validation result:

### 3. Performance Data [Required]

Test environment (GPU, driver, MACA, PyTorch, TileLang):

Reproduction command:

Before optimization:

After optimization:

Speedup:

Bottleneck analysis (mcProfiler observations):

### 4. Submission Checklist

- [ ] All test cases pass and correctness meets the required tolerance
- [ ] Temporary debug output and redundant test code have been removed
- [ ] Performance results are locally reproducible and based on valid measurements
