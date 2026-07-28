# Summer Camp Roofline Checklist

For every workload, define semantic FLOPs `F` and minimum global-memory bytes
`B`, including dtype size and all required input/output streams. Do not use
implementation tile sizes in the specification.

Given peak compute `P_peak` and memory bandwidth `BW_peak`:

```text
t_compute = F / P_peak
t_memory = B / BW_peak
t_roofline = max(t_compute, t_memory)
roofline_efficiency = t_roofline / measured_latency
```

For transpose/copy-like operators, `F` may be zero and the useful bound is
`B / BW_peak`.

Record the source and units for device peaks, warmup/measurement iterations,
synchronization, workload, commit SHA, software versions, absolute latency,
independent-baseline latency, speedup, and Roofline efficiency. Exclude
compilation from steady-state latency. If efficiency exceeds one, audit units,
traffic/FLOP counts, peak values, and synchronization before drawing a
performance conclusion.

## sGPU slices

Record the sGPU slice quota alongside the device peaks. A container may hold a
GPU slice rather than the whole card: check the Sliced GPU section of `mx-smi`
for `Vram Quota` and the `Compute` percentage (for example 16000 MiB at 25%
compute, where the first section still shows 65536 MiB for the whole card).
`torch.cuda.get_device_properties(0).total_memory` reports the slice value.

State explicitly whether `P_peak` and `BW_peak` are whole-card figures or scaled
to the slice, and keep that choice consistent across every workload. Dividing a
slice measurement by a whole-card theoretical peak produces an efficiency far
below one for reasons that have nothing to do with the kernel.
