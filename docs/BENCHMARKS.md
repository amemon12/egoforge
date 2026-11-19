# Executed benchmarks

Benchmark `54d58eeb004f668b` · 2026-11-19T21:48:09.633010+00:00

Processed **48,272 pose frames** and **964,596 IMU samples** across **9 episodes**, repeated in **3 separate worker processes**.

**11.092 seconds median** per pose/IMU pass (**4,352 frames/s**); observed range 10.913–11.571 s. Maximum whole-worker peak RSS: **783.8 MB**. This excludes video decoding, downloading, exports, manual review and training.

Host: Apple M4 / arm64, 10 logical CPUs, macOS-26.6.2-arm64-arm-64bit; Python 3.12.15. CPU only. Local inputs were cached; no cache flush or workstation-load isolation. The median is computed from whole-pass wall times, not by summing stage medians.

## Stage timing

| Stage | Median s | Observed p95 s | Range s |
|---|---:|---:|---:|
| Parquet load + source hashing | 6.317 | 6.655 | 6.163–6.693 |
| Geometry, cleaning + sensor alignment | 2.129 | 2.158 | 2.119–2.161 |
| Temporal annotation | 1.522 | 1.570 | 1.521–1.575 |
| Repeated-motion detection | 0.622 | 0.641 | 0.609–0.643 |
| Review selection | 0.239 | 0.246 | 0.237–0.246 |

Whole-pass wall time also includes result-signature hashing, ablation accounting and explicit garbage collection (about 0.26 s per pass); these are outside the stage timers.

The p95 column is a descriptive percentile of three measurements, not a validated tail-latency estimate. Stage timing excludes process import startup; each raw pass also records complete subprocess wall time. Peak RSS includes imported libraries and retained allocator pages.

## Sensitivity to injected faults

Executed **126 episode-level sweep cases**: six fault families × seven magnitudes × episodes 2, 4 and 8. Each curve pools target counts, not percentages. Baseline-positive targets are excluded from incremental sensitivity; a zero eligible-target count is reported as null. Source flags remain unadjudicated.

| Fault | Magnitude | Previously negative targets | Newly flagged | Incremental sensitivity |
|---|---:|---:|---:|---:|
| world_translation | 0 m | 270 | 0 | 0.0% |
| world_translation | 0.002 m | 270 | 0 | 0.0% |
| world_translation | 0.005 m | 270 | 0 | 0.0% |
| world_translation | 0.01 m | 270 | 214 | 79.3% |
| world_translation | 0.02 m | 270 | 270 | 100.0% |
| world_translation | 0.05 m | 270 | 270 | 100.0% |
| world_translation | 0.15 m | 270 | 270 | 100.0% |
| keypoint_displacement | 0 px | 270 | 0 | 0.0% |
| keypoint_displacement | 5 px | 270 | 0 | 0.0% |
| keypoint_displacement | 15 px | 270 | 0 | 0.0% |
| keypoint_displacement | 30 px | 270 | 31 | 11.5% |
| keypoint_displacement | 50 px | 270 | 263 | 97.4% |
| keypoint_displacement | 80 px | 270 | 270 | 100.0% |
| keypoint_displacement | 120 px | 270 | 270 | 100.0% |
| bone_extension | 0 m | 268 | 0 | 0.0% |
| bone_extension | 0.002 m | 268 | 0 | 0.0% |
| bone_extension | 0.005 m | 268 | 0 | 0.0% |
| bone_extension | 0.01 m | 268 | 56 | 20.9% |
| bone_extension | 0.02 m | 268 | 208 | 77.6% |
| bone_extension | 0.05 m | 268 | 210 | 78.4% |
| bone_extension | 0.2 m | 268 | 268 | 100.0% |
| quaternion_scale_error | 0 fraction | 270 | 0 | 0.0% |
| quaternion_scale_error | 0.001 fraction | 270 | 0 | 0.0% |
| quaternion_scale_error | 0.005 fraction | 270 | 0 | 0.0% |
| quaternion_scale_error | 0.009 fraction | 270 | 0 | 0.0% |
| quaternion_scale_error | 0.011 fraction | 270 | 270 | 100.0% |
| quaternion_scale_error | 0.02 fraction | 270 | 270 | 100.0% |
| quaternion_scale_error | 0.2 fraction | 270 | 270 | 100.0% |
| shared_imu_clock_shift | 0 ms | 15,116 | 0 | 0.0% |
| shared_imu_clock_shift | 5 ms | 15,116 | 0 | 0.0% |
| shared_imu_clock_shift | 15 ms | 15,116 | 0 | 0.0% |
| shared_imu_clock_shift | 25 ms | 15,116 | 7,457 | 49.3% |
| shared_imu_clock_shift | 40 ms | 15,116 | 15,116 | 100.0% |
| shared_imu_clock_shift | 100 ms | 15,116 | 15,116 | 100.0% |
| shared_imu_clock_shift | 200 ms | 15,116 | 15,116 | 100.0% |
| imu_burst_removal | 0 s | 0 | 0 | — |
| imu_burst_removal | 0.01 s | 3 | 0 | 0.0% |
| imu_burst_removal | 0.025 s | 3 | 0 | 0.0% |
| imu_burst_removal | 0.05 s | 3 | 3 | 100.0% |
| imu_burst_removal | 0.1 s | 9 | 3 | 33.3% |
| imu_burst_removal | 0.25 s | 21 | 21 | 100.0% |
| imu_burst_removal | 1 s | 87 | 87 | 100.0% |

`bone_extension` is the internal family identifier for displacing the left index fingertip along world X; its magnitude is displacement, not a guaranteed increase in bone length. Orientation can make the same displacement lengthen or shorten a segment.

World/keypoint/bone/quaternion perturbations affect three fixed 30-frame windows at 20%, 50% and 80% of each episode. Shared-clock offsets affect the full auxiliary-left IMU timeline. Burst targets are video capture timestamps strictly inside the span of actually removed head-IMU samples; very short bursts can contain no video-frame target. Full rows include baseline flags and newly flagged frames outside the target interval.

At exact numeric thresholds (10 mm world translation and 25 ms clock shift), inherited residuals and floating-point comparisons can split detections; use the neighboring magnitudes to interpret the transition. Burst percentages need not increase monotonically: the number of frame timestamps inside the removed sample span changes with width, and frames near a hole edge may still have a remaining sample within the permitted 25 ms.

The original severe suite was also rerun: **27/27 cases** detected every injected target frame. This separate baseline reports post-fault rule hits and may include pre-existing positives; it is not semantic accuracy or precision.

## Temporal ablation

Raw framewise switches: **10,027**; constrained-decoder switches: **1,457** (85.5% fewer). Short interior runs below the 9-frame dwell: **8,737 → 0**. Episode, gap and unknown-boundary fragments are excluded from that count.

Abstained frames: **1,863 / 48,272 (3.86%)**. Geometric quarantine: 1,852. Produced 1,804 motion proposals and 2 experimental recurrence bouts / 10 side-specific cycles. Repeated passes returned identical state/QA/recurrence signatures: **True**.

These quantify regularization, functional constraints and repeatability. A smoother sequence can still be wrong; no human phase reference exists to measure semantic accuracy.

## Review coverage

The 36-item queue includes 9 episodes and 6 states. Coverage is a scheduling property, not a claim that a person reviewed those examples.

## Reproduce

```bash
uv run python scripts/run_benchmarks.py --repeats 3
```

Committed snapshots of raw passes, benchmark JSON, CSV tables and hashes are in [evidence/benchmarks](../evidence/benchmarks/). Reproduction writes the same artifact types to `output/benchmarks/`. Core module hashes, source manifest/revision, exact config hash, runtime versions and machine details are included. UI/static changes are intentionally outside the computational benchmark fingerprint.

## Limits

- No independent human reference labels: phase accuracy, QA precision and action success are unmeasured.
- Throughput covers cached local pose/IMU loading, QC, temporal annotation, recurrence and review selection; excludes video decode, downloads, export serialization, manual review and training.
- Three isolated worker processes; OS file cache was not flushed. Workstation load was uncontrolled. Import startup is excluded from pipeline timing but recorded separately.
- The reported p95 is a descriptive percentile of only three observations, not a reliable tail-latency estimate or confidence bound.
- Peak RSS is the maximum whole-worker process high-water mark including imports and allocator memory; it is not module-specific or a system-wide memory estimate. MB uses 1,000,000 bytes.
- Corruption sweeps measure rule sensitivity on previously rule-negative injection targets. Untouched-source flags are unadjudicated, not false positives.
- The separate severe suite reports post-fault hits and may include baseline-positive targets; use the baseline-adjusted curves for incremental sensitivity.
- Lower-magnitude faults may deliberately fall below screening thresholds. Neither high synthetic sensitivity nor fewer phase switches establishes semantic accuracy.
