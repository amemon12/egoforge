# EgoForge — executed data-quality and annotation report

Run `60b3583947acfaa4`. All numbers below are computed from the downloaded source files.

Processed **48,272 frames** across **9 episodes**, with **964,596 IMU samples**. Produced **1,804 machine-proposed motion intervals** and a **36-item review queue**. No human-reviewed phase labels are claimed.

**1,852 frames quarantined** by configured rules; **26,514 frames have at least one flag**. Flags indicate review candidates, not verified errors.

## Source and attribution

[Humanola egocentric-sample](https://huggingface.co/datasets/humanola-inc/egocentric-sample), CC BY 4.0. Existing joints, camera poses, keypoints and task labels are inherited. Added outputs include reversible masking/filtering, geometry/time checks, motion proposals and review prioritization. See the source manifest and THIRD_PARTY_NOTICES.md.

## Episode results

| Episode | Source task (reconciled) | Frames | Quarantined | Proposed intervals |
|---|---|---:|---:|---:|
| 0 | ironing clothes | 8,371 | 964 | 384 |
| 1 | folding laundry | 3,931 | 58 | 124 |
| 2 | wiping | 7,201 | 565 | 294 |
| 3 | washing dishes | 9,241 | 87 | 194 |
| 4 | slicing carrots | 2,904 | 0 | 123 |
| 5 | peeling and slicing cucumbers | 4,801 | 0 | 219 |
| 6 | pouring cereal into a container | 3,211 | 89 | 158 |
| 7 | washing dishes | 3,601 | 0 | 117 |
| 8 | cutting an apple | 5,011 | 89 | 191 |

## Cleaning and source conflicts

Source bytes remain unchanged. Derived task indices follow the episode task table; conflicting frame-table values are retained beside them. This override is evidence-based metadata reconciliation and still requires visual semantic review.

- `meta/info.json.total_tasks`: 1 → 8; affected rows: metadata only.

- `task_index`: 0 → 2; affected rows: 7201.

Absent/inconsistent observations are masked, never interpolated across. Smoothing runs only inside valid contiguous spans. No source rows are deleted. Quaternion conversion uses camera-to-world `xyzw`; residuals against inherited camera-space points are internal consistency checks, not independent physical accuracy.

## Synthetic corruption evaluation

**9/9 seeded cases** detected every injected target frame. These are deliberately threshold-visible corruptions and do not estimate real-world precision, semantic label accuracy, or human agreement.

| Injected corruption | Target frames | Detected |
|---|---:|---:|
| world_coordinate_translation_15cm | 90 | 90 |
| image_keypoints_displaced_120px | 90 | 90 |
| nonunit_quaternion_scale_1.2 | 90 | 90 |
| presence_flag_contradiction | 90 | 90 |
| tracking_quality_contradiction | 90 | 90 |
| finger_segment_stretched_20cm | 90 | 90 |
| duplicate_frame_timestamp | 1 | 1 |
| imu_timestamp_frame_clock_mismatch_200ms | 2904 | 2904 |
| imu_burst_removed_1s | 27 | 27 |

## Temporal regularization

Duration-constrained decoding reduces framewise phase switches. This is a stability statistic; a smoother sequence is not necessarily a more accurate one. Full before/after counts are in `ablation.json`. Entropy and margin are uncertainty proxies, not calibrated correctness probabilities.

## Recurrence candidates

The experimental recurrence layer finds periodic wrist motion using principal-axis projection and autocorrelation. A cycle is not a confirmed cut, grasp, or successful action. See `recurrence.json` and the method document.

## Review and acceptance

Use the local review console to inspect source video, inherited skeletons, phase intervals and QA evidence. Approve, reject or correct with an explicit reviewer identifier. Decisions append to a hash-chained log; no proposal becomes human ground truth automatically. The log is tamper-evident within a retained chain, not externally signed or tamper-proof.

## Limits

- Motion proposals are heuristic and uncalibrated; no independent human gold labels.
- Existing 3D/2D poses and task names are inherited from the source.
- No inferred physical contact, force, successful action count, or task success.
- Splits are episode-disjoint only; participant/session IDs unavailable.
- Synthetic fault tests do not establish annotation accuracy.
