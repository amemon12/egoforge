# EgoForge data card

Version: 0.1.0. Source audit: 19 November 2026.

## Purpose and scope

EgoForge adds an auditable temporal motion annotation and quality-control layer to an existing egocentric demonstration dataset. The technical problem is recovering useful bimanual motion intervals while exposing inconsistent metadata, tracking interruptions, implausible kinematics, geometric residuals, and synchronization uncertainty. The new labels are automatic proposals requiring adjudication. They are not human-certified activity or contact ground truth.

The input is [Humanola, Inc., Humanola Egocentric Hand-Pose Dataset — Sample Delivery](https://huggingface.co/datasets/humanola-inc/egocentric-sample), pinned at revision `3c6e782911b4e82a56b288f6830d3a43bdb85d49`. The publisher's dataset card declares [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Its original card, repository metadata, file tree, and checksums are preserved alongside downloaded files.

## Material actually acquired

| Material | Local coverage |
|---|---:|
| Pose episodes | 9 of 9 |
| Pose timesteps | 48,272 |
| Nominal duration, sum of frame counts / 30 | 1,609.067 seconds, or 26.818 minutes |
| Distinct tasks in task dictionary | 8 |
| IMU sidecars | 27, three per episode |
| IMU samples | 964,596 |
| RGB video streams | Four cameras for episode 4 only |
| Episode 4 visual coverage | 2,904 frames per stream, 96.8 seconds, 30 fps, 1920 × 1200 |
| Original downloaded source files | 47 |
| Original source bytes | 787,644,615 |

The byte and file counts exclude the three locally generated provenance JSON files. The full upstream release contains additional videos and visualization aids that are not part of this local subset. Dataset-wide pose/IMU statistics must not be described as dataset-wide visual inspection.

| Episode | Task from episode metadata | Frames | Nominal duration (s) | Absent-hand observations |
|---|---|---:|---:|---|
| 0 | ironing clothes | 8,371 | 279.033 | None |
| 1 | folding laundry | 3,931 | 131.033 | Left at frame 3930 |
| 2 | wiping | 7,201 | 240.033 | Right at frames 7199 and 7200 |
| 3 | washing dishes | 9,241 | 308.033 | None |
| 4 | slicing carrots | 2,904 | 96.800 | None |
| 5 | peeling and slicing cucumbers | 4,801 | 160.033 | None |
| 6 | pouring cereal into a container | 3,211 | 107.033 | None |
| 7 | washing dishes | 3,601 | 120.033 | None |
| 8 | cutting an apple | 5,011 | 167.033 | None |

Duration is sample count divided by frame rate; first-to-last timestamp span is one frame shorter. "Absent" uses the source `hand_present` flags, not a fresh visual judgment.

## Inherited versus newly derived information

The publisher supplies videos, task descriptions, 21 joints per hand in camera and world coordinates, 2D keypoints, hand boxes, 6-DoF head and wrist poses, finger angles, grip closure, hand-presence flags, tracking-quality codes, calibration, and IMU measurements. EgoForge does not claim to have recorded those videos, estimated the original poses, or manually annotated those labels.

EgoForge computes derived motion features, temporal proposals, consistency checks, cleaning decisions, and review priorities. Any pose smoothing or metadata normalization is a derived representation; raw inputs remain intact. A proposed motion category, geometric proximity, or grip closure is insufficient evidence for object contact, object identity, task success, causal intent, or safe execution. No original object-contact annotations, success labels, force measurements, or independent motion-capture reference are supplied in this subset.

## Independent source audit findings

1. `meta/info.json` declares `total_tasks = 1`, but `meta/tasks.jsonl` contains eight distinct entries.
2. Every one of episode 2's 7,201 parquet rows contains `task_index = 0`; its episode record specifies index 2, "wiping". This is a source disagreement. Resolving to episode metadata must retain both source values and the decision in provenance.
3. Episode 2 frame 5901 has both hands marked present and nonzero joints, but `tracking_quality = 2`, documented as right hand absent. It must remain a reviewable contradiction, rather than being silently treated as confirmed absence.
4. The three zero-filled missing-hand observations are episode 1 frame 3930, and episode 2 frames 7199 and 7200. They are not valid world-origin measurements.
5. All nine pose frame-index arrays are consecutive from zero, nominal timestamps follow `frame_index / 30`, and all 188-dimensional state values are finite. All 27 IMU timestamps and fractional-frame arrays are strictly increasing; no IMU gap exceeds three times that stream's median spacing. These checks establish structural integrity, not label correctness.

Head quaternions are normalized to float32 precision. On episode 4's left hand, transforming camera-space joints with the supplied camera-to-world pose reproduces supplied world joints within approximately `1.1e-7 m`. This is internal consistency of two source-derived representations, not independent 3D accuracy. Pinhole projection against source 2D keypoints has median residual 2.05 pixels and 95th-percentile residual 6.52 pixels across that hand's joints.

## Temporal and geometric conventions

World positions are in meters, with the publisher declaring world up along −Y. Head pose is `[tx, ty, tz, qx, qy, qz, qw]`, mapping camera to world. Main stereo images are declared rectified. Intrinsics vary by camera and are stored by episode. Wrist-camera extrinsics relative to the tracked wrist are not established merely by the existence of a wrist pose.

The source documents IMU `timestamp_ns` as its canonical shared-clock key. Pose `timestamp` is nominal video time. Recover episode-specific frame capture times from each IMU stream's `timestamp_ns` versus fractional `frame`; do not apply the single metadata `clock_anchor_ns` to every episode. Preserve integer nanoseconds and subtract an integer anchor before conversion to floating point.

An exploratory gyro/pose magnitude comparison finds best correlations at different lags in different episodes; the diagnostic and its limits are recorded in [PROVENANCE.md](PROVENANCE.md). No universal timing correction is justified by these results.

## Evaluation and generalization limits

There is no adjudicated human gold set for the new temporal labels yet. A QA score measures rule satisfaction, not annotation accuracy. Internal agreement among source poses, projections, and derived labels is not independent validation. The dataset's source estimation method and uncertainty are not reconstructed by this project.

The acquired metadata does not provide a reliable subject identifier or complete session grouping. An episode-disjoint split prevents direct frame overlap but cannot guarantee subject-disjoint or session-disjoint evaluation. Repeated tasks, near-identical environments, and correlated neighboring frames may cause leakage. Eight tasks and nine short episodes provide limited diversity; this is a technical pilot, not a broad benchmark of generalization.

Do not infer demographic attributes or identities from the footage. The public card's license is retained; this project does not independently verify capture consents or provenance beyond the publisher's release. Redistributed source footage, frames, or source-derived material must retain the source attribution and applicable license notice.
