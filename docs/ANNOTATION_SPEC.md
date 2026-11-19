# Annotation contract

## Units and coordinate conventions

Native frame indices are 0-based. Intervals are `[start_frame, end_frame_exclusive)`, with seconds following the nominal 30 FPS video timeline. World coordinates are source gravity-aligned meters (up is −Y). Head pose is camera→world with quaternion order `qx,qy,qz,qw`. Camera-space XYZ and source 2D keypoints are inherited observations. An absent hand is missing data, never an all-zero hand at the origin.

## Motion vocabulary

| State | Observable interpretation | Prohibited inference |
|---|---|---|
| `quiescent` | Both wrist speeds provide low-motion evidence | The person is idle or an object is held |
| `left_dominant` | Left wrist motion dominates | The left hand touches or controls a particular object |
| `right_dominant` | Right wrist motion dominates | A tool is successfully operated |
| `bilateral_coordinated` | Both wrists move with similar velocity direction | Cooperation proves physical contact |
| `bilateral_independent` | Both wrists move without strong directional alignment | The motions have separate goals |
| `unknown` | Insufficient usable evidence or quality rejection | Any class inferred through the gap |

World-frame wrist motion reduces direct sensitivity to head movement, but inherits drift and error in the source camera/hand tracking. Grip closure and wrist separation are retained as evidence, not used to assert object contact.

## Label provenance

All initial intervals are `machine_proposed`. Entropy is a diagnostic of heuristic emission ambiguity, not a calibrated error probability. A review action records the actor, run/proposal identity, decision, optional replacement class, comment and timestamp. A correction to motion class does not validate inherited geometry. Automated tests use temporary fixture logs; they do not create real human decisions.

Reviewer procedure: examine the interval and neighboring context; switch views if needed; check QA flags and missing evidence; accept only the motion interpretation supported by visible evidence; otherwise correct or reject. Record a reason for ambiguity. Boundaries shorter than the decoder dwell can appear at censored observation edges; do not silently merge across unknown gaps.

For a future gold set, independently label stratified intervals from each state and each QA stratum; double-label a subset before adjudication. Compute frame agreement, segment overlap, boundary error and class-wise error only after the reference exists. Preserve disagreement rather than averaging away incompatible judgments.

## Cleaning policy

No source row is deleted. Invalid observations are quarantined from temporal inference; short-window filtering changes only the derived world-pose copy. Metadata overrides retain both source and reconciled values. Source tasks are not used as phase supervision. The configured reprojection thresholds (warning 15 px, quarantine 50 px of per-hand median joint residual) are engineering screening thresholds, not independently calibrated acceptance tolerances.

## Recurrence vocabulary

A repeated-motion bout is an experimental interval with internally consistent oscillation evidence. Period, dominant side and peak locations describe signal shape. `cycle` does not mean cut, peel, grasp, insertion or successful operation. No action count is released as ground truth.
