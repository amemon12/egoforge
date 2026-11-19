# Quality rules and interpretation

These are engineering screening rules, not validated physical accuracy specifications. Configuration lives in `configs/default.json`; raw observations are retained. A flag may indicate a representation mismatch, estimation error, unusual but real motion, or a threshold unsuitable for a particular episode.

| Check | Default trigger | Effect |
|---|---|---|
| Frame/time continuity | Nonconsecutive frame; nonincreasing timestamp; step differs from 1/30 s by over half a frame | Quarantine the break; reset finite-span processing |
| Hand presence | Either hand absent; flag conflicts with pose payload or tracking-quality code | Quarantine bimanual inference |
| Finite geometry | Nonfinite world/camera joints, head pose or keypoints | Quarantine |
| Quaternion norm | Absolute norm error > 0.01 | Quarantine |
| World/camera consistency | Maximum joint transform residual > 0.01 m | Quarantine |
| Positive depth | Joint camera depth ≤ 0.01 m | Quarantine |
| Reprojection | Per-hand median joint residual > 15 px | Review warning |
| Extreme reprojection | Per-hand median joint residual > 50 px | Quarantine |
| Bone stability | Any bone deviates > 35% from its episode median length | Review warning |
| Extreme bone change | Relative deviation > 100% | Quarantine |
| Grip scalar | Outside [0,1] or nonfinite | Quarantine |
| Wrist speed | Derived speed > 2.5 m/s | Review warning |
| Wrist acceleration | Derived acceleration > 40 m/s² | Review warning |
| IMU coverage | No sensor sample within 25 ms of the shared-epoch frame time | Review warning |
| Shared clocks | Same fractional-frame timestamp differs from head reference by > 25 ms | Review warning |
| IMU fields | Required timing, gyro or accelerometer fields are nonfinite | Review warning |
| Clock representations | Nominal seconds and fractional-frame time differ by > 1/30 s; time is nonmonotonic | Review warning |

Warnings subtract configured fixed weights from an initial quality score of 1; quarantine sets it to 0. These weights are heuristic. The temporal engine abstains below quality 0.25 or on unusable/nonfinite evidence. An isolated remaining valid sample can therefore become `unknown` even without a geometric quarantine. This explains why unknown-frame and quarantined-frame counts need not match.

Cleaned poses use a Savitzky–Golay polynomial of order 2, with at most seven frames per window. No interpolation crosses invalid spans. Filtering can attenuate high-frequency motion and shift apparent boundaries near edges; raw poses remain available through source hashes and frame IDs. Bone baselines are computed within each episode for screening and are not human anatomical ground truth.

Video streams are fully decoded to check observed frame count against pose count. Image-detail, exposure and hash metrics are sampled once per second in each pilot camera. The image checks do not certify all frames are sharp, detect every duplicate, or establish privacy clearance. Image flags are reported separately from the full-rate pose-based phase quality score.

Pose/gyro timing comparisons use integer nanosecond differences before conversion to floating point, interpolate the camera-clock reference using fractional frame coordinates, and compare rotational speed at interval midpoints. The lag scan reports correlation peaks without shifting source timestamps. Matching inherited representations or sensors is not independent validation of physical accuracy.
