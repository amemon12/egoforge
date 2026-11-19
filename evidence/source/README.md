---
license: cc-by-4.0
---

# Humanola Egocentric Hand-Pose Dataset — Sample Delivery

## Overview

Egocentric (head-mounted) manipulation video with synchronized **3D hand-pose tracking**. Per frame: 3D hand keypoints in a gravity-aligned **world frame**, per-joint finger angles, a grip-closure scalar, the head camera's 6-DoF world pose, and a per-hand wrist pose. Four hardware-synchronized camera streams (head stereo + both wrists) and each camera's ~200 Hz IMU accompany every episode.

**Format:** LeRobot **v2.1**.

## Dataset Statistics

| Metric | Value |
|--------|-------|
| Episodes | 9 |
| Total frames | 48,272 @ 30 fps |
| Total duration | ~27 min |
| State-vector dimension | 188 |
| Camera streams | 4 (head stereo L+R, wrist L+R), 1920×1200 |
| IMU | ~200 Hz × 3 (head + both wrists), gyro + accelerometer |
| Hand-tracking coverage | Both hands ≈100%; absences flagged per-frame (`hand_present`, `tracking_quality`) |

Head stereo: rectified, zero distortion, fx=fy≈**754 px**, baseline **0.120 m**; wrist cameras mono; global shutter throughout. Exact per-episode camera intrinsics in `meta/camera_intrinsics.json`.

### Episodes

| # | Duration | Frames | Task |
|---|----------|--------|------|
| 0 | 4:39 | 8,371 | ironing clothes |
| 1 | 2:11 | 3,931 | folding laundry |
| 2 | 4:00 | 7,201 | wiping |
| 3 | 5:08 | 9,241 | washing dishes |
| 4 | 1:36 | 2,904 | slicing carrots |
| 5 | 2:40 | 4,801 | peeling and slicing cucumbers |
| 6 | 1:47 | 3,211 | pouring cereal into a container |
| 7 | 2:00 | 3,601 | washing dishes |
| 8 | 2:47 | 5,011 | cutting an apple |

`task_index` → task lookup in `meta/tasks.jsonl`.

## Directory Structure

```
humanola_sample_dataset/
├── README.md
├── meta/
│   ├── info.json                   # feature shapes + per-dim column names, collection info (v2.1)
│   ├── modality.json               # observation.state field layout
│   ├── episodes.jsonl              # per-episode: length, duration, coverage, task
│   ├── episodes_stats.jsonl        # per-episode min/max/mean/std (v2.1 normalization stats)
│   ├── tasks.jsonl                 # task_index → task
│   └── camera_intrinsics.json      # per-episode intrinsics for all 4 cameras
├── data/chunk-000/
│   └── episode_000000.parquet …    # one row per timestep
├── videos/chunk-000/
│   ├── observation.images.main_stereo_left/episode_000000.mp4 …    # head stereo, LEFT eye (rendered/primary view)
│   ├── observation.images.main_stereo_right/ …                     # head stereo, RIGHT eye
│   ├── observation.images.aux_left/ …                              # left-wrist camera
│   └── observation.images.aux_right/ …                             # right-wrist camera
├── imu/
│   ├── imu_info.json               # per-sensor rate, units, camera↔IMU transform, video_key_map
│   └── chunk-000/
│       ├── main_stereo/episode_000000.parquet …   # head-cam IMU
│       ├── aux_left/episode_000000.parquet …      # left-wrist-cam IMU
│       └── aux_right/episode_000000.parquet …     # right-wrist-cam IMU
└── extras/                          # visualization aids, not loaded by LeRobot
    ├── overlays/episode_000000_overlay.mp4 …   # 2D skeleton + box on the left eye
    └── rrd/episode_000000.rrd …                # rerun: 3D world hands + wrist-cam panels
```

All four camera streams are hardware-synchronized (same frame count, timestamps within one frame), cut to identical per-episode ranges from t=0.

## State Vector Layout

`observation.state` — 188 `float32` per frame (offsets in `meta/modality.json`):

| Field | Indices | Dims | Description |
|-------|---------|------|-------------|
| `left_hand_world_joints`  | [0:63]    | 21×3 | Left 21 joints xyz, world frame, meters |
| `left_finger_angles`      | [63:93]   | 15×2 | Left finger joints × [flexion, abduction], radians |
| `left_grip_closure`       | [93:94]   | 1    | Left grip-closure ∈ [0,1] (0 = open, 1 = fist) |
| `right_hand_world_joints` | [94:157]  | 21×3 | Right 21 joints xyz, world frame, meters |
| `right_finger_angles`     | [157:187] | 15×2 | Right finger joints × [flexion, abduction], radians |
| `right_grip_closure`      | [187:188] | 1    | Right grip-closure ∈ [0,1] |

**Joint order** (21/hand — thumb `cmc/mcp/ip/tip`, fingers `mcp/pip/dip/tip`; per-dim names for every column in `meta/info.json`):

| Index | Joint | Index | Joint | Index | Joint |
|---|---|---|---|---|---|
| 0 | wrist | 7 | index_dip | 14 | ring_pip |
| 1 | thumb_cmc | 8 | index_tip | 15 | ring_dip |
| 2 | thumb_mcp | 9 | middle_mcp | 16 | ring_tip |
| 3 | thumb_ip | 10 | middle_pip | 17 | pinky_mcp |
| 4 | thumb_tip | 11 | middle_dip | 18 | pinky_pip |
| 5 | index_mcp | 12 | middle_tip | 19 | pinky_dip |
| 6 | index_pip | 13 | ring_mcp | 20 | pinky_tip |

**World frame:** right-handed and gravity-aligned, in meters, with up along **−Y** (gravity points toward +Y). It is fixed for the whole episode and does not move with the head; its horizontal orientation is arbitrary. `observation.joints_cam.*` holds the same joints in the camera frame.

### Finger angles & grip-closure

Finger angles are 30 values, `[flexion, abduction]` per joint interleaved `[j0_flex, j0_abd, …]`, ordered level-major (proximal → distal, thumb → pinky within each level):

| Slots (flex, abd) | Joint | Slots | Joint | Slots | Joint |
|---|---|---|---|---|---|
| 0–1 | thumb_cmc | 10–11 | thumb_mcp | 20–21 | thumb_ip |
| 2–3 | index_mcp | 12–13 | index_pip | 22–23 | index_dip |
| 4–5 | middle_mcp | 14–15 | middle_pip | 24–25 | middle_dip |
| 6–7 | ring_mcp | 16–17 | ring_pip | 26–27 | ring_dip |
| 8–9 | pinky_mcp | 18–19 | pinky_pip | 28–29 | pinky_dip |

- **flexion**: palm-curl, radians — 0 = straight, ≈1.5 (≈90°) = bent. **abduction**: sideways splay, signed, ≈±0.4. (Occasional ±π = angle-wrap.)
- **grip-closure**: mean flexion of the 5 MCP + 5 PIP joints ÷ 90°, clipped to [0,1]. 0 = open, 1 = fist; hand-size-independent.

### Extra observation columns (not in `observation.state`)

| Column | Shape | Description |
|--------|-------|-------------|
| `observation.head_pose`                 | 7  | Head-camera 6-DoF pose, world frame: `[tx,ty,tz, qx,qy,qz,qw]` (xyz meters, xyzw quaternion, camera→world) |
| `observation.wrist_pose.{left,right}`   | 7  | Per-hand wrist pose, world frame, same `[tx,ty,tz, qx,qy,qz,qw]` layout (wrist→world) |
| `observation.joints_cam.{left,right}`   | 63 | The 21 joints in the camera frame, meters |
| `observation.keypoints_2d.{left,right}` | 42 | 21×2 pixel keypoints in the left-eye image |
| `observation.bbox.{left,right}`         | 4  | Hand detection box `[x1,y1,x2,y2]`, left-eye image |
| `observation.hand_present.{left,right}` | 1  | 1.0 tracked, 0.0 absent |
| `tracking_quality`                      | 1  | 0 = both hands good, 1 = left absent, 2 = right absent, 3 = both |

When a hand is absent, its joints / keypoints / bbox / wrist_pose are zero-filled and `hand_present = 0`.

## IMU

Each camera's ~200 Hz IMU is a separate parquet per episode at `imu/chunk-000/<sensor>/episode_NNNNNN.parquet` (`main_stereo` = head, `aux_left`/`aux_right` = wrists) — a sidecar, not a LeRobot feature, so read it directly. One row per sample:

| Column | Description |
|--------|-------------|
| `sample_index` | sample index within the file |
| `timestamp_ns` | sensor timestamp on the shared camera clock (ns) — the cross-stream alignment key |
| `timestamp_s`  | seconds from the episode's first frame |
| `frame`        | fractional video-frame index; snap via `round(frame) == frame_index` |
| `angular_velocity_x, _y, _z`    | gyroscope, rad/s |
| `linear_acceleration_x, _y, _z` | accelerometer, m/s² (gravity included) |

`imu/imu_info.json` has the per-sensor rate/units, the camera↔IMU transform (`camera_imu_transform`, 4×4, meters), and a `video_key_map` (sensor → video stream; `main_stereo` pairs with both head eyes).

## Loading Data

```python
import json, numpy as np, pyarrow.parquet as pq

t = pq.read_table("data/chunk-000/episode_000000.parquet")
state = np.array(t.column("observation.state").to_pylist(), dtype=np.float32)   # (N, 188)

mod = json.load(open("meta/modality.json"))["state"]
def slice_(name): s = mod[name]; return state[:, s["start"]:s["end"]]

left_joints  = slice_("left_hand_world_joints").reshape(-1, 21, 3)   # (N, 21, 3) meters, world
right_joints = slice_("right_hand_world_joints").reshape(-1, 21, 3)
left_wrist   = left_joints[:, 0]                                     # (N, 3)
left_angles  = slice_("left_finger_angles")                         # (N, 30) radians
left_grip    = slice_("left_grip_closure")                          # (N, 1)

tq = np.array(t.column("tracking_quality").to_pylist())             # tq == 0 -> both hands good
# IMU: one file per sensor; align via the `frame` column
head_imu = pq.read_table("imu/chunk-000/main_stereo/episode_000000.parquet").to_pandas()
imu_at_frame_100 = head_imu[(head_imu.frame >= 99.5) & (head_imu.frame < 100.5)]
```

### Visual checks (`extras/`)

- `overlays/episode_NNNNNN_overlay.mp4` — 2D keypoints + boxes on the left eye (green = left, magenta = right).
- `rrd/episode_NNNNNN.rrd` — rerun: 3D world hand skeletons, head-camera frustum, wrist-camera panels.

## Contact
📧 sales@humanola.com
