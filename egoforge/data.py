"""Source decoding. Original bytes never pass through a write path."""

from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq

EDGES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
]


@dataclass
class Episode:
    episode_id: int
    frame_index: np.ndarray
    time: np.ndarray
    state: np.ndarray
    head: np.ndarray
    world: np.ndarray  # N, hand, joint, xyz
    camera: np.ndarray
    keypoints: np.ndarray
    present: np.ndarray
    tracking_quality: np.ndarray
    task_index: int
    source_sha256: str
    imu: dict
    intrinsics: dict

    def copy(self):
        return replace(
            self,
            **{k: v.copy() for k, v in vars(self).items() if isinstance(v, np.ndarray)},
            imu={k: v.copy(deep=True) for k, v in self.imu.items()},
        )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def load_episode(raw: Path, episode_id: int) -> Episode:
    name = f"episode_{episode_id:06d}"
    path = raw / "data/chunk-000" / f"{name}.parquet"
    t = pq.read_table(path)
    a = lambda k: np.asarray(t[k].to_pylist())
    state = a("observation.state")
    if state.shape[1:] != (188,):
        raise ValueError("Expected 188-state source schema")
    worlds = np.stack(
        [state[:, :63].reshape(-1, 21, 3), state[:, 94:157].reshape(-1, 21, 3)], axis=1
    )
    imu = {
        sensor: pq.read_table(
            raw / "imu/chunk-000" / sensor / f"{name}.parquet"
        ).to_pandas()
        for sensor in ["main_stereo", "aux_left", "aux_right"]
    }
    return Episode(
        episode_id,
        a("frame_index"),
        a("timestamp"),
        state,
        a("observation.head_pose"),
        worlds,
        np.stack(
            [
                a(f"observation.joints_cam.{h}").reshape(-1, 21, 3)
                for h in ["left", "right"]
            ],
            axis=1,
        ),
        np.stack(
            [
                a(f"observation.keypoints_2d.{h}").reshape(-1, 21, 2)
                for h in ["left", "right"]
            ],
            axis=1,
        ),
        np.stack(
            [a(f"observation.hand_present.{h}") for h in ["left", "right"]], axis=1
        ).astype(bool),
        a("tracking_quality"),
        int(a("task_index")[0]),
        sha256(path),
        imu,
        json.loads((raw / "meta/camera_intrinsics.json").read_text())[name][
            "main_stereo"
        ],
    )


def json_safe(x):
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in x]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        return float(x) if np.isfinite(x) else None
    return x


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(obj), indent=2, allow_nan=False) + "\n")
