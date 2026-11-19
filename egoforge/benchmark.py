"""Seeded fault-injection evaluation. Measures rule sensitivity, NOT label accuracy."""

from __future__ import annotations
import numpy as np
from .quality import audit_episode


def corruption_benchmark(ep, config=None):
    n = len(ep.time)
    if n < 100:
        raise ValueError("Corruption suite requires at least 100 frames at 30fps")
    start = min(200, n // 4)
    stop = min(start + 90, n - 2)
    ids = np.arange(start, stop)
    cases = []

    def run(name, mutate, code, target=ids):
        damaged = ep.copy()
        mutate(damaged)
        f, _, _ = audit_episode(damaged, config)
        detected = np.array([code in x for x in f.qa_flags])
        hits = int(detected[target].sum())
        cases.append(
            {
                "case": name,
                "expected_rule": code,
                "injected_frame_count": len(target),
                "detected_injected_frames": hits,
                "sensitivity": hits / len(target),
                "severity": "deliberately threshold-visible synthetic corruption",
            }
        )

    run(
        "world_coordinate_translation_15cm",
        lambda e: e.world.__setitem__(ids, e.world[ids] + [0.15, 0, 0]),
        "world_camera_inconsistent",
    )
    run(
        "image_keypoints_displaced_120px",
        lambda e: e.keypoints.__setitem__(ids, e.keypoints[ids] + [120, 0]),
        "reprojection_extreme",
    )
    run(
        "nonunit_quaternion_scale_1.2",
        lambda e: e.head.__setitem__((ids, slice(3, None)), e.head[ids, 3:] * 1.2),
        "quaternion_norm_invalid",
    )
    run(
        "presence_flag_contradiction",
        lambda e: e.present.__setitem__((ids, 0), False),
        "presence_payload_conflict",
    )
    run(
        "tracking_quality_contradiction",
        lambda e: e.tracking_quality.__setitem__(ids, 3),
        "tracking_flag_conflict",
    )

    def stretch(e):
        e.world[ids, 0, 8] += [0.20, 0, 0]

    run("finger_segment_stretched_20cm", stretch, "bone_length_extreme")
    target = np.array([start])
    run(
        "duplicate_frame_timestamp",
        lambda e: e.time.__setitem__(start, e.time[start - 1]),
        "timestamp_or_frame_discontinuity",
        target,
    )

    def clock(e):
        e.imu["main_stereo"]["timestamp_s"] += 0.2

    run(
        "imu_timestamp_frame_clock_mismatch_200ms",
        clock,
        "main_stereo_imu_clock_inconsistent",
        np.arange(n),
    )

    def gap(e):
        d = e.imu["main_stereo"]
        e.imu["main_stereo"] = d[~d.timestamp_s.between(2.0, 3.0)].copy()

    run(
        "imu_burst_removed_1s",
        gap,
        "main_stereo_imu_coverage_gap",
        np.flatnonzero((ep.time > 2.05) & (ep.time < 2.95)),
    )
    original, _, _ = audit_episode(ep, config)
    return {
        "evaluation_type": "synthetic corruption sensitivity on a real source episode",
        "episode_id": ep.episode_id,
        "seed": 0,
        "cases": cases,
        "cases_passing": sum(x["sensitivity"] == 1 for x in cases),
        "total_cases": len(cases),
        "untouched_source_flag_rate": float(np.mean(original.qa_flags.map(bool))),
        "limitations": [
            "Perturbations are seeded severe faults, not naturally adjudicated errors.",
            "Sensitivity is not phase annotation accuracy or QA precision.",
            "Untouched-source flags are unadjudicated and must not be called false positives.",
            "No independent human phase reference labels are available.",
        ],
    }
