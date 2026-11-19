"""Geometry, time, skeleton and cross-sensor QC with explicit abstention.

Residuals compare inherited representations and are consistency checks, not
independent estimates of physical pose accuracy.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from .data import Episode, EDGES
from .synchronization import align_sensor_clocks

DEFAULTS = {
    "fps": 30.0,
    "reprojection_warn_px": 15.0,
    "reprojection_fail_px": 50.0,
    "transform_fail_m": 0.01,
    "bone_relative_warn": 0.35,
    "speed_warn_mps": 2.5,
    "acceleration_warn_mps2": 40.0,
    "quaternion_norm_tolerance": 0.01,
    "smoothing_window_frames": 7,
    "imu_gap_warn_s": 0.025,
}


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def safe_derivative(values, times, valid):
    result = np.full_like(values, np.nan, dtype=float)
    for a, b in spans(valid):
        if b - a > 1:
            result[a:b] = np.gradient(values[a:b], times[a:b], axis=0)
    return result


def nan_reduce(function, x, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return function(x, axis=axis)


def qsummary(x):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    return (
        {
            "median": float(np.median(x)),
            "p95": float(np.percentile(x, 95)),
            "max": float(np.max(x)),
        }
        if len(x)
        else {"median": None, "p95": None, "max": None}
    )


def audit_episode(ep: Episode, config=None):
    c = {**DEFAULTS, **(config or {})}
    n = len(ep.time)
    flags = [[] for _ in range(n)]
    fatal = np.zeros(n, bool)
    risk = np.zeros(n, float)

    def mark(code, mask, severity="warning", weight=0.12):
        mask = np.asarray(mask, bool)
        for i in np.flatnonzero(mask):
            flags[i].append(code)
        risk[mask] += weight
        if severity == "error":
            fatal[mask] = True

    dt = np.diff(ep.time)
    df = np.diff(ep.frame_index)
    broken = np.r_[
        False,
        (~np.isfinite(dt))
        | (dt <= 0)
        | (np.abs(dt - 1 / c["fps"]) > 0.5 / c["fps"])
        | (df != 1),
    ]
    mark("timestamp_or_frame_discontinuity", broken, "error", 1)
    mark("nonfinite_timestamp", ~np.isfinite(ep.time), "error", 1)
    mark("hand_unobserved", ~ep.present.all(axis=1), "error", 1)
    finite = (
        np.isfinite(ep.world).all(axis=(1, 2, 3))
        & np.isfinite(ep.camera).all(axis=(1, 2, 3))
        & np.isfinite(ep.head).all(axis=1)
        & np.isfinite(ep.keypoints).all(axis=(1, 2, 3))
    )
    mark("nonfinite_pose", ~finite, "error", 1)
    zero = np.all(ep.world == 0, axis=(2, 3))
    mark(
        "presence_payload_conflict",
        ((zero & ep.present) | (~zero & ~ep.present)).any(axis=1),
        "error",
        1,
    )
    expected = (~ep.present[:, 0]).astype(int) + 2 * (~ep.present[:, 1]).astype(int)
    mark("tracking_flag_conflict", expected != ep.tracking_quality, "error", 1)
    qnorm = np.linalg.norm(ep.head[:, 3:], axis=1)
    qbad = (~np.isfinite(qnorm)) | (np.abs(qnorm - 1) > c["quaternion_norm_tolerance"])
    mark("quaternion_norm_invalid", qbad, "error", 1)
    rotation = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    good = (~qbad) & finite
    rotation[good] = Rotation.from_quat(ep.head[good, 3:]).as_matrix()
    pred = np.einsum("nji,nhkj->nhki", rotation, ep.world - ep.head[:, None, None, :3])
    terr = np.linalg.norm(pred - ep.camera, axis=3)
    terr[~ep.present] = np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        k = ep.intrinsics["left"]
        projected = ep.camera[..., :2] / ep.camera[..., 2, None] * [
            k["fx"],
            k["fy"],
        ] + [k["cx"], k["cy"]]
        perr = np.linalg.norm(projected - ep.keypoints, axis=3)
    perr[~ep.present] = np.nan
    transform = nan_reduce(np.nanmax, terr, (1, 2))
    reproj = nan_reduce(np.nanmedian, perr, 2)
    mark("world_camera_inconsistent", transform > c["transform_fail_m"], "error", 1)
    mark(
        "behind_camera",
        (ep.camera[..., 2] <= 0.01).any(axis=(1, 2)) & ep.present.all(axis=1),
        "error",
        1,
    )
    mark(
        "reprojection_residual",
        np.nanmax(reproj, axis=1) > c["reprojection_warn_px"],
        weight=0.18,
    )
    mark(
        "reprojection_extreme",
        np.nanmax(reproj, axis=1) > c["reprojection_fail_px"],
        "error",
        1,
    )
    lengths = np.stack(
        [
            np.linalg.norm(ep.world[:, :, j] - ep.world[:, :, i], axis=2)
            for i, j in EDGES
        ],
        axis=2,
    )
    lengths[~ep.present] = np.nan
    baseline = np.nanmedian(lengths, axis=0)
    rel = np.abs(lengths - baseline[None]) / np.maximum(baseline[None], 1e-6)
    bone = np.nanmax(rel, axis=(1, 2))
    mark("bone_length_outlier", bone > c["bone_relative_warn"], weight=0.2)
    mark("bone_length_extreme", bone > 1.0, "error", 1)
    mark(
        "grip_out_of_range",
        (
            (ep.state[:, [93, 187]] < 0)
            | (ep.state[:, [93, 187]] > 1)
            | ~np.isfinite(ep.state[:, [93, 187]])
        ).any(axis=1),
        "error",
        1,
    )
    # Keep original rows. NaN masking is reversible through the source pointer.
    clean = ep.world.copy().astype(float)
    clean[~ep.present] = np.nan
    clean[fatal] = np.nan
    valid = (~fatal) & np.isfinite(clean).all(axis=(1, 2, 3))
    for a, b in spans(valid):
        w = min(int(c["smoothing_window_frames"]), b - a)
        if w % 2 == 0:
            w -= 1
        if w >= 3:
            clean[a:b] = savgol_filter(clean[a:b], w, 2, axis=0, mode="interp")
    vel = safe_derivative(clean[:, :, 0], ep.time, valid)
    speed = np.linalg.norm(vel, axis=2)
    accel = np.linalg.norm(
        safe_derivative(vel, ep.time, valid & np.isfinite(vel).all(axis=(1, 2))), axis=2
    )
    mark(
        "wrist_speed_outlier",
        nan_reduce(np.nanmax, speed, 1) > c["speed_warn_mps"],
        weight=0.2,
    )
    mark(
        "wrist_acceleration_outlier",
        nan_reduce(np.nanmax, accel, 1) > c["acceleration_warn_mps2"],
        weight=0.15,
    )
    denom = np.prod(speed, axis=1)
    cosine = np.divide(
        np.sum(vel[:, 0] * vel[:, 1], axis=1),
        denom,
        out=np.zeros(n),
        where=denom > 1e-6,
    )
    clock_report, gap_masks = align_sensor_clocks(
        ep,
        gap_warn_s=c["imu_gap_warn_s"],
        fps=c["fps"],
        quaternion_tolerance=c["quaternion_norm_tolerance"],
    )
    capture = clock_report.pop("frame_capture_time_s")
    reference_valid = clock_report.pop("reference_valid_mask")
    imu_summary = {}
    head_ang = np.full(n, np.nan)
    if n > 1:
        pair = (
            good[:-1]
            & good[1:]
            & reference_valid[:-1]
            & reference_valid[1:]
            & (~broken[1:])
            & (np.diff(capture) > 0)
        )
        dr = np.einsum("nji,njk->nik", rotation[:-1], rotation[1:])
        head_ang[1:][pair] = (
            Rotation.from_matrix(dr[pair]).magnitude() / np.diff(capture)[pair]
        )
    required = [
        "timestamp_ns",
        "timestamp_s",
        "frame",
        "angular_velocity_x",
        "angular_velocity_y",
        "angular_velocity_z",
        "linear_acceleration_x",
        "linear_acceleration_y",
        "linear_acceleration_z",
    ]
    for sensor, data in ep.imu.items():
        mark(f"{sensor}_imu_coverage_gap", gap_masks[sensor], weight=0.15)
        timing = clock_report["sensors"].get(
            sensor, {"status": "reference_unavailable"}
        )
        mismatch = timing.pop(
            "clock_mismatch_mask",
            np.ones(n, bool) if timing.get("status") != "ok" else np.zeros(n, bool),
        )
        mark(f"{sensor}_imu_shared_clock_mismatch", mismatch, weight=0.3)
        nonfinite = ~np.isfinite(data[required].to_numpy()).all(axis=1)
        if nonfinite.any():
            mark(f"{sensor}_imu_nonfinite", np.ones(n, bool), weight=0.2)
        ti = data.timestamp_s.to_numpy()
        fi = data.frame.to_numpy()
        residual = np.abs(ti - fi / c["fps"])
        clock_bad = (
            len(ti) < 2
            or not np.isfinite(residual).all()
            or np.max(residual) > 1 / c["fps"]
            or np.any(np.diff(ti) <= 0)
            or timing.get("status") != "ok"
        )
        if clock_bad:
            mark(f"{sensor}_imu_clock_inconsistent", np.ones(n, bool), weight=0.3)
        imu_summary[sensor] = {
            **timing,
            "timestamp_frame_residual_s": qsummary(residual),
            "motion_alignment": clock_report["motion_alignment"]
            if sensor == "main_stereo"
            else {},
        }
    quality = np.clip(1 - risk, 0, 1)
    quality[fatal] = 0
    features = pd.DataFrame(
        {
            "episode_id": ep.episode_id,
            "frame_index": ep.frame_index,
            "timestamp_s": ep.time,
            "left_speed_mps": speed[:, 0],
            "right_speed_mps": speed[:, 1],
            "left_accel_mps2": accel[:, 0],
            "right_accel_mps2": accel[:, 1],
            "wrist_distance_m": np.linalg.norm(clean[:, 0, 0] - clean[:, 1, 0], axis=1),
            "left_grip": ep.state[:, 93],
            "right_grip": ep.state[:, 187],
            "velocity_cosine": cosine,
            "quality_score": quality,
            "usable": ~fatal,
            "left_reprojection_px": reproj[:, 0],
            "right_reprojection_px": reproj[:, 1],
            "world_camera_residual_m": transform,
            "bone_relative_deviation": bone,
            "head_angular_speed_rps": head_ang,
            "capture_time_from_head_anchor_s": capture,
            "qa_flags": flags,
        }
    )
    codes = sorted(set(code for f in flags for code in f))
    report = {
        "episode_id": ep.episode_id,
        "frames": n,
        "usable_frames": int((~fatal).sum()),
        "quarantined_frames": int(fatal.sum()),
        "flagged_frames": int(sum(bool(f) for f in flags)),
        "rule_counts": {code: sum(code in f for f in flags) for code in codes},
        "world_camera_residual_m": qsummary(terr),
        "left_reprojection_px": qsummary(perr[:, 0]),
        "right_reprojection_px": qsummary(perr[:, 1]),
        "bone_relative_deviation": qsummary(bone),
        "left_speed_mps": qsummary(speed[:, 0]),
        "right_speed_mps": qsummary(speed[:, 1]),
        "imu": imu_summary,
        "synchronization": {k: v for k, v in clock_report.items() if k != "sensors"},
        "cleaning": {
            "input_rows": n,
            "output_rows": n,
            "rows_deleted": 0,
            "masked_frames": int(fatal.sum()),
            "absent_hand_samples": int((~ep.present).sum()),
            "smoothing": "Savitzky-Golay order 2, window <=7 frames, finite contiguous spans only; used for kinematic proposals, source poses retained",
            "source_sha256": ep.source_sha256,
            "status": "derived copy; no source modifications",
        },
    }
    return features, clean, report
