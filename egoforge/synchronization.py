"""Shared-epoch IMU alignment and diagnostic pose/gyro lag estimation.

The canonical source clock is integer timestamp_ns. Nominal video timestamp_s
is intentionally not used here. No input time, pose, or sensor sample is edited.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def _summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"median": None, "p95": None, "max": None}
    return {
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _nearest_distance(query, samples):
    if not len(samples):
        return np.full(np.shape(query), np.inf)
    index = np.searchsorted(samples, query).clip(0, len(samples) - 1)
    previous = np.maximum(index - 1, 0)
    return np.minimum(np.abs(samples[index] - query), np.abs(samples[previous] - query))


def _frame_times(query, frames, seconds, tolerance):
    """Interpolate capture times; bound edge extrapolation and unsupported gaps."""
    result = np.interp(query, frames, seconds)
    before = query < frames[0]
    after = query > frames[-1]
    result[before] = seconds[0] + (query[before] - frames[0]) * (
        seconds[1] - seconds[0]
    ) / (frames[1] - frames[0])
    result[after] = seconds[-1] + (query[after] - frames[-1]) * (
        seconds[-1] - seconds[-2]
    ) / (frames[-1] - frames[-2])
    supported = np.isfinite(result) & (_nearest_distance(result, seconds) <= tolerance)
    result[~supported] = np.nan
    return result, supported, before | after


def align_sensor_clocks(
    ep,
    gap_warn_s=0.025,
    fps=30.0,
    max_lag_frames=15.0,
    lag_step_frames=0.25,
    quaternion_tolerance=0.01,
):
    """Return ``(report, gap_masks)`` using head IMU as frame-clock reference.

    ``report['frame_capture_time_s']`` is relative to the integer head-clock
    anchor, not nominal episode zero. Unsupported frame times are NaN.
    ``gap_masks[sensor]`` marks absent shared-clock coverage or an unavailable
    reference. Same-frame timestamp disagreement is additionally reported per
    sensor so a continuous but shifted sensor clock cannot appear aligned.
    """
    for value in [gap_warn_s, fps, lag_step_frames]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(
                "Clock tolerance, fps and lag step must be finite and positive"
            )
    if not np.isfinite(max_lag_frames) or max_lag_frames < 0:
        raise ValueError("Maximum lag must be finite and nonnegative")
    frames = np.asarray(ep.frame_index, dtype=float)
    n = len(frames)
    report = {
        "reference_sensor": "main_stereo",
        "canonical_key": "timestamp_ns",
        "anchor_timestamp_ns": None,
        "frame_capture_time_s": np.full(n, np.nan),
        "reference_valid_mask": np.zeros(n, dtype=bool),
        "reference_extrapolated_frame_count": 0,
        "sensors": {},
        "motion_alignment": {},
        "status": "reference_unavailable",
    }
    masks = {name: np.ones(n, dtype=bool) for name in ep.imu}
    reference = ep.imu.get("main_stereo")
    if reference is None or len(reference) < 2 or not n:
        return report, masks
    raw_ns = reference.timestamp_ns.to_numpy()
    if not np.issubdtype(raw_ns.dtype, np.integer):
        report["status"] = "reference_timestamp_ns_must_be_integer"
        return report, masks
    anchor = int(raw_ns[0])
    report["anchor_timestamp_ns"] = anchor
    parsed = {}
    for name, table in ep.imu.items():
        ns = table.timestamp_ns.to_numpy()
        sensor_frames = table.frame.to_numpy(dtype=float)
        valid = (
            len(table) >= 2
            and np.issubdtype(ns.dtype, np.integer)
            and np.isfinite(sensor_frames).all()
            and np.all(np.diff(ns) > 0)
            and np.all(np.diff(sensor_frames) > 0)
        )
        if not valid:
            report["sensors"][name] = {
                "status": "invalid_or_nonmonotonic_clock",
                "samples": len(table),
                "clock_mismatch_frame_count": n,
            }
            continue
        # Subtract integers BEFORE floating-point conversion: epoch ≈ 1e18 ns.
        seconds = (ns - np.int64(anchor)).astype(np.float64) / 1e9
        parsed[name] = (sensor_frames, seconds)
    if "main_stereo" not in parsed:
        report["status"] = "reference_clock_invalid"
        return report, masks
    head_frames, head_seconds = parsed["main_stereo"]
    capture, supported, extrapolated = _frame_times(
        frames, head_frames, head_seconds, gap_warn_s
    )
    report.update(
        status="ok",
        frame_capture_time_s=capture,
        reference_valid_mask=supported,
        reference_extrapolated_frame_count=int((extrapolated & supported).sum()),
    )
    for name, (sensor_frames, seconds) in parsed.items():
        offset = _nearest_distance(capture, seconds)
        gap = ~supported | ~np.isfinite(offset) | (offset > gap_warn_s)
        masks[name] = gap
        same_frame_time, same_frame_supported, _ = _frame_times(
            frames, sensor_frames, seconds, gap_warn_s
        )
        same_frame_delta = same_frame_time - capture
        comparable = supported & same_frame_supported & np.isfinite(same_frame_delta)
        mismatch = comparable & (np.abs(same_frame_delta) > gap_warn_s)
        report["sensors"][name] = {
            "status": "ok",
            "samples": len(seconds),
            "coverage_gap_frames": int(gap.sum()),
            "nearest_frame_offset_s": _summary(offset),
            "same_frame_clock_offset_s": _summary(same_frame_delta[comparable]),
            "same_frame_absolute_clock_offset_s": _summary(
                np.abs(same_frame_delta[comparable])
            ),
            "clock_comparable_frames": int(comparable.sum()),
            "clock_mismatch_frame_count": int(mismatch.sum()),
            "clock_mismatch_mask": mismatch,
            "gap_count": int((np.diff(seconds) > gap_warn_s).sum()),
            "max_gap_s": float(np.diff(seconds).max()),
            "median_rate_hz": float(1 / np.median(np.diff(seconds))),
        }

    # Diagnose motion timing on interval midpoints and a fixed support for all
    # lags. Magnitudes avoid relying on unverified sensor-axis conventions.
    if n < 3:
        return report, masks
    head = np.asarray(ep.head, dtype=float)
    qnorm = np.linalg.norm(head[:, 3:], axis=1)
    pose_valid = np.isfinite(head).all(axis=1) & (
        np.abs(qnorm - 1) <= quaternion_tolerance
    )
    pair_valid = (
        pose_valid[:-1]
        & pose_valid[1:]
        & supported[:-1]
        & supported[1:]
        & (np.diff(frames) == 1)
        & (np.diff(capture) > 0)
    )
    angular = np.full(n - 1, np.nan)
    if pair_valid.any():
        previous = Rotation.from_quat(head[:-1, 3:][pair_valid])
        following = Rotation.from_quat(head[1:, 3:][pair_valid])
        angular[pair_valid] = (previous.inv() * following).magnitude() / np.diff(
            capture
        )[pair_valid]
    midpoint = (frames[:-1] + frames[1:]) / 2
    gyro = np.linalg.norm(
        reference[
            ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
        ].to_numpy(),
        axis=1,
    )
    if not np.isfinite(gyro).all():
        report["motion_alignment"] = {"status": "nonfinite_gyro"}
        return report, masks
    lags = np.arange(
        -max_lag_frames, max_lag_frames + lag_step_frames / 2, lag_step_frames
    )
    lags = np.unique(
        np.r_[lags[(lags >= -max_lag_frames) & (lags <= max_lag_frames)], 0.0]
    )
    keep = (
        pair_valid
        & (midpoint - max_lag_frames >= head_frames[0])
        & (midpoint + max_lag_frames <= head_frames[-1])
    )
    # Never interpolate across a head-IMU hole at any tested lag.
    for lag in lags:
        _, lag_supported, _ = _frame_times(
            midpoint + lag, head_frames, head_seconds, gap_warn_s
        )
        keep &= lag_supported
    if keep.sum() < 12 or np.std(angular[keep]) < 1e-6:
        report["motion_alignment"] = {
            "status": "insufficient_motion_or_support",
            "compared_intervals": int(keep.sum()),
        }
        return report, masks
    candidates = []
    for lag in lags:
        sampled = np.interp(midpoint[keep] + lag, head_frames, gyro)
        if np.std(sampled) > 1e-6:
            candidates.append(
                (float(lag), float(np.corrcoef(angular[keep], sampled)[0, 1]))
            )
    if not candidates:
        report["motion_alignment"] = {"status": "insufficient_gyro_variation"}
        return report, masks
    best = max(candidates, key=lambda item: (item[1], -abs(item[0])))
    report["motion_alignment"] = {
        "status": "diagnostic_only",
        "best_lag_frames": best[0],
        "best_lag_s": best[0] / fps,
        "correlation_at_best_lag": best[1],
        "correlation_at_zero_lag": dict(candidates).get(0.0),
        "compared_intervals": int(keep.sum()),
        "search_boundary_hit": abs(best[0]) == max_lag_frames,
        "method": "SO(3) angular speed over measured capture intervals versus gyro magnitude at fractional-frame midpoints; fixed support; no retiming",
        "lag_convention": "positive lag compares pose speed at midpoint f to gyro at f + lag",
        "seconds_conversion": "nominal fps; lag search is in fractional frames",
    }
    return report, masks
