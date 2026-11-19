"""Experimental recurrence proposals from cleaned world-coordinate wrist paths.

Oscillations describe motion, not semantic actions. A cycle is never asserted to
be a cut, grasp, successful manipulation, or independent ground-truth event.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import math

import numpy as np
import pandas as pd
from scipy.signal import detrend, find_peaks

from .data import Episode


METHOD = "world_wrist_pca_autocorrelation_v1"


@dataclass(frozen=True)
class RecurrenceConfig:
    window_s: float = 5.0
    stride_s: float = 1.0
    minimum_period_s: float = 0.25
    maximum_period_s: float = 2.0
    minimum_quality: float = 0.50
    minimum_extent_m: float = 0.015
    minimum_axis_variance_fraction: float = 0.55
    minimum_autocorrelation: float = 0.60
    maximum_interval_cv: float = 0.30
    minimum_cycles_per_window: float = 2.5
    period_merge_tolerance: float = 0.25
    minimum_bout_windows: int = 2
    maximum_sample_jitter_fraction: float = 0.25

    @classmethod
    def from_dict(cls, values: dict | None) -> "RecurrenceConfig":
        values = values or {}
        extra = set(values) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError(f"Unknown recurrence configuration keys: {sorted(extra)}")
        cfg = cls(**values)
        if any(
            not isinstance(v, (int, float)) or not np.isfinite(v)
            for v in asdict(cfg).values()
        ):
            raise ValueError("Recurrence configuration values must be finite numbers")
        if not 4 <= cfg.window_s <= 6 or not 0 < cfg.stride_s <= cfg.window_s:
            raise ValueError(
                "window_s must lie in [4, 6]; stride must lie in (0, window_s]"
            )
        if not 0 < cfg.minimum_period_s < cfg.maximum_period_s < cfg.window_s:
            raise ValueError(
                "Require 0 < minimum_period_s < maximum_period_s < window_s"
            )
        for name in (
            "minimum_quality",
            "minimum_axis_variance_fraction",
            "minimum_autocorrelation",
        ):
            if not 0 <= getattr(cfg, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        if cfg.minimum_extent_m <= 0 or cfg.maximum_interval_cv <= 0:
            raise ValueError("Extent and interval CV limits must be positive")
        if cfg.minimum_cycles_per_window < 2 or cfg.period_merge_tolerance <= 0:
            raise ValueError(
                "At least two cycles are required; period merge tolerance must be positive"
            )
        if (
            cfg.minimum_bout_windows < 1
            or int(cfg.minimum_bout_windows) != cfg.minimum_bout_windows
        ):
            raise ValueError("minimum_bout_windows must be a positive integer")
        if not 0 < cfg.maximum_sample_jitter_fraction < 1:
            raise ValueError("maximum_sample_jitter_fraction must lie in (0, 1)")
        return cfg


def _axis_signal(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Remove translation/linear drift and use the first spatial PCA component."""
    centered = detrend(np.asarray(xyz, float), axis=0, type="linear")
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    axis = axes[0]
    # SVD sign is arbitrary; pin it to an episode-independent spatial convention.
    if axis[np.argmax(np.abs(axis))] < 0:
        axis = -axis
    signal = centered @ axis
    energy = float(np.sum(singular**2))
    explained = float(singular[0] ** 2 / energy) if energy > 1e-16 else 0.0
    extent = float(np.percentile(signal, 95) - np.percentile(signal, 5))
    return signal, axis, explained, extent


def _correlation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """Cosine-normalized overlapping autocorrelation, without zero padding."""
    n = len(signal)
    numerator = np.correlate(signal, signal, mode="full")[n - 1 : n + max_lag]
    squares = np.r_[0, np.cumsum(signal**2)]
    lags = np.arange(max_lag + 1)
    denominator = np.sqrt(squares[n - lags] * (squares[n] - squares[lags]))
    return np.divide(
        numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-15
    )


def _measure_window(
    xyz: np.ndarray,
    times: np.ndarray,
    quality: np.ndarray,
    dt: float,
    cfg: RecurrenceConfig,
) -> dict:
    signal, axis, explained, extent = _axis_signal(xyz)
    evidence = {
        "axis_world_xyz": axis.tolist(),
        "axis_variance_fraction": explained,
        "motion_extent_m": extent,
        "mean_quality": float(quality.mean()),
    }
    if extent < cfg.minimum_extent_m:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "insufficient_motion_extent",
        }
    if explained < cfg.minimum_axis_variance_fraction:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "no_dominant_motion_axis",
        }
    low = max(2, int(math.ceil(cfg.minimum_period_s / dt)))
    high = min(int(math.floor(cfg.maximum_period_s / dt)), len(signal) - 3)
    correlation = _correlation(signal, min(high + 1, len(signal) - 2))
    peaks, _ = find_peaks(correlation, prominence=0.05)
    candidates = peaks[(peaks >= low) & (peaks <= high)]
    if not len(candidates):
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "no_periodic_correlation_peak",
        }
    strongest = float(correlation[candidates].max())
    if strongest < cfg.minimum_autocorrelation:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "weak_autocorrelation",
            "autocorrelation": strongest,
        }
    # Earliest comparably strong local peak favors the fundamental over multiples.
    eligible = candidates[
        (correlation[candidates] >= cfg.minimum_autocorrelation)
        & (correlation[candidates] >= 0.90 * strongest)
    ]
    lag = int(eligible[0])
    refinement = 0.0
    if 0 < lag < len(correlation) - 1:
        curvature = correlation[lag - 1] - 2 * correlation[lag] + correlation[lag + 1]
        if abs(curvature) > 1e-12:
            refinement = float(
                np.clip(
                    0.5 * (correlation[lag - 1] - correlation[lag + 1]) / curvature,
                    -0.5,
                    0.5,
                )
            )
    period = (lag + refinement) * dt
    evidence.update({"period_s": period, "autocorrelation": float(correlation[lag])})
    if len(signal) * dt / period < cfg.minimum_cycles_per_window:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "too_few_supported_cycles",
        }
    extrema, _ = find_peaks(
        signal,
        distance=max(1, int(0.60 * period / dt)),
        prominence=max(0.20 * extent, cfg.minimum_extent_m * 0.10),
    )
    if len(extrema) < 3:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "too_few_observed_peaks",
        }
    intervals = np.diff(times[extrema])
    cv = float(intervals.std() / max(intervals.mean(), 1e-12))
    median_interval = float(np.median(intervals))
    mismatch = abs(median_interval - period) / period
    evidence.update(
        {
            "interval_cv": cv,
            "observed_median_period_s": median_interval,
            "period_peak_mismatch_fraction": mismatch,
        }
    )
    if cv > cfg.maximum_interval_cv or mismatch > cfg.period_merge_tolerance:
        return {
            **evidence,
            "accepted": False,
            "rejection_reason": "inconsistent_peak_spacing",
        }
    coherence = float(
        np.clip(
            np.sqrt(max(0.0, correlation[lag]) * explained)
            * (1 - min(cv, 1))
            * quality.mean(),
            0,
            1,
        )
    )
    return {
        **evidence,
        "accepted": True,
        "rejection_reason": None,
        "coherence": coherence,
        "uncertainty": 1 - coherence,
        "peak_offsets": extrema.tolist(),
    }


def _make_bout(
    group: list[dict],
    ep: Episode,
    clean: np.ndarray,
    cfg: RecurrenceConfig,
    bout_index: int,
) -> dict | None:
    period = float(np.median([w["period_s"] for w in group]))
    # Cluster same extrema discovered in overlapping windows. Select a real
    # source-frame extremum, never an interpolated event timestamp.
    candidates = sorted(
        [(p, w["coherence"]) for w in group for p in w["peak_positions"]]
    )
    clusters: list[list[tuple[int, float]]] = []
    for candidate in candidates:
        if (
            not clusters
            or ep.time[candidate[0]] - np.median([ep.time[p] for p, _ in clusters[-1]])
            > 0.25 * period
        ):
            clusters.append([candidate])
        else:
            clusters[-1].append(candidate)
    peaks = sorted(
        {max(cluster, key=lambda x: (x[1], -x[0]))[0] for cluster in clusters}
    )
    # If nearby alternatives escaped clustering, retain the earliest one. Such
    # phase disagreement raises mismatch and prevents an invented full cycle.
    separated: list[int] = []
    for peak in peaks:
        if not separated or ep.time[peak] - ep.time[separated[-1]] >= 0.45 * period:
            separated.append(peak)
    cycles = []
    for start, end in zip(separated[:-1], separated[1:]):
        duration = float(ep.time[end] - ep.time[start])
        mismatch = abs(duration - period) / period
        if mismatch <= cfg.period_merge_tolerance:
            cycles.append(
                {
                    "start_frame": int(ep.frame_index[start]),
                    "end_frame_exclusive": int(ep.frame_index[end]),
                    "start_time_s": float(ep.time[start]),
                    "end_time_s": float(ep.time[end]),
                    "duration_s": duration,
                    "period_residual_fraction": float(mismatch),
                    "status": "machine_proposed",
                }
            )
    if len(cycles) < 2:
        return None
    first = min(w["start_position"] for w in group)
    last = max(w["end_position_exclusive"] for w in group)
    extents = [
        _axis_signal(clean[first:last, hand, 0])[3]
        if np.isfinite(clean[first:last, hand, 0]).all()
        else None
        for hand in (0, 1)
    ]
    if any(extent is None for extent in extents):
        dominant = "unknown"
    elif extents[0] > 1.25 * extents[1]:
        dominant = "left"
    elif extents[1] > 1.25 * extents[0]:
        dominant = "right"
    else:
        dominant = "balanced"
    coherence = float(np.mean([w["coherence"] for w in group]))
    dt = float(np.median(np.diff(ep.time[first:last])))
    return {
        "bout_id": f"ep{ep.episode_id:06d}-rec{bout_index:04d}",
        "episode_id": ep.episode_id,
        "side": group[0]["side"],
        "dominant_side": dominant,
        "dominance_definition": "relative detrended PCA path extent, not semantic role",
        "left_motion_extent_m": extents[0],
        "right_motion_extent_m": extents[1],
        "start_frame": int(ep.frame_index[first]),
        "end_frame_exclusive": int(ep.frame_index[last - 1]) + 1,
        "start_time_s": float(ep.time[first]),
        "end_time_s": float(ep.time[last - 1] + dt),
        "period_s": period,
        "period_iqr_s": float(
            np.subtract(*np.percentile([w["period_s"] for w in group], [75, 25]))
        ),
        "coherence": coherence,
        "uncertainty": 1 - coherence,
        "supporting_windows": len(group),
        "window_ids": [w["window_id"] for w in group],
        "peak_frames": [int(ep.frame_index[p]) for p in separated],
        "peak_times_s": [float(ep.time[p]) for p in separated],
        "cycle_count": len(cycles),
        "cycles": cycles,
        "status": "machine_proposed",
        "layer": "experimental_recurrence",
        "method": METHOD,
        "interpretation": "Repeated wrist-path oscillations; no object/action semantics asserted.",
    }


def _suppress_competing_bouts(bouts: list[dict]) -> tuple[list[dict], list[dict]]:
    """Preserve alternative hypotheses while removing same-hand double counts."""
    ordered = sorted(
        bouts,
        key=lambda b: (
            -b["coherence"] * math.sqrt(b["supporting_windows"]),
            b["bout_id"],
        ),
    )
    retained: list[dict] = []
    suppressed: list[dict] = []
    for bout in ordered:
        conflict = None
        for other in retained:
            if bout["side"] != other["side"]:
                continue
            overlap = max(
                0.0,
                min(bout["end_time_s"], other["end_time_s"])
                - max(bout["start_time_s"], other["start_time_s"]),
            )
            shorter = min(
                bout["end_time_s"] - bout["start_time_s"],
                other["end_time_s"] - other["start_time_s"],
            )
            if shorter > 0 and overlap / shorter > 0.25:
                conflict = other["bout_id"]
                break
        if conflict is not None:
            suppressed.append(
                {
                    **bout,
                    "status": "suppressed_alternative",
                    "suppressed_by": conflict,
                    "suppression_reason": "same-hand temporal overlap exceeds 25% of shorter bout",
                }
            )
        else:
            retained.append(bout)
    return sorted(retained, key=lambda b: (b["start_time_s"], b["side"])), suppressed


def detect_recurrence(
    ep: Episode,
    clean_world: np.ndarray,
    features: pd.DataFrame,
    config: dict | None = None,
) -> dict:
    """Detect quality-gated candidate rhythmic bouts; never bridge data gaps.

    Uses only inherited world-coordinate wrist positions from ``clean_world``.
    Projection to a camera image or subtracting head translation is unnecessary
    here; camera-independent coordinates already remove image-plane head motion.
    Tracking drift and hand/body co-motion can still affect these proposals.
    """
    cfg = RecurrenceConfig.from_dict(config)
    clean = np.asarray(clean_world, dtype=float)
    n = len(ep.time)
    if clean.shape != (n, 2, 21, 3) or len(features) != n:
        raise ValueError(
            "clean_world must be N-by-2-by-21-by-3 and features must have N rows"
        )
    if not {"usable", "quality_score", "frame_index", "timestamp_s"} <= set(features):
        raise ValueError(
            "features require usable, quality_score, frame_index, timestamp_s"
        )
    if not np.array_equal(
        features.frame_index.to_numpy(), ep.frame_index
    ) or not np.allclose(features.timestamp_s.to_numpy(), ep.time, equal_nan=True):
        raise ValueError("Feature and episode rows must align exactly")
    quality = features.quality_score.to_numpy(float)
    if not np.isfinite(quality).all() or np.any((quality < 0) | (quality > 1)):
        raise ValueError("quality_score must be finite and within [0, 1]")
    if (
        features.usable.isna().any()
        or not features.usable.isin([0, 1, True, False]).all()
    ):
        raise ValueError("usable must be boolean or binary integer")
    windows: list[dict] = []
    bouts: list[dict] = []
    result = {
        "method": METHOD,
        "episode_id": ep.episode_id,
        "config": asdict(cfg),
        "windows": windows,
        "bouts": bouts,
        "suppressed_bouts": [],
        "interpretation": "Experimental periodic-motion proposals; cycles are not action counts.",
    }
    if n < 2:
        result["statistics"] = {
            "windows": 0,
            "accepted_windows": 0,
            "bouts": 0,
            "cycles": 0,
            "suppressed_bouts": 0,
            "rejection_reasons": {},
            "insufficient_episode_length": True,
        }
        return result
    delta = np.diff(ep.time)
    consecutive = np.diff(ep.frame_index) == 1
    good_delta = np.isfinite(delta) & (delta > 0) & consecutive
    dt = float(np.median(delta[good_delta])) if good_delta.any() else 1 / 30
    width = max(4, int(round(cfg.window_s / dt)))
    stride = max(1, int(round(cfg.stride_s / dt)))
    starts = list(range(0, n - width + 1, stride))
    if starts and starts[-1] != n - width:
        starts.append(n - width)
    usable = features.usable.to_numpy(bool) & (quality >= cfg.minimum_quality)
    breaks = (~good_delta) | (
        np.abs(delta - dt) > cfg.maximum_sample_jitter_fraction * dt
    )
    for window_index, start in enumerate(starts):
        end = start + width
        for hand, side in enumerate(("left", "right")):
            record = {
                "window_id": f"ep{ep.episode_id:06d}-win{window_index:05d}-{side}",
                "episode_id": ep.episode_id,
                "window_index": window_index,
                "side": side,
                "start_position": start,
                "end_position_exclusive": end,
                "start_frame": int(ep.frame_index[start]),
                "end_frame_exclusive": int(ep.frame_index[end - 1]) + 1,
                "start_time_s": float(ep.time[start]),
                "end_time_s": float(ep.time[end - 1] + dt),
                "method": METHOD,
                "status": "machine_proposed",
            }
            if (
                not usable[start:end].all()
                or not np.isfinite(clean[start:end, hand, 0]).all()
                or not np.isfinite(ep.time[start:end]).all()
            ):
                record.update(
                    {
                        "accepted": False,
                        "rejection_reason": "unusable_or_low_quality_window",
                    }
                )
            elif breaks[start : end - 1].any():
                record.update(
                    {"accepted": False, "rejection_reason": "timestamp_or_frame_gap"}
                )
            else:
                measured = _measure_window(
                    clean[start:end, hand, 0],
                    ep.time[start:end],
                    quality[start:end],
                    dt,
                    cfg,
                )
                offsets = measured.pop("peak_offsets", [])
                record.update(measured)
                if offsets:
                    positions = [start + p for p in offsets]
                    record["peak_positions"] = positions
                    record["peak_frames"] = [int(ep.frame_index[p]) for p in positions]
                    record["peak_times_s"] = [float(ep.time[p]) for p in positions]
            windows.append(record)
    for side in ("left", "right"):
        accepted = [w for w in windows if w["side"] == side and w["accepted"]]
        groups: list[list[dict]] = []
        for window in accepted:
            compatible = False
            if groups:
                prev = groups[-1][-1]
                median_period = float(np.median([w["period_s"] for w in groups[-1]]))
                compatible = (
                    window["window_index"] == prev["window_index"] + 1
                    and window["start_position"] < prev["end_position_exclusive"]
                    and abs(window["period_s"] - median_period) / median_period
                    <= cfg.period_merge_tolerance
                )
            if compatible:
                groups[-1].append(window)
            else:
                groups.append([window])
        for group in groups:
            if len(group) >= cfg.minimum_bout_windows:
                bout = _make_bout(group, ep, clean, cfg, len(bouts))
                if bout is not None:
                    bouts.append(bout)
    bouts, suppressed = _suppress_competing_bouts(bouts)
    result["bouts"] = bouts
    result["suppressed_bouts"] = suppressed
    result["statistics"] = {
        "windows": len(windows),
        "accepted_windows": sum(w["accepted"] for w in windows),
        "bouts": len(bouts),
        "cycles": sum(b["cycle_count"] for b in bouts),
        "suppressed_bouts": len(suppressed),
        "rejection_reasons": dict(
            Counter(w["rejection_reason"] for w in windows if not w["accepted"])
        ),
        "insufficient_episode_length": n < width,
        "note": "Overlapping hands can describe the same physical activity; cycle totals are side-specific motion events.",
    }
    return result
