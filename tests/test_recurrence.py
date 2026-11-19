from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from egoforge.recurrence import detect_recurrence, _suppress_competing_bouts


def episode(period=0.7, seconds=12, amplitude=0.04):
    t = np.arange(round(seconds * 30)) / 30
    xyz = np.zeros((len(t), 2, 21, 3))
    xyz[:, 0, :, 0] = (amplitude * np.sin(2 * np.pi * t / period))[:, None]
    # Constant offsets and linear movement are removed before recurrence scoring.
    xyz[:, :, :, 1] = (0.2 + 0.015 * t)[:, None, None]
    ep = SimpleNamespace(episode_id=4, time=t, frame_index=np.arange(len(t)))
    features = pd.DataFrame(
        {
            "frame_index": ep.frame_index,
            "timestamp_s": t,
            "quality_score": np.ones(len(t)),
            "usable": np.ones(len(t), bool),
        }
    )
    return ep, xyz, features


def test_sinusoidal_world_trajectory_recovers_known_period():
    ep, clean, features = episode(period=0.7)
    result = detect_recurrence(ep, clean, features)
    assert result["bouts"]
    assert all(b["side"] == "left" for b in result["bouts"])
    assert all(b["dominant_side"] == "left" for b in result["bouts"])
    assert np.median([b["period_s"] for b in result["bouts"]]) == pytest.approx(
        0.7, abs=0.025
    )
    assert all(b["status"] == "machine_proposed" for b in result["bouts"])
    assert sum(b["cycle_count"] for b in result["bouts"]) >= 10
    assert all(
        abs(c["duration_s"] - 0.7) < 0.07 for b in result["bouts"] for c in b["cycles"]
    )


def test_stationary_and_linear_trajectories_rejected():
    ep, clean, features = episode(amplitude=0)
    result = detect_recurrence(ep, clean, features)
    assert not result["bouts"]
    assert result["statistics"]["accepted_windows"] == 0
    assert set(result["statistics"]["rejection_reasons"]) == {
        "insufficient_motion_extent"
    }


def test_invalid_tracking_window_is_not_interpolated():
    ep, clean, features = episode(seconds=15)
    features.loc[210, "usable"] = False
    clean[210] = np.nan
    result = detect_recurrence(ep, clean, features)
    assert any(
        w["rejection_reason"] == "unusable_or_low_quality_window"
        for w in result["windows"]
    )
    assert result["bouts"]
    assert not any(
        b["start_frame"] <= 210 < b["end_frame_exclusive"] for b in result["bouts"]
    )


def test_frame_and_clock_gaps_cannot_bridge_bouts():
    ep, clean, features = episode(seconds=15)
    ep.time[210:] += 0.5
    ep.frame_index[210:] += 15
    features.timestamp_s = ep.time
    features.frame_index = ep.frame_index
    result = detect_recurrence(ep, clean, features)
    assert any(
        w["rejection_reason"] == "timestamp_or_frame_gap" for w in result["windows"]
    )
    assert not any(
        b["start_frame"] < 225 < b["end_frame_exclusive"] for b in result["bouts"]
    )


def test_quality_gate_and_determinism():
    ep, clean, features = episode()
    first = detect_recurrence(ep, clean, features)
    assert first == detect_recurrence(ep, clean, features)
    features.quality_score = 0.2
    result = detect_recurrence(ep, clean, features)
    assert not result["bouts"]
    assert result["statistics"]["accepted_windows"] == 0


def test_rigid_world_rotation_preserves_detected_period():
    ep, clean, features = episode(period=0.9)
    from scipy.spatial.transform import Rotation

    rotation = Rotation.from_euler("xyz", [0.3, -0.4, 0.5]).as_matrix()
    rotated = np.einsum("ij,nhkj->nhki", rotation, clean)
    a = detect_recurrence(ep, clean, features)
    b = detect_recurrence(ep, rotated, features)
    assert np.median([x["period_s"] for x in a["bouts"]]) == pytest.approx(
        np.median([x["period_s"] for x in b["bouts"]]), abs=1e-10
    )


def test_short_episode_is_explicitly_unsupported():
    ep, clean, features = episode(seconds=2)
    result = detect_recurrence(ep, clean, features)
    assert result["statistics"]["insufficient_episode_length"]
    assert not result["windows"] and not result["bouts"]


def test_misaligned_rows_fail():
    ep, clean, features = episode()
    features.loc[0, "timestamp_s"] += 0.1
    with pytest.raises(ValueError, match="align"):
        detect_recurrence(ep, clean, features)


def test_competing_same_hand_hypotheses_do_not_double_count():
    primary = {
        "bout_id": "a",
        "side": "left",
        "coherence": 0.8,
        "supporting_windows": 4,
        "start_time_s": 0,
        "end_time_s": 8,
    }
    alternative = {
        **primary,
        "bout_id": "b",
        "coherence": 0.6,
        "start_time_s": 4,
        "end_time_s": 10,
    }
    opposite = {**alternative, "bout_id": "c", "side": "right"}
    kept, suppressed = _suppress_competing_bouts([alternative, opposite, primary])
    assert {b["bout_id"] for b in kept} == {"a", "c"}
    assert suppressed[0]["bout_id"] == "b"
    assert suppressed[0]["suppressed_by"] == "a"
    assert suppressed[0]["status"] == "suppressed_alternative"
