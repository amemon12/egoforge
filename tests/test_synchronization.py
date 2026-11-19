from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from egoforge.synchronization import align_sensor_clocks


def clock_episode(
    n=240, clock_scale=1.0, gyro_lag_frames=0.0, epoch=1_780_000_000_000_000_123
):
    frame = np.arange(n, dtype=float)
    capture = frame / 30 * clock_scale
    frequency_a, frequency_b = 2 * np.pi * 0.6, 2 * np.pi * 1.2

    def angle(t):
        return (
            2 * t
            - 0.4 / frequency_a * np.cos(frequency_a * t)
            - 0.25 / frequency_b * np.cos(frequency_b * t)
        )

    def angular_speed(t):
        return 2 + 0.4 * np.sin(frequency_a * t) + 0.25 * np.sin(frequency_b * t)

    head = np.zeros((n, 7))
    head[:, 3:] = Rotation.from_rotvec(
        np.stack([np.zeros(n), np.zeros(n), angle(capture)], axis=1)
    ).as_quat()
    imu_frame = np.arange(0, n, 0.15)
    imu_time = imu_frame / 30 * clock_scale
    gyro = angular_speed((imu_frame - gyro_lag_frames) / 30 * clock_scale)
    table = pd.DataFrame(
        {
            "frame": imu_frame,
            "timestamp_ns": epoch + np.rint(imu_time * 1e9).astype(np.int64),
            "timestamp_s": imu_frame
            / 30,  # Deliberately nominal, not used for alignment.
            "angular_velocity_x": np.zeros(len(imu_frame)),
            "angular_velocity_y": np.zeros(len(imu_frame)),
            "angular_velocity_z": gyro,
        }
    )
    return SimpleNamespace(
        frame_index=frame.astype(int),
        time=frame / 30,
        head=head,
        imu={
            name: table.copy(deep=True)
            for name in ["main_stereo", "aux_left", "aux_right"]
        },
    )


def test_integer_epoch_subtraction_preserves_submicrosecond_precision():
    ep = clock_episode(n=10, clock_scale=1.00000003)
    report, masks = align_sensor_clocks(ep)
    assert report["anchor_timestamp_ns"] == 1_780_000_000_000_000_123
    expected = ep.frame_index / 30 * 1.00000003
    np.testing.assert_allclose(
        report["frame_capture_time_s"], expected, atol=1e-9, rtol=0
    )
    assert all(not mask.any() for mask in masks.values())


def test_measured_clock_drift_does_not_become_false_missing_coverage():
    ep = clock_episode(clock_scale=1.05)
    report, masks = align_sensor_clocks(ep)
    np.testing.assert_allclose(
        report["frame_capture_time_s"], ep.time * 1.05, atol=1e-9
    )
    assert all(not mask.any() for mask in masks.values())
    assert all(
        sensor["clock_mismatch_frame_count"] == 0
        for sensor in report["sensors"].values()
    )
    assert report["motion_alignment"]["best_lag_frames"] == 0
    assert report["motion_alignment"]["correlation_at_zero_lag"] > 0.999


def test_dense_sensor_with_shifted_absolute_epoch_is_detected():
    ep = clock_episode()
    ep.imu["aux_left"]["timestamp_ns"] += 200_000_000
    report, masks = align_sensor_clocks(ep)
    sensor = report["sensors"]["aux_left"]
    assert sensor["clock_mismatch_frame_count"] == len(ep.frame_index)
    assert sensor["same_frame_clock_offset_s"]["median"] == pytest.approx(0.2)
    assert masks["aux_left"][:5].all()
    assert not masks["main_stereo"].any()


def test_missing_sensor_burst_marks_shared_time_coverage_only_where_missing():
    ep = clock_episode()
    table = ep.imu["aux_right"]
    ep.imu["aux_right"] = table[~table.frame.between(60, 90)].copy()
    report, masks = align_sensor_clocks(ep)
    assert masks["aux_right"][62:89].all()
    assert not masks["aux_right"][:59].any()
    assert not masks["main_stereo"].any()
    assert report["sensors"]["aux_right"]["gap_count"] == 1


def test_head_clock_hole_abstains_from_inventing_frame_capture_times():
    ep = clock_episode()
    table = ep.imu["main_stereo"]
    ep.imu["main_stereo"] = table[~table.frame.between(60, 90)].copy()
    report, masks = align_sensor_clocks(ep)
    assert np.isnan(report["frame_capture_time_s"][62:89]).all()
    assert all(mask[62:89].all() for mask in masks.values())


@pytest.mark.parametrize("lag", [-2.0, 0.0, 2.0])
def test_lag_estimation_uses_rotation_interval_midpoint_and_documented_sign(lag):
    report, _ = align_sensor_clocks(clock_episode(gyro_lag_frames=lag))
    result = report["motion_alignment"]
    assert result["best_lag_frames"] == lag
    assert result["correlation_at_best_lag"] > 0.999
    assert not result["search_boundary_hit"]


def test_invalid_reference_and_empty_sensor_abstain_without_crashing():
    ep = clock_episode()
    ep.imu["aux_left"] = ep.imu["aux_left"].iloc[:0]
    report, masks = align_sensor_clocks(ep)
    assert masks["aux_left"].all()
    assert report["sensors"]["aux_left"]["status"] == "invalid_or_nonmonotonic_clock"
    ep.imu["main_stereo"].loc[10, "timestamp_ns"] = ep.imu[
        "main_stereo"
    ].timestamp_ns.iloc[9]
    report, masks = align_sensor_clocks(ep)
    assert report["status"] == "reference_clock_invalid"
    assert all(mask.all() for mask in masks.values())


def test_no_mutation_and_quaternion_sign_invariance():
    ep = clock_episode(gyro_lag_frames=2)
    ep.head[1::2, 3:] *= -1
    before = deepcopy(ep)
    report, _ = align_sensor_clocks(ep)
    assert report["motion_alignment"]["best_lag_frames"] == 2
    np.testing.assert_array_equal(ep.head, before.head)
    for sensor in ep.imu:
        pd.testing.assert_frame_equal(ep.imu[sensor], before.imu[sensor])
