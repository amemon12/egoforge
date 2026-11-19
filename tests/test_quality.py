"""Independent geometric and data-integrity checks using synthetic sensor data."""

from copy import deepcopy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from egoforge.benchmark import corruption_benchmark
from egoforge.data import Episode, load_episode
from egoforge.quality import audit_episode, safe_derivative


def synthetic_episode(n=120):
    """Known camera-to-world transform, positive depth, and intact sample clock."""
    time = np.arange(n, dtype=float) / 30
    camera = np.zeros((n, 2, 21, 3), dtype=float)
    for hand in range(2):
        camera[:, hand, :, 0] = -0.16 + hand * 0.32 + np.arange(21) * 0.002
        camera[:, hand, :, 1] = np.arange(21) * 0.003
        camera[:, hand, :, 2] = 1.3 + np.arange(21) * 0.001
    orientation = Rotation.from_euler("xyz", [23, -37, 61], degrees=True)
    translation = np.array([0.2, -0.4, 0.7])
    world = orientation.apply(camera.reshape(-1, 3)).reshape(camera.shape) + translation
    head = np.tile(np.r_[translation, orientation.as_quat()], (n, 1))
    intrinsics = {
        "left": {"fx": 700.0, "fy": 710.0, "cx": 960.0, "cy": 600.0},
        "resolution": {"width": 1920, "height": 1200},
    }
    keypoints = camera[..., :2] / camera[..., 2, None] * [700, 710] + [960, 600]
    state = np.zeros((n, 188), dtype=float)
    state[:, :63] = world[:, 0].reshape(n, 63)
    state[:, 94:157] = world[:, 1].reshape(n, 63)
    state[:, [93, 187]] = 0.4
    imu_time = np.arange(0, n / 30, 0.005)
    imu = pd.DataFrame(
        {
            "sample_index": np.arange(len(imu_time)),
            "frame": imu_time * 30,
            "timestamp_ns": 1_780_000_000_000_000_000
            + np.rint(imu_time * 1e9).astype(np.int64),
            "timestamp_s": imu_time,
            "angular_velocity_x": np.zeros(len(imu_time)),
            "angular_velocity_y": np.zeros(len(imu_time)),
            "angular_velocity_z": np.zeros(len(imu_time)),
            "linear_acceleration_x": np.zeros(len(imu_time)),
            "linear_acceleration_y": np.full(len(imu_time), 9.81),
            "linear_acceleration_z": np.zeros(len(imu_time)),
        }
    )
    return Episode(
        episode_id=4,
        frame_index=np.arange(n),
        time=time,
        state=state,
        head=head,
        world=world,
        camera=camera,
        keypoints=keypoints,
        present=np.ones((n, 2), dtype=bool),
        tracking_quality=np.zeros(n, dtype=int),
        task_index=4,
        source_sha256="a" * 64,
        imu={
            name: imu.copy(deep=True)
            for name in ["main_stereo", "aux_left", "aux_right"]
        },
        intrinsics=intrinsics,
    )


def reproject_from_world(ep):
    """Update coherent source representations after a known synthetic movement."""
    rotation = Rotation.from_quat(ep.head[:, 3:]).as_matrix()
    ep.camera[:] = np.einsum(
        "nji,nhkj->nhki", rotation, ep.world - ep.head[:, None, None, :3]
    )
    ep.keypoints[:] = ep.camera[..., :2] / ep.camera[..., 2, None] * [700, 710] + [
        960,
        600,
    ]
    ep.state[:, :63] = ep.world[:, 0].reshape(len(ep.time), 63)
    ep.state[:, 94:157] = ep.world[:, 1].reshape(len(ep.time), 63)


def audit_quietly(ep, config=None):
    # Absent/quarantined frames intentionally generate all-NaN reduction slices.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return audit_episode(ep, config)


def test_nontrivial_camera_to_world_rotation_and_pinhole_geometry():
    ep = synthetic_episode()
    features, clean, report = audit_episode(ep)
    assert features.usable.all()
    assert features.world_camera_residual_m.max() < 1e-12
    assert features.left_reprojection_px.max() < 1e-12
    assert features.right_reprojection_px.max() < 1e-12
    assert report["quarantined_frames"] == 0
    np.testing.assert_allclose(clean, ep.world, rtol=0, atol=1e-12)


def test_quaternion_sign_equivalence_does_not_create_motion_or_geometry_error():
    ep = synthetic_episode()
    ep.head[1::2, 3:] *= -1  # q and -q represent exactly the same orientation.
    features, _, _ = audit_episode(ep)
    assert features.usable.all()
    assert features.head_angular_speed_rps.max() < 1e-10
    assert features.world_camera_residual_m.max() < 1e-12


def test_zero_filled_absent_hand_is_masked_and_never_becomes_origin_motion():
    ep = synthetic_episode()
    missing = 30
    ep.present[missing, 0] = False
    ep.tracking_quality[missing] = 1
    ep.world[missing, 0] = ep.camera[missing, 0] = 0
    ep.keypoints[missing, 0] = 0
    ep.state[missing, :94] = 0
    features, clean, report = audit_quietly(ep)
    assert not features.usable.iloc[missing]
    assert "hand_unobserved" in features.qa_flags.iloc[missing]
    assert "presence_payload_conflict" not in features.qa_flags.iloc[missing]
    assert np.isnan(clean[missing]).all()
    assert np.isnan(features.left_speed_mps.iloc[missing])
    assert np.nanmax(features.left_speed_mps.to_numpy()) < 1e-10
    assert report["cleaning"]["absent_hand_samples"] == 1
    assert report["cleaning"]["rows_deleted"] == 0
    assert len(features) == len(ep.time)


def test_missing_flag_with_nonzero_payload_is_a_quarantined_contradiction():
    ep = synthetic_episode()
    ep.present[30, 0] = False
    ep.tracking_quality[30] = 1
    features, clean, _ = audit_quietly(ep)
    assert "presence_payload_conflict" in features.qa_flags.iloc[30]
    assert not features.usable.iloc[30]
    assert np.isnan(clean[30]).all()


def test_smoothing_and_derivatives_never_bridge_a_time_break():
    ep = synthetic_episode(n=48)
    split = 24
    ep.time[split:] += 2
    ep.frame_index[split:] += 60
    ep.world[split:] += [0.1, -0.05, 0.2]
    reproject_from_world(ep)
    ep.imu = {}  # This test isolates trajectory segmentation from coverage diagnostics.
    features, clean, _ = audit_quietly(ep)
    assert "timestamp_or_frame_discontinuity" in features.qa_flags.iloc[split]
    assert np.isnan(clean[split]).all()
    np.testing.assert_allclose(clean[:split], ep.world[:split], atol=1e-12)
    np.testing.assert_allclose(clean[split + 1 :], ep.world[split + 1 :], atol=1e-12)
    assert np.nanmax(features.left_speed_mps.to_numpy()) < 1e-10
    assert np.nanmax(features.right_speed_mps.to_numpy()) < 1e-10


def test_derivative_resets_at_an_invalid_run_and_leaves_singletons_undefined():
    values = np.array([[0.0], [1.0], [999.0], [20.0], [22.0], [888.0], [42.0]])
    times = np.arange(len(values), dtype=float)
    valid = np.array([True, True, False, True, True, False, True])
    result = safe_derivative(values, times, valid)
    np.testing.assert_allclose(result[[0, 1, 3, 4], 0], [1, 1, 2, 2])
    assert np.isnan(result[[2, 5, 6]]).all()


def test_geometry_threshold_overrides_change_decisions_with_known_residual():
    ep = synthetic_episode()
    ep.world[30] += [0.02, 0, 0]
    strict, _, _ = audit_quietly(ep, {"transform_fail_m": 0.01})
    relaxed, _, _ = audit_quietly(ep, {"transform_fail_m": 0.03})
    assert strict.world_camera_residual_m.iloc[30] == pytest.approx(0.02)
    assert "world_camera_inconsistent" in strict.qa_flags.iloc[30]
    assert not strict.usable.iloc[30]
    assert "world_camera_inconsistent" not in relaxed.qa_flags.iloc[30]
    assert relaxed.usable.iloc[30]


def test_reprojection_warning_and_quarantine_have_distinct_thresholds():
    ep = synthetic_episode()
    ep.keypoints[30] += [30, 0]
    ep.keypoints[60] += [80, 0]
    features, _, _ = audit_quietly(ep)
    assert features.left_reprojection_px.iloc[30] == pytest.approx(30)
    assert features.usable.iloc[30]
    assert "reprojection_residual" in features.qa_flags.iloc[30]
    assert "reprojection_extreme" not in features.qa_flags.iloc[30]
    assert "reprojection_extreme" in features.qa_flags.iloc[60]
    assert not features.usable.iloc[60]
    relaxed, _, _ = audit_quietly(ep, {"reprojection_warn_px": 40})
    assert "reprojection_residual" not in relaxed.qa_flags.iloc[30]


def test_nonunit_quaternion_is_rejected_without_scipy_crash():
    ep = synthetic_episode()
    ep.head[30, 3:] = 0
    features, clean, _ = audit_quietly(ep)
    assert "quaternion_norm_invalid" in features.qa_flags.iloc[30]
    assert not features.usable.iloc[30]
    assert np.isnan(clean[30]).all()


def test_audit_preserves_every_input_array_and_imu_table():
    ep = synthetic_episode()
    ep.keypoints[30] += [30, 0]
    before = deepcopy(ep)
    audit_quietly(ep)
    for key, value in vars(ep).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(before, key))
    for sensor in ep.imu:
        pd.testing.assert_frame_equal(ep.imu[sensor], before.imu[sensor])
    assert ep.intrinsics == before.intrinsics


def test_benchmark_detects_deliberate_faults_and_preserves_source():
    ep = synthetic_episode(n=400)
    before = deepcopy(ep)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        report = corruption_benchmark(ep)
    assert report["total_cases"] == 9
    assert report["cases_passing"] == report["total_cases"]
    assert all(case["injected_frame_count"] > 0 for case in report["cases"])
    for key, value in vars(ep).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(before, key))
    for sensor in ep.imu:
        pd.testing.assert_frame_equal(ep.imu[sensor], before.imu[sensor])


RAW = Path(__file__).resolve().parents[1] / "data/raw/humanola"


@pytest.mark.skipif(
    not (RAW / "data/chunk-000/episode_000004.parquet").exists(),
    reason="Optional pinned source acquisition is not installed",
)
def test_optional_real_source_coordinate_convention_and_task_disagreement():
    ep = load_episode(RAW, 4)
    rotation = Rotation.from_quat(ep.head[:, 3:]).as_matrix()
    recovered = (
        np.einsum("nij,nhkj->nhki", rotation, ep.camera) + ep.head[:, None, None, :3]
    )
    assert np.linalg.norm(recovered - ep.world, axis=-1).max() < 2e-7
    assert len(ep.time) == 2904
    np.testing.assert_allclose(ep.time, ep.frame_index / 30, atol=1e-12)
    # The untouched source disagreement is intentional evidence, not normalized here.
    if (RAW / "data/chunk-000/episode_000002.parquet").exists():
        assert load_episode(RAW, 2).task_index == 0
