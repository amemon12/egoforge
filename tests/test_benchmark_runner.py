"""Accounting checks that prevent inflated benchmark conclusions."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_benchmarks.py"
spec = importlib.util.spec_from_file_location("benchmark_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_preexisting_flags_are_not_credited_as_new_detections():
    result = runner.detection_counts(
        [True, False, False, True], [True, True, False, False], [0, 1, 2]
    )
    assert result["target_frames"] == 3
    assert result["baseline_flagged_targets"] == 1
    assert result["eligible_targets"] == 2
    assert result["newly_detected_targets"] == 1
    assert result["delta_sensitivity"] == 0.5
    assert result["post_fault_sensitivity"] == pytest.approx(2 / 3)


def test_no_eligible_targets_is_unmeasured_not_perfect():
    result = runner.detection_counts([True, True], [True, True], [0, 1])
    assert result["eligible_targets"] == 0
    assert result["delta_sensitivity"] is None
    assert result["post_fault_sensitivity"] == 1
    empty = runner.detection_counts([False], [False], [])
    assert empty["delta_sensitivity"] is None
    assert empty["post_fault_sensitivity"] is None


def test_new_flags_outside_injected_targets_are_retained_separately():
    result = runner.detection_counts([False] * 4, [True, True, True, False], [1])
    assert result["newly_detected_targets"] == 1
    assert result["newly_flagged_outside_targets"] == 2


@pytest.mark.parametrize("target", [[1, 1], [-1], [2]])
def test_invalid_target_denominators_rejected(target):
    with pytest.raises(ValueError):
        runner.detection_counts([False, False], [True, True], target)


def test_dwell_count_excludes_observation_boundaries():
    labels = ["a", "a", "b", "a", "a", "unknown", "b", "a", "a"]
    assert runner.short_interior_runs(labels, np.arange(9), 2) == 1
    # The same isolated b is now at a frame gap and must be censored.
    frames = np.array([0, 1, 3, 4, 5, 6, 7, 8, 9])
    assert runner.short_interior_runs(labels, frames, 2) == 0


def test_dwell_count_ignores_start_and_end_fragments():
    assert runner.short_interior_runs(["a", "b", "b", "a"], np.arange(4), 3) == 1
    assert runner.short_interior_runs(["a", "b", "b", "a"], np.arange(4), 2) == 0
    assert runner.short_interior_runs([], np.array([]), 9) == 0


@pytest.mark.parametrize("system,expected", [("darwin", 1000), ("linux", 1024000)])
def test_rss_units_are_platform_correct(monkeypatch, system, expected):
    monkeypatch.setattr(runner.sys, "platform", system)
    monkeypatch.setattr(
        runner.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1000)
    )
    assert runner.rss_bytes() == expected
