"""Release-artifact integration checks; skipped until a real pipeline run exists.

These inspect actual independently generated tables, intervals, and exchange
formats. They do not synthesize expected output by rerunning the implementation.
"""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import pytest


PROJECT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    root = PROJECT / "output" / "run"
    if not (root / "artifact_manifest.json").is_file():
        pytest.skip(
            "Generate a complete release with `egoforge run` before integration checks"
        )

    def load(name):
        return json.loads((root / name).read_text())

    return {
        "root": root,
        "summary": load("summary.json"),
        "episodes": load("episodes.json"),
        "segments": load("segments.json"),
        "queue": load("review_queue.json"),
        "frames": pd.read_parquet(root / "annotations.parquet"),
        "manifest": load("artifact_manifest.json"),
        "provenance": load("provenance.json"),
        "label_studio": load("exports/label_studio_predictions.json"),
    }


def test_release_artifact_hashes_and_run_identity(release):
    root, manifest, summary = release["root"], release["manifest"], release["summary"]
    assert manifest["run_id"] == summary["run_id"] == release["provenance"]["run_id"]
    assert set(release["frames"].run_id) == {summary["run_id"]}
    assert {segment["run_id"] for segment in release["segments"]} == {summary["run_id"]}
    for relative, expected in manifest["artifacts"].items():
        path = (root / relative).resolve()
        assert path.is_relative_to(root.resolve()), relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, relative
    for relative, expected in release["provenance"]["identity"]["code"].items():
        assert (
            hashlib.sha256((PROJECT / "egoforge" / relative).read_bytes()).hexdigest()
            == expected
        ), relative


def test_segments_partition_frames_and_match_per_frame_annotations(release):
    all_frames, segments = release["frames"], release["segments"]
    assert len(all_frames) == release["summary"]["total_frames"]
    assert len(segments) == release["summary"]["segments"]
    assert all_frames[["episode_id", "frame_index"]].duplicated().sum() == 0
    assert len({segment["segment_id"] for segment in segments}) == len(segments)
    for episode in release["episodes"]:
        frames = all_frames[all_frames.episode_id == episode["episode_id"]].sort_values(
            "frame_index"
        )
        intervals = sorted(
            (s for s in segments if s["episode_id"] == episode["episode_id"]),
            key=lambda s: s["start_frame"],
        )
        assert len(frames) == episode["frame_count"]
        assert intervals[0]["start_frame"] == int(frames.frame_index.iloc[0])
        assert (
            intervals[-1]["end_frame_exclusive"] == int(frames.frame_index.iloc[-1]) + 1
        )
        assert all(
            a["end_frame_exclusive"] == b["start_frame"]
            for a, b in zip(intervals, intervals[1:])
        ), episode["episode_id"]
        assert all(
            a["end_time_s"] == pytest.approx(b["start_time_s"])
            for a, b in zip(intervals, intervals[1:])
        ), episode["episode_id"]
        for segment in intervals:
            rows = frames[
                (frames.frame_index >= segment["start_frame"])
                & (frames.frame_index < segment["end_frame_exclusive"])
            ]
            assert len(rows) == segment["frame_count"] > 0, segment["segment_id"]
            assert set(rows.state) == {segment["state"]}, segment["segment_id"]
            assert set(rows.segment_id) == {segment["segment_id"]}, segment[
                "segment_id"
            ]
            assert segment["status"] == "machine_proposed"
            assert (
                np.isfinite(segment["uncertainty"]) and 0 <= segment["uncertainty"] <= 1
            )
            assert segment["end_time_s"] > segment["start_time_s"]


def test_review_queue_is_bounded_unique_and_covers_release(release):
    queue, segments = release["queue"], release["segments"]
    known = {segment["segment_id"] for segment in segments}
    assert len(queue) == release["summary"]["review_queue_size"]
    assert len({segment["segment_id"] for segment in queue}) == len(queue)
    assert all(segment["segment_id"] in known for segment in queue)
    assert {segment["episode_id"] for segment in queue} == {
        episode["episode_id"] for episode in release["episodes"]
    }


def test_label_studio_predictions_round_trip_to_native_frame_labels(release):
    tasks = release["label_studio"]
    assert len(tasks) == 1
    assert tasks[0]["data"]["episode_id"] == 4
    assert not tasks[0].get("annotations"), (
        "Machine proposals must not be presented as completed human annotations"
    )
    assert len(tasks[0]["predictions"]) == 1
    regions = tasks[0]["predictions"][0]["result"]
    frames = release["frames"][release["frames"].episode_id == 4].sort_values(
        "frame_index"
    )
    decoded = np.empty(len(frames), dtype=object)
    decoded[:] = None
    coverage = np.zeros(len(frames), dtype=int)
    # HumanSignal's official converter independently defines LS as start-1:end:
    # https://github.com/HumanSignal/label-studio-ml-backend/blob/master/
    # label_studio_ml/examples/yolo/utils/converter.py#L46-L52
    for region in regions:
        assert region["type"] == "timelinelabels"
        assert region["from_name"] == "phase" and region["to_name"] == "video"
        (span,) = region["value"]["ranges"]
        (label,) = region["value"]["timelinelabels"]
        assert 1 <= span["start"] <= span["end"] <= len(frames)
        decoded[span["start"] - 1 : span["end"]] = label
        coverage[span["start"] - 1 : span["end"]] += 1
    np.testing.assert_array_equal(coverage, np.ones(len(frames), dtype=int))
    np.testing.assert_array_equal(decoded, frames.state.to_numpy())
    config = ElementTree.parse(
        release["root"] / "exports" / "label_studio_config.xml"
    ).getroot()
    video = config.find("Video")
    assert video is not None
    # The official Video tag defaults to24 fps; source streams are30 fps.
    assert float(video.attrib["frameRate"]) == 30
    labels = {label.attrib["value"] for label in config.findall("TimelineLabels/Label")}
    assert set(decoded).issubset(labels)
