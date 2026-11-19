import numpy as np
import pandas as pd
import pytest

from egoforge.temporal import annotate_episode, duration_viterbi, select_review_segments


def frames(left, right=None, cosine=0.0):
    left = np.asarray(left, float)
    n = len(left)
    return pd.DataFrame(
        {
            "episode_id": np.full(n, 4),
            "frame_index": np.arange(n),
            "timestamp_s": np.arange(n) / 30,
            "left_speed_mps": left,
            "right_speed_mps": np.zeros(n) if right is None else right,
            "velocity_cosine": np.broadcast_to(cosine, n),
            "wrist_distance_m": np.full(n, 0.25),
            "left_grip": np.full(n, 0.4),
            "right_grip": np.full(n, 0.5),
            "quality_score": np.ones(n),
            "usable": np.ones(n, bool),
        }
    )


def test_known_motion_regimes_and_half_open_segments():
    left = np.r_[np.zeros(30), np.full(30, 0.3), np.zeros(30), np.full(30, 0.3)]
    right = np.r_[np.zeros(60), np.full(60, 0.3)]
    out, segments, stats = annotate_episode(frames(left, right, cosine=1))
    assert out.state.iloc[15] == "quiescent"
    assert out.state.iloc[45] == "left_dominant"
    assert out.state.iloc[75] == "right_dominant"
    assert out.state.iloc[105] == "bilateral_coordinated"
    assert segments[0]["start_frame"] == 0
    assert segments[-1]["end_frame_exclusive"] == 120
    assert segments[-1]["end_time_s"] == pytest.approx(4)
    assert sum(s["frame_count"] for s in segments) == len(out)
    assert all(s["status"] == "machine_proposed" for s in segments)
    assert stats["known_frames"] == 120


def test_short_spikes_suppressed_and_interior_dwell_enforced():
    values = np.zeros(150)
    values[20:22] = 0.25
    values[50:52] = 0.25
    values[80:110] = 0.25
    _, segments, stats = annotate_episode(frames(values))
    assert stats["raw_transitions"] == 6
    assert stats["temporal_transitions"] < stats["raw_transitions"]
    assert any(s["state"] == "left_dominant" for s in segments)
    assert all(
        s["frame_count"] >= 9 for s in segments if not s["boundary_censored_right"]
    )


def test_invalid_tracking_and_time_gaps_reset_decoder():
    x = frames(np.full(60, 0.3))
    x.loc[20:24, "usable"] = False
    x.loc[40:, "timestamp_s"] += 2
    x.loc[40:, "frame_index"] += 60
    out, segments, stats = annotate_episode(x)
    assert set(out.state.iloc[20:25]) == {"unknown"}
    assert out.uncertainty.iloc[20:25].eq(1).all()
    assert stats["gaps"] == 1
    assert not any(s["start_frame"] < 100 < s["end_frame_exclusive"] for s in segments)
    assert len([s for s in segments if s["state"] == "left_dominant"]) == 3


def test_nonfinite_speed_and_low_quality_abstain():
    x = frames(np.zeros(20))
    x.loc[5, "left_speed_mps"] = np.nan
    x.loc[9, "quality_score"] = 0.1
    out, _, _ = annotate_episode(x)
    assert out.state.iloc[5] == out.state.iloc[9] == "unknown"


def test_unknown_does_not_join_across_dropped_frames():
    x = frames(np.zeros(10))
    x.usable = False
    x.loc[5:, "frame_index"] += 1
    x.loc[5:, "timestamp_s"] += 1 / 30
    _, segments, _ = annotate_episode(x)
    assert len(segments) == 2
    assert segments[0]["end_frame_exclusive"] == 5
    assert segments[1]["start_frame"] == 6


def test_quality_tempering_increases_uncertainty():
    x = frames(np.full(50, 0.3))
    a, _, _ = annotate_episode(x)
    x.quality_score = 0.3
    b, _, _ = annotate_episode(x)
    assert b.uncertainty.mean() > a.uncertainty.mean()


def test_decoder_matches_exhaustive_optimum_for_small_sequence():
    import itertools

    rng = np.random.default_rng(19)
    logp = np.log(rng.dirichlet([1, 1], size=7))
    penalty = 0.8
    decoded = duration_viterbi(logp, 2, penalty)
    best = -np.inf
    for path in itertools.product(range(2), repeat=7):
        changes = [0] + [i for i in range(1, 7) if path[i] != path[i - 1]] + [7]
        # The last segment may be right-censored; all earlier ones must dwell.
        if any(b - a < 2 for a, b in zip(changes[:-2], changes[1:-1])):
            continue
        score = sum(logp[i, state] for i, state in enumerate(path)) - penalty * (
            len(changes) - 2
        )
        best = max(best, score)
    score = logp[np.arange(7), decoded].sum() - penalty * np.count_nonzero(
        np.diff(decoded)
    )
    assert score == pytest.approx(best)


def test_diverse_queue_covers_episodes_and_is_deterministic():
    pool = []
    for episode in range(3):
        x = frames(np.r_[np.zeros(30), np.full(30, 0.3)])
        x.episode_id = episode
        _, seg, _ = annotate_episode(x)
        pool.extend(seg)
    selected = select_review_segments(pool, budget=3)
    assert {s["episode_id"] for s in selected} == {0, 1, 2}
    assert selected == select_review_segments(list(reversed(pool)), budget=3)
    assert [s["review_rank"] for s in selected] == [1, 2, 3]
    assert all("review_rank" not in s for s in pool)


@pytest.mark.parametrize("field,value", [("quality_score", 2), ("usable", "False")])
def test_malformed_input_fails_loudly(field, value):
    x = frames(np.zeros(3))
    x[field] = value
    with pytest.raises(ValueError):
        annotate_episode(x)


def test_duplicate_indices_are_not_silently_sorted():
    x = frames(np.zeros(3))
    x.loc[1, "frame_index"] = 0
    with pytest.raises(ValueError, match="strictly increasing"):
        annotate_episode(x)


def test_empty_and_singleton_inputs():
    out, segments, stats = annotate_episode(frames([]))
    assert len(out) == len(segments) == stats["frames"] == 0
    out, segments, _ = annotate_episode(frames([0]))
    assert out.state.iloc[0] == "quiescent"
    assert segments[0]["short_boundary_fragment"]


def test_duration_one_can_follow_every_frame():
    emissions = np.log(np.array([[0.99, 0.01], [0.01, 0.99], [0.99, 0.01]]))
    assert list(duration_viterbi(emissions, 1, 0)) == [0, 1, 0]


def review_segment(identifier, episode, state, risk):
    return {
        "segment_id": identifier,
        "episode_id": episode,
        "state": state,
        "start_time_s": 0,
        "end_time_s": 1,
        "review_priority": risk,
    }


def test_review_coverage_cannot_be_overwhelmed_by_high_risk_unknowns():
    all_states = {
        "unknown",
        "quiescent",
        "left_dominant",
        "right_dominant",
        "bilateral_coordinated",
        "bilateral_independent",
    }
    pool = [
        review_segment(f"u{episode}", episode, "unknown", 1.0) for episode in range(12)
    ]
    pool += [
        review_segment(state, 0, state, 0.01)
        for state in sorted(all_states - {"unknown"})
    ]
    selected = select_review_segments(pool, budget=6)
    assert {s["state"] for s in selected} == all_states
    full = select_review_segments(pool, budget=36)
    assert {s["state"] for s in full} == all_states
    assert {s["episode_id"] for s in full} == set(range(12))


def test_minimum_edge_cover_fits_tight_feasible_budget():
    # A naive risk-first seed A/episode0 would need three records. Matching
    # chooses A/episode1 and B/episode0 to cover all vertices in two records.
    pool = [
        review_segment("a0", 0, "A", 1.0),
        review_segment("a1", 1, "A", 0.1),
        review_segment("b0", 0, "B", 0.1),
    ]
    selected = select_review_segments(pool, budget=2)
    assert {s["segment_id"] for s in selected} == {"a1", "b0"}
    assert all(
        s["selection_reason"]["minimum_full_coverage_budget"] == 2 for s in selected
    )
    assert selected == select_review_segments(list(reversed(pool)), budget=2)
