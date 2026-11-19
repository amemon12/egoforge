"""Quality-aware bimanual motion segmentation and diverse review scheduling.

This module proposes labels from inherited pose tracks.  Its emission scores and
uncertainty are heuristic evidence, not calibrated probabilities or ground truth.
No label in this ontology asserts physical contact, intent, or task success.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
import math
from typing import Any

import numpy as np
import pandas as pd


STATES = (
    "quiescent",
    "left_dominant",
    "right_dominant",
    "bilateral_coordinated",
    "bilateral_independent",
)
METHOD = "quality_tempered_duration_viterbi_v1"


@dataclass(frozen=True)
class TemporalConfig:
    moving_speed_mps: float = 0.055
    speed_softness_mps: float = 0.020
    coordination_cosine: float = 0.65
    coordination_softness: float = 0.15
    minimum_duration_s: float = 0.30
    transition_penalty: float = 2.2
    minimum_quality: float = 0.25
    maximum_gap_factor: float = 2.5
    nominal_fps: float = 30.0

    @classmethod
    def from_dict(cls, overrides: dict | None) -> "TemporalConfig":
        values = overrides or {}
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown temporal configuration keys: {sorted(unknown)}")
        cfg = cls(**values)
        for name, value in asdict(cfg).items():
            if not isinstance(value, (int, float)) or not np.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if cfg.speed_softness_mps <= 0 or cfg.coordination_softness <= 0:
            raise ValueError("Emission softness values must be positive")
        if cfg.moving_speed_mps < 0 or cfg.transition_penalty < 0:
            raise ValueError(
                "Speed threshold and transition penalty must be nonnegative"
            )
        if not 0 <= cfg.minimum_quality <= 1:
            raise ValueError("minimum_quality must lie in [0, 1]")
        if not 0 <= cfg.minimum_duration_s <= 5:
            raise ValueError("minimum_duration_s must lie in [0, 5]")
        if cfg.nominal_fps <= 0 or cfg.maximum_gap_factor <= 1:
            raise ValueError(
                "nominal_fps must be positive; maximum_gap_factor must exceed one"
            )
        return cfg


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40, 40)))


def emission_scores(features: pd.DataFrame, cfg: TemporalConfig) -> np.ndarray:
    """Five normalized heuristic scores, tempered by sensor/geometry quality.

    Coordination requires both similar directions and a reasonably balanced
    speed ratio.  These are *instantaneous motion* cues, not evidence of a
    shared object.  Near-zero velocity cosine is uninformative, so its NaNs
    become zero; absent/nonfinite wrist speed is rejected by the caller.
    """
    left = np.maximum(np.nan_to_num(features.left_speed_mps.to_numpy(float)), 0)
    right = np.maximum(np.nan_to_num(features.right_speed_mps.to_numpy(float)), 0)
    a_left = _sigmoid((left - cfg.moving_speed_mps) / cfg.speed_softness_mps)
    a_right = _sigmoid((right - cfg.moving_speed_mps) / cfg.speed_softness_mps)
    cosine = np.nan_to_num(features.velocity_cosine.to_numpy(float), nan=0.0)
    directional = _sigmoid(
        (np.clip(cosine, -1, 1) - cfg.coordination_cosine) / cfg.coordination_softness
    )
    balance = np.minimum(left + 1e-4, right + 1e-4) / np.maximum(
        left + 1e-4, right + 1e-4
    )
    coherence = directional * np.sqrt(balance)
    both = a_left * a_right
    scores = np.column_stack(
        [
            (1 - a_left) * (1 - a_right),
            a_left * (1 - a_right),
            (1 - a_left) * a_right,
            both * coherence,
            both * (1 - coherence),
        ]
    )
    quality = features.quality_score.to_numpy(float)
    scores = quality[:, None] * scores + (1 - quality[:, None]) / len(STATES)
    return scores / scores.sum(axis=1, keepdims=True)


def duration_viterbi(
    log_emissions: np.ndarray, min_steps: int, transition_penalty: float
) -> np.ndarray:
    """Expanded-state Viterbi with an enforced minimum interior dwell time.

    Every semantic state has min_steps age states.  The mature age may stay
    or pay a penalty to enter a different state at age zero.  Ending at an
    immature age is allowed because the final segment is right-censored.
    Time O(T*K*(D+K)), backtrace memory O(T*K*D).
    """
    emissions = np.asarray(log_emissions, dtype=float)
    if emissions.ndim != 2 or not np.all(np.isfinite(emissions)):
        raise ValueError("log_emissions must be a finite T-by-K array")
    n, k = emissions.shape
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if min_steps < 1 or int(min_steps) != min_steps:
        raise ValueError("min_steps must be a positive integer")
    d = min(int(min_steps), n)
    if transition_penalty < 0:
        raise ValueError("transition_penalty must be nonnegative")
    dtype = np.int16 if k * d < 32767 else np.int32
    previous = np.full((n, k, d), -1, dtype=dtype)
    dp = np.full((k, d), -np.inf)
    dp[:, 0] = emissions[0]
    states = np.arange(k)
    for t in range(1, n):
        nxt = np.full((k, d), -np.inf)
        mature = dp[:, -1]
        switches = np.broadcast_to(mature[:, None], (k, k)).copy()
        switches -= transition_penalty
        np.fill_diagonal(switches, -np.inf)
        from_state = np.argmax(switches, axis=0)
        nxt[:, 0] = switches[from_state, states]
        previous[t, :, 0] = from_state * d + d - 1
        if d > 1:
            nxt[:, 1:] = dp[:, :-1]
            previous[t, :, 1:] = states[:, None] * d + np.arange(d - 1)
            stay = mature >= nxt[:, -1]
            nxt[stay, -1] = mature[stay]
            previous[t, stay, -1] = states[stay] * d + d - 1
        else:
            stay = mature >= nxt[:, 0]
            nxt[stay, 0] = mature[stay]
            previous[t, stay, 0] = states[stay]
        dp = nxt + emissions[t, :, None]
    flat_state = int(np.argmax(dp))
    result = np.empty(n, dtype=np.int64)
    for t in range(n - 1, -1, -1):
        result[t] = flat_state // d
        if t:
            flat_state = int(previous[t].flat[flat_state])
            if flat_state < 0:
                raise RuntimeError("Unreachable duration-decoder state")
    return result


def _mean(values: pd.Series) -> float | None:
    array = values.to_numpy(float)
    good = array[np.isfinite(array)]
    return float(good.mean()) if len(good) else None


def _runs(values: np.ndarray, breaks: np.ndarray) -> list[tuple[int, int]]:
    if not len(values):
        return []
    starts = np.r_[0, np.flatnonzero((values[1:] != values[:-1]) | breaks[1:]) + 1]
    ends = np.r_[starts[1:], len(values)]
    return list(zip(starts.tolist(), ends.tolist()))


def annotate_episode(
    features: pd.DataFrame, config: dict | None = None
) -> tuple[pd.DataFrame, list[dict], dict]:
    """Return per-frame proposals, half-open temporal segments, and diagnostics.

    Input rows must belong to one episode, have increasing integer frame
    indices and strictly increasing timestamps.  Gaps are allowed and split
    decoding; missing rows are never silently synthesized.  Invalid tracking
    becomes ``unknown``.  The returned frame table retains all input columns.
    """
    cfg = TemporalConfig.from_dict(config)
    required = {
        "episode_id",
        "frame_index",
        "timestamp_s",
        "left_speed_mps",
        "right_speed_mps",
        "wrist_distance_m",
        "left_grip",
        "right_grip",
        "velocity_cosine",
        "quality_score",
        "usable",
    }
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    out = features.copy().reset_index(drop=True)
    n = len(out)
    if not n:
        for col in ("raw_state", "state", "segment_id"):
            out[col] = pd.Series(dtype=str)
        for col in ("uncertainty", "emission_margin", "review_priority"):
            out[col] = pd.Series(dtype=float)
        return (
            out,
            [],
            {"method": METHOD, "frames": 0, "segments": 0, "config": asdict(cfg)},
        )
    if out.episode_id.nunique(dropna=False) != 1 or pd.isna(out.episode_id.iloc[0]):
        raise ValueError("annotate_episode requires exactly one non-null episode_id")
    episode_id = int(out.episode_id.iloc[0])
    if float(out.episode_id.iloc[0]) != episode_id:
        raise ValueError("episode_id must be an integer")
    indices = out.frame_index.to_numpy(float)
    times = out.timestamp_s.to_numpy(float)
    if not np.all(np.isfinite(indices)) or not np.all(indices == np.floor(indices)):
        raise ValueError("frame_index must contain finite integers")
    if (
        np.any(np.diff(indices) <= 0)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("frame indices and timestamps must be strictly increasing")
    quality = out.quality_score.to_numpy(float)
    if np.any(~np.isfinite(quality)) or np.any((quality < 0) | (quality > 1)):
        raise ValueError("quality_score must contain finite values in [0, 1]")
    # bool coercion of strings ('False' -> True) would silently accept bad data.
    if out.usable.isna().any() or not out.usable.isin([True, False, 0, 1]).all():
        raise ValueError("usable must contain booleans or binary integers")
    speed = out[["left_speed_mps", "right_speed_mps"]].to_numpy(float)
    known = (
        out.usable.to_numpy(bool)
        & (quality >= cfg.minimum_quality)
        & np.isfinite(speed).all(axis=1)
        & (speed >= 0).all(axis=1)
    )
    consecutive = np.diff(indices) == 1
    increments = np.diff(times)
    ordinary_dt = increments[consecutive & (increments > 0)]
    dt = float(np.median(ordinary_dt)) if len(ordinary_dt) else 1 / cfg.nominal_fps
    min_steps = max(1, int(math.ceil(cfg.minimum_duration_s / dt - 1e-8)))
    gap = np.r_[True, (~consecutive) | (increments > cfg.maximum_gap_factor * dt)]
    scores = emission_scores(out, cfg)
    raw_ids = scores.argmax(axis=1)
    smooth_ids = np.full(n, -1, dtype=int)
    for start, end in _runs(known, gap):
        if known[start]:
            smooth_ids[start:end] = duration_viterbi(
                np.log(np.clip(scores[start:end], 1e-12, 1)),
                min_steps,
                cfg.transition_penalty,
            )
    names = np.array(STATES, dtype=object)
    out["raw_state"] = np.where(known, names[raw_ids], "unknown")
    out["state"] = np.where(known, names[np.maximum(smooth_ids, 0)], "unknown")
    entropy = -(scores * np.log(np.clip(scores, 1e-12, 1))).sum(axis=1) / math.log(
        len(STATES)
    )
    ordered = np.sort(scores, axis=1)
    out["uncertainty"] = np.where(known, entropy, 1.0)
    out["emission_margin"] = np.where(known, ordered[:, -1] - ordered[:, -2], 0.0)
    disagreement = out.raw_state.to_numpy() != out.state.to_numpy()
    out["review_priority"] = np.where(
        known,
        np.clip(0.5 * entropy + 0.3 * (1 - quality) + 0.2 * disagreement, 0, 1),
        1.0,
    )
    segments: list[dict[str, Any]] = []
    segment_ids = np.empty(n, dtype=object)
    state_values = out.state.to_numpy()
    for number, (start, end) in enumerate(_runs(state_values, gap)):
        sub = out.iloc[start:end]
        segment_id = f"ep{episode_id:06d}-seg{number:05d}"
        segment_ids[start:end] = segment_id
        end_time = float(times[end - 1] + dt)
        # Do not bridge an absent interval or claim evidence beyond the next sample.
        if end < n:
            end_time = min(end_time, float(times[end]))
        duration = max(0.0, end_time - float(times[start]))
        censored_left = bool(
            start == 0 or gap[start] or (start > 0 and not known[start - 1])
        )
        censored_right = bool(end == n or (end < n and (gap[end] or not known[end])))
        evidence = {
            name: _mean(sub[name])
            for name in (
                "left_speed_mps",
                "right_speed_mps",
                "wrist_distance_m",
                "left_grip",
                "right_grip",
                "velocity_cosine",
                "quality_score",
                "uncertainty",
                "emission_margin",
            )
        }
        evidence["raw_decoder_disagreement_fraction"] = float(
            disagreement[start:end].mean()
        )
        qa_flags = (
            sorted({str(flag) for row in sub["qa_flags"] for flag in row})
            if "qa_flags" in sub
            else []
        )
        segment = {
            "segment_id": segment_id,
            "episode_id": episode_id,
            "state": str(state_values[start]),
            "start_frame": int(indices[start]),
            "end_frame": int(indices[end - 1]) + 1,
            "end_frame_exclusive": int(indices[end - 1]) + 1,
            "start_time_s": float(times[start]),
            "end_time_s": end_time,
            "duration_s": duration,
            "frame_count": int(end - start),
            "status": "machine_proposed",
            "method": METHOD,
            "uncertainty": float(sub.uncertainty.mean()),
            "review_priority": float(sub.review_priority.mean()),
            "boundary_censored_left": censored_left,
            "boundary_censored_right": censored_right,
            "short_boundary_fragment": bool(
                end - start < min_steps and (censored_left or censored_right)
            ),
            "qa_flags": qa_flags,
            "evidence": evidence,
        }
        segments.append(segment)
    out["segment_id"] = segment_ids
    contiguous_known = known[1:] & known[:-1] & ~gap[1:]
    raw_transitions = int(
        (
            (out.raw_state.to_numpy()[1:] != out.raw_state.to_numpy()[:-1])
            & contiguous_known
        ).sum()
    )
    smooth_transitions = int(
        ((state_values[1:] != state_values[:-1]) & contiguous_known).sum()
    )
    usable_seconds = float(known.sum() * dt)
    stats = {
        "method": METHOD,
        "frames": n,
        "segments": len(segments),
        "known_frames": int(known.sum()),
        "unknown_frames": int((~known).sum()),
        "state_frame_counts": dict(Counter(str(x) for x in state_values)),
        "nominal_frame_interval_s": dt,
        "minimum_dwell_frames": min_steps,
        "gaps": int(gap.sum() - 1),
        "raw_transitions": raw_transitions,
        "temporal_transitions": smooth_transitions,
        "raw_transitions_per_minute": raw_transitions * 60 / usable_seconds
        if usable_seconds
        else None,
        "temporal_transitions_per_minute": smooth_transitions * 60 / usable_seconds
        if usable_seconds
        else None,
        "decoder_changed_frames": int(disagreement.sum()),
        "mean_uncertainty": float(out.uncertainty.mean()),
        "evaluation_note": "Transition reduction measures temporal regularization, not semantic accuracy.",
        "config": asdict(cfg),
    }
    return out, segments, stats


def select_review_segments(segments: list[dict], budget: int = 24) -> list[dict]:
    """Guarantee feasible episode/state coverage, then fill by risk/novelty.

    An episode-state bipartite edge cover provides the smallest seed set that
    covers every available episode and state. A deterministic augmenting-path
    maximum matching constructs that edge cover. If budget is smaller, greedily
    maximize newly covered vertices, preferring a new state on ties. Remaining
    slots follow the risk/temporal-novelty policy. Selection never approves labels.
    """
    if budget < 0 or int(budget) != budget:
        raise ValueError("budget must be a nonnegative integer")
    if len({s["segment_id"] for s in segments}) != len(segments):
        raise ValueError("segment_id values must be unique across the review pool")
    candidates = sorted(
        segments, key=lambda s: (s["episode_id"], s["start_time_s"], s["segment_id"])
    )
    if not candidates or budget == 0:
        return []

    def risk_of(segment):
        return float(
            np.clip(
                segment.get("review_priority", segment.get("uncertainty", 0.5)), 0, 1
            )
        )

    # One deterministic, high-risk representative for each episode-state edge.
    pair_best: dict[tuple, dict] = {}
    for segment in candidates:
        pair = (segment["episode_id"], segment["state"])
        if pair not in pair_best or risk_of(segment) > risk_of(pair_best[pair]):
            pair_best[pair] = segment
    available_states = sorted({state for _, state in pair_best})
    available_episodes = sorted({episode for episode, _ in pair_best})
    adjacency = {
        state: sorted(
            [episode for episode, s in pair_best if s == state],
            key=lambda episode: (-risk_of(pair_best[episode, state]), episode),
        )
        for state in available_states
    }
    matched_episode: dict[Any, str] = {}

    def augment(state, seen):
        for episode in adjacency[state]:
            if episode in seen:
                continue
            seen.add(episode)
            if episode not in matched_episode or augment(
                matched_episode[episode], seen
            ):
                matched_episode[episode] = state
                return True
        return False

    for state in sorted(available_states, key=lambda s: (len(adjacency[s]), s)):
        augment(state, set())
    cover_pairs = {(episode, state) for episode, state in matched_episode.items()}
    covered_states = {state for _, state in cover_pairs}
    covered_episodes = {episode for episode, _ in cover_pairs}
    for state in available_states:
        if state not in covered_states:
            episode = adjacency[state][0]
            cover_pairs.add((episode, state))
            covered_states.add(state)
            covered_episodes.add(episode)
    for episode in available_episodes:
        if episode not in covered_episodes:
            pairs = [pair for pair in pair_best if pair[0] == episode]
            pair = min(pairs, key=lambda p: (-risk_of(pair_best[p]), p[1]))
            cover_pairs.add(pair)
            covered_episodes.add(episode)
    minimum_coverage_budget = len(cover_pairs)
    seed_ids = {pair_best[pair]["segment_id"] for pair in cover_pairs}
    feasible_full_coverage = budget >= minimum_coverage_budget
    chosen: list[dict] = []
    episodes: set[int] = set()
    states: set[str] = set()
    while candidates and len(chosen) < budget:
        need_coverage = len(episodes) < len(available_episodes) or len(states) < len(
            available_states
        )
        best = None
        for i, segment in enumerate(candidates):
            if (
                need_coverage
                and feasible_full_coverage
                and segment["segment_id"] not in seed_ids
            ):
                continue
            episode = segment["episode_id"]
            state = segment["state"]
            center = (segment["start_time_s"] + segment["end_time_s"]) / 2
            same_episode = [s for s in chosen if s["episode_id"] == episode]
            distance = min(
                (
                    abs(center - (s["start_time_s"] + s["end_time_s"]) / 2)
                    for s in same_episode
                ),
                default=float("inf"),
            )
            novelty = min(distance / 15.0, 1.0)
            risk = risk_of(segment)
            new_episode, new_state = episode not in episodes, state not in states
            score = (
                0.50 * risk
                + 0.22 * (episode not in episodes)
                + 0.16 * (state not in states)
                + 0.12 * novelty
            )
            # Coverage is a constraint, not a small additive risk bonus. New-state
            # tie priority preserves all states even when episodes exceed budget.
            priority = (
                (int(new_episode) + int(new_state), int(new_state), score)
                if need_coverage
                else (score,)
            )
            # Stable input ordering resolves exact score ties reproducibly.
            if best is None or priority > best[0]:
                best = (priority, score, i, novelty, new_episode, new_state)
        assert best is not None
        _, score, index, novelty, new_episode, new_state = best
        segment = dict(candidates.pop(index))
        segment.update(
            {
                "review_rank": len(chosen) + 1,
                "selection_score": float(score),
                "selection_reason": {
                    "new_episode": bool(new_episode),
                    "new_state": bool(new_state),
                    "temporal_novelty": float(novelty),
                    "phase": "coverage_seed" if need_coverage else "risk_novelty_fill",
                    "minimum_full_coverage_budget": minimum_coverage_budget,
                    "full_coverage_feasible": feasible_full_coverage,
                },
                "selection_method": "minimum_cover_then_greedy_risk_v2",
            }
        )
        chosen.append(segment)
        episodes.add(segment["episode_id"])
        states.add(segment["state"])
    return chosen
