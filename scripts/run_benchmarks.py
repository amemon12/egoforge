#!/usr/bin/env python3
"""Reproducible CPU throughput, rule sensitivity and temporal-stability benchmarks.

No gold semantic labels exist in the sample. This suite therefore never reports
phase accuracy, error precision, or successful action recognition.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np

from egoforge.data import json_safe, load_episode, sha256
from egoforge.quality import audit_episode
from egoforge.temporal import annotate_episode, select_review_segments
from egoforge.recurrence import detect_recurrence
from egoforge.benchmark import corruption_benchmark

ROOT = Path(__file__).resolve().parents[1]
CORE_MODULES = [
    "data.py",
    "quality.py",
    "synchronization.py",
    "temporal.py",
    "recurrence.py",
    "benchmark.py",
]
STAGES = ["load", "quality", "temporal", "recurrence", "review_selection"]
STAGE_LABELS = {
    "load": "Parquet load + source hashing",
    "quality": "Geometry, cleaning + sensor alignment",
    "temporal": "Temporal annotation",
    "recurrence": "Repeated-motion detection",
    "review_selection": "Review selection",
}


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def detection_counts(baseline, perturbed, target):
    """Count only newly detected, previously rule-negative injection targets."""
    baseline = np.asarray(baseline, bool)
    perturbed = np.asarray(perturbed, bool)
    target = np.asarray(target, dtype=int)
    if baseline.shape != perturbed.shape or baseline.ndim != 1:
        raise ValueError("Detection arrays must be matching vectors")
    if (
        len(np.unique(target)) != len(target)
        or np.any(target < 0)
        or np.any(target >= len(baseline))
    ):
        raise ValueError("Targets must be unique valid frame positions")
    eligible = target[~baseline[target]]
    newly = int(perturbed[eligible].sum())
    outside = np.ones(len(baseline), bool)
    outside[target] = False
    return {
        "target_frames": len(target),
        "baseline_flagged_targets": int(baseline[target].sum()),
        "eligible_targets": len(eligible),
        "newly_detected_targets": newly,
        "delta_sensitivity": newly / len(eligible) if len(eligible) else None,
        "post_fault_detected_targets": int(perturbed[target].sum()),
        "post_fault_sensitivity": float(perturbed[target].mean())
        if len(target)
        else None,
        "newly_flagged_outside_targets": int((perturbed & ~baseline & outside).sum()),
        "baseline_flags_total": int(baseline.sum()),
    }


def short_interior_runs(labels, frame_indices, min_frames):
    """Count short runs only when neither boundary touches unknown/gap/end."""
    labels = np.asarray(labels, object)
    frames = np.asarray(frame_indices)
    if not len(labels):
        return 0
    breaks = np.r_[True, (labels[1:] != labels[:-1]) | (np.diff(frames) != 1)]
    starts = np.flatnonzero(breaks)
    ends = np.r_[starts[1:], len(labels)]
    count = 0
    for start, end in zip(starts, ends):
        if labels[start] == "unknown" or end - start >= min_frames:
            continue
        left_open = (
            start == 0
            or frames[start] != frames[start - 1] + 1
            or labels[start - 1] == "unknown"
        )
        right_open = (
            end == len(labels)
            or frames[end] != frames[end - 1] + 1
            or labels[end] == "unknown"
        )
        count += not left_open and not right_open
    return int(count)


def rss_bytes():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def performance_worker(raw: Path, config, episodes):
    totals = {stage: 0.0 for stage in STAGES}
    rows, all_segments, ablations = [], [], []
    frames = imu_samples = 0
    signature = hashlib.sha256()
    started = time.perf_counter()
    for episode_id in episodes:
        local = {}
        t = time.perf_counter()
        ep = load_episode(raw, episode_id)
        local["load"] = time.perf_counter() - t
        t = time.perf_counter()
        features, clean, qa = audit_episode(ep, config)
        local["quality"] = time.perf_counter() - t
        t = time.perf_counter()
        annotated, segments, stats = annotate_episode(features, config.get("temporal"))
        local["temporal"] = time.perf_counter() - t
        t = time.perf_counter()
        recurrence = detect_recurrence(ep, clean, annotated, config.get("recurrence"))
        local["recurrence"] = time.perf_counter() - t
        for name, value in local.items():
            totals[name] += value
        frames += len(ep.time)
        imu_samples += sum(len(t) for t in ep.imu.values())
        all_segments.extend(segments)
        ablations.append(
            {
                "episode_id": episode_id,
                "frames": len(ep.time),
                "raw_transitions": stats["raw_transitions"],
                "temporal_transitions": stats["temporal_transitions"],
                "minimum_dwell_frames": stats["minimum_dwell_frames"],
                "raw_short_interior_runs": short_interior_runs(
                    annotated.raw_state, ep.frame_index, stats["minimum_dwell_frames"]
                ),
                "temporal_short_interior_runs": short_interior_runs(
                    annotated.state, ep.frame_index, stats["minimum_dwell_frames"]
                ),
                "abstained_frames": stats["unknown_frames"],
                "quarantined_frames": qa["quarantined_frames"],
                "proposed_segments": len(segments),
                "recurrence_bouts": len(recurrence["bouts"]),
                "recurrence_cycles": recurrence["statistics"]["cycles"],
            }
        )
        signature.update(
            json.dumps(
                json_safe(
                    {
                        "states": annotated.state.tolist(),
                        "qa_flags": features.qa_flags.tolist(),
                        "bouts": recurrence["bouts"],
                    }
                ),
                sort_keys=True,
            ).encode()
        )
        rows.append({"episode_id": episode_id, "frames": len(ep.time), **local})
        del ep, features, clean, annotated, segments, recurrence
        gc.collect()
    t = time.perf_counter()
    queue = select_review_segments(all_segments, config["review_budget"])
    totals["review_selection"] = time.perf_counter() - t
    elapsed = time.perf_counter() - started
    return {
        "frames": frames,
        "imu_samples": imu_samples,
        "stage_seconds": totals,
        "pipeline_wall_s": elapsed,
        "timed_stage_sum_s": sum(totals.values()),
        "peak_rss_bytes": rss_bytes(),
        "episode_timings": rows,
        "ablation": ablations,
        "deterministic_result_sha256": signature.hexdigest(),
        "review_coverage": {
            "queue_size": len(queue),
            "episodes": sorted({s["episode_id"] for s in queue}),
            "states": sorted({s["state"] for s in queue}),
            "state_counts": dict(Counter(s["state"] for s in queue)),
        },
    }


def target_indices(ep):
    # Three distributed windows, rather than selecting only visually/QA-clean data.
    return np.unique(
        np.concatenate(
            [
                np.arange(int(len(ep.time) * p), int(len(ep.time) * p) + 30)
                for p in [0.2, 0.5, 0.8]
            ]
        )
    )


def flag_vector(features, rule):
    return np.array([rule in flags for flags in features.qa_flags], bool)


def robustness_suite(raw, config, episodes):
    rows, severe = [], []
    grids = {
        "world_translation": (
            [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.15],
            "m",
            "world_camera_inconsistent",
        ),
        "keypoint_displacement": (
            [0, 5, 15, 30, 50, 80, 120],
            "px",
            "reprojection_extreme",
        ),
        "bone_extension": (
            [0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.2],
            "m",
            "bone_length_outlier",
        ),
        "quaternion_scale_error": (
            [0, 0.001, 0.005, 0.009, 0.011, 0.02, 0.2],
            "fraction",
            "quaternion_norm_invalid",
        ),
        "shared_imu_clock_shift": (
            [0, 5, 15, 25, 40, 100, 200],
            "ms",
            "aux_left_imu_shared_clock_mismatch",
        ),
        "imu_burst_removal": (
            [0, 0.01, 0.025, 0.05, 0.1, 0.25, 1],
            "s",
            "main_stereo_imu_coverage_gap",
        ),
    }
    for episode_id in episodes:
        print(f"Fault sweeps: episode {episode_id}", flush=True)
        ep = load_episode(raw, episode_id)
        baseline, _, _ = audit_episode(ep, config)
        indices = target_indices(ep)
        head_imu = ep.imu["main_stereo"]
        anchor = int(head_imu.timestamp_ns.iloc[0])
        seconds = (head_imu.timestamp_ns.to_numpy() - np.int64(anchor)).astype(
            float
        ) / 1e9
        capture = np.interp(ep.frame_index, head_imu.frame.to_numpy(), seconds)
        for family, (magnitudes, unit, rule) in grids.items():
            before = flag_vector(baseline, rule)
            for magnitude in magnitudes:
                damaged = ep.copy()
                target = indices
                details = {
                    "target_definition": "Three fixed 30-frame windows at 20%, 50%, 80% of the episode; no QA-based selection"
                }
                if family == "world_translation":
                    damaged.world[indices] += [magnitude, 0, 0]
                elif family == "keypoint_displacement":
                    damaged.keypoints[indices] += [magnitude, 0]
                elif family == "bone_extension":
                    damaged.world[indices, 0, 8] += [magnitude, 0, 0]
                elif family == "quaternion_scale_error":
                    damaged.head[indices, 3:] *= 1 + magnitude
                elif family == "shared_imu_clock_shift":
                    damaged.imu["aux_left"]["timestamp_ns"] += round(magnitude * 1e6)
                    target = np.arange(len(ep.time))
                    details = {
                        "target_definition": "All video frames: the entire auxiliary sensor shared-epoch clock is displaced"
                    }
                elif family == "imu_burst_removal":
                    remove = (
                        head_imu.timestamp_s.between(
                            3 - magnitude / 2, 3 + magnitude / 2
                        )
                        if magnitude
                        else np.zeros(len(head_imu), bool)
                    )
                    damaged.imu["main_stereo"] = head_imu[~remove].copy()
                    removed_seconds = seconds[np.asarray(remove)]
                    target = (
                        np.flatnonzero(
                            (capture > removed_seconds.min())
                            & (capture < removed_seconds.max())
                        )
                        if len(removed_seconds) > 1
                        else np.array([], int)
                    )
                    details = {
                        "target_definition": "Video capture times strictly between the first and last actually removed IMU samples; endpoints excluded",
                        "removed_imu_samples": int(np.sum(remove)),
                        "removed_support_start_s": float(removed_seconds.min())
                        if len(removed_seconds)
                        else None,
                        "removed_support_end_s": float(removed_seconds.max())
                        if len(removed_seconds)
                        else None,
                    }
                after, _, _ = audit_episode(damaged, config)
                counts = detection_counts(before, flag_vector(after, rule), target)
                rows.append(
                    {
                        "episode_id": episode_id,
                        "family": family,
                        "magnitude": magnitude,
                        "unit": unit,
                        "rule": rule,
                        **counts,
                        **details,
                    }
                )
        # Keep the original deliberately severe suite as a separately labeled baseline.
        severe.append(corruption_benchmark(ep, config))
    grouped = []
    for family, (magnitudes, unit, rule) in grids.items():
        for magnitude in magnitudes:
            selected = [
                r for r in rows if r["family"] == family and r["magnitude"] == magnitude
            ]
            n = sum(r["eligible_targets"] for r in selected)
            hits = sum(r["newly_detected_targets"] for r in selected)
            grouped.append(
                {
                    "family": family,
                    "magnitude": magnitude,
                    "unit": unit,
                    "rule": rule,
                    "episodes": len(selected),
                    "target_frames": sum(r["target_frames"] for r in selected),
                    "eligible_targets": n,
                    "newly_detected_targets": hits,
                    "baseline_flagged_targets": sum(
                        r["baseline_flagged_targets"] for r in selected
                    ),
                    "delta_sensitivity": hits / n if n else None,
                }
            )
    return rows, grouped, severe


def environment():
    def sysctl(key):
        if sys.platform != "darwin":
            return None
        result = subprocess.run(
            ["/usr/sbin/sysctl", "-n", key], text=True, capture_output=True
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "os": platform.platform(),
        "architecture": platform.machine(),
        "cpu": sysctl("machdep.cpu.brand_string")
        or platform.processor()
        or "not reported",
        "logical_cpu_count": os.cpu_count(),
        "physical_memory_bytes": int(sysctl("hw.memsize") or 0) or None,
        "python": platform.python_version(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ["numpy", "pandas", "pyarrow", "scipy"]
        },
        "accelerator": "CPU only; no GPU inference",
        "thread_environment": {
            name: os.environ.get(name)
            for name in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
    }


def run(raw, output, config_path, repeats):
    if repeats < 3:
        raise ValueError("At least three repetitions are required")
    raw = raw.resolve()
    output = output.resolve()
    config = json.loads(config_path.read_text())
    source = json.loads((raw / "source_manifest.json").read_text())
    for entry in source["files"]:
        if (
            not entry["path"].startswith("videos/")
            and sha256(raw / entry["path"]) != entry["sha256"]
        ):
            raise ValueError(f"Source integrity failure: {entry['path']}")
    episodes = [
        json.loads(line)["episode_index"]
        for line in (raw / "meta/episodes.jsonl").read_text().splitlines()
    ]
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "source_revision": source["revision"],
        "source_manifest_sha256": sha256(raw / "source_manifest.json"),
        "config_sha256": sha256(config_path),
        "core_code_sha256": {
            name: sha256(ROOT / "egoforge" / name) for name in CORE_MODULES
        },
        "benchmark_script_sha256": sha256(Path(__file__)),
        "source_episodes": episodes,
        "robustness_episodes": [2, 4, 8],
        "repeats": repeats,
    }
    passes = []
    for repeat in range(repeats):
        print(
            f"Performance pass {repeat + 1}/{repeats}: {len(episodes)} episodes",
            flush=True,
        )
        target = output / f"performance_pass_{repeat + 1:02d}.json"
        t = time.perf_counter()
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--raw",
                str(raw),
                "--config",
                str(config_path.resolve()),
                "--worker-output",
                str(target),
            ],
            check=True,
        )
        result = json.loads(target.read_text())
        result["process_wall_s_including_imports"] = time.perf_counter() - t
        atomic_json(target, result)
        passes.append(result)
    robust, curve, severe = robustness_suite(raw, config, [2, 4, 8])
    for name, digest in provenance["core_code_sha256"].items():
        if sha256(ROOT / "egoforge" / name) != digest:
            raise RuntimeError(
                f"Core module changed during benchmark: {name}; rerun required"
            )
    frames = passes[0]["frames"]
    elapsed = [p["pipeline_wall_s"] for p in passes]
    performance = []
    for stage in STAGES:
        values = [p["stage_seconds"][stage] for p in passes]
        median = float(np.median(values))
        performance.append(
            {
                "stage": stage,
                "label": STAGE_LABELS[stage],
                "median_s": median,
                "p95_s": float(np.percentile(values, 95)),
                "min_s": min(values),
                "max_s": max(values),
                "fps": frames / median,
                "samples_s": values,
            }
        )
    ablation_rows = passes[0]["ablation"]
    sum_keys = [
        "raw_transitions",
        "temporal_transitions",
        "raw_short_interior_runs",
        "temporal_short_interior_runs",
        "abstained_frames",
        "quarantined_frames",
        "proposed_segments",
        "recurrence_bouts",
        "recurrence_cycles",
    ]
    ablation = {key: sum(row[key] for row in ablation_rows) for key in sum_keys}
    ablation.update(
        {
            "switch_reduction_fraction": 1
            - ablation["temporal_transitions"] / ablation["raw_transitions"],
            "abstention_rate": ablation["abstained_frames"] / frames,
            "episodes": ablation_rows,
            "note": "Stability and abstention statistics, not semantic accuracy; short-run count excludes episode/gap/unknown-censored boundaries.",
        }
    )
    summary = {
        "episodes": len(episodes),
        "frames": frames,
        "imu_samples": passes[0]["imu_samples"],
        "repeats": repeats,
        "total_median_s": float(np.median(elapsed)),
        "total_p95_s": float(np.percentile(elapsed, 95)),
        "total_min_s": min(elapsed),
        "total_max_s": max(elapsed),
        "fps": frames / float(np.median(elapsed)),
        "peak_rss_mb": max(p["peak_rss_bytes"] for p in passes) / 1e6,
        "synthetic_cases_passed": sum(p["cases_passing"] for p in severe),
        "synthetic_cases_total": sum(p["total_cases"] for p in severe),
        "sweep_cases": len(robust),
        "sweep_families": 6,
        "core_results_identical": len(
            {p["deterministic_result_sha256"] for p in passes}
        )
        == 1,
        "human_reference_labels": 0,
    }
    limitations = [
        "No independent human reference labels: phase accuracy, QA precision and action success are unmeasured.",
        "Throughput covers cached local pose/IMU loading, QC, temporal annotation, recurrence and review selection; excludes video decode, downloads, export serialization, manual review and training.",
        "Three isolated worker processes; OS file cache was not flushed. Workstation load was uncontrolled. Import startup is excluded from pipeline timing but recorded separately.",
        "The reported p95 is a descriptive percentile of only three observations, not a reliable tail-latency estimate or confidence bound.",
        "Peak RSS is the maximum whole-worker process high-water mark including imports and allocator memory; it is not module-specific or a system-wide memory estimate. MB uses 1,000,000 bytes.",
        "Corruption sweeps measure rule sensitivity on previously rule-negative injection targets. Untouched-source flags are unadjudicated, not false positives.",
        "The separate severe suite reports post-fault hits and may include baseline-positive targets; use the baseline-adjusted curves for incremental sensitivity.",
        "Lower-magnitude faults may deliberately fall below screening thresholds. Neither high synthetic sensitivity nor fewer phase switches establishes semantic accuracy.",
    ]
    result = {
        "schema_version": "2.0",
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_id": hashlib.sha256(
            json.dumps(provenance, sort_keys=True).encode()
        ).hexdigest()[:16],
        "summary": summary,
        "performance": performance,
        "performance_passes": passes,
        "robustness": robust,
        "robustness_curves": curve,
        "severe_suite": severe,
        "ablation": ablation,
        "review_coverage": passes[0]["review_coverage"],
        "environment": environment(),
        "provenance": provenance,
        "limitations": limitations,
    }
    atomic_json(output / "benchmark.json", result)
    write_csv(
        output / "performance.csv",
        performance,
        ["stage", "label", "median_s", "p95_s", "min_s", "max_s", "fps"],
    )
    write_csv(
        output / "robustness.csv",
        robust,
        [
            "episode_id",
            "family",
            "magnitude",
            "unit",
            "target_frames",
            "baseline_flagged_targets",
            "eligible_targets",
            "newly_detected_targets",
            "delta_sensitivity",
            "post_fault_sensitivity",
            "newly_flagged_outside_targets",
        ],
    )
    write_report(ROOT / "docs/BENCHMARKS.md", result)
    atomic_json(
        output / "artifact_manifest.json",
        {
            "benchmark_id": result["benchmark_id"],
            "artifacts": {
                p.name: sha256(p)
                for p in sorted(output.glob("*"))
                if p.is_file() and p.name != "artifact_manifest.json"
            },
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


def write_csv(path, rows, fields):
    import csv

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path, result):
    s = result["summary"]
    e = result["environment"]
    a = result["ablation"]
    lines = [
        "# Executed benchmarks",
        "",
        f"Benchmark `{result['benchmark_id']}` · {result['created_utc']}",
        "",
        f"Processed **{s['frames']:,} pose frames** and **{s['imu_samples']:,} IMU samples** across **{s['episodes']} episodes**, repeated in **{s['repeats']} separate worker processes**.",
        "",
        f"**{s['total_median_s']:.3f} seconds median** per pose/IMU pass (**{s['fps']:,.0f} frames/s**); observed range {s['total_min_s']:.3f}–{s['total_max_s']:.3f} s. Maximum whole-worker peak RSS: **{s['peak_rss_mb']:.1f} MB**. This excludes video decoding, downloading, exports, manual review and training.",
        "",
        f"Host: {e['cpu']} / {e['architecture']}, {e['logical_cpu_count']} logical CPUs, {e['os']}; Python {e['python']}. CPU only. Local inputs were cached; no cache flush or workstation-load isolation. The median is computed from whole-pass wall times, not by summing stage medians.",
        "",
        "## Stage timing",
        "",
        "| Stage | Median s | Observed p95 s | Range s |",
        "|---|---:|---:|---:|",
    ]
    for r in result["performance"]:
        lines.append(
            f"| {r['label']} | {r['median_s']:.3f} | {r['p95_s']:.3f} | {r['min_s']:.3f}–{r['max_s']:.3f} |"
        )
    lines += [
        "",
        "The p95 column is a descriptive percentile of three measurements, not a validated tail-latency estimate. Stage timing excludes process import startup; each raw pass also records complete subprocess wall time. Peak RSS includes imported libraries and retained allocator pages.",
        "",
        "## Sensitivity to injected faults",
        "",
        f"Executed **{s['sweep_cases']} episode-level sweep cases**: six fault families × seven magnitudes × episodes 2, 4 and 8. Each curve pools target counts, not percentages. Baseline-positive targets are excluded from incremental sensitivity; a zero eligible-target count is reported as null. Source flags remain unadjudicated.",
        "",
        "| Fault | Magnitude | Previously negative targets | Newly flagged | Incremental sensitivity |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in result["robustness_curves"]:
        sensitivity = (
            "—" if r["delta_sensitivity"] is None else f"{r['delta_sensitivity']:.1%}"
        )
        lines.append(
            f"| {r['family']} | {r['magnitude']:g} {r['unit']} | {r['eligible_targets']:,} | {r['newly_detected_targets']:,} | {sensitivity} |"
        )
    lines += [
        "",
        "World/keypoint/bone/quaternion perturbations affect three fixed 30-frame windows at 20%, 50% and 80% of each episode. Shared-clock offsets affect the full auxiliary-left IMU timeline. Burst targets are video capture timestamps strictly inside the span of actually removed head-IMU samples; very short bursts can contain no video-frame target. Full rows include baseline flags and newly flagged frames outside the target interval.",
        "",
        f"The original severe suite was also rerun: **{s['synthetic_cases_passed']}/{s['synthetic_cases_total']} cases** detected every injected target frame. This separate baseline reports post-fault rule hits and may include pre-existing positives; it is not semantic accuracy or precision.",
        "",
        "## Temporal ablation",
        "",
        f"Raw framewise switches: **{a['raw_transitions']:,}**; constrained-decoder switches: **{a['temporal_transitions']:,}** ({a['switch_reduction_fraction']:.1%} fewer). Short interior runs below the 9-frame dwell: **{a['raw_short_interior_runs']:,} → {a['temporal_short_interior_runs']:,}**. Episode, gap and unknown-boundary fragments are excluded from that count.",
        "",
        f"Abstained frames: **{a['abstained_frames']:,} / {s['frames']:,} ({a['abstention_rate']:.2%})**. Geometric quarantine: {a['quarantined_frames']:,}. Produced {a['proposed_segments']:,} motion proposals and {a['recurrence_bouts']} experimental recurrence bouts / {a['recurrence_cycles']} side-specific cycles. Repeated passes returned identical state/QA/recurrence signatures: **{s['core_results_identical']}**.",
        "",
        "These quantify regularization, functional constraints and repeatability. A smoother sequence can still be wrong; no human phase reference exists to measure semantic accuracy.",
        "",
        "## Review coverage",
        "",
        f"The {result['review_coverage']['queue_size']}-item queue includes {len(result['review_coverage']['episodes'])} episodes and {len(result['review_coverage']['states'])} states. Coverage is a scheduling property, not a claim that a person reviewed those examples.",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python scripts/run_benchmarks.py --repeats 3",
        "```",
        "",
        "Raw passes, benchmark JSON, CSV tables and hashes are in `output/benchmarks/`. Core module hashes, source manifest/revision, exact config hash, runtime versions and machine details are included. UI/static changes are intentionally outside the computational benchmark fingerprint.",
        "",
        "## Limits",
        "",
    ] + [f"- {x}" for x in result["limitations"]]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw/humanola")
    parser.add_argument("--output", type=Path, default=ROOT / "output/benchmarks")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/default.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        config = json.loads(args.config.read_text())
        episodes = [
            json.loads(line)["episode_index"]
            for line in (args.raw / "meta/episodes.jsonl").read_text().splitlines()
        ]
        atomic_json(args.worker_output, performance_worker(args.raw, config, episodes))
    else:
        run(args.raw, args.output, args.config, args.repeats)


if __name__ == "__main__":
    main()
