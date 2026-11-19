"""Deterministic, local source→audit→clean→annotate→review package."""

from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
from .data import load_episode, write_json, json_safe, sha256
from .quality import audit_episode
from .temporal import annotate_episode, select_review_segments
from .benchmark import corruption_benchmark
from .media import inspect_media


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), allow_nan=False) + "\n")


def build(raw, output, config_path, with_media=True):
    raw = Path(raw).resolve()
    out = Path(output).resolve()
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    source = json.loads((raw / "source_manifest.json").read_text())
    for entry in source["files"]:
        path = (raw / entry["path"]).resolve()
        if (
            not path.is_relative_to(raw)
            or not path.is_file()
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"Source integrity failure: {entry['path']}")
    from .recurrence import detect_recurrence

    code_files = sorted(Path(__file__).parent.glob("*.py")) + sorted(
        (Path(__file__).parent / "static").glob("*")
    )
    code_hashes = {
        str(p.relative_to(Path(__file__).parent)): sha256(p)
        for p in code_files
        if p.is_file()
    }
    identity = {
        "config": config,
        "source_manifest_sha256": sha256(raw / "source_manifest.json"),
        "code": code_hashes,
        "with_media": with_media,
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[
        :16
    ]
    if (out / "reviews.jsonl").exists() and (out / "summary.json").exists():
        old = json.loads((out / "summary.json").read_text()).get("run_id")
        if old != run_id:
            raise ValueError(
                "Existing reviews belong to another run. Choose a new output directory."
            )
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", config)
    episode_meta = [
        json.loads(x) for x in (raw / "meta/episodes.jsonl").read_text().splitlines()
    ]
    tasks = {
        x["task_index"]: x["task"]
        for x in [
            json.loads(line)
            for line in (raw / "meta/tasks.jsonl").read_text().splitlines()
        ]
    }
    info = json.loads((raw / "meta/info.json").read_text())
    fixes = []
    if info["total_tasks"] != len(tasks):
        fixes.append(
            {
                "field": "meta/info.json.total_tasks",
                "source_value": info["total_tasks"],
                "derived_value": len(tasks),
                "reason": "count of indexed tasks in source tasks.jsonl",
                "method": "metadata_reconciliation",
                "status": "derived_override",
            }
        )
    episodes = []
    segments = []
    reports = []
    ablations = []
    frames = []
    recurrences = []
    imu_total = 0
    for meta in episode_meta:
        i = meta["episode_index"]
        ep = load_episode(raw, i)
        print(f"Auditing episode {i} ({len(ep.time):,} frames)", flush=True)
        f, clean, report = audit_episode(ep, config)
        f, segs, ablation = annotate_episode(f, config.get("temporal"))
        canonical = meta.get("task_index", ep.task_index)
        if canonical != ep.task_index:
            fixes.append(
                {
                    "episode_id": i,
                    "field": "task_index",
                    "source_value": ep.task_index,
                    "derived_value": canonical,
                    "affected_rows": len(ep.time),
                    "reason": "episode task table and task lookup agree; frame table conflicts",
                    "status": "derived_override_requires_semantic_review",
                    "source_sha256": ep.source_sha256,
                }
            )
        f["task_index_source"] = ep.task_index
        f["task_index_reconciled"] = canonical
        f["label_status"] = "machine_proposed"
        f["run_id"] = run_id
        for seg in segs:
            seg.update(
                {
                    "run_id": run_id,
                    "source_task": tasks.get(canonical, str(canonical)),
                    "source_sha256": ep.source_sha256,
                    "label_namespace": "bimanual_motion.v1",
                }
            )
        episode = {
            "episode_id": i,
            "task": tasks.get(canonical, str(canonical)),
            "duration_s": len(ep.time) / config["fps"],
            "frame_count": len(ep.time),
            "video_available": i == 4,
            "quarantined_frames": report["quarantined_frames"],
            "flagged_frames": report["flagged_frames"],
            "qa_flags": list(report["rule_counts"]),
            "source_task_index": ep.task_index,
            "reconciled_task_index": canonical,
            "segments": len(segs),
            "imu_samples": sum(len(v) for v in ep.imu.values()),
        }
        (out / "cleaned").mkdir(exist_ok=True)
        pd.DataFrame(
            {
                "episode_id": i,
                "frame_index": ep.frame_index,
                "timestamp_s": ep.time,
                "world_hands_xyz": list(clean.reshape(len(clean), -1)),
                "usable": f.usable,
                "task_index": canonical,
                "source_sha256": ep.source_sha256,
            }
        ).to_parquet(out / "cleaned" / f"episode_{i:06d}.parquet", index=False)
        rows = []
        for j in range(0, len(f), config["ui_frame_stride"]):
            row = f.iloc[j].to_dict()
            if i == 4:
                row["keypoints_2d"] = {
                    "left": ep.keypoints[j, 0].round(2).tolist(),
                    "right": ep.keypoints[j, 1].round(2).tolist(),
                }
            rows.append(row)
        write_json(out / "frames" / f"episode_{i:06d}.json", rows)
        write_json(out / "qa" / f"episode_{i:06d}.json", report)
        episodes.append(episode)
        segments.extend(segs)
        reports.append(report)
        ablations.append(ablation)
        frames.append(f)
        imu_total += episode["imu_samples"]
        recurrences.append(detect_recurrence(ep, clean, f, config.get("recurrence")))
        if i == 4:
            benchmark = corruption_benchmark(ep, config)
            if with_media:
                media = inspect_media(raw, out, ep, f, config)
                write_json(out / "media_qa.json", media)
    all_frames = pd.concat(frames, ignore_index=True)
    all_frames.to_parquet(out / "annotations.parquet", index=False)
    write_json(out / "episodes.json", episodes)
    write_json(out / "segments.json", segments)
    write_jsonl(out / "annotations.jsonl", segments)
    write_jsonl(
        out / "cleaning_log.jsonl",
        fixes + [{"episode_id": r["episode_id"], **r["cleaning"]} for r in reports],
    )
    queue = select_review_segments(segments, config["review_budget"])
    write_json(out / "review_queue.json", queue)
    write_json(out / "benchmark.json", benchmark)
    write_json(out / "ablation.json", ablations)
    write_json(out / "recurrence.json", recurrences)
    rule_counts = Counter()
    for report in reports:
        rule_counts.update(report["rule_counts"])
    summary = {
        "project": "EgoForge",
        "version": "0.1.0",
        "run_id": run_id,
        "source_name": "Humanola egocentric-sample",
        "source_url": "https://huggingface.co/datasets/humanola-inc/egocentric-sample",
        "source_license": "CC BY 4.0",
        "total_episodes": len(episodes),
        "total_frames": len(all_frames),
        "total_imu_samples": imu_total,
        "duration_s": len(all_frames) / config["fps"],
        "segments": len(segments),
        "quarantined_frames": sum(r["quarantined_frames"] for r in reports),
        "flagged_frames": sum(r["flagged_frames"] for r in reports),
        "review_queue_size": len(queue),
        "human_reviewed": 0,
        "label_status": "machine_proposed",
        "metadata_overrides": fixes,
        "states": dict(Counter(all_frames.state)),
        "qa_rule_counts": dict(rule_counts),
        "source_bytes": source["total_source_bytes"],
        "recurrence_bouts": sum(len(r["bouts"]) for r in recurrences),
        "synthetic_cases_passing": benchmark["cases_passing"],
        "synthetic_cases_total": benchmark["total_cases"],
        "scope": "9 episodes of pose/IMU; all four video cameras decoded for episode 4"
        if with_media
        else "9 episodes pose/IMU; media decoding skipped",
        "limitations": [
            "Motion proposals are heuristic and uncalibrated; no independent human gold labels.",
            "Existing 3D/2D poses and task names are inherited from the source.",
            "No inferred physical contact, force, successful action count, or task success.",
            "Splits are episode-disjoint only; participant/session IDs unavailable.",
            "Synthetic fault tests do not establish annotation accuracy.",
        ],
    }
    write_json(out / "summary.json", summary)
    split = {
        "train": [0, 1, 3, 5, 7],
        "validation": [2, 6],
        "test": [4, 8],
        "unit": "episode",
        "purpose": "fixed downstream experimentation partition; this run audits all episodes without fitting parameters",
        "limitations": "Not verified subject/session-disjoint. No accuracy estimated on these partitions.",
    }
    write_json(out / "splits.json", split)
    write_json(
        out / "provenance.json",
        {
            "run_id": run_id,
            "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {
                x: importlib.metadata.version(x)
                for x in ["numpy", "pandas", "pyarrow", "scipy", "Pillow"]
            },
            "lineage": {
                "inherited": [
                    "3D joints",
                    "camera pose",
                    "2D keypoints",
                    "source tasks",
                    "IMU",
                    "video",
                ],
                "new": [
                    "cleaned kinematic features",
                    "QA flags",
                    "motion phase proposals",
                    "recurrence candidates",
                    "risk-ranked review queue",
                ],
                "human_annotation_count": 0,
            },
        },
    )
    # Exports carry proposals in predictions, not completed human annotations.
    pilot_segments = [s for s in segments if s["episode_id"] == 4]
    label_studio = [
        {
            "id": 4,
            "data": {
                "video": "http://127.0.0.1:8765/media/episode_000004/main_stereo_left.mp4",
                "episode_id": 4,
            },
            "meta": {
                "run_id": run_id,
                "status": "machine_proposed",
                "frame_convention": "1-based inclusive ranges",
            },
            "predictions": [
                {
                    "model_version": "egoforge-v1",
                    "result": [
                        {
                            "id": s["segment_id"],
                            "from_name": "phase",
                            "to_name": "video",
                            "type": "timelinelabels",
                            "value": {
                                "ranges": [
                                    {
                                        "start": s["start_frame"] + 1,
                                        "end": s["end_frame_exclusive"],
                                    }
                                ],
                                "timelinelabels": [s["state"]],
                            },
                        }
                        for s in pilot_segments
                    ],
                }
            ],
        }
    ]
    write_json(out / "exports/label_studio_predictions.json", label_studio)
    (out / "exports/label_studio_config.xml").write_text(
        '<View><Video name="video" value="$video" frameRate="30"/><TimelineLabels name="phase" toName="video">'
        + "".join(f'<Label value="{x}"/>' for x in summary["states"])
        + "</TimelineLabels></View>\n"
    )
    make_report(out, summary, episodes, reports, ablations, benchmark, recurrences)
    hashes = {
        str(p.relative_to(out)): sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name not in ["artifact_manifest.json", "reviews.jsonl"]
    }
    write_json(
        out / "artifact_manifest.json",
        {
            "run_id": run_id,
            "artifacts": hashes,
            "mutable_exclusions": ["reviews.jsonl"],
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "run_id",
                    "total_frames",
                    "segments",
                    "quarantined_frames",
                    "synthetic_cases_passing",
                ]
            }
        ),
        flush=True,
    )
    return summary


def make_report(out, s, episodes, reports, ablations, benchmark, recurrences):
    lines = [
        "# EgoForge — executed data-quality and annotation report",
        "",
        f"Run `{s['run_id']}`. All numbers below are computed from the downloaded source files.",
        "",
        f"Processed **{s['total_frames']:,} frames** across **{s['total_episodes']} episodes**, with **{s['total_imu_samples']:,} IMU samples**. Produced **{s['segments']:,} machine-proposed motion intervals** and a **{s['review_queue_size']}-item review queue**. No human-reviewed phase labels are claimed.",
        "",
        f"**{s['quarantined_frames']:,} frames quarantined** by configured rules; **{s['flagged_frames']:,} frames have at least one flag**. Flags indicate review candidates, not verified errors.",
        "",
        "## Source and attribution",
        "",
        "[Humanola egocentric-sample](https://huggingface.co/datasets/humanola-inc/egocentric-sample), CC BY 4.0. Existing joints, camera poses, keypoints and task labels are inherited. Added outputs include reversible masking/filtering, geometry/time checks, motion proposals and review prioritization. See the source manifest and THIRD_PARTY_NOTICES.md.",
        "",
        "## Episode results",
        "",
        "| Episode | Source task (reconciled) | Frames | Quarantined | Proposed intervals |",
        "|---|---|---:|---:|---:|",
    ]
    for e in episodes:
        lines.append(
            f"| {e['episode_id']} | {e['task']} | {e['frame_count']:,} | {e['quarantined_frames']:,} | {e['segments']} |"
        )
    lines += [
        "",
        "## Cleaning and source conflicts",
        "",
        "Source bytes remain unchanged. Derived task indices follow the episode task table; conflicting frame-table values are retained beside them. This override is evidence-based metadata reconciliation and still requires visual semantic review.",
    ]
    for x in s["metadata_overrides"]:
        lines += [
            "",
            f"- `{x['field']}`: {x['source_value']} → {x['derived_value']}; affected rows: {x.get('affected_rows', 'metadata only')}.",
        ]
    lines += [
        "",
        "Absent/inconsistent observations are masked, never interpolated across. Smoothing runs only inside valid contiguous spans. No source rows are deleted. Quaternion conversion uses camera-to-world `xyzw`; residuals against inherited camera-space points are internal consistency checks, not independent physical accuracy.",
        "",
        "## Synthetic corruption evaluation",
        "",
        f"**{benchmark['cases_passing']}/{benchmark['total_cases']} seeded cases** detected every injected target frame. These are deliberately threshold-visible corruptions and do not estimate real-world precision, semantic label accuracy, or human agreement.",
        "",
        "| Injected corruption | Target frames | Detected |",
        "|---|---:|---:|",
    ]
    for b in benchmark["cases"]:
        lines.append(
            f"| {b['case']} | {b['injected_frame_count']} | {b['detected_injected_frames']} |"
        )
    lines += [
        "",
        "## Temporal regularization",
        "",
        "Duration-constrained decoding reduces framewise phase switches. This is a stability statistic; a smoother sequence is not necessarily a more accurate one. Full before/after counts are in `ablation.json`. Entropy and margin are uncertainty proxies, not calibrated correctness probabilities.",
        "",
        "## Recurrence candidates",
        "",
        "The experimental recurrence layer finds periodic wrist motion using principal-axis projection and autocorrelation. A cycle is not a confirmed cut, grasp, or successful action. See `recurrence.json` and the method document.",
        "",
        "## Review and acceptance",
        "",
        "Use the local review console to inspect source video, inherited skeletons, phase intervals and QA evidence. Approve, reject or correct with an explicit reviewer identifier. Decisions append to a hash-chained log; no proposal becomes human ground truth automatically. The log is tamper-evident within a retained chain, not externally signed or tamper-proof.",
        "",
        "## Limits",
        "",
    ] + [f"- {x}" for x in s["limitations"]]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
