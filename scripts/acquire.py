#!/usr/bin/env python3
"""Download a pinned, hash-verified subset of the public Humanola release.

No credentials are required. Original files remain unchanged. LFS objects are
verified using the publisher's SHA-256; Git objects using Git's blob SHA-1.
Every file additionally receives a local SHA-256 recorded in source_manifest.json.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

REPO = "humanola-inc/egocentric-sample"
REVISION = "3c6e782911b4e82a56b288f6830d3a43bdb85d49"
LICENSE = "CC-BY-4.0"
DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data/raw/humanola"
META = [
    "README.md",
    "meta/info.json",
    "meta/modality.json",
    "meta/episodes.jsonl",
    "meta/tasks.jsonl",
    "meta/camera_intrinsics.json",
    "imu/imu_info.json",
]


def read_json(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return json.load(response)


def digests(path, size):
    sha256 = hashlib.sha256()
    git = hashlib.sha1(f"blob {size}\0".encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha256.update(block)
            git.update(block)
    return sha256.hexdigest(), git.hexdigest()


def acquire_file(item, root):
    relative = item["path"]
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/{relative}"
    expected = item.get("lfs", {}).get("oid", item["oid"])
    algorithm = "sha256" if "lfs" in item else "git-blob-sha1"
    if not target.exists():
        partial = target.with_name(target.name + ".partial")
        try:
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                partial.open("wb") as out,
            ):
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
            sha256, git = digests(partial, item["size"])
            actual = sha256 if algorithm == "sha256" else git
            if partial.stat().st_size != item["size"] or actual != expected:
                raise ValueError(f"Upstream checksum/size mismatch: {relative}")
            partial.replace(target)
        finally:
            if partial.exists():
                partial.unlink()
    else:
        sha256, git = digests(target, item["size"])
        actual = sha256 if algorithm == "sha256" else git
        if target.stat().st_size != item["size"] or actual != expected:
            raise ValueError(f"Existing file does not match pinned source: {relative}")
    print(f"verified {relative}: {item['size']:,} bytes", flush=True)
    return {
        "path": relative,
        "url": url,
        "bytes": item["size"],
        "sha256": sha256,
        "upstream_checksum_algorithm": algorithm,
        "upstream_checksum": expected,
        "license": LICENSE,
        "modified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--episodes", type=int, nargs="+", default=list(range(9)))
    parser.add_argument("--video-episodes", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--video", choices=["none", "left", "stereo", "all"], default="all"
    )
    parser.add_argument("--max-bytes", type=int, default=900_000_000)
    args = parser.parse_args()
    if any(ep < 0 or ep > 8 for ep in args.episodes + args.video_episodes):
        parser.error("Release contains episode indices 0 through 8")
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    metadata_url = f"https://huggingface.co/api/datasets/{REPO}/revision/{REVISION}"
    tree_url = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}?recursive=true&limit=1000"
    repo_meta = read_json(metadata_url)
    if repo_meta.get("cardData", {}).get("license") != "cc-by-4.0":
        raise ValueError("Pinned release does not declare expected CC BY 4.0 license")
    tree = read_json(tree_url)
    lookup = {entry["path"]: entry for entry in tree if entry["type"] == "file"}
    wanted = list(META)
    for ep in sorted(set(args.episodes)):
        stem = f"episode_{ep:06d}"
        wanted.append(f"data/chunk-000/{stem}.parquet")
        for sensor in ["main_stereo", "aux_left", "aux_right"]:
            wanted.append(f"imu/chunk-000/{sensor}/{stem}.parquet")
        if ep not in args.video_episodes:
            continue
        if args.video != "none":
            wanted.append(
                f"videos/chunk-000/observation.images.main_stereo_left/{stem}.mp4"
            )
        if args.video in ["stereo", "all"]:
            wanted.append(
                f"videos/chunk-000/observation.images.main_stereo_right/{stem}.mp4"
            )
        if args.video == "all":
            for camera in ["aux_left", "aux_right"]:
                wanted.append(
                    f"videos/chunk-000/observation.images.{camera}/{stem}.mp4"
                )
    selected = [lookup[path] for path in wanted]
    total = sum(item["size"] for item in selected)
    if total > args.max_bytes:
        raise ValueError(
            f"Requested {total:,} bytes exceeds explicit budget {args.max_bytes:,}"
        )
    (root / "upstream_repository_metadata.json").write_text(
        json.dumps(repo_meta, indent=2) + "\n"
    )
    (root / "upstream_file_tree.json").write_text(json.dumps(tree, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=4) as pool:
        files = list(pool.map(lambda item: acquire_file(item, root), selected))
    manifest = {
        "schema_version": "1.0.0",
        "dataset": REPO,
        "revision": REVISION,
        "source_url": f"https://huggingface.co/datasets/{REPO}",
        "license": LICENSE,
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "Humanola, Inc. — Humanola Egocentric Hand-Pose Dataset — Sample Delivery",
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_metadata_url": metadata_url,
        "retrieval_tree_url": tree_url,
        "episodes": sorted(set(args.episodes)),
        "video_selection": args.video,
        "video_episodes": sorted(set(args.episodes) & set(args.video_episodes))
        if args.video != "none"
        else [],
        "total_source_bytes": total,
        "files": files,
    }
    (root / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Complete: {len(files)} verified files, {total:,} bytes", flush=True)


if __name__ == "__main__":
    main()
