#!/usr/bin/env python3
"""Bundle source + computed evidence; source videos/pose inputs rehydrate by hash."""

from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import hashlib

root = Path(__file__).resolve().parents[1]
paths = []
for directory in [
    "egoforge",
    "tests",
    "docs",
    "scripts",
    "configs",
    "evidence",
    "output/run",
    "output/benchmarks",
    "output/previews",
]:
    paths.extend(
        p
        for p in (root / directory).rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "reviews.jsonl"
    )
for name in [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CITATION.cff",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitignore",
    "data/raw/humanola/source_manifest.json",
]:
    paths.append(root / name)
destination = root / "dist"
destination.mkdir(exist_ok=True)
archive = destination / "egoforge-v0.1.0.zip"
with ZipFile(archive, "w", ZIP_DEFLATED, compresslevel=6) as z:
    for p in sorted(set(paths)):
        z.write(p, Path("egoforge") / p.relative_to(root))
h = hashlib.sha256(archive.read_bytes()).hexdigest()
(destination / "SHA256SUMS").write_text(f"{h}  {archive.name}\n")
print(f"{archive} ({archive.stat().st_size:,} bytes, {len(paths)} files)")
