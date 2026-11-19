# Third-party notices

## Humanola source data

**Publisher:** Humanola, Inc.  
**Title:** Humanola Egocentric Hand-Pose Dataset — Sample Delivery  
**Source:** https://huggingface.co/datasets/humanola-inc/egocentric-sample  
**Pinned revision:** `3c6e782911b4e82a56b288f6830d3a43bdb85d49`  
**Declared license:** [Creative Commons Attribution 4.0 International, CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
**Legal code:** https://creativecommons.org/licenses/by/4.0/legalcode  
**Retrieved:** 19 November 2026

The source dataset card and repository metadata declare `cc-by-4.0`. The original README is retained at `data/raw/humanola/README.md`; file-level integrity and source URLs are recorded in `data/raw/humanola/source_manifest.json`.

The original recordings, tracking outputs, pose and keypoint annotations, task descriptions, calibration, and IMU measurements are supplied by Humanola. EgoForge adds software, derived motion proposals, quality diagnostics, and visualizations. Extracted frames, crops, recompressed previews, and overlays are modifications or derivatives of the source material and must identify those changes. The raw downloaded source files are unmodified.

Retain source attribution and a license link when redistributing source material or derived visual/data artifacts. Do not imply endorsement by Humanola. A software license for EgoForge does not replace the source dataset's license or represent additional rights to the underlying recordings.

## Runtime dependencies

EgoForge uses separately licensed Python packages listed in `pyproject.toml`, including NumPy, pandas, PyArrow, SciPy, OpenCV, Pillow, and, for development, pytest. Their licenses and notices remain with their installed distributions. This project does not bundle model weights or claim ownership of those dependencies. A distributable binary or environment bundle should include notices from its exact installed versions.
