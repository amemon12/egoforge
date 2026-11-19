# Provenance and acquisition record

## Source identity

| Field | Recorded value |
|---|---|
| Publisher | Humanola, Inc. |
| Dataset | Humanola Egocentric Hand-Pose Dataset — Sample Delivery |
| Repository | `humanola-inc/egocentric-sample` |
| Source URL | https://huggingface.co/datasets/humanola-inc/egocentric-sample |
| Immutable revision | `3c6e782911b4e82a56b288f6830d3a43bdb85d49` |
| Source repository creation | 2026-07-04T19:48:31Z, as returned by upstream API |
| Source revision last modification | 2026-07-04T19:54:09Z, as returned by upstream API |
| Acquisition completed | 2026-11-19T21:00:49.341614Z |
| Declared source license | CC BY 4.0 |
| Access at acquisition | Public; API reported `gated: false` |

The source README's YAML license field and pinned repository API card both declare `cc-by-4.0`. The corresponding license is [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/). The acquisition does not substitute a software repository's license for a dataset license.

## Reproduction

From the project root:

```sh
python scripts/acquire.py
```

Defaults acquire all nine pose tables, all 27 IMU tables, seven source metadata/card files, and four camera videos for episode 4. The downloader first checks the license declaration, builds a selection from the revision-pinned file tree, and refuses requests exceeding its default 900,000,000-byte budget. This scope totals 47 source files and 787,644,615 bytes. Source objects download to temporary paths and are moved into place only after size and checksum verification. Repeated runs reverify existing source files.

The downloader supports `--episodes`, `--video-episodes`, `--video`, `--root`, and `--max-bytes` for explicit alternate subsets. Any different selection creates a different manifest; claims about this specific audit apply to the pinned default acquisition.

Three local provenance files live under `data/raw/humanola/`:

- `source_manifest.json`: each selected path, immutable URL, byte count, local SHA-256, upstream checksum and its algorithm, license, source revision, acquisition time, and attribution.
- `upstream_repository_metadata.json`: upstream repository API response at the pinned revision.
- `upstream_file_tree.json`: revision-specific upstream file listing, object IDs, sizes, and LFS hashes.

Large LFS objects are checked against their upstream SHA-256. Small Git-tracked objects are checked using `SHA1("blob " + decimal_byte_length + NUL + original_bytes)`. Every source file also receives a local SHA-256. These checks demonstrate that the downloaded bytes match the pinned release; they do not certify the source's semantic correctness or capture history.

## Lineage boundaries

The source directory contains unmodified original payloads and explicitly named locally generated provenance files. New annotations, reports, visual overlays, extracted previews, cleaned representations, and adjudication decisions belong outside the source payload paths. Any derived copy should identify the pinned source revision, source episode/frame identifiers, generating configuration, and relevant software version.

The original 3D pose, 2D keypoint, hand-presence, task, grip, and calibration fields are inherited. New temporal labels are automatic proposals. Human review is a separate step and must only be reported when an identifiable review record exists. In particular, a derived QA score or automatic reviewer assignment does not establish that a person validated the labels.

The source task metadata conflict in episode 2 and the quality/presence contradiction at frame 5901 are retained in the original data. Any downstream resolution must be reported as a documented decision, not attributed to an upstream corrected release. Known source limitations are recorded in [DATASET_CARD.md](DATASET_CARD.md).

## Exploratory synchronization audit

This independent diagnostic compared the magnitude of head-pose angular velocity against the magnitude of the head-camera gyro. It did not use the task label or newly proposed temporal classes.

For each consecutive pose pair, compute the SO(3) relative rotation angle and divide by the measured interval obtained by interpolating head IMU `timestamp_ns` over fractional `frame`. Associate that angular speed with the frame midpoint `f = i + 0.5`. Interpolate the raw gyro magnitude at `f + lag`, scan lags from −15 to +15 frames in 0.25-frame steps, and compute Pearson correlation. Use fixed common support excluding 16 frames at each edge. No smoothing or automatic retiming was applied in this diagnostic.

Positive lag means the comparison samples later gyro values against the pose-speed midpoint; negative lag samples earlier gyro values. Nominal milliseconds below use 30 fps.

| Episode | Correlation at zero lag | Best correlation | Best lag (frames) | Nominal lag (ms) |
|---|---:|---:|---:|---:|
| 0 | 0.934390 | 0.966674 | −1.25 | −41.67 |
| 1 | 0.831470 | 0.960777 | −2.50 | −83.33 |
| 2 | 0.944749 | 0.988668 | −2.00 | −66.67 |
| 3 | 0.955002 | 0.970507 | −1.25 | −41.67 |
| 4 | 0.992467 | 0.997744 | +1.00 | +33.33 |
| 5 | 0.933290 | 0.979834 | −2.00 | −66.67 |
| 6 | 0.957630 | 0.994870 | −2.00 | −66.67 |
| 7 | 0.865463 | 0.960369 | −2.25 | −75.00 |
| 8 | 0.875879 | 0.965030 | −2.00 | −66.67 |

These are diagnostic maxima on a searched grid, not calibrated estimates of clock error. Pose filtering, upstream tracking latency, similar motion profiles, and loss of direction in a magnitude comparison can all influence the optimum. The pose and gyro may also share upstream estimation inputs, so their agreement is not independent validation. Episode 4's opposite lag direction specifically argues against applying a single global shift. Preserve source clocks and expose suspected offsets for review.

## Attribution for derived material

Suggested notice:

> Contains material from Humanola, Inc., Humanola Egocentric Hand-Pose Dataset — Sample Delivery, revision 3c6e782911b4e82a56b288f6830d3a43bdb85d49, available under CC BY 4.0. EgoForge adds derived motion annotations, quality diagnostics, and visualization overlays. The original publisher does not endorse these additions.

Include the [source link](https://huggingface.co/datasets/humanola-inc/egocentric-sample) and [license link](https://creativecommons.org/licenses/by/4.0/) with that notice when distributing material. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
