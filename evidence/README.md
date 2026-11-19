# Executed evidence

These are snapshots from the executed EgoForge run and benchmark suite on 19 November 2026. They can be inspected without downloading the source dataset.

- [Run report](run/REPORT.md) and [summary](run/summary.json): 48,272 frames, nine episodes, 1,804 machine-proposed intervals.
- [Measured benchmarks](benchmarks/benchmark.json), raw timing passes and CSV tables: three CPU passes, 126 threshold sweeps, 27 severe corruption cases.
- [Review interface](previews/review.jpg), [benchmark interface](previews/benchmarks.jpg), and [annotated contact sheet](previews/contact_sheet.jpg).
- [Source manifest](source/source_manifest.json) and [original dataset card](source/README.md): pinned upstream files, checksums and attribution.

`manifest.json` checksums every retained file. `benchmarks/artifact_manifest.json` retains the complete original benchmark-file manifest. `run/` is a selected report subset; the full run manifest and large generated annotation tables are available in the release bundle or after reproduction, and are not represented as complete here.

Raw recordings and pose/IMU inputs are downloaded by `scripts/acquire.py`. The runtime reads `output/run/`, not this snapshot directory. Follow the repository quick start to rebuild it, or unpack the release evidence bundle and acquire its source inputs.

Source footage and inherited tracking are from Humanola, Inc., under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The previews add skeleton overlays and EgoForge interface/phase labels. See [third-party notices](../THIRD_PARTY_NOTICES.md). No human-adjudicated phase accuracy is claimed.
