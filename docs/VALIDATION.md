# Executed validation record

Validated 19 November 2026. Run `60b3583947acfaa4`.

## Execution evidence

- Source integrity: all 47 selected files match their pinned manifest hashes; 787,644,615 source bytes.
- Pose coverage: 48,272 distinct episode/frame rows across nine episodes.
- IMU coverage: 964,596 samples; no shared-clock coverage gaps or same-frame clock conflicts in the acquired source.
- Video decoding: all four pilot cameras decode exactly 2,904 frames at 30 FPS (11,616 decoded camera-frames). Image QA samples 97 frames per camera, 388 total.
- Native interval coverage: 1,804 half-open intervals cover every source frame exactly once and agree with per-frame labels.
- Rule outcomes: 1,852 quarantined frames; 26,514 frames with one or more review flags. These are configured screening outcomes, not adjudicated error counts.
- Metadata reconciliation: declared task count 1 → 8; episode 2 task index 0 → 2 in 7,201 derived rows, preserving source values and a semantic-review requirement.
- Temporal stability: 10,027 raw transitions → 1,457 decoded transitions on adjacent known observations. This statistic is not accuracy.
- Repetition: 3,162 windows, 17 accepted windows, two retained bouts / ten side-specific cycle candidates; one overlapping alternative retained as suppressed. Episode 4 abstains.
- Review selection: 36 unique proposals cover all nine episodes and all six states, with nine minimum-cover seeds.

## Checks completed

`uv run pytest -q`: **77 passed**. Includes quaternion/coordinate math, absence and discontinuity handling, shared-clock precision and shifts, exact decoder optimality, duration/censoring, recurrence recovery and rejection, review coverage, journal concurrency/tamper checks, HTTP ranges/origins, and real-artifact integration checks.

`uv run egoforge verify`: **51 artifacts checked, zero mismatches**. Production code hashes in provenance match the delivered files.

`ruff check --select E9,F63,F7,F82`: passed; Python files formatted with Ruff 0.17.6. Formatting tooling is not a runtime dependency.

Synthetic corruption suite: **9/9 seeded cases** detected every injected target frame. Tests cover coordinate translation, image-keypoint displacement, quaternion scaling, presence contradiction, tracking-quality contradiction, stretched finger geometry, duplicate timestamp, nominal IMU clock mismatch and missing IMU burst. Severe synthetic corruption sensitivity does not establish real-world QA precision.

Browser inspection confirmed real video loading and playback, camera switching, inherited skeleton overlays, phase timeline and motion curves. The console was left on the pilot episode. No real review decisions were submitted.

Label Studio range round-trip and `frameRate=30` configuration pass integration tests. The running media endpoint returns HTTP 206 with the expected restricted CORS header for localhost:8080. Import into a separately running Label Studio application was not performed.

## Extended benchmark run

The measured benchmark report is [BENCHMARKS.md](BENCHMARKS.md). Three isolated CPU processes each handled all nine episodes: median **11.092 seconds**, **4,352 pose frames/second**, maximum whole-worker peak RSS **783.8 MB** on Apple M4. Timings exclude video decode, download, export serialization, manual review and training; inputs were locally cached.

Executed **126 threshold-sweep experiments** across three episodes and **27 severe corruption cases**. All severe cases were detected, and 18 zero-magnitude controls produced no new flags. Incremental sensitivity excludes targets already flagged by the same rule. These results do not measure semantic annotation accuracy.

The minimal light interface and its Benchmarks tab were checked in the browser. Camera switching, video playback, episode selection and measured result rendering work; no browser console errors were observed. Preview screenshots are saved in `output/previews/`.

## Review status

**Zero human-adjudicated phase labels.** All delivered phases and repetition events are machine proposals. Source poses and tasks are inherited. `reviews.jsonl` is created only when a real decision is submitted; temporary automated-test journals do not contribute to this release.

The next scientific evaluation requires an independently annotated reference set. Semantic accuracy, human agreement, physical pose error, contact, force and successful task completion remain unmeasured.
