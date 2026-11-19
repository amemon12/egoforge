# Evidence review

The local console joins proposed motion phases to source video, inherited 2D hand skeletons, world-frame wrist speeds, and QA findings. It starts on the four-camera episode when present. Other episodes expose their kinematic evidence without implying that their videos have been downloaded.

From the project directory:

```sh
python -m egoforge.review --run-dir output/run --raw-dir data/raw/humanola --port 8765
```

Open <http://127.0.0.1:8765>. The server binds only to loopback. It rejects foreign Host/Origin values except the explicitly permitted media readers described below, and has no remote scripts, analytics, or network writes. Do not expose it through a public reverse proxy: it is a local review application, not an authenticated multi-user service.

1. Select a priority segment or click the phase timeline. Left/right arrow keys move between segments; Space toggles playback.
2. Inspect the head-left view with its inherited skeleton. Camera switching preserves the current time. The supplied 2D keypoints belong to the head-left image, so the overlay is disabled for other cameras.
3. Compare motion, flags, and quality. The UI uses sampled frame records; machine exports retain the full frame resolution. Uncertainty is an uncalibrated heuristic. A high score is a review cue, not a probability that the label is wrong.
4. Enter an actual reviewer identifier. Approve, reject, or select a corrected phase. Rejection/correction requires a written reason. Never approve a sequence merely because the geometric consistency checks pass: source tracks can be consistently wrong.
5. Export the journal from the top navigation. The source proposals remain immutable; adjudication is stored separately.

`reviews.jsonl` is an append-only hash chain. Each event includes its run ID, source-proposal SHA-256, actor identifier and declared actor type, decision, evidence note, previous hash, event hash, and UTC timestamp. The server refuses changed or truncated journals and rejects decisions based on stale runs or proposals. Request IDs make retries idempotent. The chain detects modification relative to a trusted tip; it is not a digital signature, external timestamp, or authenticated identity proof. Preserve the exported tip independently when handing off a release. A local process with filesystem access can replace the entire journal; the application does not claim otherwise.

The UI records explicitly submitted user decisions as `actor_type: human`. Programmatic reviews must submit `actor_type: machine`. Automated test decisions use temporary fixtures and are never added to the project's real review journal. A fresh run has zero human decisions.

## API artifacts

`GET /api/summary`, `/api/episodes`, `/api/segments`, `/api/review-queue`, `/api/frames?episode=4`, `/api/reviews`, and `/api/export` expose the local release. Segments returned by the API include a transient `_review` token with `run_id` and `proposal_sha256`; it is not written back to the proposal file. `POST /api/reviews` requires these two values, a known `segment_id`, `actor`, `actor_type`, `decision`, and optionally `note`, `corrected_state`, and a unique `request_id`. Allowed corrected states are `quiescent`, `left_dominant`, `right_dominant`, `bilateral_coordinated`, `bilateral_independent`, and `unknown`.

Video is served through `/media/episode_000004/main_stereo_left.mp4` and the corresponding `main_stereo_right`, `aux_left`, and `aux_right` paths. Single byte ranges support seeking without reloading entire source recordings. Arbitrary filesystem paths are not exposed.

## Label Studio media access

The media endpoints permit read-only CORS from exactly `http://localhost:8080` and `http://127.0.0.1:8080` to support a local Label Studio installation. The whitelist applies only to GET/HEAD media requests and their range preflights. API reads, review mutations, static assets, other ports, and external origins remain blocked. Label Studio config uses an explicit 30 fps value; prediction ranges are 1-based inclusive, converted from EgoForge's 0-based half-open intervals.

Import compatibility has been checked against the official converter and artifact schema, but a complete Label Studio application import has not been executed. If Label Studio runs on another port, host, in a remote browser, or behind a media proxy, import/rehost the source video through that installation and replace the task's `data.video` URL with its reachable media URL. A loopback URL always refers to the machine running the browser. Do not widen the review API's origin policy to make an external video host work.
