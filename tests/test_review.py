"""Review journal integrity, stale-proposal rejection, and real HTTP media/API behavior."""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from egoforge.review import (
    GENESIS,
    ReviewStore,
    _canonical,
    create_server,
    parse_range,
    read_reviews,
)


@pytest.fixture
def artifacts(tmp_path):
    run = tmp_path / "run"
    raw = tmp_path / "raw"
    run.mkdir()
    raw.mkdir()
    segment = {
        "segment_id": "ep004-s000",
        "episode_id": 4,
        "state": "left_dominant",
        "start_frame": 0,
        "end_frame": 30,
        "start_time_s": 0.0,
        "end_time_s": 1.0,
    }
    (run / "segments.json").write_text(json.dumps([segment]))
    (run / "summary.json").write_text(
        json.dumps({"run_id": "test-run-1", "total_frames": 30})
    )
    payload = {
        "segment_id": segment["segment_id"],
        "actor": "test-reviewer",
        "actor_type": "machine",
        "decision": "approve",
        "run_id": "test-run-1",
        "proposal_sha256": hashlib.sha256(_canonical(segment)).hexdigest(),
    }
    return run, raw, segment, payload


def test_journal_records_provenance_and_detects_tampering(artifacts):
    run, _, _, payload = artifacts
    store = ReviewStore(run)
    first = store.append({**payload, "request_id": "request-1"})
    second = store.append(
        {
            **payload,
            "decision": "correct",
            "corrected_state": "quiescent",
            "note": "No visible wrist motion",
            "request_id": "request-2",
        }
    )
    assert first["previous_hash"] == GENESIS
    assert second["previous_hash"] == first["event_hash"]
    assert second["proposed_state"] == "left_dominant"
    assert second["run_id"] == "test-run-1"
    assert second["actor_type"] == "machine"
    assert len(read_reviews(store.path)) == 2
    store.path.write_text(
        store.path.read_text().replace("No visible wrist motion", "Changed evidence")
    )
    with pytest.raises(ValueError, match="integrity failure"):
        read_reviews(store.path)
    with pytest.raises(ValueError, match="integrity failure"):
        store.append(payload)


def test_review_validation_and_stale_proposals(artifacts):
    run, _, segment, payload = artifacts
    store = ReviewStore(run)
    for change in [
        {"actor": "  "},
        {"actor_type": None},
        {"segment_id": "missing"},
        {"decision": "accepted"},
        {"decision": "correct", "corrected_state": "cutting"},
        {"decision": "reject", "note": ""},
        {"run_id": "old-run"},
        {"proposal_sha256": "wrong"},
        {"corrected_state": "quiescent"},
    ]:
        with pytest.raises(ValueError):
            store.append({**payload, **change})
    (run / "segments.json").write_text(json.dumps([{**segment, "state": "quiescent"}]))
    with pytest.raises(ValueError, match="Stale"):
        store.append(payload)
    assert not store.path.exists()


def test_idempotency_and_concurrent_chain(artifacts):
    run, _, _, payload = artifacts
    store = ReviewStore(run)
    first = store.append({**payload, "request_id": "same-request"})
    assert store.append({**payload, "request_id": "same-request"}) == first
    with pytest.raises(ValueError, match="different decision"):
        store.append({**payload, "request_id": "same-request", "note": "changed"})
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: store.append({**payload, "request_id": f"parallel-{i}"}),
                range(10),
            )
        )
    assert len(read_reviews(store.path)) == 11


@pytest.mark.parametrize(
    "header, expected",
    [
        (None, None),
        ("bytes=2-5", (2, 5)),
        ("bytes=7-", (7, 9)),
        ("bytes=-3", (7, 9)),
        ("bytes=0-99", (0, 9)),
        ("bytes=-99", (0, 9)),
    ],
)
def test_ranges(header, expected):
    assert parse_range(header, 10) == expected


@pytest.mark.parametrize(
    "header",
    [
        "bytes=",
        "bytes=-",
        "bytes=-0",
        "bytes=10-11",
        "bytes=5-2",
        "bytes=0-1,3-4",
        "items=0-2",
    ],
)
def test_bad_ranges(header):
    with pytest.raises(ValueError):
        parse_range(header, 10)


def test_real_http_range_origin_path_and_review(artifacts):
    run, raw, _, payload = artifacts
    video = (
        raw / "videos/chunk-000/observation.images.main_stereo_left/episode_000004.mp4"
    )
    video.parent.mkdir(parents=True)
    video.write_bytes(b"0123456789")
    with create_server(run, raw) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with urlopen(base + "/") as response:
                assert response.status == 200
                document = response.read()
                assert b"EgoForge" in document and b'id="video"' in document
                assert (
                    "frame-ancestors 'none'"
                    in response.headers["Content-Security-Policy"]
                )
            with urlopen(base + "/api/benchmarks") as response:
                assert json.load(response) == {"status": "not_run"}
            benchmark_dir = run.parent / "benchmarks"
            benchmark_dir.mkdir()
            measured = {"summary": {"frames": 30, "total_median_s": 0.25}}
            (benchmark_dir / "benchmark.json").write_text(json.dumps(measured))
            with urlopen(base + "/api/benchmarks") as response:
                assert json.load(response) == measured
            with urlopen(base + "/api/segments") as response:
                segment = json.load(response)[0]
                assert (
                    segment["_review"]["proposal_sha256"] == payload["proposal_sha256"]
                )
            with urlopen(
                Request(
                    base + "/media/episode_000004/main_stereo_left.mp4",
                    headers={"Range": "bytes=3-6"},
                )
            ) as response:
                assert response.status == 206
                assert response.headers["Content-Range"] == "bytes 3-6/10"
                assert response.read() == b"3456"
            for origin in ["http://localhost:8080", "http://127.0.0.1:8080"]:
                with urlopen(
                    Request(
                        base + "/media/episode_000004/main_stereo_left.mp4",
                        headers={"Origin": origin, "Range": "bytes=3-6"},
                    )
                ) as response:
                    assert response.status == 206
                    assert response.headers["Access-Control-Allow-Origin"] == origin
                    assert response.headers["Vary"] == "Origin"
                    assert response.read() == b"3456"
                with urlopen(
                    Request(
                        base + "/media/episode_000004/main_stereo_left.mp4",
                        method="OPTIONS",
                        headers={
                            "Origin": origin,
                            "Access-Control-Request-Method": "GET",
                            "Access-Control-Request-Headers": "Range",
                        },
                    )
                ) as response:
                    assert response.status == 204
                    assert response.headers["Access-Control-Allow-Origin"] == origin
                    assert response.headers["Access-Control-Allow-Headers"] == "Range"
            for path, headers, status in [
                ("/assets/../../review.py", {}, 404),
                ("/api/frames?episode=../secret", {}, 400),
                (
                    "/media/episode_000004/main_stereo_left.mp4",
                    {"Range": "bytes=20-"},
                    416,
                ),
                ("/api/summary", {"Origin": "https://attacker.example"}, 403),
                ("/api/benchmarks", {"Origin": "https://attacker.example"}, 403),
                ("/api/summary", {"Origin": "http://localhost:8080"}, 403),
                (
                    "/media/episode_000004/main_stereo_left.mp4",
                    {"Origin": "https://attacker.example"},
                    403,
                ),
                (
                    "/media/episode_000004/main_stereo_left.mp4",
                    {"Origin": "http://localhost:8081"},
                    403,
                ),
                ("/api/summary", {"Host": "attacker.example"}, 403),
            ]:
                with pytest.raises(HTTPError) as exc:
                    urlopen(Request(base + path, headers=headers))
                assert exc.value.code == status
            with pytest.raises(HTTPError) as exc:
                urlopen(
                    Request(
                        base + "/api/reviews",
                        data=json.dumps(payload).encode(),
                        headers={
                            "Content-Type": "application/json",
                            "Origin": "http://localhost:8080",
                        },
                    )
                )
            assert exc.value.code == 403
            assert not (run / "reviews.jsonl").exists()
            request = Request(
                base + "/api/reviews",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request) as response:
                assert response.status == 201
                assert json.load(response)["actor_type"] == "machine"
            with urlopen(base + "/api/export") as response:
                exported = json.load(response)
                assert len(exported["events"]) == 1
                assert exported["journal_tip"] == exported["events"][0]["event_hash"]
        finally:
            server.shutdown()
            thread.join(timeout=2)
