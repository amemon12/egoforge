"""Loopback-only, dependency-free evidence review and append-only adjudication API."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

STATES = frozenset(
    {
        "quiescent",
        "left_dominant",
        "right_dominant",
        "bilateral_coordinated",
        "bilateral_independent",
        "unknown",
    }
)
CAMERAS = frozenset({"main_stereo_left", "main_stereo_right", "aux_left", "aux_right"})
LABEL_STUDIO_ORIGINS = frozenset({"http://localhost:8080", "http://127.0.0.1:8080"})
GENESIS = "0" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def read_json(path: Path, default: object) -> object:
    return json.loads(path.read_text()) if path.is_file() else default


def read_reviews(path: Path) -> list[dict]:
    """Fail closed on a truncated, altered, or reordered review journal."""
    events: list[dict] = []
    previous = GENESIS
    if not path.exists():
        return events
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        try:
            event = json.loads(line)
            digest = event.pop("event_hash")
            if (
                event["previous_hash"] != previous
                or hashlib.sha256(_canonical(event)).hexdigest() != digest
            ):
                raise ValueError("hash chain mismatch")
            event["event_hash"] = digest
            events.append(event)
            previous = digest
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(
                f"Review journal integrity failure on line {line_number}"
            ) from exc
    return events


class ReviewStore:
    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir).resolve()
        self.path = self.run_dir / "reviews.jsonl"
        self.lock = threading.Lock()

    def append(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ValueError("Review must be a JSON object")
        actor = payload.get("actor", "")
        if not isinstance(actor, str) or not actor.strip() or len(actor) > 120:
            raise ValueError(
                "A reviewer name or identifier is required (maximum 120 characters)"
            )
        actor_type = payload.get("actor_type")
        if actor_type not in {"human", "machine"}:
            raise ValueError("actor_type must explicitly be human or machine")
        decision = payload.get("decision")
        if decision not in {"approve", "reject", "correct"}:
            raise ValueError("decision must be approve, reject, or correct")
        segments = read_json(self.run_dir / "segments.json", [])
        if isinstance(segments, dict):
            segments = segments.get("segments", [])
        segment = next(
            (
                item
                for item in segments
                if str(item.get("segment_id")) == str(payload.get("segment_id"))
            ),
            None,
        )
        if segment is None:
            raise ValueError("Unknown segment_id")
        run_id = read_json(self.run_dir / "summary.json", {}).get("run_id")
        proposal_hash = hashlib.sha256(_canonical(segment)).hexdigest()
        if not run_id or payload.get("run_id") != run_id:
            raise ValueError(
                "Stale or missing run_id; reload the current run before reviewing"
            )
        if payload.get("proposal_sha256") != proposal_hash:
            raise ValueError(
                "Stale or missing proposal_sha256; reload the current segment before reviewing"
            )
        corrected_state = payload.get("corrected_state")
        if decision == "correct" and corrected_state not in STATES:
            raise ValueError("A correction requires a valid corrected_state")
        if decision != "correct" and corrected_state is not None:
            raise ValueError("corrected_state is only valid for a correction")
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > 4000:
            raise ValueError("note must be text, at most 4000 characters")
        if decision in {"correct", "reject"} and not note.strip():
            raise ValueError("A rejection or correction requires a reason")
        request_id = payload.get("request_id") or str(uuid.uuid4())
        if not isinstance(request_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,100}", request_id
        ):
            raise ValueError("Invalid request_id")
        with self.lock:
            events = read_reviews(self.path)
            existing = next(
                (item for item in events if item["request_id"] == request_id), None
            )
            if existing:
                if any(
                    existing.get(key) != value
                    for key, value in {
                        "actor": actor.strip(),
                        "actor_type": actor_type,
                        "decision": decision,
                        "segment_id": str(segment["segment_id"]),
                        "corrected_state": corrected_state,
                        "note": note.strip(),
                        "run_id": run_id,
                        "proposal_sha256": proposal_hash,
                    }.items()
                ):
                    raise ValueError("request_id already used for a different decision")
                return existing
            event = {
                "schema_version": "1.0",
                "event_id": str(uuid.uuid4()),
                "request_id": request_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "actor": actor.strip(),
                "actor_type": actor_type,
                "decision": decision,
                "segment_id": str(segment["segment_id"]),
                "episode_id": segment.get("episode_id"),
                "proposed_state": segment.get("state"),
                "run_id": run_id,
                "corrected_state": corrected_state,
                "note": note.strip(),
                "proposal_sha256": proposal_hash,
                "previous_hash": events[-1]["event_hash"] if events else GENESIS,
            }
            event["event_hash"] = hashlib.sha256(_canonical(event)).hexdigest()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("ab") as handle:
                handle.write(_canonical(event) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            return event


def parse_range(header: str | None, size: int) -> tuple[int, int] | None:
    """Return an inclusive single byte range, rejecting malformed/multipart requests."""
    if header is None:
        return None
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
    if not match or not any(match.groups()) or size == 0:
        raise ValueError("Invalid byte range")
    first, last = match.groups()
    if not first:
        suffix = int(last)
        if suffix <= 0:
            raise ValueError("Invalid suffix range")
        return max(size - suffix, 0), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or start > end:
        raise ValueError("Unsatisfiable byte range")
    return start, end


class ReviewServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, run_dir: Path, raw_dir: Path, port: int):
        self.run_dir = Path(run_dir).resolve()
        self.raw_dir = Path(raw_dir).resolve()
        self.static_dir = Path(__file__).parent / "static"
        self.store = ReviewStore(self.run_dir)
        super().__init__(("127.0.0.1", port), ReviewHandler)


class ReviewHandler(BaseHTTPRequestHandler):
    server: ReviewServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args: object) -> None:
        # Avoid logging reviewer identities, decision text, or filesystem paths.
        pass

    def _headers(
        self, status: int, content_type: str, length: int, extra: dict | None = None
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; media-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        if media_origin := self._media_origin():
            self.send_header("Access-Control-Allow-Origin", media_origin)
            self.send_header("Vary", "Origin")
            self.send_header(
                "Access-Control-Expose-Headers",
                "Content-Length, Content-Range, Accept-Ranges",
            )
        for key, value in (extra or {}).items():
            self.send_header(key, str(value))
        self.end_headers()

    def _json(self, value: object, status: int = 200) -> None:
        data = _canonical(value)
        self._headers(status, "application/json; charset=utf-8", len(data))
        if self.command != "HEAD":
            self.wfile.write(data)

    def _same_origin(self) -> bool:
        expected = {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }
        if self.headers.get("Host") not in expected:
            self._json({"error": "Only the local review origin is allowed"}, 403)
            return False
        origin = self.headers.get("Origin")
        if (
            origin
            and origin not in {f"http://{host}" for host in expected}
            and not self._media_origin()
        ):
            self._json({"error": "Cross-origin requests are not allowed"}, 403)
            return False
        return True

    def _media_origin(self) -> str | None:
        """Only allow the two documented local Label Studio readers, never API writes."""
        if self.command not in {"GET", "HEAD", "OPTIONS"}:
            return None
        route = unquote(urlsplit(self.path).path)
        media = re.fullmatch(r"/media/episode_(\d{6})/([a-z_]+)\.mp4", route)
        origin = self.headers.get("Origin")
        if media and media.group(2) in CAMERAS and origin in LABEL_STUDIO_ORIGINS:
            return origin
        return None

    def do_OPTIONS(self) -> None:
        if not self._same_origin():
            return
        requested_headers = {
            name.strip().lower()
            for name in self.headers.get("Access-Control-Request-Headers", "").split(
                ","
            )
            if name.strip()
        }
        if (
            not self._media_origin()
            or self.headers.get("Access-Control-Request-Method") not in {"GET", "HEAD"}
            or not requested_headers.issubset({"range"})
        ):
            self._json(
                {
                    "error": "Only local Label Studio media reads support cross-origin access"
                },
                403,
            )
            return
        self._headers(
            204,
            "text/plain",
            0,
            {
                "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
                "Access-Control-Allow-Headers": "Range",
            },
        )

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._same_origin():
            return
        try:
            request = urlsplit(self.path)
            route = unquote(request.path)
            if route == "/api/benchmarks":
                self._json(
                    read_json(
                        self.server.run_dir.parent / "benchmarks" / "benchmark.json",
                        {"status": "not_run"},
                    )
                )
                return
            if route in {
                "/api/summary",
                "/api/episodes",
                "/api/segments",
                "/api/review-queue",
            }:
                filename = route.split("/")[-1].replace("-", "_") + ".json"
                result = read_json(
                    self.server.run_dir / filename,
                    {} if route.endswith("summary") else [],
                )
                if route == "/api/segments":
                    run_id = read_json(self.server.run_dir / "summary.json", {}).get(
                        "run_id"
                    )
                    segments = (
                        result
                        if isinstance(result, list)
                        else result.get("segments", [])
                    )
                    segments = [
                        {
                            **segment,
                            "_review": {
                                "run_id": run_id,
                                "proposal_sha256": hashlib.sha256(
                                    _canonical(segment)
                                ).hexdigest(),
                            },
                        }
                        for segment in segments
                    ]
                    result = (
                        segments
                        if isinstance(result, list)
                        else {**result, "segments": segments}
                    )
                self._json(result)
                return
            if route == "/api/frames":
                episode = parse_qs(request.query).get("episode", [""])[0]
                if not re.fullmatch(r"\d{1,6}", episode):
                    self._json({"error": "episode must be an integer"}, 400)
                    return
                self._json(
                    read_json(
                        self.server.run_dir
                        / "frames"
                        / f"episode_{int(episode):06d}.json",
                        [],
                    )
                )
                return
            if route in {"/api/reviews", "/api/export"}:
                events = read_reviews(self.server.store.path)
                if route == "/api/reviews":
                    self._json(events)
                else:
                    latest = {item["segment_id"]: item for item in events}
                    self._json(
                        {
                            "schema_version": "1.0",
                            "events": events,
                            "latest_decisions": list(latest.values()),
                            "journal_tip": events[-1]["event_hash"]
                            if events
                            else GENESIS,
                            "note": "Review decisions are separate from unchanged machine proposals. Actor type is explicitly declared by the reviewer; this local journal has no identity authentication.",
                        }
                    )
                return
            video = re.fullmatch(r"/media/episode_(\d{6})/([a-z_]+)\.mp4", route)
            if video and video.group(2) in CAMERAS:
                path = (
                    self.server.raw_dir
                    / "videos"
                    / "chunk-000"
                    / ("observation.images." + video.group(2))
                    / ("episode_" + video.group(1) + ".mp4")
                )
                self._file(path, self.server.raw_dir, "video/mp4")
                return
            if route == "/":
                self._file(
                    self.server.static_dir / "index.html",
                    self.server.static_dir,
                    "text/html; charset=utf-8",
                )
                return
            if route.startswith("/assets/"):
                self._file(
                    self.server.static_dir / route[len("/assets/") :],
                    self.server.static_dir,
                )
                return
            self._json({"error": "Not found"}, 404)
        except (ValueError, OSError, TypeError) as exc:
            self._json(
                {
                    "error": str(exc)
                    if isinstance(exc, ValueError)
                    else "Unable to read the requested artifact"
                },
                409 if "integrity" in str(exc) else 500,
            )

    def do_POST(self) -> None:
        if not self._same_origin():
            self.close_connection = True
            return
        if urlsplit(self.path).path != "/api/reviews":
            self._json({"error": "Not found"}, 404)
            self.close_connection = True
            return
        if (
            self.headers.get("Content-Type", "").split(";")[0].strip()
            != "application/json"
        ):
            self._json({"error": "Content-Type must be application/json"}, 415)
            self.close_connection = True
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("Review body must contain 1–16384 bytes")
            payload = json.loads(self.rfile.read(length))
            self._json(self.server.store.append(payload), 201)
        except (ValueError, UnicodeError) as exc:
            self._json({"error": str(exc)}, 400)
            self.close_connection = True

    def _file(self, path: Path, root: Path, content_type: str | None = None) -> None:
        path = path.resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            self._json({"error": "File not found"}, 404)
            return
        size = path.stat().st_size
        try:
            byte_range = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self._headers(416, "text/plain", 0, {"Content-Range": f"bytes */{size}"})
            return
        start, end = byte_range or (0, size - 1)
        extra = {"Accept-Ranges": "bytes"}
        if byte_range:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self._headers(
            206 if byte_range else 200,
            content_type
            or mimetypes.guess_type(path.name)[0]
            or "application/octet-stream",
            max(end - start + 1, 0),
            extra,
        )
        if self.command == "HEAD":
            return
        with path.open("rb") as handle:
            handle.seek(start)
            remaining = end - start + 1
            try:
                while remaining > 0:
                    chunk = handle.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # Seeking or switching cameras cancels the previous media transfer.


def create_server(
    run_dir: Path | str, raw_dir: Path | str, port: int = 0
) -> ReviewServer:
    return ReviewServer(Path(run_dir), Path(raw_dir), port)


def serve(run_dir: Path | str, raw_dir: Path | str, port: int = 8765) -> None:
    with create_server(run_dir, raw_dir, port) as server:
        print(
            f"EgoForge review console: http://127.0.0.1:{server.server_port}",
            flush=True,
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("output/run"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/humanola"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    serve(args.run_dir, args.raw_dir, args.port)
