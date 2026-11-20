"""Serve the local BaddieVision experiment viewer."""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from webbrowser import open as open_browser

from .experiments import READ_ERRORS, discover, recording_metrics, recording_overlays


WEB = Path(__file__).with_name("web")
ASSETS = {
    "/": "index.html", "/styles.css": "styles.css", "/diagnostics.js": "diagnostics.js",
    "/failures.js": "failures.js", "/experiments.js": "experiments.js", "/overlays.js": "overlays.js",
}
BYTE_RANGE = re.compile(r"bytes=(\d*)-(\d*)")


def response_range(header, size):
    if not header:
        return 0, size - 1
    match = BYTE_RANGE.fullmatch(header)
    if match is None or not any(match.groups()) or not size:
        raise ValueError("Invalid range")
    start, end = match.groups()
    if not start:
        if int(end) == 0:
            raise ValueError("Empty suffix range")
        return max(0, size - int(end)), size - 1
    lo = int(start)
    hi = min(int(end), size - 1) if end else size - 1
    if lo > hi:
        raise ValueError("Unsatisfiable range")
    return lo, hi


class Server(ThreadingHTTPServer):
    def __init__(self, root, port):
        self.root = root
        self.recordings = {}
        super().__init__(("127.0.0.1", port), Handler)

    def handle_error(self, request, client_address):
        if not isinstance(sys.exception(), (BrokenPipeError, ConnectionResetError, TimeoutError)):
            super().handle_error(request, client_address)

    @property
    def origin(self):
        return f"http://127.0.0.1:{self.server_port}"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(25)

    def do_GET(self):
        self.serve(head=False)

    def do_HEAD(self):
        self.serve(head=True)

    def serve(self, head):
        if self.headers.get("Host") != f"127.0.0.1:{self.server.server_port}":
            self.send_error(403)
            return
        url = urlsplit(self.path)
        try:
            if url.path in ASSETS:
                asset = WEB / ASSETS[url.path]
                media = mimetypes.guess_type(asset.name)[0] or "application/octet-stream"
                return self.reply(200, asset.read_bytes(), media, head)
            if not url.path.startswith("/api/"):
                self.send_error(404)
                return
            if (
                self.headers.get("Origin") not in (None, self.server.origin)
                or self.headers.get("Sec-Fetch-Site") not in (None, "same-origin", "none")
            ):
                self.send_error(403)
                return
            if url.path == "/api/catalog":
                recordings = discover(self.server.root)
                self.server.recordings = recordings
                payload = [
                    {
                        "id": item.id, "experiment": item.experiment, "source": item.source_id,
                        "runs": [{"id": run.id, "name": run.name, "playable": run.source.video is not None} for run in item.runs],
                    }
                    for item in recordings.values()
                ]
                return self.json(200, {"recordings": payload}, head)
            request = parse_qs(url.query, max_num_fields=6)
            recording = self.server.recordings.get(request.get("recording", [""])[0])
            runs = {run.id: run for run in recording.runs} if recording else {}
            selected = request.get("run", [])
            if not (1 <= len(selected) <= 4 and len(set(selected)) == len(selected)) or any(r not in runs for r in selected):
                return self.json(404, {"error": "Choose a recording and its available runs."}, head)
            if url.path == "/api/metrics":
                data, warnings = recording_metrics(recording, [runs[run] for run in selected])
                return self.json(200, {"data": data, "warnings": warnings}, head)
            if url.path == "/api/overlays" and len(selected) == 1:
                start = int(request.get("start", ["0"])[0])
                overlays = recording_overlays(recording, runs[selected[0]], start)
                return self.json(200, overlays, head)
            if url.path == "/api/video" and len(selected) == 1:
                path = runs[selected[0]].source.video
                if path is None or not path.resolve().is_relative_to(self.server.root):
                    return self.json(404, {"error": "Source video is unavailable."}, head)
                return self.video(path, head)
            self.send_error(404)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            return
        except READ_ERRORS as error:
            self.json(422, {"error": str(error)}, head)

    def headers_for(self, status, size, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "frame-ancestors 'none'")

    def reply(self, status, body, content_type, head):
        self.headers_for(status, len(body), content_type)
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def json(self, status, payload, head):
        self.reply(status, json.dumps(payload, separators=(",", ":")).encode(), "application/json", head)

    def video(self, path, head):
        with path.open("rb") as stream:
            size = stream.seek(0, 2)
            header = self.headers.get("Range")
            try:
                a, b = response_range(header, size)
            except ValueError:
                self.headers_for(416, 0, "text/plain")
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            content_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
            remaining = b - a + 1
            self.headers_for(206 if header else 200, remaining, content_type)
            self.send_header("Accept-Ranges", "bytes")
            if header:
                self.send_header("Content-Range", f"bytes {a}-{b}/{size}")
            self.end_headers()
            if head:
                return
            stream.seek(a)
            while remaining:
                chunk = stream.read(min(remaining, 256 * 1024))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument("--open", action="store_true", help="open the local viewer in your default browser")
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        parser.error("workspace root must be a directory")
    with Server(root, args.port) as server:
        print(f"BaddieVision Rally Diagnostics · {server.origin}", flush=True)
        if args.open:
            open_browser(server.origin)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
