"""Interactively calibrate a static camera to badminton court coordinates."""

from __future__ import annotations

import argparse
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from court_projection import (
    COURT_CALIBRATION_LINES,
    COURT_LANDMARKS,
    CourtHomography,
    court_line_segments,
    draw_court_overlay,
)

DEFAULT_LANDMARKS = [
    "near_left_doubles",
    "near_right_doubles",
    "far_right_doubles",
    "far_left_doubles",
]
DEFAULT_LINES = [
    "left_doubles_sideline",
    "right_doubles_sideline",
    "near_short_service",
    "far_short_service",
]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_frame(source: Path, frame_index: int):
    if source.suffix.lower() in IMAGE_SUFFIXES:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"could not read image: {source}")
        return frame

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_index} from {source}")
    return frame


def frame_count(source: Path) -> int:
    if source.suffix.lower() in IMAGE_SUFFIXES:
        return 1
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open video: {source}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    if count < 1:
        raise RuntimeError(f"could not determine frame count for {source}")
    return count


def collect_points_opencv(frame, landmark_names: list[str]):
    window = "Court calibration"
    points: list[tuple[float, float]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < len(landmark_names):
            points.append((float(x), float(y)))

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, on_mouse)
    while True:
        display = frame.copy()
        for index, point in enumerate(points):
            x, y = map(int, point)
            cv2.circle(display, (x, y), 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                landmark_names[index],
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        next_name = (
            landmark_names[len(points)] if len(points) < len(landmark_names) else "done"
        )
        cv2.putText(
            display,
            f"Click: {next_name} | U: undo | Enter: save | Esc: cancel",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("u"), 8) and points:
            points.pop()
        elif key in (10, 13) and len(points) == len(landmark_names):
            break
        elif key == 27:
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("calibration cancelled")
    cv2.destroyAllWindows()
    return dict(zip(landmark_names, points))


def _valid_point(point) -> bool:
    return (
        isinstance(point, list)
        and len(point) == 2
        and all(isinstance(value, (int, float)) and math.isfinite(value) for value in point)
    )


def _validate_lines(lines, expected_names: list[str]) -> bool:
    return (
        isinstance(lines, dict)
        and set(lines) == set(expected_names)
        and all(
            isinstance(endpoints, list)
            and len(endpoints) == 2
            and all(_valid_point(point) for point in endpoints)
            for endpoints in lines.values()
        )
    )


def collect_browser_selection(
    source: Path,
    initial_frame_index: int,
    *,
    mode: str,
    landmark_names: list[str],
    line_names: list[str],
    open_browser: bool = False,
):
    """Collect point or line calibration data in a browser."""

    total_frames = frame_count(source)
    initial_frame_index = min(max(initial_frame_index, 0), total_frames - 1)
    result: dict = {}
    config = {
        "mode": mode,
        "landmarkNames": landmark_names,
        "lineNames": line_names,
        "totalFrames": total_frames,
        "initialFrame": initial_frame_index,
    }
    template_path = Path(__file__).with_name("calibration_ui.html")
    page = template_path.read_text(encoding="utf-8").replace(
        "__CALIBRATION_CONFIG__", json.dumps(config)
    )
    page_bytes = page.encode("utf-8")

    class CalibrationHandler(BaseHTTPRequestHandler):
        def _read_json(self):
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length))

        def _send(self, content: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def _send_json(self, data: dict, status: int = 200):
            self._send(
                json.dumps(data).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(page_bytes, "text/html; charset=utf-8")
                return
            if parsed.path != "/frame.jpg":
                self.send_error(404)
                return
            requested = parse_qs(parsed.query).get(
                "index", [str(initial_frame_index)]
            )[0]
            try:
                index = min(max(int(requested), 0), total_frames - 1)
                selected_frame = read_frame(source, index)
                ok, encoded = cv2.imencode(".jpg", selected_frame)
                if not ok:
                    raise RuntimeError("could not encode frame")
            except (RuntimeError, ValueError) as error:
                self.send_error(400, str(error))
                return
            self._send(encoded.tobytes(), "image/jpeg")

        def do_POST(self):
            if self.path == "/preview":
                submitted = self._read_json()
                lines = submitted.get("lines")
                if not _validate_lines(lines, line_names):
                    self._send_json({"error": "invalid court lines"}, 400)
                    return
                try:
                    calibration, _, _ = CourtHomography.from_lines(lines)
                    segments = [
                        calibration.project_to_image(segment).tolist()
                        for segment in court_line_segments()
                    ]
                except ValueError as error:
                    self._send_json({"error": str(error)}, 422)
                    return
                self._send_json({"segments": segments})
                return

            if self.path != "/submit":
                self.send_error(404)
                return
            submitted = self._read_json()
            submitted_index = submitted.get("frame_index")
            if (
                not isinstance(submitted_index, int)
                or not 0 <= submitted_index < total_frames
                or submitted.get("mode") != mode
            ):
                self._send_json({"error": "invalid calibration submission"}, 400)
                return

            if mode == "lines":
                selection = submitted.get("lines")
                if not _validate_lines(selection, line_names):
                    self._send_json({"error": "invalid court lines"}, 400)
                    return
                try:
                    CourtHomography.from_lines(selection)
                except ValueError as error:
                    self._send_json({"error": str(error)}, 422)
                    return
            else:
                selection = submitted.get("points")
                if (
                    not isinstance(selection, dict)
                    or list(selection) != landmark_names
                    or not all(_valid_point(point) for point in selection.values())
                ):
                    self._send_json({"error": "invalid landmark points"}, 400)
                    return

            result["frame_index"] = submitted_index
            result["selection"] = selection
            self._send_json({"ok": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()

        def log_message(self, _format, *_args):
            pass

    server = HTTPServer(("127.0.0.1", 0), CalibrationHandler)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Open {url} in your browser to calibrate the court.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        raise KeyboardInterrupt("calibration cancelled") from None
    finally:
        server.server_close()
    selected_index = result["frame_index"]
    return result["selection"], read_frame(source, selected_index), selected_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit image-to-court homography from draggable lines or points."
    )
    parser.add_argument("source", type=Path, help="image or video from a static camera")
    parser.add_argument("output", type=Path, help="output calibration JSON")
    parser.add_argument("--frame", type=int, default=0, help="initial video frame")
    parser.add_argument(
        "--mode",
        choices=("lines", "points"),
        default="lines",
        help="line mode supports off-screen court corners (default: lines)",
    )
    parser.add_argument(
        "--ui",
        choices=("browser", "opencv"),
        default="browser",
        help="browser is required for line mode",
    )
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="ask the OS to open the calibration URL automatically",
    )
    parser.add_argument(
        "--landmarks",
        nargs="+",
        default=DEFAULT_LANDMARKS,
        choices=sorted(COURT_LANDMARKS),
        help="point-mode landmarks to click (at least four)",
    )
    parser.add_argument(
        "--court-lines",
        nargs="+",
        default=DEFAULT_LINES,
        choices=sorted(COURT_CALIBRATION_LINES),
        help="line-mode guides (at least two sidelines and two cross-court lines)",
    )
    parser.add_argument("--preview", type=Path, help="optional overlay image output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "points" and len(args.landmarks) < 4:
        raise ValueError("point mode requires at least four landmarks")
    if args.mode == "lines":
        axes = [COURT_CALIBRATION_LINES[name][0] for name in args.court_lines]
        if axes.count("x") < 2 or axes.count("y") < 2:
            raise ValueError("line mode requires at least two sidelines and two cross lines")
        if args.ui == "opencv":
            raise ValueError("line mode requires the browser UI")

    if args.ui == "opencv":
        frame = read_frame(args.source, args.frame)
        selection = collect_points_opencv(frame, args.landmarks)
        selected_frame_index = args.frame
    else:
        selection, frame, selected_frame_index = collect_browser_selection(
            args.source,
            args.frame,
            mode=args.mode,
            landmark_names=args.landmarks,
            line_names=args.court_lines,
            open_browser=args.open_browser,
        )

    image_lines = None
    if args.mode == "lines":
        image_lines = selection
        homography, inliers, image_landmarks = CourtHomography.from_lines(image_lines)
        court_landmarks = []
        for name in image_landmarks:
            x_line, y_line = name.split("__", 1)
            court_landmarks.append(
                (
                    COURT_CALIBRATION_LINES[x_line][1],
                    COURT_CALIBRATION_LINES[y_line][1],
                )
            )
    else:
        image_landmarks = selection
        homography, inliers = CourtHomography.from_landmarks(image_landmarks)
        court_landmarks = [COURT_LANDMARKS[name] for name in image_landmarks]

    height, width = frame.shape[:2]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    homography.save(
        args.output,
        image_landmarks,
        (width, height),
        image_lines=image_lines,
    )
    errors = homography.reprojection_errors(
        image_landmarks.values(), court_landmarks
    )
    print(f"Saved {args.output}")
    print(f"Calibration frame: {selected_frame_index}")
    print(
        f"Inliers: {int(inliers.sum())}/{len(inliers)}; "
        f"mean reprojection error: {errors[inliers].mean():.2f}px"
    )
    if args.preview:
        args.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.preview), draw_court_overlay(frame, homography))
        print(f"Saved preview {args.preview}")


if __name__ == "__main__":
    main()
