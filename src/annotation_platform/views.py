"""Headless view-model and frame-rendering helpers shared with Dash callbacks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import AnnotationRegistry, SourceRegistration, file_sha256

# Matplotlib's tab20 palette, copied here so rendering and the Dash controls do
# not depend on Matplotlib's import-time configuration or version.
MARKER_COLORS = (
    (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
    (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
    (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
    (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
    (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229),
)

_DRAW_SCALE = 4
_FRAME_MARGIN = 2
_LABEL_PADDING = 5
_LABEL_GAP = 6
_FONT_SIZE = 20


def candidate_display_items(candidates: Any) -> tuple[dict[str, Any], ...]:
    """Return the exact numbered candidates shown to an annotator."""
    items: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = candidate.get("candidate_id")
        center = candidate.get("center")
        if not candidate_id or not isinstance(center, (list, tuple)) or len(center) != 2:
            continue
        try:
            point = (float(center[0]), float(center[1]))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in point):
            continue
        number = len(items) + 1
        items.append({"number": number, "candidate_id": str(candidate_id),
                      "center": point, "color": MARKER_COLORS[(number - 1) % len(MARKER_COLORS)],
                      "candidate": candidate})
    return tuple(items)


@lru_cache(maxsize=2)
def _candidate_font(scale: int = 1) -> ImageFont.FreeTypeFont:
    """Return the same bundled DejaVu mono face for every rendered frame."""
    size = _FONT_SIZE * scale
    try:
        return ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        # Matplotlib is an existing runtime dependency and provides DejaVu even
        # when Pillow cannot resolve the font by family name on the host.
        from matplotlib import font_manager

        return ImageFont.truetype(
            font_manager.findfont("DejaVu Sans Mono", fallback_to_default=False), size
        )


def _finite_candidate_value(candidate: Mapping[str, Any], fields: tuple[str, ...]) -> float | None:
    for field in fields:
        if field not in candidate:
            continue
        try:
            value = float(candidate[field])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _candidate_confidence(candidate: Mapping[str, Any]) -> float | None:
    return _finite_candidate_value(candidate, ("peak_activation", "peak_value", "peak"))


def _candidate_label(item: Mapping[str, Any], verbose: bool) -> str:
    label = f"#{item['number']}"
    if not verbose:
        return label
    candidate = item["candidate"]
    peak = _candidate_confidence(candidate)
    if peak is not None:
        formatted_peak = f"{peak:.2f}"
        if formatted_peak.startswith("0."):
            formatted_peak = formatted_peak[1:]
        elif formatted_peak.startswith("-0."):
            formatted_peak = "-" + formatted_peak[2:]
        label += f" P={formatted_peak}"
    area = _finite_candidate_value(candidate, ("area",))
    if area is not None and area >= 0:
        label += f" A={round(area):d}"
    return label


def _intersection_area(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _label_size(text: str) -> tuple[int, int]:
    font = _candidate_font(_DRAW_SCALE)
    bounds = font.getbbox(text)
    width = math.ceil((bounds[2] - bounds[0]) / _DRAW_SCALE) + 2 * _LABEL_PADDING
    height = math.ceil((bounds[3] - bounds[1]) / _DRAW_SCALE) + 2 * _LABEL_PADDING
    return width, height


def _label_options(
    point: tuple[int, int], radius: int, size: tuple[int, int], frame_size: tuple[int, int]
) -> tuple[tuple[int, int, int, int], ...]:
    """Return clamped boxes in the documented deterministic preference order."""
    x, y = point
    width, height = size
    gap = radius + _LABEL_GAP
    raw = (
        (x + gap, y - gap - height),        # top-right
        (x - gap - width, y - gap - height),  # top-left
        (x + gap, y + gap),                 # bottom-right
        (x - gap - width, y + gap),         # bottom-left
        (x + gap, y - height // 2),         # right
        (x - gap - width, y - height // 2), # left
        (x - width // 2, y - gap - height), # above
        (x - width // 2, y + gap),          # below
    )
    frame_width, frame_height = frame_size
    max_x = max(_FRAME_MARGIN, frame_width - _FRAME_MARGIN - width)
    max_y = max(_FRAME_MARGIN, frame_height - _FRAME_MARGIN - height)
    return tuple(
        (
            min(max(_FRAME_MARGIN, left), max_x),
            min(max(_FRAME_MARGIN, top), max_y),
            min(max(_FRAME_MARGIN, left), max_x) + width,
            min(max(_FRAME_MARGIN, top), max_y) + height,
        )
        for left, top in raw
    )


def _layout_candidate_labels(
    items: tuple[dict[str, Any], ...], frame_size: tuple[int, int], verbose: bool
) -> tuple[dict[str, Any], ...]:
    confidences = [_candidate_confidence(item["candidate"]) for item in items]
    finite = [(value, index) for index, value in enumerate(confidences) if value is not None]
    strongest = max(finite, key=lambda pair: (pair[0], -pair[1]))[1] if finite else None
    markers = []
    for index, item in enumerate(items):
        radius = 6 if index == strongest else 4
        outline = 2 if index == strongest else 1
        x, y = (int(round(value)) for value in item["center"])
        exclusion = radius + outline
        markers.append((x - exclusion, y - exclusion, x + exclusion + 1, y + exclusion + 1))

    placed: list[dict[str, Any]] = []
    boxes: list[tuple[int, int, int, int]] = []
    obstacles = markers
    for index, item in enumerate(items):
        text = _candidate_label(item, verbose)
        radius = 6 if index == strongest else 4
        point = tuple(int(round(value)) for value in item["center"])
        options = _label_options(point, radius, _label_size(text), frame_size)
        penalties = [
            sum(_intersection_area(box, other) for other in (*boxes, *obstacles))
            for box in options
        ]
        option_index = next((i for i, penalty in enumerate(penalties) if penalty == 0), None)
        if option_index is None:
            option_index = min(range(len(options)), key=lambda i: penalties[i])
        box = options[option_index]
        boxes.append(box)
        placed.append({
            **item,
            "point": point,
            "radius": radius,
            "outline": 2 if index == strongest else 1,
            "label": text,
            "label_box": box,
            "offset_index": option_index,
        })
    return tuple(placed)


def _nearest_box_point(point: tuple[int, int], box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y = point
    clamped_x = min(max(x, box[0]), box[2])
    clamped_y = min(max(y, box[1]), box[3])
    if box[0] < x < box[2] and box[1] < y < box[3]:
        return min(
            ((box[0], y), (box[2], y), (x, box[1]), (x, box[3])),
            key=lambda edge: (edge[0] - x) ** 2 + (edge[1] - y) ** 2,
        )
    return clamped_x, clamped_y


def draw_candidates(
    frame: np.ndarray,
    candidates: Any,
    *,
    verbose: bool = False,
    highlighted_candidate_id: str | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with readable, deterministic candidate labels."""
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be an HxWx3 NumPy array")
    if frame.dtype != np.uint8:
        raise ValueError("frame must use uint8 pixels")
    result = frame.copy()
    items = candidate_display_items(candidates)
    if not items:
        return result

    height, width = result.shape[:2]
    placed = _layout_candidate_labels(items, (width, height), bool(verbose))
    scale = _DRAW_SCALE
    overlay = Image.new("RGBA", (width * scale, height * scale), (0, 0, 0, 0))
    drawing = ImageDraw.Draw(overlay)

    def scaled_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(value * scale for value in box)

    # Leaders are beneath every label and marker.
    for item in placed:
        x, y = item["point"]
        edge_x, edge_y = _nearest_box_point(item["point"], item["label_box"])
        dx, dy = edge_x - x, edge_y - y
        distance = math.hypot(dx, dy)
        if distance:
            start_distance = item["radius"] + item["outline"]
            start = (x + dx * start_distance / distance, y + dy * start_distance / distance)
            drawing.line(
                (start[0] * scale, start[1] * scale, edge_x * scale, edge_y * scale),
                fill=(*item["color"], 255), width=scale,
            )

    font = _candidate_font(scale)
    for item in placed:
        box = item["label_box"]
        highlighted = item["candidate_id"] == highlighted_candidate_id
        drawing.rounded_rectangle(
            scaled_box(box),
            radius=4 * scale,
            fill=(0, 0, 0, 179),
            outline=(255, 213, 79, 255) if highlighted else None,
            width=2 * scale if highlighted else 1,
        )
        text_bounds = font.getbbox(item["label"])
        drawing.text(
            (
                (box[0] + _LABEL_PADDING) * scale - text_bounds[0],
                (box[1] + _LABEL_PADDING) * scale - text_bounds[1],
            ),
            item["label"],
            font=font,
            fill=(255, 213, 79, 255) if highlighted else (255, 255, 255, 255),
        )

    for item in placed:
        x, y = item["point"]
        radius = item["radius"]
        outline = item["outline"]
        outer = (x - radius - outline, y - radius - outline,
                 x + radius + outline, y + radius + outline)
        inner = (x - radius, y - radius, x + radius, y + radius)
        drawing.ellipse(scaled_box(outer), fill=(255, 255, 255, 255))
        drawing.ellipse(scaled_box(inner), fill=(*item["color"], 255))

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    overlay = overlay.resize((width, height), resampling)
    base = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)).convert("RGBA")
    annotated = Image.alpha_composite(base, overlay).convert("RGB")
    return cv2.cvtColor(np.asarray(annotated), cv2.COLOR_RGB2BGR).copy()


@dataclass(frozen=True)
class PlaybackView:
    task: str
    source_id: str
    center_frame: int
    center_seconds: float
    clip_start_seconds: float
    clip_end_seconds: float
    candidate_artifact_sha256: str
    source_video_sha256: str
    overlays_by_frame: Mapping[int, tuple[Mapping[str, Any], ...]]

    @property
    def center_candidates(self) -> tuple[Mapping[str, Any], ...]:
        return self.overlays_by_frame.get(self.center_frame, ())


def validate_source_video(source: SourceRegistration, *, force: bool = False) -> None:
    if not source.video_path.is_file():
        raise ValueError(f"registered source video is missing: {source.video_path}")
    stat = source.video_path.stat()
    unchanged_stat = (stat.st_size, stat.st_mtime_ns) == (
        source.video_size,
        source.video_mtime_ns,
    )
    if unchanged_stat and not force:
        return
    current = file_sha256(source.video_path)
    if current != source.video_sha256:
        raise ValueError(
            f"source video changed after registration for {source.source_id}; "
            "restart with a new source registration"
        )


def build_playback_view(
    registry: AnnotationRegistry,
    *,
    task: str,
    source_id: str,
    center_frame: int,
    context_seconds: float = 1.0,
) -> PlaybackView:
    """Preload candidate overlays for every frame in the ±context clip."""
    plugin, source = registry.resolve(task, source_id)
    center_frame = int(center_frame)
    if not 0 <= center_frame < source.frame_count:
        raise ValueError(f"center frame outside source bounds: {center_frame}")
    if context_seconds <= 0:
        raise ValueError("context_seconds must be positive")
    validate_source_video(source)
    eligible = {int(frame) for frame in plugin.eligible_frames(source)}
    if center_frame not in eligible:
        raise ValueError(
            f"task {task!r} has no payload record for {source_id}:{center_frame}"
        )
    run_start = center_frame
    while run_start - 1 in eligible:
        run_start -= 1
    run_end = center_frame
    while run_end + 1 in eligible:
        run_end += 1
    center_seconds = center_frame / source.fps
    source_end = source.frame_count / source.fps
    start_seconds = max(0.0, run_start / source.fps, center_seconds - context_seconds)
    end_seconds = min(
        source_end,
        (run_end + 1) / source.fps,
        center_seconds + context_seconds,
    )
    start_frame = max(0, math.floor(start_seconds * source.fps))
    end_frame = min(run_end, source.frame_count - 1, math.ceil(end_seconds * source.fps))
    overlays = {
        frame: tuple(plugin.overlays(source, frame))
        for frame in range(start_frame, end_frame + 1)
    }
    return PlaybackView(
        task=task,
        source_id=source_id,
        center_frame=center_frame,
        center_seconds=center_seconds,
        clip_start_seconds=start_seconds,
        clip_end_seconds=end_seconds,
        candidate_artifact_sha256=plugin.artifact_sha256(source),
        source_video_sha256=source.video_sha256,
        overlays_by_frame=overlays,
    )


def decode_video_frame(source: SourceRegistration, frame: int) -> np.ndarray:
    """Decode the requested source frame exactly for center-frame selection."""
    validate_source_video(source)
    frame = int(frame)
    if not 0 <= frame < source.frame_count:
        raise ValueError(f"frame outside source bounds: {frame}")
    capture = cv2.VideoCapture(str(source.video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, image = capture.read()
    finally:
        capture.release()
    if not ok or image is None:
        raise ValueError(f"could not decode {source.source_id} frame {frame}")
    return image


def render_raw_frame(
    source: SourceRegistration,
    frame: int,
    *,
    image_format: str = ".jpg",
) -> bytes:
    """Render an unannotated native frame for motion preview."""
    image = decode_video_frame(source, frame)
    ok, encoded = cv2.imencode(image_format, image)
    if not ok:
        raise ValueError(f"could not encode preview frame as {image_format}")
    return encoded.tobytes()


def render_center_frame(
    registry: AnnotationRegistry,
    *,
    task: str,
    source_id: str,
    frame: int,
    image_format: str = ".jpg",
    verbose: bool = False,
    highlighted_candidate_id: str | None = None,
    candidate_view: str = "grouped",
    minimum_threshold: float | None = None,
) -> bytes:
    """Render a labeled center frame for UI and non-UI smoke tests."""
    plugin, source = registry.resolve(task, source_id)
    image = decode_video_frame(source, frame)
    if hasattr(plugin, "display_overlays"):
        overlays = plugin.display_overlays(
            source, frame, view=candidate_view, minimum_threshold=minimum_threshold
        )
    else:
        overlays = plugin.overlays(source, frame)
    image = draw_candidates(
        image,
        overlays,
        verbose=verbose,
        highlighted_candidate_id=highlighted_candidate_id,
    )
    ok, encoded = cv2.imencode(image_format, image)
    if not ok:
        raise ValueError(f"could not encode center frame as {image_format}")
    return encoded.tobytes()
