"""Render synchronized video and model-state diagnostics for selected rallies."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


DEFAULT_RALLY_NUMBERS = (23, 24, 26, 27, 29)
GRAPH_HEIGHT = 300


def selection_summary(row: Mapping[str, Any]) -> tuple[str, float, float]:
    """Return the softmax winner, its probability, and margin over runner-up."""
    logits = {
        str(candidate_id): float(value)
        for candidate_id, value in row.get("candidate_logits", {}).items()
    }
    logits["NULL"] = float(row["null_logit"])
    maximum = max(logits.values())
    weights = {
        outcome: math.exp(value - maximum) for outcome, value in logits.items()
    }
    total = sum(weights.values())
    probabilities = sorted(
        ((weight / total, outcome) for outcome, weight in weights.items()),
        reverse=True,
    )
    score, outcome = probabilities[0]
    runner_up = probabilities[1][0] if len(probabilities) > 1 else 0.0
    return outcome, score, score - runner_up


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_diagnostics(
    path: Path, rally_numbers: Sequence[int]
) -> list[dict[str, str]]:
    wanted = set(rally_numbers)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [
        row
        for row in rows
        if int(str(row["rally_id"]).rsplit("-", 1)[-1]) in wanted
    ]
    found = {int(str(row["rally_id"]).rsplit("-", 1)[-1]) for row in selected}
    missing = wanted - found
    if missing:
        raise ValueError(f"diagnostic table does not contain rallies: {sorted(missing)}")
    return sorted(selected, key=lambda row: int(row["start_frame"]))


def _source_videos(config_path: Path) -> dict[str, Path]:
    value = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    return {
        str(source["source_id"]): (base / source["video_path"]).resolve()
        for source in value["dataset"]["sources"]
    }


def _alpha_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)


def _draw_candidate(frame: np.ndarray, row: Mapping[str, Any]) -> None:
    outcome, score, margin = selection_summary(row)
    positions = row.get("candidate_positions", {})
    height, width = frame.shape[:2]
    for candidate_id, position in positions.items():
        x, y = position["peak_position_normalized"]
        center = int(float(x) * width), int(float(y) * height)
        color = (0, 255, 255) if candidate_id == outcome else (130, 130, 130)
        radius = 13 if candidate_id == outcome else 5
        cv2.circle(frame, center, radius, color, 2, cv2.LINE_AA)
    if outcome != "NULL" and outcome in positions:
        bbox = positions[outcome].get("bbox_normalized")
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(
                frame,
                (int(x1 * width), int(y1 * height)),
                (int(x2 * width), int(y2 * height)),
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
    label = (
        f"selection: {outcome}  score={score:.3f}  margin={margin:.3f}  "
        f"candidates={int(row['candidate_count'])}"
    )
    _alpha_rectangle(frame, (12, 12), (min(width - 12, 1160), 62), (0, 0, 0), 0.65)
    cv2.putText(
        frame, label, (25, 47), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
        (0, 255, 255), 2, cv2.LINE_AA,
    )


def _runs(
    rows: Sequence[Mapping[str, Any]], field: str
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    start: int | None = None
    previous = 0
    for row in rows:
        frame = int(row["frame"])
        state = bool(row[field])
        if state and start is None:
            start = frame
        if not state and start is not None:
            output.append((start, previous))
            start = None
        previous = frame
    if start is not None:
        output.append((start, previous))
    return output


def _draw_graph(
    width: int,
    panel_rows: Sequence[Mapping[str, Any]],
    current_frame: int,
    diagnostic: Mapping[str, str],
    threshold: float,
    counterfactual_rows: Mapping[int, Mapping[str, Any]] | None = None,
) -> np.ndarray:
    graph = np.full((GRAPH_HEIGHT, width, 3), 245, dtype=np.uint8)
    left, right, top, bottom = 85, width - 30, 32, 202
    frames = [int(row["frame"]) for row in panel_rows]
    start, end = frames[0], frames[-1]

    def x(frame: int) -> int:
        return left + round((frame - start) * (right - left) / max(1, end - start))

    def y(probability: float) -> int:
        return bottom - round(probability * (bottom - top))

    gt_start, gt_end = int(diagnostic["start_frame"]), int(diagnostic["end_frame"])
    _alpha_rectangle(graph, (x(gt_start), top), (x(gt_end), bottom), (90, 200, 90), 0.22)
    for probability in (0.0, 0.5, 1.0):
        cv2.line(graph, (left, y(probability)), (right, y(probability)), (215, 215, 215), 1)
        cv2.putText(
            graph, f"{probability:.1f}", (34, y(probability) + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA,
        )
    points = np.array(
        [(x(int(row["frame"])), y(float(row["in_play_probability"]))) for row in panel_rows],
        dtype=np.int32,
    )
    cv2.polylines(graph, [points], False, (180, 85, 25), 2, cv2.LINE_AA)
    for field, color in (
        ("rally_start_probability", (40, 150, 40)),
        ("rally_end_probability", (160, 70, 160)),
    ):
        if all(row.get(field) is not None for row in panel_rows):
            values = np.array(
                [(x(int(row["frame"])), y(float(row[field]))) for row in panel_rows],
                dtype=np.int32,
            )
            cv2.polylines(graph, [values], False, color, 1, cv2.LINE_AA)
    if counterfactual_rows is not None:
        masked_points = np.array(
            [
                (x(frame), y(float(counterfactual_rows[frame]["in_play_probability"])))
                for frame in frames
            ],
            dtype=np.int32,
        )
        cv2.polylines(graph, [masked_points], False, (20, 150, 220), 2, cv2.LINE_AA)
    cv2.line(graph, (left, y(threshold)), (right, y(threshold)), (30, 30, 220), 1, cv2.LINE_AA)
    cv2.line(graph, (x(current_frame), top), (x(current_frame), 276), (30, 30, 30), 2)
    for run_start, run_end in _runs(panel_rows, "decoded_inplay"):
        cv2.line(graph, (x(run_start), 241), (x(run_end), 241), (150, 80, 180), 10)
    cv2.line(graph, (x(gt_start), 270), (x(gt_end), 270), (40, 150, 40), 10)
    current = next(row for row in panel_rows if int(row["frame"]) == current_frame)
    masked_text = ""
    if counterfactual_rows is not None:
        masked_text = (
            f"  P(masked)={float(counterfactual_rows[current_frame]['in_play_probability']):.3f}"
        )
    cv2.putText(
        graph,
        f"{diagnostic['rally_id']}  frame {current_frame}  "
        f"P(InPlay)={float(current['in_play_probability']):.3f}  "
        f"{masked_text}"
        f"GT={'IN' if bool(current['inplay_target']) else 'OUT'}  "
        f"decoded={'IN' if bool(current['decoded_inplay']) else 'OUT'}",
        (left, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA,
    )
    cv2.putText(graph, "decoded", (8, 246), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1)
    cv2.putText(graph, "GT", (50, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (80, 80, 80), 1)
    cv2.putText(
        graph, f"threshold {threshold:.2f}", (right - 135, y(threshold) - 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 180), 1, cv2.LINE_AA,
    )
    cv2.putText(
        graph, "normal  masked  start  end", (left, 222),
        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (70, 70, 70), 1, cv2.LINE_AA,
    )
    return graph


def render_rally_videos(
    *,
    config_path: Path,
    predictions_path: Path,
    diagnostics_path: Path,
    output_dir: Path,
    rally_numbers: Sequence[int] = DEFAULT_RALLY_NUMBERS,
    context_frames: int = 30,
    counterfactual_predictions_path: Path | None = None,
) -> list[Path]:
    """Render one source-local diagnostic video per requested rally."""
    rows = _load_jsonl(predictions_path)
    counterfactual = (
        {
            (str(row["prediction_source_id"]), int(row["frame"])): row
            for row in _load_jsonl(counterfactual_predictions_path)
        }
        if counterfactual_predictions_path is not None else None
    )
    diagnostics = _load_diagnostics(diagnostics_path, rally_numbers)
    videos = _source_videos(config_path)
    decoder_configs = {json.dumps(row["decoder_config"], sort_keys=True) for row in rows}
    if len(decoder_configs) != 1:
        raise ValueError("predictions do not share one decoder configuration")
    threshold = float(json.loads(next(iter(decoder_configs)))["threshold"])
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for diagnostic in diagnostics:
        source_id = str(diagnostic["source_id"])
        view_start = int(diagnostic["start_frame"]) - context_frames
        view_end = int(diagnostic["end_frame"]) + context_frames
        panel_rows = [
            row for row in rows
            if str(row["prediction_source_id"]) == source_id
            and view_start <= int(row["frame"]) <= view_end
        ]
        counterfactual_panel = (
            {
                int(row["frame"]): counterfactual[(source_id, int(row["frame"]))]
                for row in panel_rows
            }
            if counterfactual is not None else None
        )
        expected = list(range(view_start, view_end + 1))
        if [int(row["frame"]) for row in panel_rows] != expected:
            raise ValueError(f"predictions are incomplete for {diagnostic['rally_id']}")
        capture = cv2.VideoCapture(str(videos[source_id]))
        if not capture.isOpened():
            raise ValueError(f"cannot open source video: {videos[source_id]}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        render_width = min(source_width, 1280)
        render_height = round(source_height * render_width / source_width)
        output_path = output_dir / f"{diagnostic['rally_id']}-diagnostic.mp4"
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (render_width, render_height + GRAPH_HEIGHT),
        )
        if not writer.isOpened():
            capture.release()
            raise ValueError(f"cannot open diagnostic video writer: {output_path}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, view_start)
        for row in panel_rows:
            ok, frame = capture.read()
            if not ok:
                writer.release()
                capture.release()
                raise ValueError(f"cannot read source frame {row['frame']}")
            if frame.shape[1] != render_width:
                frame = cv2.resize(frame, (render_width, render_height))
            _draw_candidate(frame, row)
            graph = _draw_graph(
                render_width, panel_rows, int(row["frame"]), diagnostic, threshold
                , counterfactual_panel
            )
            writer.write(np.vstack((frame, graph)))
        writer.release()
        capture.release()
        outputs.append(output_path)
    return outputs


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rallies", type=int, nargs="+", default=list(DEFAULT_RALLY_NUMBERS)
    )
    parser.add_argument("--context-frames", type=int, default=30)
    parser.add_argument("--counterfactual-predictions", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputs = render_rally_videos(
        config_path=args.config,
        predictions_path=args.predictions,
        diagnostics_path=args.diagnostics,
        output_dir=args.output_dir,
        rally_numbers=args.rallies,
        context_frames=args.context_frames,
        counterfactual_predictions_path=args.counterfactual_predictions,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
