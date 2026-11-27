from experiments.temporal_selector.rally_context_grid import break_predictions


def test_break_predictions_requires_a_complete_prediction_in_reviewed_break():
    def rows(first_frame, labels, decoded):
        return [
            {"frame": frame, "inplay_target": label, "decoded_inplay": bool(predicted)}
            for frame, (label, predicted) in enumerate(zip(labels, decoded, strict=True), first_frame)
        ]

    predictions = (
        rows(0, [0, 0, 0, 0], [0, 1, 1, 0])
        + rows(6, [0, 0, 0], [1, 1, 0])
        + rows(10, [0, 0, -100, 0, 0], [0, 1, 1, 1, 0])
        + rows(16, [0, 0, 1, 0], [0, 1, 1, 0])
        + rows(21, [0, 0, 0], [0, 1, 1])
    )
    assert break_predictions(predictions, fps=4) == [
        {"start_frame": 1, "end_frame": 2, "start_seconds": 0.25, "duration_seconds": 0.5}
    ]
