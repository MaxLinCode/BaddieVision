import pytest

from src.temporal_selector import (
    build_two_source_crossfit,
    partition_metrics_by_queue,
    validate_out_of_source_predictions,
)


def test_two_fold_predictions_are_strictly_out_of_source_and_stable():
    manifest = build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=71)
    assert manifest == build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=71)
    assert manifest.folds[0].train_sources == ("malaysia",)
    assert manifest.folds[0].predict_sources == ("max-vs-nik",)
    assert set(vars(manifest.folds[0])) == {
        "fold_id",
        "training_source_ids",
        "evaluation_source_ids",
    }
    predictions = [
        {
            "fold_id": "A",
            "training_source_ids": ["malaysia"],
            "evaluation_source_ids": ["max-vs-nik"],
            "prediction_source_id": "max-vs-nik",
            "label_queue": "audit",
        },
        {
            "fold_id": "B",
            "training_source_ids": ["max-vs-nik"],
            "evaluation_source_ids": ["malaysia"],
            "prediction_source_id": "malaysia",
            "label_queue": "adaptive",
        },
    ]
    validate_out_of_source_predictions(manifest, predictions)
    split = partition_metrics_by_queue(manifest, predictions)
    assert len(split["audit"]) == len(split["adaptive"]) == 1
    with pytest.raises(ValueError, match="out-of-source"):
        validate_out_of_source_predictions(
            manifest,
            [
                {
                    "fold_id": "A",
                    "training_source_ids": ["malaysia"],
                    "evaluation_source_ids": ["max-vs-nik"],
                    "prediction_source_id": "malaysia",
                    "label_queue": "audit",
                }
            ],
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("training_source_ids", ["malaysia", "max-vs-nik"]),
        ("evaluation_source_ids", ["malaysia"]),
        ("prediction_source_id", "malaysia"),
    ],
)
def test_audit_metric_compilation_rejects_in_source_or_manifest_drift(field, value):
    manifest = build_two_source_crossfit(("malaysia", "max-vs-nik"), seed=71)
    prediction = {
        "fold_id": "A",
        "training_source_ids": ["malaysia"],
        "evaluation_source_ids": ["max-vs-nik"],
        "prediction_source_id": "max-vs-nik",
        "label_queue": "audit",
    }
    prediction[field] = value
    with pytest.raises(ValueError, match="out-of-source"):
        partition_metrics_by_queue(manifest, [prediction])


def test_crossfit_requires_exactly_two_sources():
    with pytest.raises(ValueError, match="exactly two"):
        build_two_source_crossfit(("only-one",))
