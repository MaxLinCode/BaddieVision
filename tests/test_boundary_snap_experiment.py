from src.temporal_selector.boundary_snap_experiment import (
    OperationalSnapConfig,
    snap_prediction_rows,
)


def _rows(states, starts, ends):
    return [{"source_id": "one", "frame": i, "decoded_inplay": state,
             "inplay_target": int(state), "rally_start_logit": starts[i],
             "rally_end_logit": ends[i],
             "decoder_configuration": {"minimum_duration_seconds": 0.2}}
            for i, state in enumerate(states)]


def test_operational_snap_moves_edges_but_preserves_interval_count():
    rows = _rows([0, 0, 0, 1, 1, 1, 1, 0, 0],
                 [-9, 8, -9, -9, -9, -9, -9, -9, -9],
                 [-9, -9, -9, -9, -9, -9, -9, -9, 8])
    config = OperationalSnapConfig(.5, .5, .5, .5, .5)
    snapped = snap_prediction_rows(rows, fps_by_source={"one": 10}, config=config)
    assert [row["decoded_inplay"] for row in snapped] == [0, 1, 1, 1, 1, 1, 1, 1, 1]


def test_operational_snap_has_stable_ties_and_respects_maximum_shift():
    starts = [-9.0] * 12
    starts[2] = starts[6] = 8.0
    rows = _rows([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0], starts, [-9.0] * 12)
    config = OperationalSnapConfig(.5, .5, .5, .5, .2)
    snapped = snap_prediction_rows(rows, fps_by_source={"one": 10}, config=config)
    # Equal peaks at equal distance: the earlier frame wins deterministically.
    assert next(i for i, row in enumerate(snapped) if row["decoded_inplay"]) == 2
    limited = snap_prediction_rows(
        rows, fps_by_source={"one": 10},
        config=OperationalSnapConfig(.5, .5, .5, .5, .1),
    )
    assert next(i for i, row in enumerate(limited) if row["decoded_inplay"]) == 4


def test_operational_snap_reverts_overlap_deterministically():
    states = [0, 0, 1, 1, 0, 0, 1, 1, 0]
    starts = [-9.0] * 9
    ends = [-9.0] * 9
    ends[5] = starts[4] = 8.0
    rows = _rows(states, starts, ends)
    config = OperationalSnapConfig(.5, .5, .5, .5, .5)
    snapped = snap_prediction_rows(rows, fps_by_source={"one": 10}, config=config)
    assert [row["decoded_inplay"] for row in snapped] == states
