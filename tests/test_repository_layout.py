"""Behavioral checks for the maintained/research layout."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_python(source: str) -> None:
    subprocess.run([sys.executable, "-c", source], cwd=ROOT, check=True)


def test_interval_utilities_do_not_initialize_ml_or_studies() -> None:
    _run_python("""
import sys
from src.temporal_selector import RallyInterval, read_rally_intervals
assert RallyInterval.__name__ == 'RallyInterval'
assert callable(read_rally_intervals)
assert 'torch' not in sys.modules
assert not any(name.startswith('experiments.') for name in sys.modules)
""")


def test_legacy_lstm_definition_is_available_without_training_arrays() -> None:
    _run_python("""
import numpy as np

def forbidden_load(*args, **kwargs):
    raise AssertionError('import must not load training arrays')
np.load = forbidden_load
from archive.prototypes.inplay_train_lstm_model import LSTMClassifier
model = LSTMClassifier(input_size=4, hidden_size=2, num_layers=2)
assert model.lstm.input_size == 4
""")
