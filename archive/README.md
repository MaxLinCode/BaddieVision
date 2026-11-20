# Historical prototypes

These fixed-input scripts predate the source-based observation workflow and the
independent rally segmenter. They are kept for historical reproduction and are
not interchangeable components of the maintained pipeline. Their data/model
paths and experiment choices remain fixed. Invoke them from the repository root
with `python -m archive.prototypes.<module>` after supplying the original assets.

The [historical Colab setup notebook](notebooks/BaddieVision_Colab.ipynb)
reproduces the original package-based workflows.

Obsolete command wrappers have been removed. Import the historical LSTM model
from `archive.prototypes.inplay_train_lstm_model`; this does not load training
arrays, and best-model weights are copied when selected.

Use `python -m src.workflow extract --help` for maintained extraction and
`python -m src.temporal_selector.rally_experiment --help` for rally experiments.
The whole-frame pose implementation lives in `src/pose_extraction.py`.
