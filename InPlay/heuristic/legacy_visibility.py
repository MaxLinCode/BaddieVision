"""Legacy visibility-based temporal decoder.

The learned InPlay probability path intentionally imports this module.  The
implementation remains in :mod:`segment` for CLI/backwards compatibility,
but heuristic v2 never calls it.
"""

from .segment import Rally, segment_tracks

__all__ = ["Rally", "segment_tracks"]
