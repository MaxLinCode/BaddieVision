"""Convenience workflow for registering sources and launching annotation."""

from .registry import SourceCatalog, SourceEntry, SourceStatus
from .identity import ParentVideoProvenance, SourceIdentity

__all__ = [
    "ParentVideoProvenance",
    "SourceCatalog",
    "SourceEntry",
    "SourceIdentity",
    "SourceStatus",
]
