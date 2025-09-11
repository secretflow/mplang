"""Analysis and visualization utilities for mplang.

This subpackage hosts non-core developer aids: diagram rendering, IR dumps,
profiling helpers (future), etc.
"""

from mplang.analysis.diagram import dump, to_flowchart, to_sequence_diagram

__all__ = [
    "dump",
    "to_flowchart",
    "to_sequence_diagram",
]
