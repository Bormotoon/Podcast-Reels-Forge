"""RU: Фактический интервал нарезки клипа с учётом padding.

EN: The interval a clip is actually cut from, padding included. The cut, its
burned subtitles and later subtitle re-syncs must all agree on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def moment_bounds(moment: Mapping[str, Any]) -> tuple[float, float]:
    """A moment's (start, end); (0, 0) when they are not numbers."""

    try:
        start = float(moment.get("start", 0) or 0)
        end = float(moment.get("end", 0) or 0)
    except (TypeError, ValueError):
        return 0.0, 0.0
    return start, end


def padded_intervals(
    bounds: Sequence[tuple[float, float]],
    padding: float,
) -> list[tuple[float, float]]:
    """The interval each clip is actually cut from, padding included.

    Padding widens a clip on both sides, but never into a neighbouring clip:
    towards a neighbour it is capped at half the gap between them, and it is
    dropped on a side that already overlaps one (the selection allows a small
    overlap). Otherwise two adjacent moments would share the same seconds of
    footage — and the same subtitles — in two reels. The subtitles, the
    encode and the QA duration check all use this one interval.
    """

    padding = max(0.0, float(padding))
    result: list[tuple[float, float]] = []
    for position, (start, end) in enumerate(bounds):
        before = after = padding
        if padding > 0:
            for other, (o_start, o_end) in enumerate(bounds):
                if other == position or o_end <= o_start:
                    continue
                if o_end <= start:
                    before = min(before, (start - o_end) / 2.0)
                elif o_start >= end:
                    after = min(after, (o_start - end) / 2.0)
                else:
                    if o_start < start:
                        before = 0.0
                    if o_end > end:
                        after = 0.0
        result.append((max(0.0, start - before), end + after))
    return result
