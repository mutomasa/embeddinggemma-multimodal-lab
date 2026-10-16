"""Human event annotations for videos: labelled time segments.

CSV columns: ``video_id, start_s, end_s, event`` and an optional ``annotator``.
Segments may overlap; a timestamp can therefore carry several events.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

REQUIRED_COLUMNS = ("video_id", "start_s", "end_s", "event")


@dataclass(frozen=True)
class EventSegment:
    video_id: str
    start_s: float
    end_s: float
    event: str
    annotator: str = ""

    def contains(self, timestamp_s: float, tolerance_s: float = 0.0) -> bool:
        return self.start_s - tolerance_s <= timestamp_s <= self.end_s + tolerance_s


def _segment(row: dict[str, str], line_no: int) -> EventSegment:
    start, end = float(row["start_s"]), float(row["end_s"])
    if end < start:
        raise ValueError(f"line {line_no}: end_s ({end}) is before start_s ({start})")
    return EventSegment(
        video_id=row["video_id"].strip(),
        start_s=start,
        end_s=end,
        event=row["event"].strip(),
        annotator=(row.get("annotator") or "").strip(),
    )


def parse_annotations(text: str) -> list[EventSegment]:
    reader = csv.DictReader(StringIO(text))
    missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"annotations CSV is missing columns {sorted(missing)}")
    return [_segment(row, line_no) for line_no, row in enumerate(reader, start=2)]


def load_annotations(path: str | Path) -> list[EventSegment]:
    return parse_annotations(Path(path).read_text(encoding="utf-8-sig"))


def events_at(
    segments: Sequence[EventSegment], video_id: str, timestamp_s: float, tolerance_s: float = 0.0
) -> frozenset[str]:
    """Events a human marked at ``timestamp_s`` (± tolerance) in one video."""
    return frozenset(
        s.event for s in segments if s.video_id == video_id and s.contains(timestamp_s, tolerance_s)
    )


def video_events(segments: Sequence[EventSegment], video_id: str) -> frozenset[str]:
    return frozenset(s.event for s in segments if s.video_id == video_id)


def annotated_video_ids(segments: Sequence[EventSegment]) -> list[str]:
    return sorted({s.video_id for s in segments})
