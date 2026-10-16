"""Event queries and relevance judgments derived from human annotations.

A query names the events it is looking for with a grade, e.g.
``{"query_id": "r03", "query": "the block slips", "events": {"slip": 2}}``.
Judgments are not written by hand per item: they are derived from the
annotated segments, at video level or at frame level.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from embeddinggemma_lab.annotations import EventSegment, events_at, video_events
from embeddinggemma_lab.evaluation import EvalQuery
from embeddinggemma_lab.ingestion import Item

EventGrades = Mapping[str, int]


@dataclass(frozen=True)
class EventQuery:
    query_id: str
    query: str
    events: dict[str, int]

    @property
    def primary_event(self) -> str:
        """The event with the highest grade (ties broken alphabetically)."""
        return min(self.events, key=lambda e: (-self.events[e], e))


def parse_event_queries(text: str) -> list[EventQuery]:
    queries = []
    for line_no, line in enumerate(StringIO(text), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"query_id", "query", "events"} - set(row)
        if missing:
            raise ValueError(f"line {line_no}: missing keys {sorted(missing)}")
        events = {str(k): int(v) for k, v in row["events"].items()}
        queries.append(EventQuery(str(row["query_id"]), str(row["query"]), events))
    return queries


def load_event_queries(path: str | Path) -> list[EventQuery]:
    return parse_event_queries(Path(path).read_text(encoding="utf-8"))


def grade_of(events: frozenset[str], grades: EventGrades) -> int:
    """Highest grade among the events present (0 when none match)."""
    return max((grades.get(e, 0) for e in events), default=0)


def video_qrels(
    grades: EventGrades, video_ids: Sequence[str], segments: Sequence[EventSegment]
) -> dict[str, int]:
    """A video is relevant if a human marked one of the events anywhere in it."""
    judged = {vid: grade_of(video_events(segments, vid), grades) for vid in video_ids}
    return {vid: g for vid, g in judged.items() if g > 0}


def frame_qrels(
    grades: EventGrades,
    frames: Sequence[Item],
    segments: Sequence[EventSegment],
    tolerance_s: float,
) -> dict[str, int]:
    """A frame is relevant if its timestamp lies in a matching segment (± tolerance)."""
    judged = {f.item_id: grade_of(frame_events(f, segments, tolerance_s), grades) for f in frames}
    return {fid: g for fid, g in judged.items() if g > 0}


def frame_events(
    frame: Item, segments: Sequence[EventSegment], tolerance_s: float = 0.0
) -> frozenset[str]:
    return events_at(
        segments, frame.metadata["video_id"], float(frame.metadata["timestamp_s"]), tolerance_s
    )


def to_eval_queries(
    queries: Sequence[EventQuery], qrels_for: Callable[[EventGrades], dict[str, int]]
) -> list[EvalQuery]:
    return [EvalQuery(q.query_id, q.query, qrels_for(q.events)) for q in queries]
