"""Checking retrieved frames against human event labels."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import pandas as pd

from embeddinggemma_lab.annotations import EventSegment
from embeddinggemma_lab.event_queries import EventGrades, frame_events, grade_of
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.search_service import RankedItem

UNLABELED = "(unlabeled)"


@dataclass(frozen=True)
class LabeledHit:
    rank: int
    score: float
    frame: Item
    human_events: frozenset[str]
    grade: int

    @property
    def matched(self) -> bool:
        return self.grade > 0

    @property
    def video_id(self) -> str:
        return self.frame.metadata["video_id"]

    @property
    def timestamp_s(self) -> float:
        return float(self.frame.metadata["timestamp_s"])


def label_hit(
    ranked: RankedItem, grades: EventGrades, segments: Sequence[EventSegment], tolerance_s: float
) -> LabeledHit:
    events = frame_events(ranked.item, segments, tolerance_s)
    return LabeledHit(
        ranked.hit.rank, ranked.hit.score, ranked.item, events, grade_of(events, grades)
    )


def label_hits(
    results: Iterable[RankedItem],
    grades: EventGrades,
    segments: Sequence[EventSegment],
    tolerance_s: float,
) -> list[LabeledHit]:
    return [label_hit(r, grades, segments, tolerance_s) for r in results]


def agreement_at_k(hits: Sequence[LabeledHit], k: int) -> float:
    """Share of the top-k frames whose human label matches the query."""
    top = hits[:k]
    return sum(h.matched for h in top) / len(top) if top else 0.0


def events_or_unlabeled(events: frozenset[str]) -> list[str]:
    return sorted(events) or [UNLABELED]


def confusion_table(pairs: Iterable[tuple[str, frozenset[str]]]) -> pd.DataFrame:
    """Query event (rows) x human events of the retrieved frame (columns), as counts.

    A frame carrying several events (overlapping segments) counts once per event.
    """
    rows = [(asked, seen) for asked, events in pairs for seen in events_or_unlabeled(events)]
    frame = pd.DataFrame(rows, columns=["query_event", "human_label"])
    if frame.empty:
        return pd.DataFrame()
    return pd.crosstab(frame["query_event"], frame["human_label"])


def hits_frame(hits: Sequence[LabeledHit]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": h.rank,
                "score": round(h.score, 4),
                "video_id": h.video_id,
                "timestamp_s": h.timestamp_s,
                "human_label": ", ".join(events_or_unlabeled(h.human_events)),
                "match": "✅" if h.matched else "❌",
            }
            for h in hits
        ]
    )
