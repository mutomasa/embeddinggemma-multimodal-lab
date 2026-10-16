"""Research-demo dataset: grasp videos, their frames, human labels and queries.

A dataset directory looks like::

    <root>/
    ├── videos/            # *.mp4 (optional metadata.csv: media_id, file, caption, label)
    ├── annotations.csv    # video_id, start_s, end_s, event[, annotator]
    └── queries.jsonl      # {"query_id", "query", "events": {event: grade}}

The built-in sample reuses the synthetic gripper videos in ``datasets/sample/videos``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embeddinggemma_lab.annotations import EventSegment, annotated_video_ids, load_annotations
from embeddinggemma_lab.config import DATASETS_DIR
from embeddinggemma_lab.event_queries import EventQuery, load_event_queries
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.media_ingestion import VIDEO, all_frame_items, load_media_items
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.sample_media import SAMPLE_VIDEOS_DIR

SAMPLE_RESEARCH_DIR = DATASETS_DIR / "sample" / "research"
ANNOTATIONS_FILE = "annotations.csv"
QUERIES_FILE = "queries.jsonl"


@dataclass(frozen=True)
class ResearchPaths:
    videos: Path
    annotations: Path
    queries: Path

    @classmethod
    def from_root(cls, root: Path) -> ResearchPaths:
        return cls(root / "videos", root / ANNOTATIONS_FILE, root / QUERIES_FILE)

    @classmethod
    def sample(cls) -> ResearchPaths:
        root = SAMPLE_RESEARCH_DIR
        return cls(SAMPLE_VIDEOS_DIR, root / ANNOTATIONS_FILE, root / QUERIES_FILE)


@dataclass(frozen=True)
class ResearchCorpus:
    videos: list[Item]
    frames: list[Item]
    segments: list[EventSegment]
    queries: list[EventQuery]

    @property
    def video_ids(self) -> list[str]:
        return [v.item_id for v in self.videos]

    @property
    def events(self) -> list[str]:
        return sorted({s.event for s in self.segments})


def unannotated_videos(corpus: ResearchCorpus) -> list[str]:
    labelled = set(annotated_video_ids(corpus.segments))
    return [vid for vid in corpus.video_ids if vid not in labelled]


def unknown_annotated_videos(corpus: ResearchCorpus) -> list[str]:
    known = set(corpus.video_ids)
    return [vid for vid in annotated_video_ids(corpus.segments) if vid not in known]


def load_research_corpus(
    paths: ResearchPaths, sampling: FrameSampling, frames_root: Path
) -> ResearchCorpus:
    videos = load_media_items(paths.videos, VIDEO)
    return ResearchCorpus(
        videos=videos,
        frames=all_frame_items(videos, sampling, frames_root),
        segments=load_annotations(paths.annotations),
        queries=load_event_queries(paths.queries),
    )
