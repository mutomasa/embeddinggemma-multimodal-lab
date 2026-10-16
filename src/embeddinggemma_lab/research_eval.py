"""Evaluating event retrieval against human labels at frame or video level.

Both granularities reuse :func:`evaluation.run_evaluation`; only the
judgments (derived from annotations) and the ranked corpus differ.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.evaluation import EvalQuery, EvalTarget, ItemTarget, VideoTarget
from embeddinggemma_lab.event_queries import frame_qrels, to_eval_queries, video_qrels
from embeddinggemma_lab.label_matching import confusion_table, label_hits
from embeddinggemma_lab.research_corpus import ResearchCorpus
from embeddinggemma_lab.search_service import SearchService
from embeddinggemma_lab.video_search import MAX_FRAME, VideoSearchService

FRAME = "frame"
VIDEO = "video"
GRANULARITIES = (FRAME, VIDEO)


@dataclass(frozen=True)
class ResearchEvalSpec:
    granularity: str = FRAME
    video_method: str = MAX_FRAME
    tolerance_s: float = 0.25

    @property
    def name(self) -> str:
        if self.granularity == FRAME:
            return f"research-frame-tol{self.tolerance_s:g}s"
        return f"research-video-{self.video_method}"


def frame_eval_queries(corpus: ResearchCorpus, tolerance_s: float) -> list[EvalQuery]:
    return to_eval_queries(
        corpus.queries, lambda g: frame_qrels(g, corpus.frames, corpus.segments, tolerance_s)
    )


def video_eval_queries(corpus: ResearchCorpus) -> list[EvalQuery]:
    return to_eval_queries(
        corpus.queries, lambda g: video_qrels(g, corpus.video_ids, corpus.segments)
    )


def build_research_eval(
    corpus: ResearchCorpus, embedder: CachedEmbedder, spec: ResearchEvalSpec
) -> tuple[list[EvalQuery], EvalTarget]:
    if spec.granularity == FRAME:
        return frame_eval_queries(corpus, spec.tolerance_s), ItemTarget(embedder, corpus.frames)
    if spec.granularity == VIDEO:
        service = VideoSearchService(embedder, corpus.videos, corpus.frames)
        return video_eval_queries(corpus), VideoTarget(service, spec.video_method)
    raise ValueError(f"unknown granularity {spec.granularity!r}; choose from {GRANULARITIES}")


def top1_confusion(
    corpus: ResearchCorpus, frame_search: SearchService, dim: int, tolerance_s: float
) -> pd.DataFrame:
    """For every query: its main event vs the human label of the top-1 frame."""
    pairs = []
    for query in corpus.queries:
        top = frame_search.search(query.query, dim=dim, k=1).results
        hits = label_hits(top, query.events, corpus.segments, tolerance_s)
        pairs.append((query.primary_event, hits[0].human_events if hits else frozenset()))
    return confusion_table(pairs)
