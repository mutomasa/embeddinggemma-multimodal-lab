"""Video retrieval by aggregating frame embeddings, or via native video embeddings.

Three strategies are compared (development policy 7.4):

* ``max_frame``   - score of the best-matching frame (keeps short events)
* ``mean_vector`` - cosine with the re-normalised mean frame embedding
* ``native``      - the model's own video input (frames sampled by the processor)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.matryoshka import l2_normalize
from embeddinggemma_lab.timing import stopwatch

MAX_FRAME = "max_frame"
MEAN_VECTOR = "mean_vector"
NATIVE = "native"
METHODS = (MAX_FRAME, MEAN_VECTOR, NATIVE)


@dataclass(frozen=True)
class VideoHit:
    video: Item
    score: float
    rank: int
    best_timestamp_s: float
    best_frame: Item | None


@dataclass(frozen=True)
class VideoSearchResult:
    method: str
    dim: int
    hits: list[VideoHit]
    encode_ms: float
    search_ms: float


def group_frames(frames: Sequence[Item]) -> dict[str, list[int]]:
    """video_id -> frame positions, preserving order."""
    groups: dict[str, list[int]] = defaultdict(list)
    for pos, frame in enumerate(frames):
        groups[frame.metadata["video_id"]].append(pos)
    return dict(groups)


def best_frames(frame_scores: np.ndarray, groups: dict[str, list[int]]) -> dict[str, int]:
    """video_id -> position of its highest-scoring frame."""
    return {vid: pos[int(np.argmax(frame_scores[pos]))] for vid, pos in groups.items()}


def mean_vectors(frame_vectors: np.ndarray, groups: dict[str, list[int]]) -> dict[str, np.ndarray]:
    return {vid: l2_normalize(frame_vectors[pos].mean(axis=0)) for vid, pos in groups.items()}


def video_scores(
    method: str,
    query: np.ndarray,
    frame_vectors: np.ndarray,
    groups: dict[str, list[int]],
    native_vectors: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    if method == MAX_FRAME:
        frame_scores = frame_vectors @ query
        return {vid: float(frame_scores[pos].max()) for vid, pos in groups.items()}
    if method == MEAN_VECTOR:
        return {vid: float(vec @ query) for vid, vec in mean_vectors(frame_vectors, groups).items()}
    if method == NATIVE:
        if native_vectors is None:
            raise ValueError("native method needs native video vectors")
        return {vid: float(vec @ query) for vid, vec in native_vectors.items()}
    raise ValueError(f"unknown method {method!r}; choose from {METHODS}")


class VideoSearchService:
    def __init__(
        self, embedder: CachedEmbedder, videos: Sequence[Item], frames: Sequence[Item]
    ) -> None:
        self.embedder = embedder
        self.videos = {v.item_id: v for v in videos}
        self.frames = list(frames)
        self.groups = group_frames(self.frames)
        self._frame_vectors: dict[int, np.ndarray] = {}
        self._native_vectors: dict[int, dict[str, np.ndarray]] = {}

    def frame_vectors(self, dim: int) -> np.ndarray:
        """Frame embeddings at ``dim``, loaded once per dim (like an index)."""
        if dim not in self._frame_vectors:
            self._frame_vectors[dim] = self.embedder.embed_items(self.frames, dim)
        return self._frame_vectors[dim]

    def native_vectors(self, dim: int) -> dict[str, np.ndarray]:
        if dim not in self._native_vectors:
            videos = list(self.videos.values())
            vectors = self.embedder.embed_items(videos, dim)
            self._native_vectors[dim] = {
                v.item_id: vec for v, vec in zip(videos, vectors, strict=True)
            }
        return self._native_vectors[dim]

    def search_vector(self, query: np.ndarray, dim: int, k: int, method: str) -> list[VideoHit]:
        frame_vectors = self.frame_vectors(dim)
        native = self.native_vectors(dim) if method == NATIVE else None
        scores = video_scores(method, query, frame_vectors, self.groups, native)
        best = best_frames(frame_vectors @ query, self.groups)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
        return [self._hit(vid, score, rank, best.get(vid)) for rank, (vid, score) in
                enumerate(ranked, start=1)]  # fmt: skip

    def _hit(self, video_id: str, score: float, rank: int, frame_pos: int | None) -> VideoHit:
        frame = self.frames[frame_pos] if frame_pos is not None else None
        timestamp = float(frame.metadata["timestamp_s"]) if frame else 0.0
        return VideoHit(self.videos[video_id], score, rank, timestamp, frame)

    def search(
        self, query: str, dim: int, k: int = 5, method: str = MAX_FRAME
    ) -> VideoSearchResult:
        self.frame_vectors(dim)  # warm caches outside the timed region
        if method == NATIVE:
            self.native_vectors(dim)
        with stopwatch() as encode:
            vector = self.embedder.embed_queries([query], dim)[0]
        with stopwatch() as search:
            hits = self.search_vector(vector, dim, k, method)
        return VideoSearchResult(method, dim, hits, encode.elapsed_ms, search.elapsed_ms)
