"""Application service: query text -> ranked corpus items with timings."""

from __future__ import annotations

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np

from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.retriever import Hit, NumpyIndex
from embeddinggemma_lab.timing import stopwatch


@dataclass(frozen=True)
class RankedItem:
    hit: Hit
    item: Item


@dataclass(frozen=True)
class SearchResult:
    query: str
    dim: int
    results: list[RankedItem]
    encode_ms: float
    search_ms: float


class SearchService:
    """Builds one index per Matryoshka dim over a fixed corpus."""

    def __init__(self, embedder: CachedEmbedder, items: Sequence[Item]) -> None:
        self.embedder = embedder
        self.items = list(items)
        self._indexes: dict[int, NumpyIndex] = {}
        self.index_build_ms: dict[int, float] = {}

    def index(self, dim: int) -> NumpyIndex:
        if dim not in self._indexes:
            with stopwatch() as watch:
                vectors = self.embedder.embed_items(self.items, dim)
                self._indexes[dim] = NumpyIndex([x.item_id for x in self.items], vectors)
            self.index_build_ms[dim] = watch.elapsed_ms
        return self._indexes[dim]

    def label_mask(self, labels: Collection[str] | None) -> np.ndarray | None:
        if not labels:
            return None
        return np.array([item.label in labels for item in self.items])

    def search_vector(
        self, query: np.ndarray, dim: int, k: int, labels: Collection[str] | None = None
    ) -> list[RankedItem]:
        hits = self.index(dim).search(query, k, mask=self.label_mask(labels))
        return [RankedItem(hit=h, item=self.items[h.index]) for h in hits]

    def search(
        self, query: str, dim: int, k: int = 5, labels: Collection[str] | None = None
    ) -> SearchResult:
        self.index(dim)  # build outside the timed region
        with stopwatch() as encode:
            vector = self.embedder.embed_queries([query], dim)[0]
        with stopwatch() as search:
            results = self.search_vector(vector, dim, k, labels)
        return SearchResult(query, dim, results, encode.elapsed_ms, search.elapsed_ms)
