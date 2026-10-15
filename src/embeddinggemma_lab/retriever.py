"""Exact cosine-similarity retrieval over L2-normalised vectors (NumPy)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Hit:
    index: int
    item_id: str
    score: float
    rank: int  # 1-based


def top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest scores, sorted descending (ties by index)."""
    k = min(k, scores.shape[-1])
    if k <= 0:
        return np.zeros(0, dtype=int)
    part = np.argpartition(-scores, k - 1)[:k] if k < scores.shape[-1] else np.arange(k)
    return part[np.lexsort((part, -scores[part]))]


class NumpyIndex:
    """Brute-force inner-product index; equals cosine for unit vectors."""

    def __init__(self, item_ids: Sequence[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(item_ids) != len(vectors):
            raise ValueError("vectors must be (N, D) and match item_ids")
        self.item_ids = list(item_ids)
        self.vectors = vectors

    @property
    def dim(self) -> int:
        return self.vectors.shape[1]

    def __len__(self) -> int:
        return len(self.item_ids)

    def scores(self, query: np.ndarray) -> np.ndarray:
        query = np.asarray(query, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.dim:
            raise ValueError(f"query dim {query.shape[0]} != index dim {self.dim}")
        return self.vectors @ query

    def search(self, query: np.ndarray, k: int, mask: np.ndarray | None = None) -> list[Hit]:
        scores = self.scores(query)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
            k = min(k, int(mask.sum()))
        order = top_k_indices(scores, k)
        return [
            Hit(index=int(i), item_id=self.item_ids[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(order, start=1)
        ]
