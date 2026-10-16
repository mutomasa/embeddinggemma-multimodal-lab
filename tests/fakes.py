"""Deterministic, model-free encoder used by unit tests."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np

from embeddinggemma_lab.matryoshka import l2_normalize


def _token_vector(token: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:8], "little")
    return np.random.default_rng(seed).standard_normal(dim).astype(np.float32)


def bag_of_words(text: str, dim: int = 768) -> np.ndarray:
    tokens = text.lower().split() or [""]
    return l2_normalize(np.sum([_token_vector(t, dim) for t in tokens], axis=0))


class FakeEncoder:
    """Bag-of-words hashing encoder: shared words -> higher cosine similarity."""

    model_id = "fake/bag-of-words"
    revision = "test"

    def __init__(self) -> None:
        self.calls = 0

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        return np.stack([bag_of_words(t) for t in texts]) if texts else np.zeros((0, 768))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_documents(self, texts, titles=None) -> np.ndarray:  # noqa: ANN001
        return self._encode(texts)

    def encode_media(self, items: Sequence[dict[str, Any]]) -> np.ndarray:
        return self._encode([" ".join(str(v) for v in item.values()) for item in items])
