"""Matryoshka truncation helpers.

The model card requires re-normalising after truncation, otherwise cosine
scores silently degrade. Truncating a 768-d unit vector to its first ``dim``
components and re-normalising is exactly what ``encode(truncate_dim=...,
normalize_embeddings=True)`` does, so a single 768-d encode can serve all
supported dimensions.
"""

from __future__ import annotations

import numpy as np

from embeddinggemma_lab.config import validate_dim


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def truncate(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Keep the leading ``dim`` components and re-normalise."""
    validate_dim(dim)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape[-1] < dim:
        raise ValueError(f"Cannot truncate {vectors.shape[-1]}-d vectors to {dim}")
    return l2_normalize(vectors[..., :dim])


def bytes_per_vector(dim: int, itemsize: int = 4) -> int:
    """Approximate raw storage for one float32 vector (no metadata/index)."""
    return validate_dim(dim) * itemsize
