"""Dimension-wise comparisons: neighbourhood stability and rank changes.

All functions take L2-normalised vectors and are pure (no model access).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from embeddinggemma_lab.retriever import top_k_indices


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    return np.asarray(vectors, dtype=np.float32) @ np.asarray(vectors, dtype=np.float32).T


def knn_indices(vectors: np.ndarray, k: int) -> np.ndarray:
    """k nearest neighbours of every row, excluding itself."""
    sims = cosine_matrix(vectors)
    np.fill_diagonal(sims, -np.inf)
    k = min(k, len(vectors) - 1)
    return np.stack([top_k_indices(row, k) for row in sims]) if k > 0 else np.zeros((len(sims), 0))


def overlap_at_k(a: Sequence[int], b: Sequence[int]) -> float:
    """|A ∩ B| / k for two top-k lists of equal length."""
    if len(a) == 0:
        return 1.0
    return len(set(a) & set(b)) / len(a)


def neighbor_overlap(
    vectors_by_dim: Mapping[int, np.ndarray], k: int = 5, reference_dim: int = 768
) -> pd.DataFrame:
    """Mean overlap of each item's k-NN set with the reference dimension's k-NN set."""
    reference = knn_indices(vectors_by_dim[reference_dim], k)
    rows = []
    for dim, vectors in sorted(vectors_by_dim.items()):
        neighbours = knn_indices(vectors, k)
        overlaps = [overlap_at_k(r, n) for r, n in zip(reference, neighbours, strict=True)]
        rows.append({"dim": dim, f"overlap@{k}": float(np.mean(overlaps))})
    return pd.DataFrame(rows)


def ranks_by_dim(scores_by_dim: Mapping[int, np.ndarray], item_ids: Sequence[str]) -> pd.DataFrame:
    """Full ranking (1 = best) of every item for each dimension; rows = items."""
    table = {}
    for dim, scores in sorted(scores_by_dim.items()):
        order = top_k_indices(np.asarray(scores), len(scores))
        ranks = np.empty(len(scores), dtype=int)
        ranks[order] = np.arange(1, len(scores) + 1)
        table[f"rank@{dim}"] = ranks
    return pd.DataFrame(table, index=list(item_ids))


def rank_change_summary(
    scores_by_dim: Mapping[int, np.ndarray], k: int = 10, reference_dim: int = 768
) -> pd.DataFrame:
    """Per dimension: top-k overlap and Kendall's tau versus the reference ranking."""
    reference = np.asarray(scores_by_dim[reference_dim])
    ref_top = top_k_indices(reference, k)
    rows = []
    for dim, scores in sorted(scores_by_dim.items()):
        scores = np.asarray(scores)
        tau = kendalltau(reference, scores).statistic
        rows.append(
            {
                "dim": dim,
                f"top{k}_overlap": overlap_at_k(ref_top, top_k_indices(scores, k)),
                "kendall_tau": float(tau) if np.isfinite(tau) else float("nan"),
                "top1_same": bool(top_k_indices(scores, 1)[0] == ref_top[0]),
            }
        )
    return pd.DataFrame(rows)
