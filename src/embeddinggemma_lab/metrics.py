"""Ranking metrics over graded relevance judgments.

``ranked`` is a list of item ids, best first. ``qrels`` maps item id to a
relevance grade; grade > 0 counts as relevant for binary metrics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

Qrels = Mapping[str, int]


def _relevant(qrels: Qrels) -> set[str]:
    return {item for item, grade in qrels.items() if grade > 0}


def recall_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    relevant = _relevant(qrels)
    if not relevant:
        return 0.0
    return len(relevant & set(ranked[:k])) / len(relevant)


def precision_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    if k <= 0:
        return 0.0
    return len(_relevant(qrels) & set(ranked[:k])) / k


def reciprocal_rank(ranked: Sequence[str], qrels: Qrels, k: int | None = None) -> float:
    relevant = _relevant(qrels)
    for position, item in enumerate(ranked[:k] if k else ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    """Discounted cumulative gain with exponential gain ``2^rel - 1``."""
    gains = np.asarray(gains, dtype=float)
    if gains.size == 0:
        return 0.0
    discounts = np.log2(np.arange(2, gains.size + 2))
    return float(np.sum((2.0**gains - 1.0) / discounts))


def ndcg_at_k(ranked: Sequence[str], qrels: Qrels, k: int) -> float:
    ideal = dcg(sorted(qrels.values(), reverse=True)[:k])
    if ideal == 0.0:
        return 0.0
    return dcg([qrels.get(item, 0) for item in ranked[:k]]) / ideal


def evaluate_ranking(ranked: Sequence[str], qrels: Qrels, ks: Sequence[int]) -> dict[str, float]:
    """All metrics for one query; MRR is computed over the full ranking."""
    scores = {"MRR": reciprocal_rank(ranked, qrels)}
    for k in ks:
        scores[f"Recall@{k}"] = recall_at_k(ranked, qrels, k)
        scores[f"P@{k}"] = precision_at_k(ranked, qrels, k)
        scores[f"nDCG@{k}"] = ndcg_at_k(ranked, qrels, k)
    return scores


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if len(values) else float("nan")
