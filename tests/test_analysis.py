import numpy as np
import pytest

from embeddinggemma_lab.analysis import (
    knn_indices,
    neighbor_overlap,
    overlap_at_k,
    rank_change_summary,
    ranks_by_dim,
)
from embeddinggemma_lab.matryoshka import l2_normalize, truncate


def test_knn_excludes_self():
    vectors = l2_normalize(np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]]))
    nn = knn_indices(vectors, 1)
    assert nn[:, 0].tolist() == [1, 0, 3, 2]


def test_overlap_at_k():
    assert overlap_at_k([1, 2, 3], [3, 2, 9]) == pytest.approx(2 / 3)
    assert overlap_at_k([], []) == 1.0


def test_neighbor_overlap_reference_is_one():
    rng = np.random.default_rng(0)
    native = l2_normalize(rng.standard_normal((20, 768)))
    by_dim = {d: truncate(native, d) for d in (128, 256, 512, 768)}
    table = neighbor_overlap(by_dim, k=3).set_index("dim")
    assert table.loc[768, "overlap@3"] == pytest.approx(1.0)
    assert ((table["overlap@3"] >= 0) & (table["overlap@3"] <= 1)).all()


def test_ranks_by_dim_is_one_based():
    ranks = ranks_by_dim({128: np.array([0.1, 0.9, 0.5]), 768: np.array([0.9, 0.1, 0.5])}, "abc")
    assert ranks["rank@128"].tolist() == [3, 1, 2]
    assert ranks["rank@768"].tolist() == [1, 3, 2]


def test_rank_change_summary():
    ref = np.array([0.9, 0.8, 0.7, 0.1])
    summary = rank_change_summary({768: ref, 128: ref[::-1].copy()}, k=2).set_index("dim")
    assert summary.loc[768, "kendall_tau"] == pytest.approx(1.0)
    assert summary.loc[768, "top2_overlap"] == 1.0 and summary.loc[768, "top1_same"]
    assert summary.loc[128, "kendall_tau"] == pytest.approx(-1.0)
    assert summary.loc[128, "top2_overlap"] == 0.0
