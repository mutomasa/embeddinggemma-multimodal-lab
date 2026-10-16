import math

import pytest

from embeddinggemma_lab.metrics import (
    dcg,
    evaluate_ranking,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

RANKED = ["a", "x", "b", "y", "c"]
QRELS = {"a": 2, "b": 1, "c": 1, "z": 2}


def test_recall_at_k():
    assert recall_at_k(RANKED, QRELS, 1) == pytest.approx(1 / 4)
    assert recall_at_k(RANKED, QRELS, 3) == pytest.approx(2 / 4)
    assert recall_at_k(RANKED, QRELS, 5) == pytest.approx(3 / 4)
    assert recall_at_k(RANKED, {}, 5) == 0.0


def test_precision_at_k():
    assert precision_at_k(RANKED, QRELS, 2) == pytest.approx(1 / 2)
    assert precision_at_k(RANKED, QRELS, 5) == pytest.approx(3 / 5)
    assert precision_at_k(RANKED, QRELS, 0) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RANKED, QRELS) == 1.0
    assert reciprocal_rank(["x", "y", "b"], QRELS) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "b"], QRELS, k=2) == 0.0
    assert reciprocal_rank(["x"], QRELS) == 0.0


def test_zero_grade_is_not_relevant():
    assert recall_at_k(["a"], {"a": 0, "b": 1}, 1) == 0.0
    assert reciprocal_rank(["a", "b"], {"a": 0, "b": 1}) == 0.5


def test_dcg_known_value():
    # gains 3 (rel 2) and 1 (rel 1): 3/log2(2) + 1/log2(3)
    assert dcg([2, 1]) == pytest.approx(3 + 1 / math.log2(3))
    assert dcg([]) == 0.0


def test_ndcg_hand_computed():
    # top3 = a(2), x(0), b(1); ideal top3 = 2, 2, 1
    actual = 3 / math.log2(2) + 1 / math.log2(4)
    ideal = 3 / math.log2(2) + 3 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RANKED, QRELS, 3) == pytest.approx(actual / ideal)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["z", "a", "b", "c"], QRELS, 4) == pytest.approx(1.0)
    assert ndcg_at_k(RANKED, {}, 5) == 0.0


def test_evaluate_ranking_keys():
    scores = evaluate_ranking(RANKED, QRELS, ks=[1, 5])
    assert set(scores) == {"MRR", "Recall@1", "P@1", "nDCG@1", "Recall@5", "P@5", "nDCG@5"}
    assert scores["nDCG@1"] == pytest.approx(1.0)


def test_percentile():
    assert percentile([1, 2, 3, 4], 50) == pytest.approx(2.5)
    assert math.isnan(percentile([], 95))
