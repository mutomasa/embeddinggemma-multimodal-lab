import numpy as np
import pytest

from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.retriever import NumpyIndex, top_k_indices
from embeddinggemma_lab.search_service import SearchService


def test_top_k_sorted_descending_with_stable_ties():
    scores = np.array([0.1, 0.9, 0.5, 0.9, 0.3])
    assert top_k_indices(scores, 3).tolist() == [1, 3, 2]
    assert top_k_indices(scores, 10).tolist() == [1, 3, 2, 4, 0]
    assert top_k_indices(scores, 0).tolist() == []


def test_index_search_returns_ranked_hits():
    vectors = np.eye(3, dtype=np.float32)
    index = NumpyIndex(["a", "b", "c"], vectors)
    hits = index.search(np.array([0.1, 0.8, 0.6]), k=2)
    assert [(h.item_id, h.rank) for h in hits] == [("b", 1), ("c", 2)]
    assert hits[0].score == pytest.approx(0.8)


def test_index_mask_filters_candidates():
    index = NumpyIndex(["a", "b", "c"], np.eye(3))
    hits = index.search(np.array([1.0, 0.5, 0.2]), k=3, mask=np.array([False, True, True]))
    assert [h.item_id for h in hits] == ["b", "c"]


def test_index_rejects_dim_mismatch():
    index = NumpyIndex(["a"], np.ones((1, 4)))
    with pytest.raises(ValueError):
        index.search(np.ones(8), 1)
    with pytest.raises(ValueError):
        NumpyIndex(["a", "b"], np.ones((1, 4)))


@pytest.fixture
def service(tmp_path, fake_encoder):
    items = [
        Item("r1", "robot drops the box", label="drop"),
        Item("r2", "robot grasps the cup", label="grasp"),
        Item("w1", "rain and wind tonight", label="weather"),
    ]
    return SearchService(CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path)), items)


@pytest.mark.parametrize("dim", [128, 256, 512, 768])
def test_search_service_finds_lexical_match(service, dim):
    result = service.search("robot drops", dim=dim, k=2)
    assert result.results[0].item.item_id == "r1"
    assert result.dim == dim and result.encode_ms >= 0 and result.search_ms >= 0
    assert service.index_build_ms[dim] >= 0


def test_search_service_label_filter(service):
    result = service.search("robot", dim=128, k=3, labels={"weather"})
    assert [r.item.item_id for r in result.results] == ["w1"]
