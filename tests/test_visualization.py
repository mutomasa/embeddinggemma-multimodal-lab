import numpy as np

from embeddinggemma_lab.analysis import ranks_by_dim
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.matryoshka import l2_normalize
from embeddinggemma_lab.visualization import (
    heatmap_figure,
    order_by_label,
    points_frame,
    rank_bump_figure,
    scatter_figure,
    umap_2d,
)

ITEMS = [Item(f"i{i}", f"text {i}", label="b" if i % 2 else "a") for i in range(12)]


def test_umap_shape_and_determinism():
    vectors = l2_normalize(np.random.default_rng(0).standard_normal((12, 32)))
    first = umap_2d(vectors, seed=1)
    assert first.shape == (12, 2)
    np.testing.assert_allclose(first, umap_2d(vectors, seed=1))


def test_umap_tiny_corpus_returns_zeros():
    assert umap_2d(np.ones((2, 8))).shape == (2, 2)


def test_scatter_and_heatmap_build():
    points = points_frame(np.zeros((12, 2)), ITEMS)
    assert set(points.columns) >= {"x", "y", "id", "label", "modality"}
    assert len(scatter_figure(points).data) == 2  # one trace per label
    heat = heatmap_figure(np.eye(12), ITEMS)
    assert heat.data[0].z.shape == (12, 12)


def test_order_by_label_groups_labels():
    order = order_by_label(ITEMS)
    assert [ITEMS[i].label for i in order] == ["a"] * 6 + ["b"] * 6


def test_rank_bump_only_shows_top_items():
    ranks = ranks_by_dim({128: np.arange(12.0), 768: np.arange(12.0)}, [x.item_id for x in ITEMS])
    assert len(rank_bump_figure(ranks, top_n=3).data) == 3
