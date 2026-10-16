import numpy as np
import pytest

from embeddinggemma_lab.embedding_cache import CacheNamespace, EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item


def ns(dim=128, role="document", revision="r1"):
    return CacheNamespace(model_id="m", revision=revision, dim=dim, role=role)


def test_put_get_roundtrip(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put(ns(), "k", np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(cache.get(ns(), "k"), np.arange(4))
    assert cache.get(ns(), "missing") is None


def test_namespace_separates_model_revision_dim_and_role():
    digests = {ns().digest(), ns(dim=256).digest(), ns(role="query").digest()}
    digests.add(ns(revision="r2").digest())
    assert len(digests) == 4


def test_corrupt_entry_is_dropped(tmp_path):
    cache = EmbeddingCache(tmp_path)
    cache.put(ns(), "k", np.ones(3))
    next(tmp_path.glob("*/k.npy")).write_bytes(b"garbage")
    assert cache.get(ns(), "k") is None


@pytest.fixture
def items():
    return [Item("a", "robot drops box"), Item("b", "sunny weather"), Item("c", "pasta recipe")]


def test_embedder_reuses_cache_across_instances(tmp_path, fake_encoder, items):
    first = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    vectors = first.embed_items(items, 256)
    assert first.stats.misses == 3 and fake_encoder.calls == 1

    # "restart": fresh embedder + cache object on the same directory
    second = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    again = second.embed_items(items, 256)
    assert second.stats.hits == 3 and fake_encoder.calls == 1
    np.testing.assert_allclose(vectors, again)


def test_embedder_populates_all_dims_from_one_encode(tmp_path, fake_encoder, items):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    embedder.embed_items(items, 768)
    for dim in (128, 256, 512):
        out = embedder.embed_items(items, dim)
        assert out.shape == (3, dim)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)
    assert fake_encoder.calls == 1


def test_embedder_encodes_only_new_items(tmp_path, fake_encoder, items):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    embedder.embed_items(items[:2], 128)
    embedder.embed_items(items, 128)
    assert (embedder.stats.hits, embedder.stats.misses) == (2, 3)


def test_queries_and_documents_do_not_share_entries(tmp_path, fake_encoder):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    embedder.embed_items([Item("a", "same text")], 128)
    embedder.embed_queries(["same text"], 128)
    assert embedder.stats.misses == 2


def test_empty_input_returns_empty_matrix(tmp_path, fake_encoder):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    assert embedder.embed_items([], 512).shape == (0, 512)
