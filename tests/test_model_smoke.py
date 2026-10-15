"""Integration smoke test against the real model (run with ``pytest -m model``)."""

import numpy as np
import pytest

pytestmark = pytest.mark.model


@pytest.fixture(scope="module")
def encoder():
    from embeddinggemma_lab.model_adapter import GemmaEncoder

    return GemmaEncoder()


def test_cross_lingual_retrieval(encoder):
    docs = encoder.encode_documents(
        ["ロボットが物体を床に落とした。", "東京の天気は晴れです。", "The cat sleeps on the sofa."]
    )
    query = encoder.encode_queries(["robot dropped an object"])
    assert np.isfinite(docs).all()
    assert int((query @ docs.T).argmax()) == 0


def test_outputs_are_native_dim_and_normalized(encoder):
    emb = encoder.encode_queries(["hello"])
    assert emb.shape == (1, 768)
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1.0, rtol=1e-3)
    assert len(encoder.revision) == 40
