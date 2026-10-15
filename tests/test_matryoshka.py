import numpy as np
import pytest

from embeddinggemma_lab.matryoshka import bytes_per_vector, l2_normalize, truncate


def test_truncate_renormalizes_to_unit_length():
    vectors = l2_normalize(np.random.default_rng(0).standard_normal((4, 768)))
    for dim in (128, 256, 512, 768):
        out = truncate(vectors, dim)
        assert out.shape == (4, dim)
        np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_truncate_keeps_leading_components_direction():
    vectors = l2_normalize(np.random.default_rng(1).standard_normal((1, 768)))
    out = truncate(vectors, 128)
    cos = float(out[0] @ l2_normalize(vectors[0, :128]))
    assert cos == pytest.approx(1.0, abs=1e-6)


def test_truncate_rejects_unsupported_dim():
    with pytest.raises(ValueError):
        truncate(np.ones((1, 768)), 100)


def test_truncate_rejects_too_short_input():
    with pytest.raises(ValueError):
        truncate(np.ones((1, 256)), 512)


def test_l2_normalize_handles_zero_vector():
    assert np.isfinite(l2_normalize(np.zeros((1, 8)))).all()


@pytest.mark.parametrize(("dim", "expected"), [(128, 512), (256, 1024), (512, 2048), (768, 3072)])
def test_bytes_per_vector_matches_policy(dim, expected):
    assert bytes_per_vector(dim) == expected
