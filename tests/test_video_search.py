import numpy as np
import pytest

from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.video_search import (
    MAX_FRAME,
    MEAN_VECTOR,
    NATIVE,
    VideoSearchService,
    best_frames,
    group_frames,
    video_scores,
)


def frame(vid, ts):
    return Item(
        f"{vid}@{ts}", modality="video_frame", metadata={"video_id": vid, "timestamp_s": ts}
    )


FRAMES = [frame("a", 0.0), frame("a", 1.0), frame("a", 2.0), frame("b", 0.0), frame("b", 1.0)]
# Video "a" has one short, strongly matching moment; "b" matches moderately throughout.
VECTORS = np.array([[0, 1], [1, 0], [0, 1], [0.6, 0.8], [0.6, 0.8]], dtype=np.float32)
QUERY = np.array([1.0, 0.0], dtype=np.float32)


def test_group_frames_preserves_order():
    assert group_frames(FRAMES) == {"a": [0, 1, 2], "b": [3, 4]}


def test_max_frame_keeps_short_event_while_mean_dilutes_it():
    groups = group_frames(FRAMES)
    max_scores = video_scores(MAX_FRAME, QUERY, VECTORS, groups)
    mean_scores = video_scores(MEAN_VECTOR, QUERY, VECTORS, groups)
    assert max_scores["a"] > max_scores["b"]
    assert mean_scores["a"] < mean_scores["b"]


def test_best_frame_positions():
    assert best_frames(VECTORS @ QUERY, group_frames(FRAMES)) == {"a": 1, "b": 3}


def test_native_requires_vectors_and_unknown_method_fails():
    groups = group_frames(FRAMES)
    with pytest.raises(ValueError):
        video_scores(NATIVE, QUERY, VECTORS, groups)
    with pytest.raises(ValueError):
        video_scores("median", QUERY, VECTORS, groups)
    native = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 0.0])}
    assert video_scores(NATIVE, QUERY, VECTORS, groups, native) == {"a": 0.0, "b": 1.0}


class StubEmbedder:
    def embed_items(self, items, dim):
        if items and items[0].modality == "video_frame":
            return VECTORS
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

    def embed_queries(self, texts, dim):
        return QUERY[None, :]


@pytest.mark.parametrize(("method", "top"), [(MAX_FRAME, "a"), (MEAN_VECTOR, "b"), (NATIVE, "b")])
def test_service_ranks_videos_with_best_timestamp(method, top):
    videos = [Item("a", modality="video"), Item("b", modality="video")]
    service = VideoSearchService(StubEmbedder(), videos, FRAMES)
    result = service.search("q", dim=128, k=2, method=method)
    assert result.hits[0].video.item_id == top
    stamps = {h.video.item_id: h.best_timestamp_s for h in result.hits}
    assert stamps == {"a": 1.0, "b": 0.0}
