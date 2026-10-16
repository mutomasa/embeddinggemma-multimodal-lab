"""Phase 2 feasibility: text -> image and text -> video (3 aggregation methods).

Usage: uv run python scripts/smoke_multimodal.py
"""

from __future__ import annotations

from embeddinggemma_lab.config import CACHE_DIR, ModelSettings
from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.media_ingestion import IMAGE, VIDEO, all_frame_items, load_media_items
from embeddinggemma_lab.model_adapter import GemmaEncoder
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.sample_media import SAMPLE_IMAGES_DIR, SAMPLE_VIDEOS_DIR, ensure_samples
from embeddinggemma_lab.search_service import SearchService
from embeddinggemma_lab.video_search import METHODS, VideoSearchService

IMAGE_QUERIES = ["a red apple", "夜空と月", "a car", "雪だるま", "robot gripper holding a block"]
VIDEO_QUERIES = [
    "the robot drops the block and it falls to the floor",
    "ロボットがブロックを落とす",
    "the block slips in the gripper",
    "the robot successfully carries the block",
]


def main() -> None:
    ensure_samples()
    settings = ModelSettings(modalities=frozenset({"text", "image", "video"}))
    embedder = CachedEmbedder(GemmaEncoder(settings), EmbeddingCache(CACHE_DIR))
    images = SearchService(embedder, load_media_items(SAMPLE_IMAGES_DIR, IMAGE))
    print("== text -> image (dim 768) ==")
    for query in IMAGE_QUERIES:
        top = images.search(query, 768, 3).results
        print(f"  {query!r}: " + ", ".join(f"{r.item.item_id}({r.hit.score:.3f})" for r in top))

    videos = load_media_items(SAMPLE_VIDEOS_DIR, VIDEO)
    frames = all_frame_items(videos, FrameSampling(0.5), CACHE_DIR.parent / "frames")
    service = VideoSearchService(embedder, videos, frames)
    for method in METHODS:
        print(f"\n== text -> video [{method}] (dim 768) ==")
        for query in VIDEO_QUERIES:
            result = service.search(query, 768, 3, method)
            print(f"  {query!r}: " + ", ".join(
                f"{h.video.label}@{h.best_timestamp_s:.1f}s({h.score:.3f})" for h in result.hits
            ))  # fmt: skip


if __name__ == "__main__":
    main()
