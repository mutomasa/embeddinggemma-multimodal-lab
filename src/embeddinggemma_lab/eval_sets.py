"""Built-in evaluation sets: (queries, target) pairs over the sample data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from embeddinggemma_lab.config import CACHE_DIR, DATASETS_DIR
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.evaluation import EvalQuery, EvalTarget, ItemTarget, VideoTarget, load_qrels
from embeddinggemma_lab.ingestion import load_sample_documents
from embeddinggemma_lab.media_ingestion import IMAGE, VIDEO, all_frame_items, load_media_items
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.sample_media import SAMPLE_IMAGES_DIR, SAMPLE_VIDEOS_DIR, ensure_samples
from embeddinggemma_lab.video_search import MAX_FRAME, VideoSearchService

EVAL_DIR = DATASETS_DIR / "sample" / "eval"
FRAMES_DIR = CACHE_DIR.parent / "frames"
TASKS = ("text", "image", "video")


@dataclass(frozen=True)
class EvalSetSpec:
    task: str = "text"
    video_method: str = MAX_FRAME
    frame_interval_s: float = 0.5

    @property
    def name(self) -> str:
        if self.task != "video":
            return self.task
        return f"video-{self.video_method}-every{self.frame_interval_s:g}s"

    @property
    def qrels_path(self) -> Path:
        return EVAL_DIR / f"{self.task}.jsonl"


def build_target(spec: EvalSetSpec, embedder: CachedEmbedder) -> EvalTarget:
    if spec.task == "text":
        return ItemTarget(embedder, load_sample_documents())
    ensure_samples()
    if spec.task == "image":
        return ItemTarget(embedder, load_media_items(SAMPLE_IMAGES_DIR, IMAGE))
    if spec.task == "video":
        videos = load_media_items(SAMPLE_VIDEOS_DIR, VIDEO)
        frames = all_frame_items(videos, FrameSampling(spec.frame_interval_s), FRAMES_DIR)
        return VideoTarget(VideoSearchService(embedder, videos, frames), spec.video_method)
    raise ValueError(f"unknown task {spec.task!r}; choose from {TASKS}")


def build_eval_set(
    spec: EvalSetSpec, embedder: CachedEmbedder
) -> tuple[list[EvalQuery], EvalTarget]:
    return load_qrels(spec.qrels_path), build_target(spec, embedder)
