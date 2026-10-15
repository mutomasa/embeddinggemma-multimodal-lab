"""Streamlit helpers for image / video corpora and result galleries."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from embeddinggemma_lab.config import CACHE_DIR, DATASETS_DIR
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.media_ingestion import (
    IMAGE,
    VIDEO,
    all_frame_items,
    load_media_items,
    save_upload,
)
from embeddinggemma_lab.preprocessing import IMAGE_SUFFIXES, VIDEO_SUFFIXES, FrameSampling
from embeddinggemma_lab.sample_media import SAMPLE_IMAGES_DIR, SAMPLE_VIDEOS_DIR, ensure_samples
from embeddinggemma_lab.search_service import RankedItem, SearchService
from embeddinggemma_lab.ui.resources import get_embedder, get_search_service
from embeddinggemma_lab.video_search import VideoSearchService

UPLOAD_IMAGES_DIR = DATASETS_DIR / "uploads" / "images"
UPLOAD_VIDEOS_DIR = DATASETS_DIR / "uploads" / "videos"
FRAMES_DIR = CACHE_DIR.parent / "frames"


@st.cache_resource(show_spinner="サンプル画像・動画を生成中…")
def _ensure_samples_once() -> bool:
    ensure_samples()
    return True


def _load_dir(directory, modality: str) -> list[Item]:
    return load_media_items(directory, modality) if directory.exists() else []


def image_corpus() -> list[Item]:
    _ensure_samples_once()
    return _load_dir(SAMPLE_IMAGES_DIR, IMAGE) + _load_dir(UPLOAD_IMAGES_DIR, IMAGE)


def video_corpus() -> list[Item]:
    _ensure_samples_once()
    return _load_dir(SAMPLE_VIDEOS_DIR, VIDEO) + _load_dir(UPLOAD_VIDEOS_DIR, VIDEO)


def upload_media_widget() -> None:
    """Sidebar uploader that adds files to the local (git-ignored) corpus."""
    suffixes = [s.lstrip(".") for s in IMAGE_SUFFIXES + VIDEO_SUFFIXES]
    files = st.sidebar.file_uploader("画像・動画を追加", type=suffixes, accept_multiple_files=True)
    for file in files or []:
        is_image = f".{file.name.rsplit('.', 1)[-1].lower()}" in IMAGE_SUFFIXES
        save_upload(
            file.name, file.getvalue(), UPLOAD_IMAGES_DIR if is_image else UPLOAD_VIDEOS_DIR
        )


def image_search_service() -> SearchService:
    return get_search_service(image_corpus())


@st.cache_data(show_spinner="フレームを抽出中…")
def _frames(video_paths: tuple[str, ...], interval_s: float) -> list[Item]:
    videos = [x for x in video_corpus() if x.path in video_paths]
    return all_frame_items(videos, FrameSampling(interval_s), FRAMES_DIR)


def video_search_service(interval_s: float) -> VideoSearchService:
    videos = video_corpus()
    frames = _frames(tuple(v.path for v in videos if v.path), interval_s)
    services = st.session_state.setdefault("_video_services", {})
    key = (tuple(v.item_id for v in videos), interval_s)
    if key not in services:
        services[key] = VideoSearchService(get_embedder(), videos, frames)
    return services[key]


def image_gallery(results: Sequence[RankedItem], columns: int = 5) -> None:
    cols = st.columns(columns)
    for i, ranked in enumerate(results):
        with cols[i % columns]:
            st.image(ranked.item.path, width="stretch")
            st.caption(
                f"**#{ranked.hit.rank}** score {ranked.hit.score:.3f}  \n"
                f"{ranked.item.title or ranked.item.item_id} · `{ranked.item.label or '-'}`"
            )
