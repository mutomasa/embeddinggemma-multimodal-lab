"""Streamlit resources for the research demo page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from embeddinggemma_lab.config import CACHE_DIR
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.research_corpus import ResearchCorpus, ResearchPaths, load_research_corpus
from embeddinggemma_lab.sample_media import ensure_samples
from embeddinggemma_lab.search_service import SearchService
from embeddinggemma_lab.ui.resources import get_search_service

FRAMES_DIR = CACHE_DIR.parent / "frames"


def research_paths(root: str) -> ResearchPaths:
    """Empty ``root`` means the built-in synthetic sample."""
    if not root.strip():
        ensure_samples()
        return ResearchPaths.sample()
    return ResearchPaths.from_root(Path(root).expanduser())


def missing_files(paths: ResearchPaths) -> list[str]:
    return [str(p) for p in (paths.videos, paths.annotations, paths.queries) if not p.exists()]


@st.cache_data(show_spinner="フレームを抽出中…")
def research_corpus(root: str, interval_s: float) -> ResearchCorpus:
    return load_research_corpus(research_paths(root), FrameSampling(interval_s), FRAMES_DIR)


def frame_search_service(corpus: ResearchCorpus) -> SearchService:
    return get_search_service(corpus.frames)
