"""Corpus choice shared by the analysis-oriented pages."""

from __future__ import annotations

import streamlit as st

from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.ui.corpus import select_corpus
from embeddinggemma_lab.ui.media import image_corpus, video_frames

TEXT = "テキスト文書"
IMAGES = "画像"
FRAMES = "動画フレーム（0.5 秒間隔）"
TEXT_AND_IMAGES = "テキスト + 画像"


def select_analysis_corpus() -> tuple[str, list[Item]]:
    kind = st.sidebar.radio(
        "分析対象", [TEXT, IMAGES, FRAMES, TEXT_AND_IMAGES], key="analysis_kind"
    )
    if kind == TEXT:
        return kind, select_corpus()
    if kind == IMAGES:
        return kind, image_corpus()
    if kind == FRAMES:
        return kind, video_frames(0.5)
    return kind, select_corpus() + image_corpus()
