"""Long-lived resources shared across Streamlit pages and reruns."""

from __future__ import annotations

from collections.abc import Sequence

import streamlit as st

from embeddinggemma_lab.config import ModelSettings
from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item, content_hash
from embeddinggemma_lab.model_adapter import GemmaEncoder
from embeddinggemma_lab.search_service import SearchService


@st.cache_resource(show_spinner="EmbeddingGemma 2 を読み込み中…")
def get_encoder() -> GemmaEncoder:
    return GemmaEncoder(ModelSettings(modalities=frozenset({"text", "image", "video"})))


@st.cache_resource
def get_embedder() -> CachedEmbedder:
    return CachedEmbedder(get_encoder(), EmbeddingCache())


def corpus_digest(items: Sequence[Item]) -> str:
    return content_hash(*(f"{x.item_id}\x1f{x.modality}\x1f{x.text}" for x in items))[:16]


def get_search_service(items: Sequence[Item]) -> SearchService:
    """One SearchService (and its per-dim indexes) per corpus, per session."""
    services: dict[str, SearchService] = st.session_state.setdefault("_services", {})
    key = corpus_digest(items)
    if key not in services:
        services[key] = SearchService(get_embedder(), items)
    return services[key]
