"""Sidebar widget for choosing the corpus shared by all pages."""

from __future__ import annotations

import streamlit as st

from embeddinggemma_lab.ingestion import Item, deduplicate, load_sample_documents, load_uploaded

SAMPLE = "サンプル文書（日英）"
UPLOAD = "アップロード（CSV / TXT）"


def _uploaded_items() -> list[Item]:
    files = st.sidebar.file_uploader(
        "CSV（text 列必須, 任意で doc_id / title / category）または TXT（1行1文書）",
        type=["csv", "txt"],
        accept_multiple_files=True,
    )
    items: list[Item] = []
    for file in files or []:
        try:
            items.extend(load_uploaded(file.name, file.getvalue()))
        except (ValueError, UnicodeDecodeError) as exc:
            st.sidebar.error(f"{file.name}: {exc}")
    return items


def select_corpus() -> list[Item]:
    source = st.sidebar.radio("データセット", [SAMPLE, UPLOAD], key="corpus_source")
    items = load_sample_documents() if source == SAMPLE else _uploaded_items()
    items = deduplicate(items)
    st.sidebar.caption(f"{len(items)} 件")
    return items
