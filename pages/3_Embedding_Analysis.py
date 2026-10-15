"""Tab B — Embedding Analysis: UMAP, similarity matrix, dimension comparison."""

import numpy as np
import pandas as pd
import streamlit as st

from embeddinggemma_lab.analysis import (
    cosine_matrix,
    neighbor_overlap,
    rank_change_summary,
    ranks_by_dim,
)
from embeddinggemma_lab.config import SUPPORTED_DIMS
from embeddinggemma_lab.ui.analysis_corpus import select_analysis_corpus
from embeddinggemma_lab.ui.resources import corpus_digest, get_embedder, get_encoder
from embeddinggemma_lab.visualization import (
    UMAP_NOTE,
    heatmap_figure,
    points_frame,
    rank_bump_figure,
    scatter_figure,
    umap_2d,
)

st.set_page_config(page_title="Embedding Analysis", page_icon="🧭", layout="wide")
st.title("🧭 Embedding Analysis")

kind, items = select_analysis_corpus()
if len(items) < 4:
    st.info("分析には 4 件以上のアイテムが必要です。")
    st.stop()
get_encoder()
embedder = get_embedder()


def vectors(dim: int) -> np.ndarray:
    with st.spinner(f"{dim} 次元の埋め込みを取得中…"):
        return embedder.embed_items(items, dim)


@st.cache_data(show_spinner="UMAP を計算中…")
def cached_umap(digest: str, dim: int, n_neighbors: int, _vectors: np.ndarray) -> np.ndarray:
    return umap_2d(_vectors, n_neighbors=n_neighbors)


def show_selection(points: pd.DataFrame, event) -> None:
    selected = [p["customdata"][0] for p in event.selection.get("points", [])] if event else []
    if not selected:
        st.caption("点をクリック、または投げ縄で囲むと詳細を表示します。")
        return
    by_id = {x.item_id: x for x in items}
    st.dataframe(points[points["id"].isin(selected)], hide_index=True, width="stretch")
    media = [by_id[i] for i in selected if by_id[i].path][:10]
    if media:
        st.image([x.path for x in media], caption=[x.item_id for x in media], width=150)


def umap_tab() -> None:
    left, mid, right = st.columns(3)
    dim = left.selectbox("次元", SUPPORTED_DIMS, index=3, key="umap_dim")
    color_by = mid.radio("色分け", ["label", "modality"], horizontal=True)
    n_neighbors = right.slider("n_neighbors", 2, 50, 15)
    coords = cached_umap(f"{corpus_digest(items)}", dim, n_neighbors, vectors(dim))
    points = points_frame(coords, items)
    event = st.plotly_chart(
        scatter_figure(points, color_by), on_select="rerun", selection_mode=("points", "lasso")
    )
    st.caption(UMAP_NOTE)
    show_selection(points, event)


def heatmap_tab() -> None:
    dim = st.selectbox("次元", SUPPORTED_DIMS, index=3, key="heat_dim")
    st.plotly_chart(heatmap_figure(cosine_matrix(vectors(dim)), items))
    st.caption(
        "ラベル順に並べたコサイン類似度行列。対角ブロックが明るいほどラベル内の類似度が高い。"
    )


def dims_tab() -> None:
    by_dim = {d: vectors(d) for d in SUPPORTED_DIMS}
    st.subheader("近傍の安定性（768 次元基準）")
    k = st.slider("k", 1, min(20, len(items) - 1), min(5, len(items) - 1))
    overlap = neighbor_overlap(by_dim, k=k)
    st.bar_chart(overlap.set_index("dim"), y=f"overlap@{k}")
    st.caption(f"各アイテムの k 近傍集合が 768 次元とどれだけ一致するか（平均 overlap@{k}）。")

    st.subheader("クエリ別の順位変動")
    query = st.text_input("クエリ", "ロボットが物体を落とす", key="rank_query")
    if not query.strip():
        return
    scores = {d: by_dim[d] @ embedder.embed_queries([query], d)[0] for d in SUPPORTED_DIMS}
    top_n = st.slider("表示する上位件数", 3, 20, 10)
    st.dataframe(rank_change_summary(scores, k=top_n), hide_index=True)
    ranks = ranks_by_dim(scores, [x.item_id for x in items])
    st.plotly_chart(rank_bump_figure(ranks, top_n=top_n))
    st.dataframe(ranks.sort_values("rank@768").head(top_n * 2))


tabs = st.tabs(["UMAP 2D", "類似度行列", "次元別の比較"])
with tabs[0]:
    umap_tab()
with tabs[1]:
    heatmap_tab()
with tabs[2]:
    dims_tab()
