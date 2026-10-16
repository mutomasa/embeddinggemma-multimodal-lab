"""Tab A — Semantic Search."""

import streamlit as st

from embeddinggemma_lab.config import SUPPORTED_DIMS
from embeddinggemma_lab.ui.corpus import select_corpus
from embeddinggemma_lab.ui.formatting import SCORE_HELP, results_frame
from embeddinggemma_lab.ui.resources import get_embedder, get_encoder, get_search_service

st.set_page_config(page_title="Search", page_icon="🔎", layout="wide")
st.title("🔎 Semantic Search")

items = select_corpus()
if not items:
    st.info("サイドバーからデータセットを選択またはアップロードしてください。")
    st.stop()

labels = sorted({x.label for x in items if x.label})
with st.sidebar:
    dim = st.segmented_control("埋め込み次元", SUPPORTED_DIMS, default=768)
    k = st.slider("Top-K", 1, min(50, len(items)), min(5, len(items)))
    label_filter = st.multiselect("ラベルで絞り込み", labels) if labels else []

query = st.text_input("クエリ（日本語・英語）", "ロボットが物体を落とす")
if not query.strip() or dim is None:
    st.stop()

encoder = get_encoder()
service = get_search_service(items)
embedder = get_embedder()
embedder.stats.reset()
with st.spinner("検索中…"):
    result = service.search(query, dim=dim, k=k, labels=label_filter)

cols = st.columns(4)
cols[0].metric("クエリエンコード", f"{result.encode_ms:.1f} ms")
cols[1].metric("検索", f"{result.search_ms:.2f} ms")
cols[2].metric("インデックス構築", f"{service.index_build_ms.get(dim, 0):.0f} ms")
cols[3].metric("キャッシュ hit / miss", f"{embedder.stats.hits} / {embedder.stats.misses}")
st.caption(
    "クエリ埋め込みもキャッシュされるため、同じクエリの 2 回目以降は"
    "エンコード時間がほぼ 0 になります。"
    " インデックス構築時間は初回（キャッシュ未作成時）のエンコードを含みます。"
)

st.dataframe(
    results_frame(result.results),
    hide_index=True,
    width="stretch",
    column_config={"score": st.column_config.NumberColumn("score", help=SCORE_HELP)},
)
info = encoder.device_info
st.caption(f"{encoder.model_id}@{encoder.revision[:7]} · {info.name} · {info.dtype_name}")
