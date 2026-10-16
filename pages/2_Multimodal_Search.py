"""Tab A (multimodal) — text -> image, image -> image, text -> video."""

import streamlit as st

from embeddinggemma_lab.config import SUPPORTED_DIMS
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.media_ingestion import IMAGE, save_upload
from embeddinggemma_lab.ui.media import (
    UPLOAD_IMAGES_DIR,
    image_corpus,
    image_gallery,
    image_search_service,
    upload_media_widget,
    video_search_service,
)
from embeddinggemma_lab.ui.resources import get_encoder
from embeddinggemma_lab.video_search import MAX_FRAME, MEAN_VECTOR, NATIVE

st.set_page_config(page_title="Multimodal Search", page_icon="🖼️", layout="wide")
st.title("🖼️ Multimodal Search")

METHOD_LABELS = {
    MAX_FRAME: "フレーム最大スコア",
    MEAN_VECTOR: "フレーム平均ベクトル",
    NATIVE: "ネイティブ動画入力",
}

with st.sidebar:
    dim = st.segmented_control("埋め込み次元", SUPPORTED_DIMS, default=768) or 768
    k = st.slider("Top-K", 1, 15, 5)
upload_media_widget()
get_encoder()


def timing_caption(encode_ms: float, search_ms: float) -> None:
    st.caption(f"クエリエンコード {encode_ms:.1f} ms · 検索 {search_ms:.2f} ms · dim {dim}")


def text_to_image() -> None:
    query = st.text_input("テキストクエリ", "a red apple", key="t2i")
    if query.strip():
        with st.spinner("画像を埋め込み中（初回のみ）…"):
            result = image_search_service().search(query, dim=dim, k=k)
        timing_caption(result.encode_ms, result.search_ms)
        image_gallery(result.results)


def image_to_image() -> None:
    corpus = image_corpus()
    uploaded = st.file_uploader("クエリ画像をアップロード", type=["png", "jpg", "jpeg", "webp"])
    if uploaded is not None:
        path = save_upload(uploaded.name, uploaded.getvalue(), UPLOAD_IMAGES_DIR / "queries")
        query = Item(item_id=uploaded.name, modality=IMAGE, path=str(path))
    else:
        names = {x.title or x.item_id: x for x in corpus}
        query = names[st.selectbox("またはコーパスから選択", list(names))]
    st.image(query.path, width=200, caption="クエリ")
    with st.spinner("検索中…"):
        result = image_search_service().search_item(query, dim=dim, k=k)
    timing_caption(result.encode_ms, result.search_ms)
    image_gallery(result.results)


def text_to_video() -> None:
    left, right = st.columns([3, 2])
    query = left.text_input("テキストクエリ", "the robot drops the block", key="t2v")
    method = right.radio(
        "集約方式", list(METHOD_LABELS), format_func=METHOD_LABELS.get, horizontal=True
    )
    interval = st.select_slider("フレーム抽出間隔（秒）", [0.25, 0.5, 1.0, 2.0], value=0.5)
    if not query.strip():
        return
    with st.spinner("動画フレームを埋め込み中（初回のみ）…"):
        result = video_search_service(interval).search(query, dim=dim, k=k, method=method)
    timing_caption(result.encode_ms, result.search_ms)
    for hit in result.hits:
        with st.container(border=True):
            info, player = st.columns([2, 3])
            info.markdown(
                f"**#{hit.rank} {hit.video.item_id}** — score {hit.score:.3f}  \n"
                f"label: `{hit.video.label or '-'}`  \n"
                f"最もスコアの高いフレーム: **{hit.best_timestamp_s:.2f}s**"
            )
            if hit.best_frame is not None:
                info.image(hit.best_frame.path, width=220)
            player.video(hit.video.path, start_time=int(hit.best_timestamp_s))
    st.caption(
        "再生はベストフレームのタイムスタンプから開始します（Streamlit の制約で秒単位）。"
        " 平均ベクトル方式では短い「落下の瞬間」が薄まる可能性があります。"
    )


tabs = st.tabs(["テキスト → 画像", "画像 → 画像", "テキスト → 動画"])
with tabs[0]:
    text_to_image()
with tabs[1]:
    image_to_image()
with tabs[2]:
    text_to_video()
