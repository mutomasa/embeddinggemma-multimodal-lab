"""Streamlit entry point: ``uv run streamlit run app.py``."""

import streamlit as st

from embeddinggemma_lab.config import MODEL_ID, SUPPORTED_DIMS

st.set_page_config(page_title="EmbeddingGemma Multimodal Lab", page_icon="🔎", layout="wide")

st.title("EmbeddingGemma Multimodal Lab")
st.markdown(
    f"""
`{MODEL_ID}` を使った意味検索と、Matryoshka 次元 {", ".join(map(str, SUPPORTED_DIMS))}
ごとの品質・性能差を確認するローカルデモです。左のページから選んでください。

| ページ | 内容 |
|---|---|
| **Search** | テキストクエリによる Top-K 意味検索、次元切り替え、検索時間 |

- 埋め込みは `.cache/embeddings/` に保存され、再起動後も再利用されます。
- スコアはコサイン類似度で、**関連度の指標であり正解確率ではありません。**
"""
)
