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
| **Multimodal Search** | テキスト → 画像、画像 → 画像、テキスト → 動画（タイムスタンプ付き再生） |
| **Embedding Analysis** | UMAP 2D、類似度行列、次元別の近傍安定性・順位変動 |
| **Evaluation** | Recall@K / MRR / nDCG@K / P@K、p50/p95 レイテンシ、保存量、CSV / JSON |
| **Research Demo** | 把持の成功 / 滑り / 落下の検索結果を人手ラベル（時間区間）と照合 |

- 埋め込みは `.cache/embeddings/` に保存され、再起動後も再利用されます。
- スコアはコサイン類似度で、**関連度の指標であり正解確率ではありません。**
"""
)
