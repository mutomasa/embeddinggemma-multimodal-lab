"""Tab C — Evaluation: Recall@K / MRR / nDCG@K / P@K, latency and storage per dim."""

from datetime import datetime

import pandas as pd
import streamlit as st

from embeddinggemma_lab.config import REPORTS_DIR, SUPPORTED_DIMS
from embeddinggemma_lab.environment import collect_environment, peak_gpu_memory_mb
from embeddinggemma_lab.eval_sets import TASKS, EvalSetSpec, build_target
from embeddinggemma_lab.evaluation import (
    EvalConfig,
    EvalReport,
    export_report,
    load_qrels,
    parse_qrels,
    report_json,
    run_evaluation,
    unknown_judgments,
)
from embeddinggemma_lab.ui.resources import get_embedder, get_encoder
from embeddinggemma_lab.video_search import METHODS

st.set_page_config(page_title="Evaluation", page_icon="📊", layout="wide")
st.title("📊 Evaluation")
st.caption(
    "同一のクエリ・コーパス・正解ラベルで 4 次元を比較します。"
    " スコアは関連度の指標であり正解確率ではありません。"
)


def sidebar_spec() -> tuple[EvalSetSpec, EvalConfig, bytes | None]:
    with st.sidebar:
        task = st.selectbox("評価セット", TASKS, format_func={
            "text": "テキスト文書（20 クエリ）",
            "image": "テキスト → 画像（14 クエリ）",
            "video": "テキスト → 動画（8 クエリ）",
        }.get)  # fmt: skip
        method, interval = METHODS[0], 0.5
        if task == "video":
            method = st.radio("動画の集約方式", METHODS)
            interval = st.select_slider("フレーム間隔（秒）", [0.25, 0.5, 1.0, 2.0], 0.5)
        dims = st.multiselect("次元", SUPPORTED_DIMS, default=list(SUPPORTED_DIMS))
        ks = st.multiselect("K", [1, 3, 5, 10, 20], default=[1, 5, 10])
        warmup = st.number_input("ウォームアップ回数", 0, 20, 2)
        repeats = st.number_input("測定回数", 1, 50, 5)
        upload = st.file_uploader("独自の正解ラベル（JSONL）", type=["jsonl"])
    config = EvalConfig(tuple(sorted(dims)), tuple(sorted(ks)), int(warmup), int(repeats))
    return EvalSetSpec(task, method, interval), config, upload.getvalue() if upload else None


def run(spec: EvalSetSpec, config: EvalConfig, custom_qrels: bytes | None) -> EvalReport | None:
    encoder, embedder = get_encoder(), get_embedder()
    queries = parse_qrels(custom_qrels.decode()) if custom_qrels else load_qrels(spec.qrels_path)
    with st.spinner("コーパスを準備中…"):
        target = build_target(spec, embedder)
    unknown = unknown_judgments(queries, target.item_ids)
    if unknown:
        st.warning(f"コーパスに存在しない ID が正解ラベルにあります: {unknown[:10]}")
    info = {**collect_environment(encoder), "eval_set": spec.name}
    with st.spinner("評価中（ウォームアップ → 複数回測定）…"):
        report = run_evaluation(embedder, target, queries, config, info)
    report.run_info["peak_gpu_memory_mb"] = peak_gpu_memory_mb()
    return report


def quality_chart(summary: pd.DataFrame) -> None:
    metrics = [c for c in summary.columns if c.split("@")[0] in {"MRR", "Recall", "nDCG", "P"}]
    chosen = st.multiselect(
        "指標", metrics, default=[m for m in metrics if "nDCG" in m or m == "MRR"]
    )
    if chosen:
        st.line_chart(summary.set_index("dim")[chosen].rename(index=str))


def cost_charts(summary: pd.DataFrame) -> None:
    left, right = st.columns(2)
    frame = summary.set_index(summary["dim"].astype(str))
    left.markdown("**検索レイテンシ (ms)**")
    left.bar_chart(frame[["search_p50_ms", "search_p95_ms"]], stack=False)
    right.markdown("**インデックス保存量 (bytes, float32)**")
    right.bar_chart(frame[["index_bytes"]])
    p50, p95 = summary["query_encode_p50_ms"].iloc[0], summary["query_encode_p95_ms"].iloc[0]
    st.caption(
        f"クエリエンコード（キャッシュなし、バッチ 1）: p50 {p50:.1f} ms / p95 {p95:.1f} ms。"
        " 切り詰めはエンコード後に行うため、エンコード時間は次元に依存しません。"
        " インデックス構築時間はキャッシュ済みベクトルの読み込み + 行列化です。"
    )


def downloads(report: EvalReport) -> None:
    cols = st.columns(5)
    files = {
        "summary.csv": report.summary.to_csv(index=False),
        "per_query.csv": report.per_query.to_csv(index=False),
        "failures.csv": report.failures.to_csv(index=False),
        "report.json": report_json(report),
    }
    for col, (name, data) in zip(cols, files.items(), strict=False):
        col.download_button(name, data, file_name=name)
    if cols[4].button("reports/ に保存"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = export_report(report, REPORTS_DIR / f"{stamp}-{report.run_info['eval_set']}")
        st.success(f"保存しました: {out}")


def show(report: EvalReport) -> None:
    st.subheader("次元別サマリー")
    st.dataframe(report.summary, hide_index=True, width="stretch")
    quality_chart(report.summary)
    cost_charts(report.summary)
    downloads(report)
    st.subheader("誤検索ケース（1 位が正解でないクエリ）")
    dim = st.selectbox("次元", report.summary["dim"].tolist(), index=len(report.summary) - 1)
    st.dataframe(report.failures[report.failures["dim"] == dim], hide_index=True, width="stretch")
    with st.expander("クエリ別の指標"):
        st.dataframe(report.per_query, hide_index=True, width="stretch")
    with st.expander("実行環境・設定"):
        st.json(report.run_info)


spec, config, custom = sidebar_spec()
if not config.dims or not config.ks:
    st.info("次元と K を 1 つ以上選んでください。")
    st.stop()
if st.button("評価を実行", type="primary"):
    st.session_state["eval_report"] = run(spec, config, custom)
report = st.session_state.get("eval_report")
if report is not None:
    show(report)
