"""Tab E — Research Demo: grasp success / slip / drop retrieval checked against human labels."""

from datetime import datetime

import streamlit as st

from embeddinggemma_lab.config import REPORTS_DIR, SUPPORTED_DIMS
from embeddinggemma_lab.environment import collect_environment
from embeddinggemma_lab.evaluation import EvalConfig, EvalReport, export_report, run_evaluation
from embeddinggemma_lab.event_queries import EventQuery
from embeddinggemma_lab.label_matching import LabeledHit, agreement_at_k, hits_frame, label_hits
from embeddinggemma_lab.research_corpus import (
    ResearchCorpus,
    unannotated_videos,
    unknown_annotated_videos,
)
from embeddinggemma_lab.research_eval import (
    FRAME,
    GRANULARITIES,
    ResearchEvalSpec,
    build_research_eval,
    top1_confusion,
)
from embeddinggemma_lab.research_plots import confusion_figure, timeline_figure
from embeddinggemma_lab.ui.research import (
    frame_search_service,
    missing_files,
    research_corpus,
    research_paths,
)
from embeddinggemma_lab.ui.resources import get_embedder, get_encoder
from embeddinggemma_lab.video_search import METHODS

st.set_page_config(page_title="Research Demo", page_icon="🦾", layout="wide")
st.title("🦾 Research Demo — 把持の成功 / 滑り / 落下")
st.caption(
    "検索で得たフレームを人手ラベル（時間区間）と照合します。"
    " スコアは関連度の指標であり、失敗の発生確率ではありません。"
)

with st.sidebar:
    root = st.text_input("データセットのディレクトリ（空欄 = 合成サンプル）", "")
    dim = st.segmented_control("埋め込み次元", SUPPORTED_DIMS, default=768) or 768
    k = st.slider("Top-K フレーム", 1, 20, 5)
    interval = st.select_slider("フレーム間隔（秒）", [0.25, 0.5, 1.0], 0.5)
    tolerance = st.select_slider("時刻の許容誤差（秒）", [0.0, 0.1, 0.25, 0.5], 0.25)


def load_corpus() -> ResearchCorpus:
    missing = missing_files(research_paths(root))
    if missing:
        st.error(f"見つかりません: {missing}")
        st.stop()
    corpus = research_corpus(root, interval)
    if unannotated_videos(corpus):
        st.warning(f"ラベルのない動画（照合では常に不一致）: {unannotated_videos(corpus)}")
    if unknown_annotated_videos(corpus):
        st.warning(f"動画が見つからないラベル: {unknown_annotated_videos(corpus)}")
    return corpus


def pick_query(corpus: ResearchCorpus) -> EventQuery:
    presets = {f"{q.query_id}: {q.query}": q for q in corpus.queries}
    preset = presets[st.selectbox("クエリ（プリセット）", list(presets))]
    left, right = st.columns([3, 2])
    text = left.text_input("クエリ文（編集可）", preset.query)
    events = right.multiselect("期待する人手ラベル", corpus.events, default=list(preset.events))
    grades = {e: preset.events.get(e, 2) for e in events}
    return EventQuery(preset.query_id, text, grades)


def hit_gallery(hits: list[LabeledHit]) -> None:
    cols = st.columns(min(len(hits), 5) or 1)
    for i, hit in enumerate(hits):
        with cols[i % len(cols)]:
            st.image(hit.frame.path, width="stretch")
            mark = "✅" if hit.matched else "❌"
            labels = ", ".join(sorted(hit.human_events)) or "(unlabeled)"
            st.caption(
                f"{mark} **#{hit.rank}** {hit.video_id} @ {hit.timestamp_s:.2f}s  \n{labels}"
            )


def player(corpus: ResearchCorpus, hits: list[LabeledHit]) -> None:
    names = [f"#{h.rank} {h.video_id} @ {h.timestamp_s:.2f}s" for h in hits]
    chosen = hits[names.index(st.selectbox("再生するフレーム", names))]
    video = next(v for v in corpus.videos if v.item_id == chosen.video_id)
    st.video(video.path, start_time=int(chosen.timestamp_s))


def search_section(corpus: ResearchCorpus) -> None:
    st.subheader("1. 検索結果を人手ラベルと照合")
    query = pick_query(corpus)
    if not query.query.strip() or not query.events:
        st.info("クエリ文と期待するラベルを指定してください。")
        return
    result = frame_search_service(corpus).search(query.query, dim=dim, k=k)
    hits = label_hits(result.results, query.events, corpus.segments, tolerance)
    a, b, c = st.columns(3)
    a.metric("一致率 @1", f"{agreement_at_k(hits, 1):.0%}")
    b.metric(f"一致率 @{k}", f"{agreement_at_k(hits, k):.0%}")
    c.metric("検索時間", f"{result.search_ms:.2f} ms")
    hit_gallery(hits)
    st.plotly_chart(timeline_figure(corpus.segments, hits), width="stretch")
    st.caption("帯 = 人手ラベルの時間区間、○ = 一致したフレーム、× = 不一致（#は順位）")
    with st.expander("結果テーブル・動画再生"):
        st.dataframe(hits_frame(hits), hide_index=True, width="stretch")
        player(corpus, hits)


def confusion_section(corpus: ResearchCorpus) -> None:
    st.subheader("2. 全クエリの 1 位フレーム × 人手ラベル")
    table = top1_confusion(corpus, frame_search_service(corpus), dim, tolerance)
    if table.empty:
        st.info("クエリがありません。")
        return
    st.plotly_chart(confusion_figure(table), width="stretch")
    st.caption("対角以外は「滑り」を「落下」と取り違えるなど、意味的に近い失敗モードの混同です。")


def run_eval(corpus: ResearchCorpus, spec: ResearchEvalSpec) -> EvalReport:
    queries, target = build_research_eval(corpus, get_embedder(), spec)
    config = EvalConfig(SUPPORTED_DIMS, (1, 5, 10), warmup=1, repeats=3)
    info = {**collect_environment(get_encoder()), "eval_set": spec.name, "tolerance_s": tolerance}
    return run_evaluation(get_embedder(), target, queries, config, info)


def evaluation_section(corpus: ResearchCorpus) -> None:
    st.subheader("3. 4 次元での定量評価（人手ラベル由来の正解）")
    left, right = st.columns(2)
    granularity = left.radio("粒度", GRANULARITIES, horizontal=True, format_func={
        "frame": "フレーム（時刻が区間内なら正解）",
        "video": "動画（イベントを含めば正解）",
    }.get)  # fmt: skip
    method = right.radio("動画の集約方式", METHODS, horizontal=True, disabled=granularity == FRAME)
    spec = ResearchEvalSpec(granularity, method, tolerance)
    if st.button("評価を実行", type="primary"):
        with st.spinner("評価中…"):
            st.session_state["research_report"] = run_eval(corpus, spec)
    report = st.session_state.get("research_report")
    if report is None:
        return
    st.dataframe(report.summary, hide_index=True, width="stretch")
    st.dataframe(report.failures, hide_index=True, width="stretch")
    if st.button("reports/ に保存"):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out = export_report(report, REPORTS_DIR / f"{stamp}-{report.run_info['eval_set']}")
        st.success(f"保存しました: {out}")


get_encoder()
corpus = load_corpus()
search_section(corpus)
confusion_section(corpus)
evaluation_section(corpus)
