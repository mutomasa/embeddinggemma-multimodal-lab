import numpy as np
import pytest

from embeddinggemma_lab.annotations import parse_annotations
from embeddinggemma_lab.evaluation import EvalConfig, ItemTarget, VideoTarget, run_evaluation
from embeddinggemma_lab.event_queries import EventQuery
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.research_corpus import (
    ResearchCorpus,
    ResearchPaths,
    unannotated_videos,
    unknown_annotated_videos,
)
from embeddinggemma_lab.research_eval import (
    FRAME,
    VIDEO,
    ResearchEvalSpec,
    build_research_eval,
    frame_eval_queries,
    top1_confusion,
    video_eval_queries,
)
from embeddinggemma_lab.search_service import SearchService

EVENTS = ("carry", "slip", "drop")
SEGMENTS = parse_annotations(
    "video_id,start_s,end_s,event\n"
    "slip,0.0,1.0,carry\nslip,1.0,2.0,slip\n"
    "drop,0.0,1.0,carry\ndrop,1.0,2.0,drop\n"
)


def one_hot(event):
    vec = np.zeros(len(EVENTS), dtype=np.float32)
    vec[EVENTS.index(event)] = 1.0
    return vec


def frame(vid, ts, looks_like):
    return Item(
        f"{vid}@{ts}",
        title=looks_like,
        modality="video_frame",
        metadata={"video_id": vid, "timestamp_s": ts},
    )


FRAMES = [
    frame("slip", 0.5, "carry"),
    frame("slip", 1.5, "slip"),
    frame("drop", 0.5, "carry"),
    frame("drop", 1.5, "slip"),  # the model confuses the drop with a slip
]
QUERIES = [EventQuery("q1", "slip", {"slip": 2}), EventQuery("q2", "drop", {"drop": 2})]
CORPUS = ResearchCorpus(
    videos=[Item("slip", modality="video"), Item("drop", modality="video")],
    frames=FRAMES,
    segments=SEGMENTS,
    queries=QUERIES,
)


class StubEncoder:
    model_id, revision = "stub", "test"

    def encode_queries(self, texts):
        # "drop" sits between drop and slip; no frame looks like a drop, so slip wins.
        return np.stack([one_hot(t) + 0.5 * one_hot("slip") * (t == "drop") for t in texts])


class StubEmbedder:
    """Frames look like their ``title`` event; a query text is its event name."""

    encoder = StubEncoder()

    def embed_items(self, items, dim):
        return np.stack([one_hot(x.title) for x in items])

    def embed_queries(self, texts, dim):
        return self.encoder.encode_queries(texts)


def test_frame_and_video_judgments_come_from_annotations():
    frame_q = frame_eval_queries(CORPUS, tolerance_s=0.0)
    assert frame_q[0].relevant == {"slip@1.5": 2}
    assert video_eval_queries(CORPUS)[1].relevant == {"drop": 2}


@pytest.mark.parametrize(
    ("granularity", "target_type"), [(FRAME, ItemTarget), (VIDEO, VideoTarget)]
)
def test_build_research_eval_picks_target(granularity, target_type):
    queries, target = build_research_eval(CORPUS, StubEmbedder(), ResearchEvalSpec(granularity))
    assert isinstance(target, target_type)
    assert [q.query_id for q in queries] == ["q1", "q2"]


def test_unknown_granularity_fails():
    with pytest.raises(ValueError):
        build_research_eval(CORPUS, StubEmbedder(), ResearchEvalSpec("clip"))


def test_frame_level_evaluation_reflects_confusion():
    embedder = StubEmbedder()
    queries, target = build_research_eval(CORPUS, embedder, ResearchEvalSpec(FRAME, tolerance_s=0))
    config = EvalConfig(dims=(128,), ks=(1,), warmup=0, repeats=1)
    report = run_evaluation(embedder, target, queries, config)
    per_query = report.per_query.set_index("query_id")
    assert per_query.loc["q1", "P@1"] == 1.0
    assert per_query.loc["q2", "P@1"] == 0.0


def test_top1_confusion_shows_drop_retrieved_as_slip():
    service = SearchService(StubEmbedder(), FRAMES)
    table = top1_confusion(CORPUS, service, dim=128, tolerance_s=0.0)
    assert table.loc["slip", "slip"] == 1
    assert table.loc["drop", "slip"] == 1


def test_spec_names_are_distinct():
    assert ResearchEvalSpec(FRAME, tolerance_s=0.5).name == "research-frame-tol0.5s"
    assert ResearchEvalSpec(VIDEO, "mean_vector").name == "research-video-mean_vector"


def test_coverage_checks():
    corpus = ResearchCorpus(
        videos=[Item("slip", modality="video"), Item("extra", modality="video")],
        frames=[],
        segments=SEGMENTS,
        queries=[],
    )
    assert unannotated_videos(corpus) == ["extra"]
    assert unknown_annotated_videos(corpus) == ["drop"]
    assert corpus.events == ["carry", "drop", "slip"]


def test_paths_layout(tmp_path):
    paths = ResearchPaths.from_root(tmp_path)
    assert paths.videos == tmp_path / "videos"
    assert paths.annotations.name == "annotations.csv"
    assert paths.queries.name == "queries.jsonl"
