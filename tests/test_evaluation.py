import json

import pytest

from embeddinggemma_lab.config import DATASETS_DIR
from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.environment import collect_environment
from embeddinggemma_lab.evaluation import (
    EvalConfig,
    EvalQuery,
    ItemTarget,
    export_report,
    failure_rows,
    load_qrels,
    parse_qrels,
    report_json,
    run_evaluation,
    timed_repeats,
    unknown_judgments,
)
from embeddinggemma_lab.ingestion import Item, load_sample_documents

ITEMS = [
    Item("r1", "robot drops the box", label="drop"),
    Item("r2", "robot grasps the cup", label="grasp"),
    Item("w1", "rain and wind tonight", label="weather"),
    Item("c1", "pasta with garlic", label="cooking"),
]
QUERIES = [
    EvalQuery("q1", "robot drops", {"r1": 2}),
    EvalQuery("q2", "rain tonight", {"w1": 2}),
    EvalQuery("q3", "garlic", {"r2": 1}),  # deliberately wrong judgment -> failure
]
CONFIG = EvalConfig(dims=(128, 768), ks=(1, 3), warmup=1, repeats=2)


@pytest.fixture
def report(tmp_path, fake_encoder):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    info = collect_environment(fake_encoder)
    return run_evaluation(embedder, ItemTarget(embedder, ITEMS), QUERIES, CONFIG, info)


def test_parse_qrels_and_validation():
    queries = parse_qrels('{"query_id": 1, "query": "x", "relevant": {"a": 2}}\n\n')
    assert queries == [EvalQuery("1", "x", {"a": 2})]
    with pytest.raises(ValueError):
        parse_qrels('{"query": "x"}')


def test_sample_qrels_reference_existing_documents():
    queries = load_qrels(DATASETS_DIR / "sample" / "eval" / "text.jsonl")
    assert len(queries) == 20
    ids = [x.item_id for x in load_sample_documents()]
    assert unknown_judgments(queries, ids) == []


def test_summary_has_one_row_per_dim_with_metrics_and_costs(report):
    summary = report.summary.set_index("dim")
    assert list(summary.index) == [128, 768]
    for col in ["MRR", "Recall@1", "nDCG@3", "P@3", "search_p50_ms", "query_encode_p95_ms"]:
        assert col in summary.columns
    assert summary.loc[128, "index_bytes"] == 128 * 4 * len(ITEMS)
    assert summary.loc[768, "bytes_per_vector"] == 3072


def test_metrics_reflect_judgments(report):
    per_query = report.per_query.set_index(["dim", "query_id"])
    assert per_query.loc[(768, "q1"), "Recall@1"] == 1.0
    assert per_query.loc[(768, "q2"), "MRR"] == 1.0
    assert set(report.failures["query_id"]) == {"q3"}


def test_rerun_gives_identical_quality(tmp_path, fake_encoder, report):
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path / "again"))
    again = run_evaluation(embedder, ItemTarget(embedder, ITEMS), QUERIES, CONFIG)
    quality = ["MRR", "Recall@1", "nDCG@3"]
    assert again.summary[quality].equals(report.summary[quality])


def test_run_info_records_environment_and_config(report):
    info = report.run_info
    assert info["model_id"] == "fake/bag-of-words" and info["model_revision"] == "test"
    assert info["config"]["dims"] == (128, 768)
    assert {"torch", "transformers", "sentence-transformers"} <= set(info["packages"])


def test_export_writes_csv_and_json(tmp_path, report):
    out = export_report(report, tmp_path / "run")
    assert {p.name for p in out.iterdir()} == {
        "summary.csv",
        "per_query.csv",
        "failures.csv",
        "report.json",
    }
    payload = json.loads(report_json(report))
    assert len(payload["summary"]) == 2 and payload["run"]["n_items"] == 4


def test_failure_rows_only_when_top1_irrelevant():
    query = EvalQuery("q", "x", {"a": 2, "b": 1})
    assert failure_rows(query, ["a", "c"], 128, 3) == []
    row = failure_rows(query, ["c", "b"], 128, 2)[0]
    assert row["expected"] == "a, b" and row["retrieved"] == "c (rel=0), b (rel=1)"


def test_timed_repeats_counts():
    calls = []
    samples = timed_repeats(lambda: calls.append(1), warmup=2, repeats=3)
    assert len(samples) == 3 and len(calls) == 5
