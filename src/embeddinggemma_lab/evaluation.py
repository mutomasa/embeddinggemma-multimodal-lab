"""Evaluation engine: quality metrics, latency and storage per Matryoshka dim.

Protocol (see docs/evaluation_protocol.md):

* Quality metrics use the same queries, corpus and judgments for every dim.
* Query-encoding latency is measured on the raw encoder (cache bypassed),
  batch size 1, after warm-up. Truncation happens after encoding, so this
  latency does not depend on the dimension and is measured once.
* Index build time and search latency are measured per dimension.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from embeddinggemma_lab.config import SUPPORTED_DIMS
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.matryoshka import bytes_per_vector
from embeddinggemma_lab.metrics import evaluate_ranking, percentile
from embeddinggemma_lab.retriever import NumpyIndex
from embeddinggemma_lab.timing import stopwatch
from embeddinggemma_lab.video_search import VideoSearchService

# --------------------------------------------------------------------------- data


@dataclass(frozen=True)
class EvalQuery:
    query_id: str
    query: str
    relevant: dict[str, int]


def parse_qrels(text: str) -> list[EvalQuery]:
    """JSONL with ``query_id``, ``query`` and ``relevant`` ({item_id: grade})."""
    queries = []
    for line_no, line in enumerate(StringIO(text), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = {"query_id", "query", "relevant"} - set(row)
        if missing:
            raise ValueError(f"line {line_no}: missing keys {sorted(missing)}")
        relevant = {str(k): int(v) for k, v in row["relevant"].items()}
        queries.append(EvalQuery(str(row["query_id"]), str(row["query"]), relevant))
    return queries


def load_qrels(path: str | Path) -> list[EvalQuery]:
    return parse_qrels(Path(path).read_text(encoding="utf-8"))


def unknown_judgments(queries: Sequence[EvalQuery], item_ids: Sequence[str]) -> list[str]:
    known = set(item_ids)
    return sorted({i for q in queries for i in q.relevant if i not in known})


# --------------------------------------------------------------------------- targets


class EvalTarget(Protocol):
    """Something that ranks corpus ids for a query vector at a given dim."""

    item_ids: list[str]

    def prepare(self, dim: int) -> None: ...

    def rank(self, query: np.ndarray, dim: int, depth: int) -> list[str]: ...


class ItemTarget:
    """Texts or images ranked by exact cosine similarity."""

    def __init__(self, embedder: CachedEmbedder, items: Sequence[Item]) -> None:
        self.embedder = embedder
        self.items = list(items)
        self.item_ids = [x.item_id for x in self.items]
        self._indexes: dict[int, NumpyIndex] = {}

    def prepare(self, dim: int) -> None:
        self._indexes[dim] = NumpyIndex(self.item_ids, self.embedder.embed_items(self.items, dim))

    def rank(self, query: np.ndarray, dim: int, depth: int) -> list[str]:
        return [h.item_id for h in self._indexes[dim].search(query, depth)]


class VideoTarget:
    """Videos ranked by one of the frame-aggregation strategies."""

    def __init__(self, service: VideoSearchService, method: str) -> None:
        self.service = service
        self.method = method
        self.item_ids = list(service.videos)

    def prepare(self, dim: int) -> None:
        self.service.frame_vectors(dim)
        if self.method == "native":
            self.service.native_vectors(dim)

    def rank(self, query: np.ndarray, dim: int, depth: int) -> list[str]:
        hits = self.service.search_vector(query, dim, depth, self.method)
        return [h.video.item_id for h in hits]


# --------------------------------------------------------------------------- config / report


@dataclass(frozen=True)
class EvalConfig:
    dims: tuple[int, ...] = SUPPORTED_DIMS
    ks: tuple[int, ...] = (1, 5, 10)
    warmup: int = 2
    repeats: int = 5
    failure_examples: int = 3


@dataclass
class EvalReport:
    summary: pd.DataFrame
    per_query: pd.DataFrame
    failures: pd.DataFrame
    run_info: dict[str, Any] = field(default_factory=dict)


def timed_repeats(fn: Callable[[], Any], warmup: int, repeats: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        with stopwatch() as watch:
            fn()
        samples.append(watch.elapsed_ms)
    return samples


def measure_query_encoding(
    encoder: Any, queries: Sequence[EvalQuery], warmup: int, repeats: int
) -> list[float]:
    """Per-query latency (batch size 1) on the raw encoder; cache is bypassed."""
    texts = [q.query for q in queries]
    for text in texts[:warmup]:
        encoder.encode_queries([text])
    samples = []
    for _ in range(repeats):
        for text in texts:
            with stopwatch() as watch:
                encoder.encode_queries([text])
            samples.append(watch.elapsed_ms)
    return samples


def failure_rows(
    query: EvalQuery, ranked: list[str], dim: int, examples: int
) -> list[dict[str, Any]]:
    if ranked and query.relevant.get(ranked[0], 0) > 0:
        return []
    expected = sorted(
        (i for i, g in query.relevant.items() if g > 0), key=lambda i: -query.relevant[i]
    )
    retrieved = [f"{i} (rel={query.relevant.get(i, 0)})" for i in ranked[:examples]]
    return [
        {
            "dim": dim,
            "query_id": query.query_id,
            "query": query.query,
            "expected": ", ".join(expected),
            "retrieved": ", ".join(retrieved),
        }
    ]


def evaluate_dim(
    embedder: CachedEmbedder,
    target: EvalTarget,
    queries: Sequence[EvalQuery],
    dim: int,
    config: EvalConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    with stopwatch() as build:
        target.prepare(dim)
    vectors = embedder.embed_queries([q.query for q in queries], dim)
    depth = len(target.item_ids)
    per_query, failures, search_ms = [], [], []
    for query, vector in zip(queries, vectors, strict=True):
        ranked = target.rank(vector, dim, depth)
        search_ms += timed_repeats(lambda v=vector: target.rank(v, dim, depth), 1, config.repeats)
        scores = evaluate_ranking(ranked, query.relevant, config.ks)
        per_query.append({"dim": dim, "query_id": query.query_id, "query": query.query, **scores})
        failures += failure_rows(query, ranked, dim, config.failure_examples)
    metrics = pd.DataFrame(per_query).drop(columns=["dim", "query_id", "query"]).mean()
    summary = {
        "dim": dim,
        **metrics.round(4).to_dict(),
        "search_p50_ms": round(percentile(search_ms, 50), 4),
        "search_p95_ms": round(percentile(search_ms, 95), 4),
        "index_build_ms": round(build.elapsed_ms, 2),
        "bytes_per_vector": bytes_per_vector(dim),
        "index_bytes": bytes_per_vector(dim) * len(target.item_ids),
    }
    return summary, per_query, failures


def run_evaluation(
    embedder: CachedEmbedder,
    target: EvalTarget,
    queries: Sequence[EvalQuery],
    config: EvalConfig | None = None,
    run_info: dict[str, Any] | None = None,
) -> EvalReport:
    config = config or EvalConfig()
    encode_ms = measure_query_encoding(embedder.encoder, queries, config.warmup, config.repeats)
    summaries, per_query, failures = [], [], []
    for dim in config.dims:
        summary, rows, fails = evaluate_dim(embedder, target, queries, dim, config)
        summary["query_encode_p50_ms"] = round(percentile(encode_ms, 50), 3)
        summary["query_encode_p95_ms"] = round(percentile(encode_ms, 95), 3)
        summaries.append(summary)
        per_query += rows
        failures += fails
    info = {
        **(run_info or {}),
        "config": asdict(config),
        "n_queries": len(queries),
        "n_items": len(target.item_ids),
    }
    failure_cols = ["dim", "query_id", "query", "expected", "retrieved"]
    return EvalReport(
        summary=pd.DataFrame(summaries),
        per_query=pd.DataFrame(per_query),
        failures=pd.DataFrame(failures, columns=failure_cols),
        run_info=info,
    )


# --------------------------------------------------------------------------- export


def report_json(report: EvalReport) -> str:
    payload = {
        "run": report.run_info,
        "summary": report.summary.to_dict(orient="records"),
        "per_query": report.per_query.to_dict(orient="records"),
        "failures": report.failures.to_dict(orient="records"),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def export_report(report: EvalReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    report.summary.to_csv(out_dir / "summary.csv", index=False)
    report.per_query.to_csv(out_dir / "per_query.csv", index=False)
    report.failures.to_csv(out_dir / "failures.csv", index=False)
    (out_dir / "report.json").write_text(report_json(report), encoding="utf-8")
    return out_dir
