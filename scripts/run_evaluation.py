"""Re-runnable evaluation of all Matryoshka dims under identical conditions.

Examples:
    uv run python scripts/run_evaluation.py --task text
    uv run python scripts/run_evaluation.py --task video --video-method mean_vector
    EGLAB_DEVICE=cpu uv run python scripts/run_evaluation.py --task image --repeats 10
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from embeddinggemma_lab.config import CACHE_DIR, REPORTS_DIR, SUPPORTED_DIMS, ModelSettings
from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.environment import collect_environment, peak_gpu_memory_mb
from embeddinggemma_lab.eval_sets import TASKS, EvalSetSpec, build_eval_set
from embeddinggemma_lab.evaluation import EvalConfig, export_report, run_evaluation
from embeddinggemma_lab.model_adapter import GemmaEncoder
from embeddinggemma_lab.video_search import METHODS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=TASKS, default="text")
    parser.add_argument("--video-method", choices=METHODS, default="max_frame")
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--dims", type=int, nargs="+", default=list(SUPPORTED_DIMS))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = EvalSetSpec(args.task, args.video_method, args.frame_interval)
    modalities = frozenset({"text"} if args.task == "text" else {"text", "image", "video"})
    encoder = GemmaEncoder(ModelSettings(modalities=modalities, batch_size=args.batch_size))
    embedder = CachedEmbedder(encoder, EmbeddingCache(CACHE_DIR))
    queries, target = build_eval_set(spec, embedder)
    config = EvalConfig(tuple(args.dims), tuple(args.ks), args.warmup, args.repeats)
    info = {**collect_environment(encoder), "eval_set": spec.name}
    report = run_evaluation(embedder, target, queries, config, info)
    report.run_info["peak_gpu_memory_mb"] = peak_gpu_memory_mb()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = export_report(report, args.out or REPORTS_DIR / f"{stamp}-{spec.name}")
    print(report.summary.to_string(index=False))
    print(f"\n{len(report.failures)} top-1 failures; report written to {out}")


if __name__ == "__main__":
    main()
