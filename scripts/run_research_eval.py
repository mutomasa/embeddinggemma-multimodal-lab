"""Re-runnable research-demo evaluation: event retrieval vs human labels, all dims.

Examples:
    uv run python scripts/run_research_eval.py --granularity frame
    uv run python scripts/run_research_eval.py --granularity video --video-method mean_vector
    uv run python scripts/run_research_eval.py --root datasets/robot_grasps --tolerance 0.5
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from embeddinggemma_lab.config import CACHE_DIR, REPORTS_DIR, SUPPORTED_DIMS, ModelSettings
from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder
from embeddinggemma_lab.environment import collect_environment, peak_gpu_memory_mb
from embeddinggemma_lab.evaluation import EvalConfig, export_report, run_evaluation
from embeddinggemma_lab.model_adapter import GemmaEncoder
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.research_corpus import ResearchPaths, load_research_corpus
from embeddinggemma_lab.research_eval import (
    GRANULARITIES,
    ResearchEvalSpec,
    build_research_eval,
    top1_confusion,
)
from embeddinggemma_lab.sample_media import ensure_samples
from embeddinggemma_lab.search_service import SearchService
from embeddinggemma_lab.video_search import METHODS

FRAMES_DIR = CACHE_DIR.parent / "frames"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=None, help="dataset dir (default: sample)")
    parser.add_argument("--granularity", choices=GRANULARITIES, default="frame")
    parser.add_argument("--video-method", choices=METHODS, default="max_frame")
    parser.add_argument("--frame-interval", type=float, default=0.5)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--dims", type=int, nargs="+", default=list(SUPPORTED_DIMS))
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def resolve_paths(root: Path | None) -> ResearchPaths:
    if root is None:
        ensure_samples()
        return ResearchPaths.sample()
    return ResearchPaths.from_root(root)


def main() -> None:
    args = parse_args()
    spec = ResearchEvalSpec(args.granularity, args.video_method, args.tolerance)
    corpus = load_research_corpus(
        resolve_paths(args.root), FrameSampling(args.frame_interval), FRAMES_DIR
    )
    encoder = GemmaEncoder(ModelSettings(modalities=frozenset({"text", "image", "video"})))
    embedder = CachedEmbedder(encoder, EmbeddingCache(CACHE_DIR))
    queries, target = build_research_eval(corpus, embedder, spec)
    config = EvalConfig(tuple(args.dims), tuple(args.ks), args.warmup, args.repeats)
    info = {**collect_environment(encoder), "eval_set": spec.name, "tolerance_s": args.tolerance}
    report = run_evaluation(embedder, target, queries, config, info)
    report.run_info["peak_gpu_memory_mb"] = peak_gpu_memory_mb()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = export_report(report, args.out or REPORTS_DIR / f"{stamp}-{spec.name}")
    confusion = top1_confusion(corpus, SearchService(embedder, corpus.frames), max(args.dims),
                               args.tolerance)  # fmt: skip
    confusion.to_csv(out / "top1_confusion.csv")
    print(report.summary.to_string(index=False))
    print(f"\nTop-1 frame vs human label (dim {max(args.dims)}):\n{confusion}")
    print(f"\nreport written to {out}")


if __name__ == "__main__":
    main()
