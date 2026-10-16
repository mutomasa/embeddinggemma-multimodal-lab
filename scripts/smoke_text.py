"""Phase 0 feasibility check: encode 5 documents and run JA/EN queries.

Usage: uv run python scripts/smoke_text.py
"""

from __future__ import annotations

import time

import numpy as np

from embeddinggemma_lab.config import SUPPORTED_DIMS
from embeddinggemma_lab.matryoshka import truncate
from embeddinggemma_lab.model_adapter import GemmaEncoder

DOCUMENTS = [
    "ロボットアームが把持していた物体を床に落とした。",
    "The robot gripper successfully picked up the red cube.",
    "北極のオーロラは太陽から来る荷電粒子によって生じる。",
    "The object slipped out of the gripper during lifting.",
    "東京の明日の天気は晴れのち曇りの予報です。",
]
QUERIES = ["ロボットが物体を落とす", "What causes the northern lights?", "把持に成功した"]


def check_truncation_equivalence(encoder: GemmaEncoder) -> None:
    """Our slice+renorm must match the model's truncate_dim API."""
    native = encoder.encode_queries(QUERIES)
    for dim in SUPPORTED_DIMS:
        ref = encoder.model.encode(
            QUERIES, prompt_name="SearchQuery", truncate_dim=dim, normalize_embeddings=True
        )
        diff = float(np.abs(truncate(native, dim) - ref).max())
        print(f"  dim={dim:<4} max|ours - truncate_dim| = {diff:.2e}")


def main() -> None:
    t0 = time.perf_counter()
    encoder = GemmaEncoder()
    info = encoder.device_info
    print(
        f"loaded {encoder.model_id}@{encoder.revision} on {info.name} ({info.dtype_name}) "
        f"in {time.perf_counter() - t0:.1f}s"
    )
    docs = encoder.encode_documents(DOCUMENTS)
    queries = encoder.encode_queries(QUERIES)
    assert np.isfinite(docs).all() and np.isfinite(queries).all(), "NaN embeddings"
    for dim in (768, 128):
        scores = truncate(queries, dim) @ truncate(docs, dim).T
        print(f"\n[dim={dim}]")
        for q, row in zip(QUERIES, scores, strict=True):
            best = int(row.argmax())
            print(f"  {q!r} -> #{best} {DOCUMENTS[best]!r} ({row[best]:.3f})")
    print("\ntruncation equivalence:")
    check_truncation_equivalence(encoder)


if __name__ == "__main__":
    main()
