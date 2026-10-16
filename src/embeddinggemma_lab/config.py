"""Static configuration shared across the lab."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

MODEL_ID = "google/embeddinggemma-2"
SUPPORTED_DIMS: tuple[int, ...] = (128, 256, 512, 768)
NATIVE_DIM = 768

QUERY_PROMPT_NAME = "SearchQuery"
DOCUMENT_PROMPT_NAME = "Document"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"
DATASETS_DIR = PROJECT_ROOT / "datasets"
REPORTS_DIR = PROJECT_ROOT / "reports"


@dataclass(frozen=True)
class ModelSettings:
    """How the embedding model is loaded."""

    model_id: str = MODEL_ID
    revision: str | None = None
    device: str | None = None  # None -> auto detect
    modalities: frozenset[str] = field(default_factory=lambda: frozenset({"text"}))
    batch_size: int = 16


def validate_dim(dim: int) -> int:
    """Return ``dim`` if it is a supported Matryoshka dimension."""
    if dim not in SUPPORTED_DIMS:
        raise ValueError(f"Unsupported dimension {dim}; choose from {SUPPORTED_DIMS}")
    return dim
