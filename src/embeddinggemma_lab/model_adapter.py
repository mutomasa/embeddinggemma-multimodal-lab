"""Embedding model adapter.

All model-specific details (prompt names, document formatting, encoder
loading flags) are confined to this module so that the rest of the code base
only depends on the :class:`Encoder` protocol.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from embeddinggemma_lab.config import (
    DOCUMENT_PROMPT_NAME,
    QUERY_PROMPT_NAME,
    ModelSettings,
)
from embeddinggemma_lab.device import DeviceInfo, resolve_device
from embeddinggemma_lab.matryoshka import l2_normalize


class Encoder(Protocol):
    """Anything that maps inputs to L2-normalised native-dimension vectors."""

    model_id: str
    revision: str

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_documents(
        self, texts: Sequence[str], titles: Sequence[str | None] | None = None
    ) -> np.ndarray: ...

    def encode_media(self, items: Sequence[dict[str, Any]]) -> np.ndarray: ...


def format_document(text: str, title: str | None = None) -> str:
    """Document prefix recommended by the model card."""
    return f"title: {title or 'none'} | text: {text}"


def modality_config_kwargs(modalities: frozenset[str]) -> dict[str, None]:
    """Disable unused encoders to save memory (model card, section 2)."""
    kwargs: dict[str, None] = {}
    if "image" not in modalities and "video" not in modalities:
        kwargs["vision_config"] = None
    if "audio" not in modalities:
        kwargs["audio_config"] = None
    return kwargs


def revision_from_path(path: str | Path) -> str:
    """Commit hash from a file path inside a Hugging Face ``snapshots/<sha>/`` dir."""
    parts = Path(path).parts
    if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts):
        return parts[parts.index("snapshots") + 1]
    return "unknown"


def resolve_revision(model_id: str, revision: str | None = None) -> str:
    """Best-effort commit hash of the locally cached model."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    if isinstance(cached, str):
        return revision_from_path(cached)
    return revision or "unknown"


class GemmaEncoder:
    """SentenceTransformer-backed implementation of :class:`Encoder`."""

    def __init__(self, settings: ModelSettings | None = None) -> None:
        self.settings = settings or ModelSettings()
        self.device_info: DeviceInfo = resolve_device(self.settings.device)
        self.model = self._load_model()
        self.model_id = self.settings.model_id
        self.revision = resolve_revision(self.settings.model_id, self.settings.revision)

    def _load_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(
            self.settings.model_id,
            revision=self.settings.revision,
            device=self.device_info.device,
            model_kwargs={"dtype": self.device_info.dtype},
            config_kwargs=modality_config_kwargs(self.settings.modalities),
        )

    def _encode(self, inputs: Sequence[Any], **kwargs: Any) -> np.ndarray:
        embeddings = self.model.encode(
            list(inputs),
            batch_size=self.settings.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            **kwargs,
        )
        return l2_normalize(np.asarray(embeddings, dtype=np.float32))

    def encode_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode(texts, prompt_name=QUERY_PROMPT_NAME)

    def encode_documents(
        self, texts: Sequence[str], titles: Sequence[str | None] | None = None
    ) -> np.ndarray:
        titles = titles or [None] * len(texts)
        if all(t is None for t in titles):
            return self._encode(texts, prompt_name=DOCUMENT_PROMPT_NAME)
        return self._encode([format_document(x, t) for x, t in zip(texts, titles, strict=True)])

    def encode_media(self, items: Sequence[dict[str, Any]]) -> np.ndarray:
        """Encode multimodal dict inputs (e.g. ``{"image": path}``) without prefixes."""
        return self._encode(items)
