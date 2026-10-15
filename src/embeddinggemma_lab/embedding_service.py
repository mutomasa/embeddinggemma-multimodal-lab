"""Cache-aware embedding of queries and corpus items."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from embeddinggemma_lab.config import NATIVE_DIM, SUPPORTED_DIMS, validate_dim
from embeddinggemma_lab.embedding_cache import CacheNamespace, EmbeddingCache
from embeddinggemma_lab.ingestion import Item, content_hash
from embeddinggemma_lab.matryoshka import truncate
from embeddinggemma_lab.model_adapter import Encoder
from embeddinggemma_lab.preprocessing import file_sha256

DOCUMENT_FORMAT_VERSION = "v1"  # bump when document formatting changes
NATIVE_VIDEO_PARAMS = "native-video:model-default-1fps"
MEDIA_INPUT_KEYS = {"image": "image", "video_frame": "image", "video": "video"}


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    def reset(self) -> None:
        self.hits = self.misses = 0


def is_media(item: Item) -> bool:
    return item.modality in MEDIA_INPUT_KEYS


def media_input(item: Item) -> dict[str, str]:
    """Model input for a media item (no task prefix, per the model card)."""
    if item.path is None:
        raise ValueError(f"media item {item.item_id} has no path")
    return {MEDIA_INPUT_KEYS[item.modality]: item.path}


def item_key(item: Item) -> str:
    """Text: hash of title+text. Media: hash of the file bytes (+ video settings)."""
    if not is_media(item):
        return content_hash(item.modality, item.title, item.text)
    extra = NATIVE_VIDEO_PARAMS if item.modality == "video" else None
    return content_hash(item.modality, file_sha256(item.path or ""), extra)


class CachedEmbedder:
    """Encodes once at native dim and caches every supported Matryoshka dim."""

    def __init__(self, encoder: Encoder, cache: EmbeddingCache) -> None:
        self.encoder = encoder
        self.cache = cache
        self.stats = CacheStats()

    def namespace(self, dim: int, role: str) -> CacheNamespace:
        return CacheNamespace(
            model_id=self.encoder.model_id,
            revision=self.encoder.revision,
            dim=dim,
            role=role,
            preprocessing=(("document_format", DOCUMENT_FORMAT_VERSION),),
        )

    def embed_queries(self, texts: Sequence[str], dim: int) -> np.ndarray:
        keys = [content_hash(t) for t in texts]
        return self._embed(keys, dim, "query", lambda idx: self._encode_queries(texts, idx))

    def embed_items(self, items: Sequence[Item], dim: int) -> np.ndarray:
        keys = [item_key(item) for item in items]
        return self._embed(keys, dim, "document", lambda idx: self._encode_items(items, idx))

    def _encode_queries(self, texts: Sequence[str], idx: list[int]) -> np.ndarray:
        return self.encoder.encode_queries([texts[i] for i in idx])

    def _encode_items(self, items: Sequence[Item], idx: list[int]) -> np.ndarray:
        """Encode text and media separately, then restore the original order."""
        subset = [items[i] for i in idx]
        text_pos = [p for p, x in enumerate(subset) if not is_media(x)]
        media_pos = [p for p, x in enumerate(subset) if is_media(x)]
        out = np.zeros((len(subset), NATIVE_DIM), dtype=np.float32)
        if text_pos:
            texts = [subset[p] for p in text_pos]
            out[text_pos] = self.encoder.encode_documents(
                [x.text for x in texts], [x.title for x in texts]
            )
        if media_pos:
            out[media_pos] = self.encoder.encode_media([media_input(subset[p]) for p in media_pos])
        return out

    def _embed(
        self,
        keys: list[str],
        dim: int,
        role: str,
        encode_missing: Callable[[list[int]], np.ndarray],
    ) -> np.ndarray:
        validate_dim(dim)
        namespace = self.namespace(dim, role)
        vectors: list[np.ndarray | None] = [self.cache.get(namespace, k) for k in keys]
        missing = [i for i, v in enumerate(vectors) if v is None]
        self.stats.hits += len(keys) - len(missing)
        self.stats.misses += len(missing)
        if missing:
            native = encode_missing(missing)
            self._store_all_dims([keys[i] for i in missing], native, role)
            for i, vec in zip(missing, truncate(native, dim), strict=True):
                vectors[i] = vec
        if not vectors:
            return np.zeros((0, dim), dtype=np.float32)
        return np.stack(vectors).astype(np.float32)

    def _store_all_dims(self, keys: list[str], native: np.ndarray, role: str) -> None:
        for dim in SUPPORTED_DIMS:
            namespace = self.namespace(dim, role)
            for key, vec in zip(keys, truncate(native, dim), strict=True):
                self.cache.put(namespace, key, vec)
