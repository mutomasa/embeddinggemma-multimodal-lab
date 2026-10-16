"""Disk cache for embeddings.

Each vector is stored as one ``.npy`` file under a *namespace* directory. The
namespace hashes everything that changes the vector except the content itself
(model id, revision, dimension, role/prompt and preprocessing settings), and
the file name is the content hash. Vectors therefore survive restarts and
are shared between corpora that contain the same item.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from embeddinggemma_lab.config import CACHE_DIR
from embeddinggemma_lab.ingestion import content_hash


@dataclass(frozen=True)
class CacheNamespace:
    model_id: str
    revision: str
    dim: int
    role: str  # "query" | "document" | "media"
    preprocessing: tuple[tuple[str, str], ...] = field(default_factory=tuple)

    def digest(self) -> str:
        return content_hash(json.dumps(asdict(self), sort_keys=True))[:16]


class EmbeddingCache:
    def __init__(self, root: Path = CACHE_DIR) -> None:
        self.root = Path(root)

    def _dir(self, namespace: CacheNamespace) -> Path:
        return self.root / namespace.digest()

    def _path(self, namespace: CacheNamespace, key: str) -> Path:
        return self._dir(namespace) / f"{key}.npy"

    def get(self, namespace: CacheNamespace, key: str) -> np.ndarray | None:
        path = self._path(namespace, key)
        if not path.exists():
            return None
        try:
            return np.load(path)
        except (OSError, ValueError):
            path.unlink(missing_ok=True)  # corrupt entry: drop and recompute
            return None

    def put(self, namespace: CacheNamespace, key: str, vector: np.ndarray) -> None:
        directory = self._dir(namespace)
        directory.mkdir(parents=True, exist_ok=True)
        self._write_manifest(directory, namespace)
        tmp = directory / f".{key}.tmp.npy"
        np.save(tmp, np.asarray(vector, dtype=np.float32))
        tmp.replace(self._path(namespace, key))

    def _write_manifest(self, directory: Path, namespace: CacheNamespace) -> None:
        manifest = directory / "namespace.json"
        if not manifest.exists():
            manifest.write_text(json.dumps(asdict(namespace), indent=2, ensure_ascii=False))

    def count(self, namespace: CacheNamespace) -> int:
        directory = self._dir(namespace)
        return len(list(directory.glob("*.npy"))) if directory.exists() else 0

    def clear(self) -> None:
        for path in self.root.glob("*/*.npy"):
            path.unlink()
