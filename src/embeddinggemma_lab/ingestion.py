"""Loading corpora into :class:`Item` records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

from embeddinggemma_lab.config import DATASETS_DIR

SAMPLE_DOCUMENTS = DATASETS_DIR / "sample" / "documents.csv"
TEXT_COLUMN_CANDIDATES = ("text", "content", "body", "document")


@dataclass(frozen=True)
class Item:
    """One searchable unit (a document, an image, or a video frame)."""

    item_id: str
    text: str = ""
    title: str | None = None
    modality: str = "text"
    label: str | None = None
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def display(self) -> str:
        return self.text or self.path or self.item_id


def content_hash(*parts: str | bytes | None) -> str:
    """Stable sha256 over the given parts (``None`` is distinguished from ``""``)."""
    digest = hashlib.sha256()
    for part in parts:
        data = b"\x00none" if part is None else part if isinstance(part, bytes) else part.encode()
        digest.update(len(data).to_bytes(8, "little") + data)
    return digest.hexdigest()


def _pick_text_column(columns: Iterable[str]) -> str:
    lowered = {c.lower(): c for c in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"CSV needs one of the columns {TEXT_COLUMN_CANDIDATES}")


def _optional(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text or None


def items_from_frame(frame: pd.DataFrame, id_prefix: str = "doc") -> list[Item]:
    text_col = _pick_text_column(frame.columns)
    items = []
    for i, row in enumerate(frame.to_dict("records")):
        text = _optional(row.get(text_col))
        if text is None:
            continue
        extra = {k: v for k, v in row.items() if k not in {text_col, "doc_id", "title", "category"}}
        items.append(
            Item(
                item_id=_optional(row.get("doc_id")) or f"{id_prefix}{i:05d}",
                text=text,
                title=_optional(row.get("title")),
                label=_optional(row.get("category")),
                metadata=extra,
            )
        )
    return items


def load_csv(source: str | Path | StringIO) -> list[Item]:
    return items_from_frame(pd.read_csv(source, dtype=str, keep_default_na=False))


def load_txt(content: str, id_prefix: str = "line") -> list[Item]:
    """One document per non-empty line."""
    lines = [line.strip() for line in content.splitlines()]
    return [Item(item_id=f"{id_prefix}{i:05d}", text=t) for i, t in enumerate(lines) if t]


def load_sample_documents() -> list[Item]:
    return load_csv(SAMPLE_DOCUMENTS)


def load_uploaded(name: str, data: bytes) -> list[Item]:
    """Dispatch on file extension for Streamlit uploads."""
    text = data.decode("utf-8-sig")
    suffix = Path(name).suffix.lower()
    if suffix == ".csv":
        return load_csv(StringIO(text))
    if suffix == ".txt":
        return load_txt(text, id_prefix=f"{Path(name).stem}-")
    raise ValueError(f"Unsupported file type: {suffix} (use .csv or .txt)")


def deduplicate(items: Iterable[Item]) -> list[Item]:
    """Drop items whose content duplicates an earlier one; keep first occurrence."""
    seen: set[str] = set()
    unique = []
    for item in items:
        key = content_hash(item.modality, item.title, item.text, item.path)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique
