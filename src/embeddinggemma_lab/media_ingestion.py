"""Turning image folders and videos into :class:`Item` records."""

from __future__ import annotations

import csv
from pathlib import Path

from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.preprocessing import (
    IMAGE_SUFFIXES,
    VIDEO_SUFFIXES,
    FrameSampling,
    extract_frames,
    save_frames,
)
from embeddinggemma_lab.sample_media import METADATA_FILE

IMAGE = "image"
VIDEO = "video"
VIDEO_FRAME = "video_frame"


def _read_metadata(directory: Path) -> dict[str, dict[str, str]]:
    path = directory / METADATA_FILE
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return {row["file"]: row for row in csv.DictReader(handle)}


def load_media_items(directory: Path, modality: str) -> list[Item]:
    """Every image (or video) file in ``directory``, annotated from ``metadata.csv``."""
    suffixes = IMAGE_SUFFIXES if modality == IMAGE else VIDEO_SUFFIXES
    metadata = _read_metadata(directory)
    items = []
    for path in sorted(p for p in directory.iterdir() if p.suffix.lower() in suffixes):
        row = metadata.get(path.name, {})
        items.append(
            Item(
                item_id=row.get("media_id") or path.stem,
                title=row.get("caption") or path.stem,
                modality=modality,
                label=row.get("label") or None,
                path=str(path),
            )
        )
    return items


def frame_items(video: Item, sampling: FrameSampling, frames_root: Path) -> list[Item]:
    """Regularly sampled frames of one video, carrying video id and timestamp."""
    if video.path is None:
        raise ValueError(f"video item {video.item_id} has no path")
    frames = extract_frames(video.path, sampling)
    out_dir = frames_root / f"{video.item_id}_every{sampling.interval_s:g}s"
    paths = save_frames(frames, out_dir, video.item_id)
    return [
        Item(
            item_id=f"{video.item_id}@{frame.timestamp_s:.1f}s",
            title=f"{video.title} @ {frame.timestamp_s:.1f}s",
            modality=VIDEO_FRAME,
            label=video.label,
            path=str(path),
            metadata={
                "video_id": video.item_id,
                "video_path": video.path,
                "timestamp_s": frame.timestamp_s,
            },
        )
        for frame, path in zip(frames, paths, strict=True)
    ]


def all_frame_items(videos: list[Item], sampling: FrameSampling, frames_root: Path) -> list[Item]:
    return [frame for video in videos for frame in frame_items(video, sampling, frames_root)]


def save_upload(name: str, data: bytes, directory: Path) -> Path:
    """Store an uploaded file under a content-addressed name (dedups re-uploads)."""
    from embeddinggemma_lab.ingestion import content_hash

    directory.mkdir(parents=True, exist_ok=True)
    stem = Path(name).stem.replace(" ", "_")[:40]
    path = directory / f"{stem}-{content_hash(data)[:10]}{Path(name).suffix.lower()}"
    if not path.exists():
        path.write_bytes(data)
    return path
