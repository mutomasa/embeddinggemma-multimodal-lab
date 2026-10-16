"""Image loading and video frame sampling (independent of the model)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".webm")


@dataclass(frozen=True)
class FrameSampling:
    """Regular sampling: one frame every ``interval_s`` seconds, at most ``max_frames``."""

    interval_s: float = 0.5
    max_frames: int = 120

    def as_params(self) -> tuple[tuple[str, str], ...]:
        return (("interval_s", f"{self.interval_s:g}"), ("max_frames", str(self.max_frames)))


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    image: Image.Image


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _file_sha256(path: str, size: int, mtime_ns: int) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_cached_sha256 = lru_cache(maxsize=4096)(_file_sha256)


def file_sha256(path: str | Path) -> str:
    """Content hash of a file, memoised on (path, size, mtime)."""
    stat = Path(path).stat()
    return _cached_sha256(str(path), stat.st_size, stat.st_mtime_ns)


def sample_timestamps(duration_s: float, sampling: FrameSampling) -> list[float]:
    """``0, interval, 2*interval, ...`` strictly before ``duration_s``."""
    if duration_s <= 0 or sampling.interval_s <= 0:
        return [0.0] if duration_s > 0 else []
    count = min(int(np.ceil(duration_s / sampling.interval_s)), sampling.max_frames)
    return [round(i * sampling.interval_s, 6) for i in range(count)]


def video_info(path: str | Path) -> VideoInfo:
    import cv2

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS)) or 0.0
        return VideoInfo(fps=fps, frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        capture.release()


def extract_frames(path: str | Path, sampling: FrameSampling) -> list[Frame]:
    import cv2

    info = video_info(path)
    capture = cv2.VideoCapture(str(path))
    frames: list[Frame] = []
    try:
        for ts in sample_timestamps(info.duration_s, sampling):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(round(ts * info.fps)))
            ok, bgr = capture.read()
            if ok:
                frames.append(Frame(ts, Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))))
    finally:
        capture.release()
    return frames


def save_frames(frames: list[Frame], out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for frame in frames:
        path = out_dir / f"{stem}_t{frame.timestamp_s:08.3f}.jpg"
        if not path.exists():
            frame.image.save(path, quality=90)
        paths.append(path)
    return paths
