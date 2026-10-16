"""Deterministic, copyright-free synthetic images and videos for the demo.

Images are simple drawings (shapes, scenes, a gripper holding a block).
Videos are 2D animations of a two-finger gripper picking a block, with four
outcomes that mirror the research use case: success, slip, drop, recovery.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from embeddinggemma_lab.config import DATASETS_DIR

SAMPLE_IMAGES_DIR = DATASETS_DIR / "sample" / "images"
SAMPLE_VIDEOS_DIR = DATASETS_DIR / "sample" / "videos"
METADATA_FILE = "metadata.csv"
IMAGE_SIZE = (320, 240)
VIDEO_FPS = 10
VIDEO_SECONDS = 4.0

Draw = Callable[[ImageDraw.ImageDraw], None]


@dataclass(frozen=True)
class MediaSpec:
    media_id: str
    caption: str
    label: str


# --------------------------------------------------------------------------- images


def _shape(kind: str, color: str) -> Draw:
    box = (110, 70, 210, 170)

    def draw(d: ImageDraw.ImageDraw) -> None:
        if kind == "circle":
            d.ellipse(box, fill=color)
        elif kind == "square":
            d.rectangle(box, fill=color)
        elif kind == "triangle":
            d.polygon([(160, 60), (100, 180), (220, 180)], fill=color)
        else:  # star
            d.regular_polygon((160, 120, 60), 5, fill=color)
            d.regular_polygon((160, 120, 60), 5, rotation=36, fill=color)

    return draw


def _sunset(d: ImageDraw.ImageDraw) -> None:
    for y in range(150):
        d.line([(0, y), (320, y)], fill=(255, 120 + y // 3, 60))
    d.pieslice((110, 100, 210, 200), 180, 360, fill=(255, 220, 60))
    d.rectangle((0, 150, 320, 240), fill=(30, 70, 150))


def _night(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((0, 0, 320, 240), fill=(10, 15, 40))
    d.ellipse((220, 30, 280, 90), fill=(240, 240, 200))
    for x, y in [(40, 40), (90, 120), (150, 60), (60, 190), (200, 170), (290, 150)]:
        d.regular_polygon((x, y, 4), 4, fill="white")


def _tree(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((145, 140, 175, 220), fill=(110, 70, 30))
    d.ellipse((90, 40, 230, 160), fill=(40, 150, 50))
    d.rectangle((0, 220, 320, 240), fill=(90, 180, 70))


def _house(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((90, 110, 230, 210), fill=(230, 200, 150))
    d.polygon([(70, 110), (160, 40), (250, 110)], fill=(170, 40, 40))
    d.rectangle((145, 160, 175, 210), fill=(100, 60, 30))
    d.rectangle((105, 130, 135, 155), fill=(150, 200, 255))


def _car(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((60, 120, 260, 170), fill=(30, 90, 200))
    d.polygon([(100, 120), (130, 85), (210, 85), (235, 120)], fill=(30, 90, 200))
    for x in (100, 220):
        d.ellipse((x - 22, 150, x + 22, 194), fill="black")
    d.rectangle((0, 194, 320, 240), fill=(90, 90, 90))


def _snowman(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((0, 0, 320, 240), fill=(200, 225, 255))
    d.ellipse((115, 130, 205, 220), fill="white", outline="gray")
    d.ellipse((130, 70, 190, 130), fill="white", outline="gray")
    d.polygon([(160, 100), (185, 105), (160, 110)], fill="orange")


def _apple(d: ImageDraw.ImageDraw) -> None:
    d.ellipse((105, 70, 215, 180), fill=(210, 20, 30))
    d.ellipse((165, 45, 200, 70), fill=(40, 140, 40))
    d.line([(160, 75), (160, 50)], fill=(90, 50, 20), width=5)


def _banana(d: ImageDraw.ImageDraw) -> None:
    d.chord((60, 20, 260, 200), 20, 160, fill=(250, 220, 40))
    d.chord((60, 0, 260, 170), 20, 160, fill="white")


def _smiley(d: ImageDraw.ImageDraw) -> None:
    d.ellipse((90, 50, 230, 190), fill=(255, 210, 0))
    d.ellipse((125, 90, 145, 110), fill="black")
    d.ellipse((175, 90, 195, 110), fill="black")
    d.arc((120, 100, 200, 165), 20, 160, fill="black", width=6)


def _traffic_light(d: ImageDraw.ImageDraw) -> None:
    d.rectangle((130, 30, 190, 210), fill=(40, 40, 40))
    for y, color in [(40, "red"), (100, (90, 70, 0)), (160, (0, 70, 0))]:
        d.ellipse((140, y, 180, y + 40), fill=color)


def _checker(d: ImageDraw.ImageDraw) -> None:
    for i in range(8):
        for j in range(6):
            if (i + j) % 2:
                d.rectangle((i * 40, j * 40, i * 40 + 40, j * 40 + 40), fill="black")


def _gripper_holding(d: ImageDraw.ImageDraw) -> None:
    _draw_gripper(d, x=160, y=110, open_px=0, block_color=(220, 40, 40), block_dy=0)


def _gripper_dropped(d: ImageDraw.ImageDraw) -> None:
    _draw_gripper(d, x=160, y=70, open_px=14, block_color=None, block_dy=0)
    d.rectangle((0, FLOOR_Y, IMAGE_SIZE[0], IMAGE_SIZE[1]), fill=(150, 120, 90))
    d.rectangle((185, FLOOR_Y - 40, 225, FLOOR_Y), fill=(220, 40, 40))


IMAGE_SPECS: list[tuple[MediaSpec, Draw]] = [
    (MediaSpec("img_red_circle", "a red circle", "shape"), _shape("circle", "red")),
    (MediaSpec("img_blue_square", "a blue square", "shape"), _shape("square", "blue")),
    (MediaSpec("img_green_triangle", "a green triangle", "shape"), _shape("triangle", "green")),
    (MediaSpec("img_yellow_star", "a yellow star", "shape"), _shape("star", "gold")),
    (MediaSpec("img_sunset", "sunset over the sea", "scene"), _sunset),
    (MediaSpec("img_night", "night sky with the moon and stars", "scene"), _night),
    (MediaSpec("img_tree", "a green tree on grass", "nature"), _tree),
    (MediaSpec("img_house", "a small house with a red roof", "building"), _house),
    (MediaSpec("img_car", "a blue car on the road", "vehicle"), _car),
    (MediaSpec("img_snowman", "a snowman in winter", "winter"), _snowman),
    (MediaSpec("img_apple", "a red apple", "food"), _apple),
    (MediaSpec("img_banana", "a yellow banana", "food"), _banana),
    (MediaSpec("img_smiley", "a smiling face", "face"), _smiley),
    (MediaSpec("img_traffic_light", "a traffic light showing red", "traffic"), _traffic_light),
    (MediaSpec("img_checker", "a black and white checkerboard", "pattern"), _checker),
    (MediaSpec("img_grip_hold", "a robot gripper holding a red block", "robot"), _gripper_holding),
    (MediaSpec("img_grip_drop", "a red block dropped on the floor", "robot"), _gripper_dropped),
]


def render_image(draw: Draw) -> Image.Image:
    image = Image.new("RGB", IMAGE_SIZE, "white")
    draw(ImageDraw.Draw(image))
    return image


# --------------------------------------------------------------------------- videos

FLOOR_Y = 215
BLOCK = 40
FINGER_W, FINGER_H = 10, 50


def _draw_gripper(
    d: ImageDraw.ImageDraw,
    x: float,
    y: float,
    open_px: float,
    block_color: tuple[int, int, int] | None,
    block_dy: float,
) -> None:
    """Gripper centred at ``x``; ``y`` is the fingertip line."""
    half = BLOCK / 2 + open_px
    d.rectangle((x - 4, 0, x + 4, y - FINGER_H), fill=(120, 120, 120))
    d.rectangle(
        (x - half - FINGER_W, y - FINGER_H - 8, x + half + FINGER_W, y - FINGER_H),
        fill=(90, 90, 90),
    )
    d.rectangle((x - half - FINGER_W, y - FINGER_H, x - half, y), fill=(60, 60, 60))
    d.rectangle((x + half, y - FINGER_H, x + half + FINGER_W, y), fill=(60, 60, 60))
    if block_color is not None:
        top = y - BLOCK + block_dy
        d.rectangle((x - BLOCK / 2, top, x + BLOCK / 2, top + BLOCK), fill=block_color)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * min(max(t, 0.0), 1.0)


@dataclass(frozen=True)
class GripperState:
    x: float
    y: float
    open_px: float
    block_x: float
    block_y: float  # bottom edge of the block
    attached: bool


def _base_motion(t: float) -> tuple[float, float, float]:
    """Descend (0-1 s), close (1-1.5 s), lift and carry (1.5-4 s)."""
    x = _lerp(100, 220, (t - 1.5) / 2.0)
    y = _lerp(120, FLOOR_Y, t) if t < 1.5 else _lerp(FLOOR_Y, 110, (t - 1.5) / 1.0)
    open_px = _lerp(14, 0, (t - 1.0) / 0.5)
    return x, y, open_px


def _scenario_state(scenario: str, t: float) -> GripperState:
    x, y, open_px = _base_motion(t)
    if t < 1.5:  # block rests on the floor until the fingers have closed
        return GripperState(x, y, open_px, 100, FLOOR_Y, False)
    slip = 0.0
    if scenario == "slip":
        slip = _lerp(0, 22, (t - 2.0) / 1.5)
    elif scenario == "recovery":
        slip = _lerp(0, 18, (t - 2.0) / 0.6) if t < 2.9 else _lerp(18, 0, (t - 2.9) / 0.5)
    if scenario == "drop" and t >= 2.4:
        x_release, y_release, _ = _base_motion(2.4)
        fall = 0.5 * 900 * (t - 2.4) ** 2
        open_px = _lerp(0, 14, (t - 2.4) / 0.2)
        return GripperState(x, y, open_px, x_release, min(y_release + fall, FLOOR_Y), False)
    return GripperState(x, y, open_px, x, y + slip, True)


def render_video_frame(scenario: str, color: tuple[int, int, int], t: float) -> Image.Image:
    image = Image.new("RGB", IMAGE_SIZE, (235, 235, 230))
    d = ImageDraw.Draw(image)
    d.rectangle((0, FLOOR_Y, IMAGE_SIZE[0], IMAGE_SIZE[1]), fill=(150, 120, 90))
    s = _scenario_state(scenario, t)
    _draw_gripper(d, s.x, s.y, s.open_px, None, 0)
    d.rectangle(
        (s.block_x - BLOCK / 2, s.block_y - BLOCK, s.block_x + BLOCK / 2, s.block_y), fill=color
    )
    return image


VIDEO_SCENARIOS = {
    "success": ("the robot grasps the block and carries it", "grasp_success"),
    "slip": ("the block slips inside the robot gripper", "grasp_slip"),
    "drop": ("the robot drops the block and it falls to the floor", "grasp_drop"),
    "recovery": ("the block slips and the robot regrasps it", "grasp_recovery"),
}
VIDEO_COLORS = {"red": (210, 40, 40), "blue": (40, 80, 210)}


def video_specs() -> Iterator[tuple[MediaSpec, str, tuple[int, int, int]]]:
    for scenario, (caption, label) in VIDEO_SCENARIOS.items():
        for color_name, rgb in VIDEO_COLORS.items():
            spec = MediaSpec(f"vid_{scenario}_{color_name}", f"{caption} ({color_name})", label)
            yield spec, scenario, rgb


def _write_mp4(frames: list[Image.Image], path: Path, fps: int) -> None:
    """H.264 via system ffmpeg (browser-playable); OpenCV mp4v as a fallback."""
    if shutil.which("ffmpeg"):
        width, height = frames[0].size
        cmd = [
            "ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
        ]  # fmt: skip
        data = b"".join(frame.tobytes() for frame in frames)
        subprocess.run(cmd, input=data, check=True)
        return
    import cv2
    import numpy as np

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frames[0].size)
    for frame in frames:
        writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    writer.release()


def render_video(scenario: str, color: tuple[int, int, int], path: Path) -> None:
    count = int(VIDEO_SECONDS * VIDEO_FPS)
    frames = [render_video_frame(scenario, color, i / VIDEO_FPS) for i in range(count)]
    _write_mp4(frames, path, VIDEO_FPS)


# --------------------------------------------------------------------------- io


def _write_metadata(directory: Path, rows: list[tuple[MediaSpec, str]]) -> None:
    with open(directory / METADATA_FILE, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["media_id", "file", "caption", "label"])
        for spec, filename in rows:
            writer.writerow([spec.media_id, filename, spec.caption, spec.label])


def generate_images(directory: Path = SAMPLE_IMAGES_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec, draw in IMAGE_SPECS:
        filename = f"{spec.media_id}.png"
        render_image(draw).save(directory / filename)
        rows.append((spec, filename))
    _write_metadata(directory, rows)
    return directory


def generate_videos(directory: Path = SAMPLE_VIDEOS_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec, scenario, color in video_specs():
        filename = f"{spec.media_id}.mp4"
        render_video(scenario, color, directory / filename)
        rows.append((spec, filename))
    _write_metadata(directory, rows)
    return directory


def ensure_samples() -> None:
    if not (SAMPLE_IMAGES_DIR / METADATA_FILE).exists():
        generate_images()
    if not (SAMPLE_VIDEOS_DIR / METADATA_FILE).exists():
        generate_videos()
