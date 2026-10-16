import pytest

from embeddinggemma_lab.preprocessing import (
    FrameSampling,
    extract_frames,
    file_sha256,
    sample_timestamps,
    save_frames,
    video_info,
)
from embeddinggemma_lab.sample_media import VIDEO_COLORS, render_video


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("vid") / "drop.mp4"
    render_video("drop", VIDEO_COLORS["red"], path)
    return path


def test_sample_timestamps_regular_and_capped():
    assert sample_timestamps(2.0, FrameSampling(0.5)) == [0.0, 0.5, 1.0, 1.5]
    assert sample_timestamps(2.1, FrameSampling(1.0)) == [0.0, 1.0, 2.0]
    assert len(sample_timestamps(100.0, FrameSampling(0.5, max_frames=10))) == 10
    assert sample_timestamps(0.0, FrameSampling()) == []


def test_sampling_params_are_stable():
    assert FrameSampling(1.0, 5).as_params() == (("interval_s", "1"), ("max_frames", "5"))


def test_video_info_and_frame_extraction(video_path, tmp_path):
    info = video_info(video_path)
    assert info.fps == pytest.approx(10, rel=0.01)
    assert info.duration_s == pytest.approx(4.0, abs=0.15)
    frames = extract_frames(video_path, FrameSampling(1.0))
    assert [f.timestamp_s for f in frames] == [0.0, 1.0, 2.0, 3.0]
    assert frames[0].image.size == (320, 240)
    paths = save_frames(frames, tmp_path, "drop")
    assert all(p.exists() for p in paths) and paths[1].name == "drop_t0001.000.jpg"


def test_video_info_rejects_missing_file(tmp_path):
    with pytest.raises(ValueError):
        video_info(tmp_path / "nope.mp4")


def test_file_sha256_changes_with_content(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"a")
    first = file_sha256(path)
    path.write_bytes(b"bb")
    assert file_sha256(path) != first
