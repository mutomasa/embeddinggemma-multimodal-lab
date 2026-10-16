import numpy as np
import pytest

from embeddinggemma_lab.embedding_cache import EmbeddingCache
from embeddinggemma_lab.embedding_service import CachedEmbedder, item_key, media_input
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.media_ingestion import (
    IMAGE,
    VIDEO,
    VIDEO_FRAME,
    all_frame_items,
    load_media_items,
)
from embeddinggemma_lab.preprocessing import FrameSampling
from embeddinggemma_lab.sample_media import (
    IMAGE_SPECS,
    _scenario_state,
    generate_images,
    generate_videos,
)
from embeddinggemma_lab.search_service import SearchService


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    return generate_images(tmp_path_factory.mktemp("images"))


@pytest.fixture(scope="module")
def video_dir(tmp_path_factory):
    return generate_videos(tmp_path_factory.mktemp("videos"))


def test_generated_images_are_annotated(image_dir):
    items = load_media_items(image_dir, IMAGE)
    assert len(items) == len(IMAGE_SPECS)
    red = next(x for x in items if x.item_id == "img_red_circle")
    assert (red.title, red.label, red.modality) == ("a red circle", "shape", IMAGE)


def test_generation_is_deterministic(image_dir, tmp_path):
    again = generate_images(tmp_path)
    first = {x.item_id: item_key(x) for x in load_media_items(image_dir, IMAGE)}
    second = {x.item_id: item_key(x) for x in load_media_items(again, IMAGE)}
    assert first == second


@pytest.mark.parametrize(
    ("scenario", "attached_at_end"),
    [("success", True), ("slip", True), ("recovery", True), ("drop", False)],
)
def test_gripper_scenarios(scenario, attached_at_end):
    assert not _scenario_state(scenario, 0.5).attached  # block rests before grasp
    end = _scenario_state(scenario, 3.9)
    assert end.attached is attached_at_end
    if scenario == "drop":
        assert end.block_y == pytest.approx(215)


def test_frame_items_carry_video_metadata(video_dir, tmp_path):
    videos = load_media_items(video_dir, VIDEO)
    assert len(videos) == 8 and {v.label for v in videos} >= {"grasp_drop", "grasp_slip"}
    frames = all_frame_items(videos[:1], FrameSampling(1.0), tmp_path)
    assert [f.metadata["timestamp_s"] for f in frames] == [0.0, 1.0, 2.0, 3.0]
    assert frames[2].item_id == f"{videos[0].item_id}@2.0s"
    assert frames[2].modality == VIDEO_FRAME and frames[2].label == videos[0].label


def test_media_input_mapping():
    assert media_input(Item("a", modality=IMAGE, path="x.png")) == {"image": "x.png"}
    assert media_input(Item("f", modality=VIDEO_FRAME, path="f.jpg")) == {"image": "f.jpg"}
    assert media_input(Item("v", modality=VIDEO, path="v.mp4")) == {"video": "v.mp4"}
    with pytest.raises(ValueError):
        media_input(Item("n", modality=IMAGE))


def test_media_key_depends_on_file_content_not_name(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    key_a = item_key(Item("a", modality=IMAGE, path=str(a)))
    assert key_a == item_key(Item("b", modality=IMAGE, path=str(b)))
    assert key_a != item_key(Item("a", modality=VIDEO, path=str(a)))


def test_mixed_text_and_media_keep_order(tmp_path, fake_encoder, image_dir):
    images = load_media_items(image_dir, IMAGE)[:2]
    items = [images[0], Item("t", "some text"), images[1]]
    embedder = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path))
    mixed = embedder.embed_items(items, 768)
    separate = CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path / "other"))
    np.testing.assert_allclose(mixed[1], separate.embed_items([items[1]], 768)[0])
    np.testing.assert_allclose(mixed[2], separate.embed_items([items[2]], 768)[0])


def test_search_item_excludes_query_itself(tmp_path, fake_encoder, image_dir):
    images = load_media_items(image_dir, IMAGE)
    service = SearchService(CachedEmbedder(fake_encoder, EmbeddingCache(tmp_path)), images)
    result = service.search_item(images[0], dim=256, k=3)
    assert images[0].item_id not in [r.item.item_id for r in result.results]
    assert [r.hit.rank for r in result.results] == [1, 2, 3]


def test_save_upload_is_content_addressed(tmp_path):
    from embeddinggemma_lab.media_ingestion import save_upload

    first = save_upload("My Photo.PNG", b"abc", tmp_path)
    assert first == save_upload("My Photo.PNG", b"abc", tmp_path)
    assert first.suffix == ".png" and first.name.startswith("My_Photo-")
    assert save_upload("My Photo.PNG", b"xyz", tmp_path) != first
