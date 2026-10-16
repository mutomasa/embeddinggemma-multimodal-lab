from embeddinggemma_lab.model_adapter import (
    format_document,
    modality_config_kwargs,
    revision_from_path,
)


def test_format_document_with_and_without_title():
    assert format_document("body", "Title") == "title: Title | text: body"
    assert format_document("body") == "title: none | text: body"


def test_text_only_disables_vision_and_audio():
    assert modality_config_kwargs(frozenset({"text"})) == {
        "vision_config": None,
        "audio_config": None,
    }


def test_image_or_video_keeps_vision_encoder():
    assert modality_config_kwargs(frozenset({"text", "image"})) == {"audio_config": None}
    assert modality_config_kwargs(frozenset({"video"})) == {"audio_config": None}


def test_revision_from_snapshot_path():
    path = "/hub/models--google--embeddinggemma-2/snapshots/abc123/config.json"
    assert revision_from_path(path) == "abc123"
    assert revision_from_path("/some/local/dir/config.json") == "unknown"
