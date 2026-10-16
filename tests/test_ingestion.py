from io import StringIO

import pytest

from embeddinggemma_lab.ingestion import (
    Item,
    content_hash,
    deduplicate,
    load_csv,
    load_sample_documents,
    load_txt,
    load_uploaded,
)


def test_sample_documents_have_ids_labels_and_unique_ids():
    items = load_sample_documents()
    assert len(items) >= 40
    assert len({x.item_id for x in items}) == len(items)
    assert all(x.label and x.text for x in items)


def test_load_csv_accepts_content_column_and_generates_ids():
    items = load_csv(StringIO("content,title\nhello,T\n,skip\nworld,\n"))
    assert [(x.item_id, x.text, x.title) for x in items] == [
        ("doc00000", "hello", "T"),
        ("doc00002", "world", None),
    ]


def test_load_csv_without_text_column_fails():
    with pytest.raises(ValueError):
        load_csv(StringIO("a,b\n1,2\n"))


def test_load_txt_one_doc_per_line():
    items = load_txt("first\n\n  second  \n")
    assert [x.text for x in items] == ["first", "second"]


def test_load_uploaded_dispatches_and_rejects_unknown():
    assert load_uploaded("notes.txt", b"a\nb")[1].item_id == "notes-00001"
    assert load_uploaded("d.csv", "﻿text\nx".encode())[0].text == "x"
    with pytest.raises(ValueError):
        load_uploaded("x.pdf", b"")


def test_content_hash_distinguishes_none_and_boundaries():
    assert content_hash("ab", "c") != content_hash("a", "bc")
    assert content_hash(None) != content_hash("")


def test_deduplicate_keeps_first():
    items = [Item("1", "same"), Item("2", "same"), Item("3", "other")]
    assert [x.item_id for x in deduplicate(items)] == ["1", "3"]
