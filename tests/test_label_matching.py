from embeddinggemma_lab.annotations import parse_annotations
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.label_matching import (
    UNLABELED,
    agreement_at_k,
    confusion_table,
    hits_frame,
    label_hits,
)
from embeddinggemma_lab.retriever import Hit
from embeddinggemma_lab.search_service import RankedItem

SEGMENTS = parse_annotations(
    "video_id,start_s,end_s,event\nv,0.0,1.0,carry\nv,1.0,2.0,slip\nv,1.5,2.0,regrasp\n"
)


def ranked(rank, ts, score=0.5):
    frame = Item(f"v@{ts}", modality="video_frame", metadata={"video_id": "v", "timestamp_s": ts})
    return RankedItem(Hit(rank - 1, frame.item_id, score, rank), frame)


def test_label_hits_attach_human_events_and_grade():
    hits = label_hits([ranked(1, 1.75), ranked(2, 0.5), ranked(3, 3.0)], {"slip": 2}, SEGMENTS, 0)
    assert [h.human_events for h in hits] == [{"slip", "regrasp"}, {"carry"}, frozenset()]
    assert [h.matched for h in hits] == [True, False, False]
    assert hits[0].video_id == "v" and hits[0].timestamp_s == 1.75


def test_tolerance_widens_the_match():
    hits = label_hits([ranked(1, 0.9)], {"slip": 2}, SEGMENTS, tolerance_s=0.2)
    assert hits[0].matched


def test_agreement_at_k():
    hits = label_hits([ranked(1, 1.5), ranked(2, 0.5), ranked(3, 1.2)], {"slip": 2}, SEGMENTS, 0)
    assert agreement_at_k(hits, 1) == 1.0
    assert agreement_at_k(hits, 3) == 2 / 3
    assert agreement_at_k([], 3) == 0.0


def test_confusion_table_counts_each_event_and_unlabeled():
    table = confusion_table(
        [("slip", frozenset({"slip"})), ("slip", frozenset({"carry"})), ("drop", frozenset())]
    )
    assert table.loc["slip", "slip"] == 1
    assert table.loc["slip", "carry"] == 1
    assert table.loc["drop", UNLABELED] == 1
    assert confusion_table([]).empty


def test_hits_frame_marks_matches():
    hits = label_hits([ranked(1, 1.5), ranked(2, 3.0)], {"slip": 2}, SEGMENTS, 0)
    frame = hits_frame(hits)
    assert frame["match"].tolist() == ["✅", "❌"]
    assert frame["human_label"].tolist() == ["regrasp, slip", UNLABELED]
