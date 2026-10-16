import pytest

from embeddinggemma_lab.annotations import parse_annotations
from embeddinggemma_lab.event_queries import (
    EventQuery,
    frame_events,
    frame_qrels,
    grade_of,
    load_event_queries,
    parse_event_queries,
    to_eval_queries,
    video_qrels,
)
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.research_corpus import ResearchPaths

SEGMENTS = parse_annotations(
    "video_id,start_s,end_s,event\n"
    "slip_vid,0.0,1.0,carry\n"
    "slip_vid,1.0,2.0,slip\n"
    "rec_vid,0.5,1.0,slip\n"
    "rec_vid,1.0,2.0,regrasp\n"
    "ok_vid,0.0,2.0,success\n"
)


def frame(vid, ts):
    return Item(
        f"{vid}@{ts}", modality="video_frame", metadata={"video_id": vid, "timestamp_s": ts}
    )


def test_parse_event_queries_and_primary_event():
    queries = parse_event_queries(
        '{"query_id": "q", "query": "x", "events": {"slip": 1, "regrasp": 2}}\n\n'
    )
    assert queries == [EventQuery("q", "x", {"slip": 1, "regrasp": 2})]
    assert queries[0].primary_event == "regrasp"
    with pytest.raises(ValueError, match="missing keys"):
        parse_event_queries('{"query_id": "q", "query": "x"}')


def test_grade_of_takes_the_best_matching_event():
    assert grade_of(frozenset({"slip", "regrasp"}), {"regrasp": 2, "slip": 1}) == 2
    assert grade_of(frozenset({"carry"}), {"slip": 2}) == 0
    assert grade_of(frozenset(), {"slip": 2}) == 0


def test_video_qrels_marks_videos_containing_the_event():
    qrels = video_qrels({"regrasp": 2, "slip": 1}, ["slip_vid", "rec_vid", "ok_vid"], SEGMENTS)
    assert qrels == {"slip_vid": 1, "rec_vid": 2}


def test_frame_qrels_uses_timestamps_and_tolerance():
    frames = [frame("slip_vid", 0.5), frame("slip_vid", 0.9), frame("slip_vid", 1.5)]
    assert frame_qrels({"slip": 2}, frames, SEGMENTS, 0.0) == {"slip_vid@1.5": 2}
    assert frame_qrels({"slip": 2}, frames, SEGMENTS, 0.1) == {"slip_vid@0.9": 2, "slip_vid@1.5": 2}


def test_frame_events_reads_frame_metadata():
    assert frame_events(frame("rec_vid", 1.0), SEGMENTS) == {"slip", "regrasp"}


def test_to_eval_queries_keeps_ids_and_text():
    queries = [EventQuery("q1", "slip", {"slip": 2})]
    out = to_eval_queries(queries, lambda grades: {"a": grades["slip"]})
    assert out[0].query_id == "q1" and out[0].relevant == {"a": 2}


def test_sample_queries_target_annotated_events():
    events = {s.event for s in parse_annotations(ResearchPaths.sample().annotations.read_text())}
    for query in load_event_queries(ResearchPaths.sample().queries):
        assert query.primary_event in events
