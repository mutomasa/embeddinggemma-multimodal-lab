import pytest

from embeddinggemma_lab.annotations import (
    EventSegment,
    annotated_video_ids,
    events_at,
    load_annotations,
    parse_annotations,
    video_events,
)
from embeddinggemma_lab.research_corpus import ResearchPaths

CSV = """video_id,start_s,end_s,event
v1,0.0,1.0,grasp
v1,1.0,2.0,slip
v1,1.5,2.0,regrasp
v2,0.0,2.0,success
"""


def test_parse_annotations_reads_rows_and_defaults_annotator():
    segments = parse_annotations(CSV)
    assert segments[0] == EventSegment("v1", 0.0, 1.0, "grasp", "")
    assert len(segments) == 4


def test_parse_annotations_rejects_missing_columns_and_reversed_segments():
    with pytest.raises(ValueError, match="missing columns"):
        parse_annotations("video_id,start_s,event\nv1,0,grasp\n")
    with pytest.raises(ValueError, match="line 2"):
        parse_annotations("video_id,start_s,end_s,event\nv1,2.0,1.0,slip\n")


def test_segment_contains_with_tolerance():
    segment = EventSegment("v1", 1.0, 2.0, "slip")
    assert segment.contains(1.0) and segment.contains(2.0)
    assert not segment.contains(0.9)
    assert segment.contains(0.9, tolerance_s=0.1 + 1e-9)


def test_events_at_returns_overlapping_events_of_one_video():
    segments = parse_annotations(CSV)
    assert events_at(segments, "v1", 1.75) == {"slip", "regrasp"}
    assert events_at(segments, "v1", 3.0) == frozenset()
    assert events_at(segments, "v2", 1.75) == {"success"}


def test_video_events_and_annotated_ids():
    segments = parse_annotations(CSV)
    assert video_events(segments, "v1") == {"grasp", "slip", "regrasp"}
    assert annotated_video_ids(segments) == ["v1", "v2"]


def test_sample_annotations_cover_the_research_outcomes():
    segments = load_annotations(ResearchPaths.sample().annotations)
    events = {s.event for s in segments}
    assert {"success", "slip", "drop", "regrasp"} <= events
    assert all(s.end_s >= s.start_s for s in segments)
