import pandas as pd

from embeddinggemma_lab.annotations import parse_annotations
from embeddinggemma_lab.ingestion import Item
from embeddinggemma_lab.label_matching import LabeledHit
from embeddinggemma_lab.research_plots import confusion_figure, segments_frame, timeline_figure

SEGMENTS = parse_annotations(
    "video_id,start_s,end_s,event\na,0,1,carry\na,1,2,slip\nb,0,2,success\n"
)


def hit(rank, vid, ts, grade):
    frame = Item(
        f"{vid}@{ts}", modality="video_frame", metadata={"video_id": vid, "timestamp_s": ts}
    )
    return LabeledHit(rank, 0.5, frame, frozenset(), grade)


def test_segments_frame_durations():
    frame = segments_frame(SEGMENTS)
    assert frame["duration_s"].tolist() == [1.0, 1.0, 2.0]


def test_timeline_shows_only_videos_with_hits():
    fig = timeline_figure(SEGMENTS, [hit(1, "a", 1.5, 2), hit(2, "a", 0.5, 0)])
    bars = [t for t in fig.data if t.type == "bar"]
    assert {t.name for t in bars} == {"carry", "slip"}
    markers = fig.data[-1]
    assert list(markers.marker.symbol) == ["circle", "x"]


def test_confusion_figure_renders_counts():
    table = pd.DataFrame([[1, 0], [1, 1]], index=["drop", "slip"], columns=["drop", "slip"])
    assert confusion_figure(table).data[0].z.tolist() == [[1, 0], [1, 1]]
