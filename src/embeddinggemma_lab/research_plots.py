"""Plotly figures for the research demo: label timelines and confusion heatmaps."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from embeddinggemma_lab.annotations import EventSegment
from embeddinggemma_lab.label_matching import LabeledHit


def segments_frame(segments: Sequence[EventSegment]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": [s.video_id for s in segments],
            "event": [s.event for s in segments],
            "start_s": [s.start_s for s in segments],
            "duration_s": [s.end_s - s.start_s for s in segments],
        }
    )


def _segment_bars(fig: go.Figure, segments: pd.DataFrame) -> None:
    for event, rows in segments.groupby("event", sort=True):
        fig.add_bar(
            name=str(event),
            y=rows["video_id"],
            x=rows["duration_s"],
            base=rows["start_s"],
            orientation="h",
            opacity=0.55,
            hovertemplate="%{y}<br>" + f"{event}" + ": %{base:.2f}s +%{x:.2f}s<extra></extra>",
        )


def _hit_markers(fig: go.Figure, hits: Sequence[LabeledHit]) -> None:
    fig.add_scatter(
        name="検索結果フレーム",
        x=[h.timestamp_s for h in hits],
        y=[h.video_id for h in hits],
        mode="markers+text",
        text=[f"#{h.rank}" for h in hits],
        textposition="top center",
        marker={
            "size": 12,
            "symbol": ["circle" if h.matched else "x" for h in hits],
            "color": ["#1a7f37" if h.matched else "#cf222e" for h in hits],
            "line": {"width": 1, "color": "white"},
        },
        hovertemplate="%{y} @ %{x:.2f}s<extra></extra>",
    )


def timeline_figure(segments: Sequence[EventSegment], hits: Sequence[LabeledHit]) -> go.Figure:
    """Human-labelled segments per video, with retrieved frames as ○ (match) / × (miss)."""
    videos = sorted({h.video_id for h in hits})
    frame = segments_frame([s for s in segments if s.video_id in videos])
    fig = go.Figure()
    _segment_bars(fig, frame)
    _hit_markers(fig, hits)
    fig.update_layout(
        barmode="overlay",
        height=120 + 45 * len(videos),
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        xaxis_title="時刻 (s)",
        legend_title="人手ラベル",
    )
    return fig


def confusion_figure(table: pd.DataFrame) -> go.Figure:
    fig = px.imshow(
        table,
        text_auto=True,
        color_continuous_scale="Blues",
        labels={"x": "1 位フレームの人手ラベル", "y": "クエリの対象イベント", "color": "件数"},
        aspect="auto",
    )
    fig.update_layout(margin={"l": 10, "r": 10, "t": 30, "b": 10})
    return fig
