"""UMAP projection and Plotly figures.

UMAP is a non-linear 2-D view: distances in the plot are *not* the
high-dimensional semantic distances. Figures label this explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from embeddinggemma_lab.ingestion import Item

UMAP_NOTE = "UMAP は非線形な 2 次元射影です。点間距離を高次元の意味的距離と断定しないでください。"


def umap_2d(vectors: np.ndarray, n_neighbors: int = 15, seed: int = 42) -> np.ndarray:
    """Cosine UMAP to 2-D; small corpora get a proportionally smaller neighbourhood."""
    import umap

    n = len(vectors)
    if n < 4:
        return np.zeros((n, 2), dtype=np.float32)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=max(2, min(n_neighbors, n - 1)),
        metric="cosine",
        random_state=seed,
        init="random" if n < 10 else "spectral",
    )
    return reducer.fit_transform(np.asarray(vectors, dtype=np.float32))


def points_frame(coords: np.ndarray, items: Sequence[Item]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "id": [x.item_id for x in items],
            "label": [x.label or "(none)" for x in items],
            "modality": [x.modality for x in items],
            "content": [(x.title or x.display)[:80] for x in items],
        }
    )


def scatter_figure(points: pd.DataFrame, color_by: str = "label") -> go.Figure:
    fig = px.scatter(
        points,
        x="x",
        y="y",
        color=color_by,
        symbol="modality",
        hover_data={"id": True, "content": True, "x": False, "y": False},
        custom_data=["id"],
    )
    fig.update_traces(marker={"size": 10, "line": {"width": 0.5, "color": "white"}})
    fig.update_layout(
        xaxis_title=None, yaxis_title=None, legend_title=color_by, height=520, dragmode="lasso"
    )
    return fig


def order_by_label(items: Sequence[Item]) -> list[int]:
    return sorted(range(len(items)), key=lambda i: (items[i].label or "", items[i].item_id))


def heatmap_figure(matrix: np.ndarray, items: Sequence[Item]) -> go.Figure:
    order = order_by_label(items)
    names = [f"{items[i].label or '-'} · {items[i].item_id}" for i in order]
    fig = go.Figure(
        go.Heatmap(
            z=matrix[np.ix_(order, order)],
            x=names,
            y=names,
            colorscale="Viridis",
            colorbar={"title": "cos"},
        )
    )
    fig.update_layout(height=650, yaxis={"autorange": "reversed"}, xaxis={"showticklabels": False})
    return fig


def rank_bump_figure(ranks: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Rank of each item across dims, for items in any dim's top-n."""
    shown = ranks[(ranks <= top_n).any(axis=1)]
    dims = [c.removeprefix("rank@") for c in ranks.columns]
    fig = go.Figure()
    for item_id, row in shown.iterrows():
        fig.add_trace(go.Scatter(x=dims, y=row.values, mode="lines+markers", name=str(item_id)))
    fig.update_layout(
        yaxis={"autorange": "reversed", "title": "rank"},
        xaxis={"title": "dim", "type": "category"},
        height=450,
    )
    return fig
