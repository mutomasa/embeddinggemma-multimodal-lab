"""Turning service results into display tables."""

from __future__ import annotations

import pandas as pd

from embeddinggemma_lab.search_service import RankedItem

SCORE_HELP = "コサイン類似度（関連度の指標であり、正解確率ではありません）"


def results_frame(results: list[RankedItem]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "rank": r.hit.rank,
                "score": round(r.hit.score, 4),
                "id": r.item.item_id,
                "label": r.item.label or "",
                "title": r.item.title or "",
                "content": r.item.display,
            }
            for r in results
        ]
    )
