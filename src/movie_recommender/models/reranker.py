"""Optional reranker that blends similarity score, rating, and popularity."""

from __future__ import annotations

import numpy as np

from movie_recommender.config import settings


def rerank(
    candidates: list[dict],
    similarity_weight: float | None = None,
    rating_weight: float | None = None,
    popularity_weight: float | None = None,
) -> list[dict]:
    """Rerank recommendation candidates using a weighted blend of three signals.

    Weights default to the values in ``settings`` (configurable via .env).

    Formula:
        final_score = w_sim * sim + w_rat * norm_rating + w_pop * norm_popularity

    Args:
        candidates: List of dicts from a recommender's recommend() call.
            Each dict must contain ``score`` and optionally ``rating``.
        similarity_weight: Override for the similarity score weight.
        rating_weight: Override for the rating weight.
        popularity_weight: Override for the popularity weight (currently unused —
            extend with a ``popularity`` field if ratings-count data is available).

    Returns:
        Candidates sorted by final_score descending, with ``final_score`` added.
    """
    w_sim = similarity_weight if similarity_weight is not None else settings.rerank_similarity_weight
    w_rat = rating_weight if rating_weight is not None else settings.rerank_rating_weight
    w_pop = popularity_weight if popularity_weight is not None else settings.rerank_popularity_weight

    if not candidates:
        return []

    ratings = np.array([c.get("rating") or 0.0 for c in candidates], dtype=float)
    r_min, r_max = ratings.min(), ratings.max()
    norm_ratings = (ratings - r_min) / (r_max - r_min + 1e-9)

    results = []
    for i, candidate in enumerate(candidates):
        sim = float(candidate.get("score", 0.0))
        final = w_sim * sim + w_rat * float(norm_ratings[i]) + w_pop * 0.0
        results.append({**candidate, "final_score": round(final, 4)})

    return sorted(results, key=lambda x: x["final_score"], reverse=True)
