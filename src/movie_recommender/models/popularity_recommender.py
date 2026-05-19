"""Popularity/rating baseline recommender."""

from __future__ import annotations

import pandas as pd

from movie_recommender.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """Recommends top-rated movies, with optional genre/language filtering.

    This is the simplest possible baseline: no personalisation, no similarity
    computation — just a sorted list of highly-rated movies.
    """

    name = "popularity"

    def fit(self, df: pd.DataFrame) -> PopularityRecommender:
        """Store the dataset; no training required."""
        self._df = df.reset_index(drop=True)
        return self

    def recommend(
        self,
        title: str,
        top_k: int = 10,
        genre: str | None = None,
        language: str | None = None,
    ) -> list[dict]:
        """Return the top_k highest-rated movies, excluding *title* itself.

        Args:
            title: The ``key`` value of the seed movie (e.g. "Parasite (2019)").
            top_k: Number of recommendations to return.
            genre: Optional genre substring filter (case-insensitive).
            language: Optional language substring filter (case-insensitive).
        """
        self._require_fitted()

        candidates = self._df[self._df["key"] != title].copy()

        if genre:
            candidates = candidates[
                candidates["genre_list"].str.contains(genre, case=False, na=False)
            ]
        if language:
            candidates = candidates[
                candidates["language"].str.contains(language, case=False, na=False)
            ]

        top = candidates.sort_values("combined_rating", ascending=False).head(top_k)

        return [
            {**self._row_to_dict(row), "score": round(float(row["combined_rating"]), 4)}
            for _, row in top.iterrows()
        ]
