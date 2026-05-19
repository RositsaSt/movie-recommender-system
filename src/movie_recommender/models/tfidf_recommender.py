"""Content-based recommender using TF-IDF vectors and cosine similarity."""

from __future__ import annotations

import pandas as pd
from sklearn.neighbors import NearestNeighbors

from movie_recommender.features.text_features import build_tfidf_matrix
from movie_recommender.models.base import BaseRecommender


class TFIDFRecommender(BaseRecommender):
    """Recommends movies with similar descriptions using TF-IDF + cosine KNN.

    Fit once on the full movie catalogue; retrieve recommendations in O(n) time
    (brute-force cosine — fast enough for the Letterboxd dataset size).
    """

    name = "tfidf"

    def __init__(self, max_features: int = 10_000) -> None:
        self._max_features = max_features

    def fit(self, df: pd.DataFrame) -> "TFIDFRecommender":
        """Vectorize descriptions and fit the KNN index."""
        self._df = df.reset_index(drop=True)
        texts = self._df["description"].fillna("").tolist()
        self._matrix, self._vectorizer = build_tfidf_matrix(texts, self._max_features)
        self._knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._knn.fit(self._matrix)
        return self

    def recommend(self, title: str, top_k: int = 10) -> list[dict]:
        """Return top_k movies most similar in description to *title*.

        Args:
            title: The ``key`` value of the seed movie (e.g. "Parasite (2019)").
            top_k: Number of recommendations to return.

        Raises:
            ValueError: If *title* is not found in the fitted catalogue.
        """
        self._require_fitted()

        matches = self._df[self._df["key"] == title]
        if matches.empty:
            raise ValueError(f"Movie '{title}' not found in the catalogue.")

        idx = matches.index[0]
        distances, indices = self._knn.kneighbors(
            self._matrix[idx], n_neighbors=top_k + 1
        )

        results = []
        for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
            row = self._df.iloc[i]
            results.append({**self._row_to_dict(row), "score": round(1.0 - float(dist), 4)})

        return results
