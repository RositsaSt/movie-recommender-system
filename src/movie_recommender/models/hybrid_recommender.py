"""Hybrid recommender combining semantic embeddings with structured metadata features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from movie_recommender.features.feature_builder import build_hybrid_features
from movie_recommender.models.base import BaseRecommender


class HybridRecommender(BaseRecommender):
    """Recommends movies by combining plot semantics with structured metadata.

    Feature vector per movie:
        [text_weight × L2_norm(embedding) | metadata_weight × L2_norm(metadata)]

    where metadata = genre one-hot + language encoding + scaled year + scaled runtime.

    Cosine similarity on this concatenated vector gives a weighted blend of
    semantic plot similarity and structural similarity.
    """

    name = "hybrid"

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        text_weight: float = 0.70,
        metadata_weight: float = 0.30,
    ) -> None:
        self._embedding_model = embedding_model
        self._text_weight = text_weight
        self._metadata_weight = metadata_weight

    def fit(
        self,
        df: pd.DataFrame,
        embeddings: np.ndarray | None = None,
        show_progress: bool = False,
    ) -> HybridRecommender:
        """Build the combined feature matrix and fit the KNN index.

        Args:
            df: Raw Letterboxd DataFrame from load_letterboxd().
            embeddings: Optional pre-computed embedding matrix (n × dim).
                Pass this to skip sentence-transformer inference (e.g. in tests).
            show_progress: Show tqdm progress bar during embedding generation.
        """
        self._df = df.reset_index(drop=True)
        texts = self._df["description"].fillna("").tolist()

        self._feature_matrix = build_hybrid_features(
            texts=texts,
            df=self._df,
            embeddings=embeddings,
            embedding_model=self._embedding_model,
            text_weight=self._text_weight,
            metadata_weight=self._metadata_weight,
            show_progress=show_progress,
        )

        self._knn = NearestNeighbors(metric="cosine", algorithm="brute")
        self._knn.fit(self._feature_matrix)
        return self

    def recommend(self, title: str, top_k: int = 10) -> list[dict]:
        """Return top_k movies most similar to *title* via the hybrid feature space.

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
            self._feature_matrix[idx].reshape(1, -1), n_neighbors=top_k + 1
        )

        results = []
        for dist, i in zip(distances.flatten()[1:], indices.flatten()[1:]):
            row = self._df.iloc[i]
            results.append({**self._row_to_dict(row), "score": round(1.0 - float(dist), 4)})

        return results
