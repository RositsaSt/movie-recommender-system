"""Tests for the hybrid semantic recommender.

Sentence-transformer inference is bypassed by passing pre-computed random
embeddings directly to fit(), so these tests run without the [ml] extras.
"""

import numpy as np
import pandas as pd
import pytest

from movie_recommender.models.hybrid_recommender import HybridRecommender
from movie_recommender.models.reranker import rerank


def _sample_df(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({
        "id": range(n),
        "name": [f"Movie {i}" for i in range(n)],
        "key": [f"Movie {i} (200{i})" for i in range(n)],
        "year": [2000 + i for i in range(n)],
        "minute": [90 + i * 5 for i in range(n)],
        "description": [f"Description of movie {i} about theme {i % 3}." for i in range(n)],
        "genre_list": ["action thriller", "comedy romance", "drama", "sci_fi action",
                        "horror", "animation", "thriller", "romance drama",
                        "action", "sci_fi", "comedy", "drama thriller"],
        "language": (["English"] * 6) + (["French"] * 3) + (["Spanish"] * 3),
        "combined_rating": [3.0 + (i % 5) * 0.4 for i in range(n)],
    })


def _random_embeddings(n: int = 12, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(42)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / norms


# ---------------------------------------------------------------------------
# HybridRecommender
# ---------------------------------------------------------------------------

class TestHybridRecommender:
    def _fitted(self, n: int = 12) -> HybridRecommender:
        df = _sample_df(n)
        emb = _random_embeddings(n)
        return HybridRecommender().fit(df, embeddings=emb)

    def test_fit_returns_self(self):
        df = _sample_df()
        emb = _random_embeddings()
        rec = HybridRecommender()
        assert rec.fit(df, embeddings=emb) is rec

    def test_recommend_returns_list(self):
        rec = self._fitted()
        assert isinstance(rec.recommend("Movie 0 (2000)"), list)

    def test_recommend_correct_count(self):
        rec = self._fitted()
        assert len(rec.recommend("Movie 0 (2000)", top_k=5)) == 5

    def test_recommend_excludes_query(self):
        rec = self._fitted()
        keys = [r["key"] for r in rec.recommend("Movie 0 (2000)", top_k=5)]
        assert "Movie 0 (2000)" not in keys

    def test_recommend_score_in_range(self):
        rec = self._fitted()
        for r in rec.recommend("Movie 0 (2000)", top_k=5):
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_recommend_result_keys(self):
        rec = self._fitted()
        r = rec.recommend("Movie 0 (2000)", top_k=1)[0]
        for field in ("title", "key", "year", "genres", "language", "rating", "score"):
            assert field in r, f"Missing field: {field}"

    def test_recommend_unknown_raises(self):
        rec = self._fitted()
        with pytest.raises(ValueError, match="not found"):
            rec.recommend("Ghost Movie (1800)")

    def test_recommend_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            HybridRecommender().recommend("Movie 0 (2000)")

    def test_name_attribute(self):
        assert HybridRecommender.name == "hybrid"

    def test_feature_matrix_shape(self):
        n, dim = 12, 32
        df = _sample_df(n)
        emb = _random_embeddings(n, dim)
        rec = HybridRecommender(text_weight=0.7, metadata_weight=0.3).fit(df, embeddings=emb)
        # feature matrix columns = weighted_emb_cols + weighted_meta_cols
        assert rec._feature_matrix.shape[0] == n


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

class TestReranker:
    def _candidates(self):
        return [
            {"key": "A", "score": 0.9, "rating": 3.0},
            {"key": "B", "score": 0.5, "rating": 5.0},
            {"key": "C", "score": 0.7, "rating": 4.0},
        ]

    def test_returns_same_length(self):
        result = rerank(self._candidates())
        assert len(result) == 3

    def test_adds_final_score(self):
        result = rerank(self._candidates())
        for r in result:
            assert "final_score" in r

    def test_sorted_by_final_score(self):
        result = rerank(self._candidates())
        scores = [r["final_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    def test_high_similarity_wins_with_default_weights(self):
        # Default weights are 0.7 sim, 0.2 rating, 0.1 pop
        # A has highest sim (0.9) and should win despite lower rating
        result = rerank(self._candidates())
        assert result[0]["key"] == "A"

    def test_custom_weights(self):
        # With 100% weight on rating, B (rating=5.0) should come first
        result = rerank(
            self._candidates(), similarity_weight=0.0, rating_weight=1.0, popularity_weight=0.0
        )
        assert result[0]["key"] == "B"

    def test_empty_input(self):
        assert rerank([]) == []
