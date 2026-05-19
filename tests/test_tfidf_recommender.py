"""Tests for the TF-IDF content-based recommender."""

import pandas as pd
import pytest

from movie_recommender.models.popularity_recommender import PopularityRecommender
from movie_recommender.models.tfidf_recommender import TFIDFRecommender


def _sample_df(n: int = 12) -> pd.DataFrame:
    descriptions = [
        "A thrilling war drama set during World War II with intense battle scenes.",
        "A romantic comedy about two strangers falling in love in Paris.",
        "A sci-fi adventure where astronauts battle aliens in deep space.",
        "A dark psychological thriller about a detective hunting a serial killer.",
        "An animated fantasy film featuring magical creatures and a young hero.",
        "A historical drama about the French Revolution and its consequences.",
        "A horror film set in an isolated haunted mansion with mysterious secrets.",
        "A coming-of-age story about teenagers navigating high school and first love.",
        "A war epic about soldiers surviving behind enemy lines in Europe.",
        "A sci-fi thriller where humanity faces extinction from an alien invasion.",
        "A romantic drama about star-crossed lovers separated by social class.",
        "A crime thriller about a heist gone wrong in a major city.",
    ]
    return pd.DataFrame({
        "id": range(n),
        "name": [f"Movie {i}" for i in range(n)],
        "key": [f"Movie {i} (200{i})" for i in range(n)],
        "year": [2000 + i for i in range(n)],
        "minute": [90 + i for i in range(n)],
        "description": descriptions,
        "genre_list": ["action war", "comedy romance", "sci_fi action", "thriller drama",
                        "animation fantasy", "drama history", "horror mystery",
                        "drama romance", "action war drama", "sci_fi thriller",
                        "romance drama", "crime thriller"],
        "language": ["English"] * n,
        "combined_rating": [3.0 + (i % 5) * 0.5 for i in range(n)],
    })


# ---------------------------------------------------------------------------
# TFIDFRecommender
# ---------------------------------------------------------------------------

class TestTFIDFRecommender:
    def test_fit_returns_self(self):
        rec = TFIDFRecommender()
        result = rec.fit(_sample_df())
        assert result is rec

    def test_recommend_returns_list(self):
        rec = TFIDFRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)")
        assert isinstance(recs, list)

    def test_recommend_correct_count(self):
        rec = TFIDFRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=5)
        assert len(recs) == 5

    def test_recommend_excludes_query_movie(self):
        rec = TFIDFRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=5)
        keys = [r["key"] for r in recs]
        assert "Movie 0 (2000)" not in keys

    def test_recommend_has_score_field(self):
        rec = TFIDFRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=3)
        for r in recs:
            assert "score" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_recommend_war_movie_favours_war(self):
        """War-description movie should rank another war movie in top results."""
        rec = TFIDFRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=10)  # war drama
        keys = [r["key"] for r in recs]
        # Movie 8 is also a war film; it must appear and rank above the romance film
        assert "Movie 8 (2008)" in keys
        if "Movie 1 (2001)" in keys:
            assert keys.index("Movie 8 (2008)") < keys.index("Movie 1 (2001)")

    def test_recommend_unknown_title_raises(self):
        rec = TFIDFRecommender().fit(_sample_df())
        with pytest.raises(ValueError, match="not found"):
            rec.recommend("Nonexistent Movie (1900)")

    def test_recommend_before_fit_raises(self):
        rec = TFIDFRecommender()
        with pytest.raises(RuntimeError):
            rec.recommend("Movie 0 (2000)")

    def test_name_attribute(self):
        assert TFIDFRecommender.name == "tfidf"


# ---------------------------------------------------------------------------
# PopularityRecommender
# ---------------------------------------------------------------------------

class TestPopularityRecommender:
    def test_fit_returns_self(self):
        rec = PopularityRecommender()
        assert rec.fit(_sample_df()) is rec

    def test_recommend_correct_count(self):
        rec = PopularityRecommender().fit(_sample_df())
        assert len(rec.recommend("Movie 0 (2000)", top_k=5)) == 5

    def test_recommend_excludes_query(self):
        rec = PopularityRecommender().fit(_sample_df())
        keys = [r["key"] for r in rec.recommend("Movie 0 (2000)")]
        assert "Movie 0 (2000)" not in keys

    def test_recommend_sorted_by_rating(self):
        rec = PopularityRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=5)
        scores = [r["score"] for r in recs]
        assert scores == sorted(scores, reverse=True)

    def test_genre_filter(self):
        rec = PopularityRecommender().fit(_sample_df())
        recs = rec.recommend("Movie 0 (2000)", top_k=10, genre="romance")
        assert all("romance" in r["genres"] or "Romance" in r["genres"] for r in recs)

    def test_name_attribute(self):
        assert PopularityRecommender.name == "popularity"
