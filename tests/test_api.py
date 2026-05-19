"""Tests for the FastAPI recommendation API.

All tests use in-memory models fitted on a tiny sample DataFrame.
No trained model files or Kaggle data are required.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from movie_recommender.api.main import app
from movie_recommender.models.hybrid_recommender import HybridRecommender
from movie_recommender.models.popularity_recommender import PopularityRecommender
from movie_recommender.models.tfidf_recommender import TFIDFRecommender

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _sample_df(n: int = 14) -> pd.DataFrame:
    descriptions = [
        "A war epic set during World War II with intense battlefield scenes.",
        "A romantic comedy about two strangers falling in love in Paris.",
        "A sci-fi thriller where humanity battles an alien invasion.",
        "A psychological thriller about a detective hunting a serial killer.",
        "An animated fantasy featuring magical creatures and a young hero.",
        "A historical drama set during the French Revolution.",
        "A horror film in an isolated haunted mansion.",
        "A coming-of-age story about teenagers and first love.",
        "Another war drama following soldiers behind enemy lines.",
        "A sci-fi adventure where astronauts explore a distant planet.",
        "A crime thriller about a heist gone wrong.",
        "A romantic drama about star-crossed lovers.",
        "A dark comedy about a dysfunctional family reunion.",
        "A biography about a pioneering scientist changing the world.",
    ]
    return pd.DataFrame({
        "id": range(n),
        "name": [f"Movie {i}" for i in range(n)],
        "key": [f"Movie {i} (200{i})" for i in range(n)],
        "year": [2000 + i for i in range(n)],
        "minute": [90 + i * 3 for i in range(n)],
        "description": descriptions,
        "genre_list": [
            "action war", "comedy romance", "sci_fi action", "thriller drama",
            "animation fantasy", "drama history", "horror mystery", "drama romance",
            "action war drama", "sci_fi adventure", "crime thriller", "romance drama",
            "comedy drama", "biography drama",
        ],
        "language": (["English"] * 8) + (["French"] * 3) + (["Spanish"] * 3),
        "combined_rating": [3.0 + (i % 5) * 0.4 for i in range(n)],
    })


def _random_embeddings(n: int = 14, dim: int = 32) -> np.ndarray:
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((n, dim)).astype(np.float32)
    return emb / np.linalg.norm(emb, axis=1, keepdims=True)


@pytest.fixture(scope="module")
def client():
    df = _sample_df()
    emb = _random_embeddings(len(df))
    registry = {
        "popularity": PopularityRecommender().fit(df),
        "tfidf": TFIDFRecommender().fit(df),
        "hybrid": HybridRecommender().fit(df, embeddings=emb),
    }
    with TestClient(app) as c:
        # Inject state AFTER lifespan has run (lifespan resets state at startup)
        app.state.registry = registry
        app.state.df = df
        app.state.ready = True
        yield c


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_status_ok(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_lists_loaded_models(self, client):
        models = client.get("/health").json()["models_loaded"]
        assert set(models) == {"popularity", "tfidf", "hybrid"}

    def test_ready_true(self, client):
        assert client.get("/health").json()["ready"] is True


# ---------------------------------------------------------------------------
# /movies/search
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_list(self, client):
        r = client.get("/movies/search", params={"query": "Movie"})
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_case_insensitive(self, client):
        upper = client.get("/movies/search", params={"query": "MOVIE 0"}).json()
        lower = client.get("/movies/search", params={"query": "movie 0"}).json()
        assert upper == lower

    def test_limit_respected(self, client):
        r = client.get("/movies/search", params={"query": "Movie", "limit": 3})
        assert len(r.json()) <= 3

    def test_result_has_required_fields(self, client):
        r = client.get("/movies/search", params={"query": "Movie 1"})
        item = r.json()[0]
        for field in ("key", "title", "year", "genres", "language", "rating"):
            assert field in item

    def test_empty_query_rejected(self, client):
        r = client.get("/movies/search", params={"query": ""})
        assert r.status_code == 422

    def test_no_match_returns_empty_list(self, client):
        r = client.get("/movies/search", params={"query": "xyzzy_no_match_12345"})
        assert r.json() == []


# ---------------------------------------------------------------------------
# /recommendations/by-title
# ---------------------------------------------------------------------------

class TestRecommendByTitle:
    @pytest.mark.parametrize("model", ["popularity", "tfidf", "hybrid"])
    def test_all_models_return_results(self, client, model):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": model, "top_k": 5},
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["recommendations"]) == 5

    def test_response_structure(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": "tfidf", "top_k": 3},
        )
        data = r.json()
        assert data["query_title"] == "Movie 0 (2000)"
        assert data["model"] == "tfidf"
        assert data["top_k"] == 3

    def test_recommendations_exclude_query(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": "hybrid", "top_k": 5},
        )
        keys = [rec["key"] for rec in r.json()["recommendations"]]
        assert "Movie 0 (2000)" not in keys

    def test_unknown_title_returns_404(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Ghost Film (1800)", "model": "tfidf"},
        )
        assert r.status_code == 404

    def test_unknown_model_returns_404(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": "svd"},
        )
        assert r.status_code == 404

    def test_top_k_default_is_10(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": "tfidf"},
        )
        assert len(r.json()["recommendations"]) == 10

    def test_top_k_above_limit_rejected(self, client):
        r = client.get(
            "/recommendations/by-title",
            params={"title": "Movie 0 (2000)", "model": "tfidf", "top_k": 999},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Unloaded-state guard
# ---------------------------------------------------------------------------

class TestUnloadedState:
    def test_503_when_not_ready(self):
        app.state.ready = False
        with TestClient(app) as c:
            r = c.get(
                "/recommendations/by-title",
                params={"title": "Movie 0 (2000)", "model": "tfidf"},
            )
        assert r.status_code == 503
        app.state.ready = True  # restore for other tests
