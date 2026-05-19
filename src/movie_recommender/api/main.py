"""FastAPI application with health, search, and recommendation endpoints."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from movie_recommender.config import settings
from movie_recommender.models.base import BaseRecommender

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_registry(app: FastAPI) -> None:
    """Load fitted models from model_dir at startup. Missing files are skipped."""
    import joblib

    model_dir = settings.model_dir
    registry: dict[str, BaseRecommender] = {}

    for name in ("popularity", "tfidf", "hybrid"):
        path = model_dir / f"{name}.joblib"
        if path.exists():
            registry[name] = joblib.load(path)
            logger.info("Loaded model: %s", name)
        else:
            logger.warning("Model file not found, skipping: %s", path)

    df_path = model_dir / "movies.joblib"
    app.state.df = joblib.load(df_path) if df_path.exists() else None
    app.state.registry = registry
    app.state.ready = bool(registry)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_registry(app)
    except Exception as exc:
        logger.warning("Model loading failed: %s — run scripts/train.py first.", exc)
        app.state.registry = {}
        app.state.df = None
        app.state.ready = False
    yield


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Movie Recommender API",
    description="Compares popularity, TF-IDF, and hybrid semantic recommenders.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

def _get_registry(model: str) -> BaseRecommender:
    """Resolve a model name to a fitted recommender, raising HTTP errors on failure."""
    if not app.state.ready:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run scripts/train.py to train and save models.",
        )
    registry: dict[str, BaseRecommender] = app.state.registry
    if model not in registry:
        available = sorted(registry.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model}' not available. Available: {available}",
        )
    return registry[model]


def _get_df() -> pd.DataFrame:
    if app.state.df is None:
        raise HTTPException(status_code=503, detail="Movie catalogue not loaded.")
    return app.state.df


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["meta"])
def health():
    """Returns service status and names of loaded models."""
    return {
        "status": "ok",
        "models_loaded": sorted(app.state.registry.keys()) if app.state.ready else [],
        "ready": app.state.ready,
    }


@app.get("/movies/search", tags=["movies"])
def search_movies(
    query: Annotated[str, Query(min_length=1, description="Partial title to search for")],
    limit: Annotated[int, Query(ge=1, le=50)] = 10,
    df: pd.DataFrame = Depends(_get_df),
):
    """Return movies whose title contains *query* (case-insensitive)."""
    mask = df["name"].str.contains(query, case=False, na=False)
    hits = df[mask].head(limit)
    return [
        {
            "key": row["key"],
            "title": row["name"],
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": row.get("genre_list", ""),
            "language": row.get("language", ""),
            "rating": round(float(row["combined_rating"]), 2)
            if pd.notna(row.get("combined_rating"))
            else None,
        }
        for _, row in hits.iterrows()
    ]


@app.get("/recommendations/by-title", tags=["recommendations"])
def recommend_by_title(
    title: Annotated[str, Query(min_length=1, description="Exact key, e.g. 'Parasite (2019)'")],
    model: Annotated[str, Query(description="popularity | tfidf | hybrid")] = "hybrid",
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """Return top_k recommendations for the given movie title using the specified model."""
    recommender = _get_registry(model)
    try:
        results = recommender.recommend(title, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "query_title": title,
        "model": model,
        "top_k": top_k,
        "recommendations": results,
    }


@app.get("/recommendations/{movie_key:path}", tags=["recommendations"])
def recommend_by_key(
    movie_key: str,
    model: Annotated[str, Query(description="popularity | tfidf | hybrid")] = "hybrid",
    top_k: Annotated[int, Query(ge=1, le=50)] = 10,
):
    """Same as /recommendations/by-title but the title is passed as a path segment.

    URL-encode the title, e.g. ``/recommendations/Parasite%20(2019)``.
    """
    recommender = _get_registry(model)
    try:
        results = recommender.recommend(movie_key, top_k=top_k)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {
        "query_title": movie_key,
        "model": model,
        "top_k": top_k,
        "recommendations": results,
    }
