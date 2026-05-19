"""Thin HTTP client for the Movie Recommender API."""

from __future__ import annotations

import requests

from movie_recommender.config import settings

_BASE = f"http://{settings.api_host}:{settings.api_port}"


def _get(path: str, params: dict | None = None, timeout: int = 10) -> dict | list:
    url = f"{_BASE}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def health() -> dict:
    return _get("/health")


def search_movies(query: str, limit: int = 20) -> list[dict]:
    return _get("/movies/search", params={"query": query, "limit": limit})


def recommend(title: str, model: str = "hybrid", top_k: int = 10) -> dict:
    return _get(
        "/recommendations/by-title",
        params={"title": title, "model": model, "top_k": top_k},
    )
