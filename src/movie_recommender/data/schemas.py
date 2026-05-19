"""Pydantic schemas for validated movie records and API responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Movie(BaseModel):
    """A single movie record as returned by the API."""

    id: int
    name: str
    year: int | None = None
    minute: int | None = None
    description: str = ""
    genre_list: str = ""
    language: str = ""
    rating: float | None = None
    combined_rating: float | None = None
    tagline: str | None = None
    poster: str | None = None


class RecommendationItem(BaseModel):
    """One recommended movie with its similarity score."""

    movie: Movie
    score: float = Field(ge=0.0, le=1.0, description="Similarity/ranking score")
    reason: str | None = None


class RecommendationResponse(BaseModel):
    """Full response from a recommendation endpoint."""

    query_title: str
    model: str
    top_k: int
    recommendations: list[RecommendationItem]
