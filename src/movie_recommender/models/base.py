"""Abstract base class shared by all recommender models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseRecommender(ABC):
    """All recommenders must implement fit() and recommend()."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in API responses and logs."""

    @abstractmethod
    def fit(self, df: pd.DataFrame) -> BaseRecommender:
        """Fit the model on a raw Letterboxd DataFrame from load_letterboxd()."""

    @abstractmethod
    def recommend(self, title: str, top_k: int = 10) -> list[dict]:
        """Return top_k recommendations for *title* as a list of dicts.

        Each dict contains at minimum: title, year, genres, language, rating, score.
        """

    def _require_fitted(self) -> None:
        if not hasattr(self, "_df") or self._df is None:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling recommend()."
            )

    @staticmethod
    def _row_to_dict(row: pd.Series) -> dict:
        return {
            "title": row.get("name", ""),
            "key": row.get("key", ""),
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres": row.get("genre_list", ""),
            "language": row.get("language", ""),
            "rating": round(float(row["combined_rating"]), 2)
            if pd.notna(row.get("combined_rating"))
            else None,
            "description": str(row.get("description", ""))[:300],
        }
