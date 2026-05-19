"""Genre, language, year, and runtime feature encoding for the hybrid recommender."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, RobustScaler


def encode_genres_matrix(df: pd.DataFrame, column: str = "genre_list") -> np.ndarray:
    """Return a binary genre matrix from a space-separated genre string column."""
    genres = df[column].fillna("").astype(str).str.split()
    mlb = MultiLabelBinarizer()
    return mlb.fit_transform(genres).astype(float)


def encode_language_vector(df: pd.DataFrame, column: str = "language") -> np.ndarray:
    """Return a MinMax-scaled integer encoding of the primary language."""
    lang = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .str.split(r"[,/;|]", regex=True)
        .str[0]
        .str.strip()
    )
    encoded = LabelEncoder().fit_transform(lang).reshape(-1, 1).astype(float)
    return MinMaxScaler().fit_transform(encoded)


def encode_year_vector(df: pd.DataFrame, column: str = "year") -> np.ndarray:
    """Return a RobustScaler-normalized year vector."""
    values = pd.to_numeric(df[column], errors="coerce").fillna(df[column].median())
    return RobustScaler().fit_transform(values.to_numpy().reshape(-1, 1))


def encode_runtime_vector(df: pd.DataFrame, column: str = "minute") -> np.ndarray:
    """Return a MinMax-scaled runtime vector."""
    values = pd.to_numeric(df[column], errors="coerce").fillna(df[column].median())
    return MinMaxScaler().fit_transform(values.to_numpy().reshape(-1, 1))


def build_metadata_matrix(
    df: pd.DataFrame,
    use_genres: bool = True,
    use_language: bool = True,
    use_year: bool = True,
    use_runtime: bool = True,
) -> np.ndarray:
    """Combine selected structured features into a single metadata matrix.

    All feature blocks are normalized before stacking so no single feature
    dominates purely due to scale.
    """
    parts: list[np.ndarray] = []

    if use_genres and "genre_list" in df.columns:
        parts.append(encode_genres_matrix(df))
    if use_language and "language" in df.columns:
        parts.append(encode_language_vector(df))
    if use_year and "year" in df.columns:
        parts.append(encode_year_vector(df))
    if use_runtime and "minute" in df.columns:
        parts.append(encode_runtime_vector(df))

    if not parts:
        return np.zeros((len(df), 1))

    return np.hstack(parts)
