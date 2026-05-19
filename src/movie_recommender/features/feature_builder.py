"""Combines text and metadata features into a single matrix for nearest-neighbor search."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import normalize

from movie_recommender.features.metadata_features import build_metadata_matrix
from movie_recommender.features.text_features import build_embeddings, build_tfidf_matrix


def build_tfidf_features(
    texts: list[str],
    max_features: int = 10_000,
) -> tuple:
    """Build a sparse TF-IDF matrix suitable for cosine-similarity KNN.

    Returns (sparse_matrix, vectorizer).
    """
    return build_tfidf_matrix(texts, max_features=max_features)


def build_hybrid_features(
    texts: list[str],
    df,
    embeddings: np.ndarray | None = None,
    embedding_model: str = "all-MiniLM-L6-v2",
    text_weight: float = 0.70,
    metadata_weight: float = 0.30,
    show_progress: bool = False,
) -> np.ndarray:
    """Build the combined feature matrix used by the hybrid recommender.

    Strategy:
        1. Produce L2-normalized sentence embeddings (or accept pre-computed ones).
        2. Produce a normalized metadata matrix (genre, language, year, runtime).
        3. Scale each block by its weight and concatenate.

    Cosine similarity on the resulting concatenated vector approximates a weighted
    combination of embedding similarity and metadata similarity.

    Args:
        texts: Raw movie descriptions in DataFrame order.
        df: Raw Letterboxd DataFrame (used for metadata columns).
        embeddings: Pre-computed embedding matrix (n_movies × dim). If None,
            embeddings are generated from *texts* using *embedding_model*.
        text_weight: Contribution weight of the embedding block (0–1).
        metadata_weight: Contribution weight of the metadata block (0–1).
    """
    if embeddings is None:
        embeddings = build_embeddings(
            texts, model_name=embedding_model, show_progress=show_progress
        )

    meta = build_metadata_matrix(df)

    emb_norm = normalize(embeddings, norm="l2")
    meta_norm = normalize(meta, norm="l2")

    return np.hstack([emb_norm * text_weight, meta_norm * metadata_weight])
