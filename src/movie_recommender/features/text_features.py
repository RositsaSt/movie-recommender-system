"""TF-IDF vectorization and sentence-embedding generation for movie descriptions."""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_matrix(
    texts: list[str],
    max_features: int = 10_000,
) -> tuple:
    """Fit a TF-IDF vectorizer on *texts* and return (sparse_matrix, vectorizer)."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return matrix, vectorizer


def build_embeddings(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = False,
) -> np.ndarray:
    """Generate L2-normalized sentence embeddings using a SentenceTransformer model.

    Requires the ``sentence-transformers`` optional dependency:
        pip install "movie-recommender-system[ml]"
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for embedding-based recommenders. "
            'Install it with: pip install "movie-recommender-system[ml]"'
        ) from exc

    model = SentenceTransformer(model_name)
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
