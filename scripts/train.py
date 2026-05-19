"""Fit all recommender models on the Letterboxd dataset and save to model_dir.

Usage:
    python scripts/train.py

Requires:
    - Raw data at settings.data_dir (download from Kaggle first)
    - pip install -e . (or -e ".[ml]" for the hybrid model)
"""

import logging

import joblib

from movie_recommender.config import settings
from movie_recommender.data.load import load_letterboxd
from movie_recommender.models.hybrid_recommender import HybridRecommender
from movie_recommender.models.popularity_recommender import PopularityRecommender
from movie_recommender.models.tfidf_recommender import TFIDFRecommender

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    model_dir = settings.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading data from %s …", settings.data_dir)
    df = load_letterboxd()
    log.info("Loaded %d movies.", len(df))

    joblib.dump(df, model_dir / "movies.joblib")
    log.info("Saved movie catalogue.")

    log.info("Fitting PopularityRecommender …")
    pop = PopularityRecommender().fit(df)
    joblib.dump(pop, model_dir / "popularity.joblib")

    log.info("Fitting TFIDFRecommender …")
    tfidf = TFIDFRecommender(max_features=settings.tfidf_max_features).fit(df)
    joblib.dump(tfidf, model_dir / "tfidf.joblib")

    log.info("Fitting HybridRecommender (this may take a few minutes) …")
    hybrid = HybridRecommender(
        embedding_model=settings.embedding_model,
        text_weight=1.0 - settings.rerank_metadata_weight
        if hasattr(settings, "rerank_metadata_weight")
        else 0.70,
    ).fit(df, show_progress=True)
    joblib.dump(hybrid, model_dir / "hybrid.joblib")

    log.info("All models saved to %s", model_dir)


if __name__ == "__main__":
    main()
