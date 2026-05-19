"""Central configuration loaded from environment variables or .env file."""

from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    data_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    model_dir: Path = Path("models")
    sample_dir: Path = Path("data/sample")

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"

    # Recommender defaults
    default_top_k: int = 10
    tfidf_max_features: int = 10_000
    embedding_model: str = "all-MiniLM-L6-v2"

    # Reranker weights
    rerank_similarity_weight: float = 0.70
    rerank_rating_weight: float = 0.20
    rerank_popularity_weight: float = 0.10

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
