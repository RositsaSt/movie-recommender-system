"""Functions for loading raw Letterboxd dataset files into a single DataFrame."""

from pathlib import Path

import numpy as np
import pandas as pd

from movie_recommender.config import settings


def _dir(data_dir: Path | None) -> Path:
    return Path(data_dir) if data_dir is not None else settings.data_dir


def load_movies(data_dir: Path | None = None) -> pd.DataFrame:
    """Load movies.csv and apply basic structural cleaning."""
    path = _dir(data_dir)
    df = pd.read_csv(path / "movies.csv")
    df = df.drop(columns=["tagline"], errors="ignore")
    df = df[df["name"].notnull() & ~df["name"].isin(["", "No Title"])]
    df = df[df["description"].notnull() & (df["description"] != "")]
    df = df.rename(columns={"date": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["minute"])
    df["minute"] = df["minute"].astype(int)
    df = df[(df["minute"] > 40) & (df["minute"] <= 240)]
    df["key"] = df["name"] + df["year"].apply(
        lambda x: "" if pd.isna(x) else f" ({int(x)})"
    )
    return df


def _deduplicate_by_key(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the best version of each movie when multiple rows share the same key.

    Priority: most recent year → highest rating → lowest id.
    """
    counts = df["key"].value_counts()
    unique_keys = counts[counts == 1].index
    dup_keys = counts[counts > 1].index

    best = (
        df[df["key"].isin(dup_keys)][["id", "year", "rating", "key"]]
        .fillna({"year": 0, "rating": 0})
        .sort_values(["key", "year", "rating", "id"], ascending=[True, False, False, True])
        .drop_duplicates(subset=["key"], keep="first")
    )
    return df[(df["key"].isin(unique_keys)) | (df["id"].isin(best["id"]))]


def _load_actors(path: Path) -> pd.DataFrame:
    actors = pd.read_csv(path / "actors.csv")
    actors = actors[actors["role"].notnull() & (actors["role"] != "")]
    actors = actors[
        ~actors["role"].str.contains(r"footage|uncredited|Ensemble/|\d", case=False, regex=True)
    ].drop(columns=["role"])
    popular = actors["name"].value_counts()
    actors = actors[actors["name"].isin(popular[popular >= 12].index)]
    return (
        actors.groupby("id")["name"]
        .apply(list)
        .reset_index(name="actor_list")
    )


def _load_crew(path: Path) -> pd.DataFrame:
    crew = pd.read_csv(path / "crew.csv")
    crew = crew[crew["role"].isin(["Director", "Writer", "Cinematography", "Composer"])]
    return (
        crew.groupby("id")
        .apply(lambda x: x.groupby("role")["name"].apply(list).to_dict())
        .reset_index(name="crew_dict")
    )


def _load_genres(path: Path) -> pd.DataFrame:
    genres = pd.read_csv(path / "genres.csv")
    genres["genre"] = genres["genre"].str.lower().str.replace(" ", "_", regex=False)
    return (
        genres.groupby("id")["genre"]
        .apply(" ".join)
        .reset_index(name="genre_list")
    )


def _load_languages(path: Path) -> pd.DataFrame:
    languages = pd.read_csv(path / "languages.csv")
    mask = languages["type"].isin(["Language", "Primary language"])
    return languages[mask].drop(columns=["type"])


def _load_studios(path: Path) -> pd.DataFrame:
    studios = pd.read_csv(path / "studios.csv")
    return (
        studios.groupby("id")["studio"]
        .apply(list)
        .reset_index(name="studio_list")
    )


def _load_imdb_ratings(data_dir: Path) -> pd.DataFrame:
    """Load IMDb title.basics + title.ratings and produce a key → averageRating lookup."""
    set_b_dir = data_dir / "set_b"
    ratings = pd.read_csv(set_b_dir / "title.ratings.tsv", sep="\t", na_values=["", "NA", "None"])
    movies = pd.read_csv(set_b_dir / "title.basics.tsv", sep="\t", na_values=["", "NA", "None"])
    movies = movies[["tconst", "primaryTitle", "startYear"]].dropna()
    movies = movies[
        ~movies["primaryTitle"].str.startswith(
            ("Episode ", "Épisode ", "Pilot", "Part 1", "Part 2", "Part 3")
        )
    ]
    movies["key"] = movies["primaryTitle"] + " (" + movies["startYear"] + ")"
    merged = movies.merge(ratings, how="inner", on="tconst")
    unique = merged["key"].value_counts()
    return merged[merged["key"].isin(unique[unique == 1].index)][["key", "averageRating"]]


def _combine_ratings(rating: float | None, average_rating: float | None) -> float | None:
    """Average Letterboxd and IMDb ratings; return whichever is available if only one exists."""
    has_r = not pd.isna(rating)
    has_a = not pd.isna(average_rating)
    if has_r and has_a:
        return round((rating + average_rating) / 2, 2)
    if has_r:
        return rating
    if has_a:
        return average_rating
    return np.nan


def load_letterboxd(data_dir: Path | None = None) -> pd.DataFrame:
    """Load and merge all Letterboxd dataset files into one DataFrame.

    Expects the raw CSVs at ``data_dir`` (defaults to ``settings.data_dir``):
        movies.csv, actors.csv, crew.csv, languages.csv, genres.csv, studios.csv

    Optionally merges IMDb ratings from ``data_dir/set_b/`` if present.
    """
    path = _dir(data_dir)

    movies = _deduplicate_by_key(load_movies(path))
    actors = _load_actors(path)
    crew = _load_crew(path)
    genres = _load_genres(path)
    languages = _load_languages(path)
    studios = _load_studios(path)

    data = (
        movies
        .merge(genres, how="left", on="id")
        .merge(actors, how="left", on="id")
        .merge(languages, how="left", on="id")
        .merge(studios, how="left", on="id")
        .merge(crew, how="left", on="id")
    )

    for col in ("actor_list", "studio_list"):
        data[col] = data[col].map(lambda x: x if isinstance(x, list) else [])

    # Merge IMDb ratings if available
    set_b_path = path / "set_b"
    if (set_b_path / "title.ratings.tsv").exists() and (set_b_path / "title.basics.tsv").exists():
        imdb = _load_imdb_ratings(path)
        data = data.merge(imdb, how="left", on="key")
        data["averageRating"] = data["averageRating"] / 2  # scale 10 → 5
        data["combined_rating"] = data.apply(
            lambda row: _combine_ratings(row["rating"], row["averageRating"]), axis=1
        )
    else:
        data["combined_rating"] = data["rating"]

    data = data.dropna(subset=["year"])
    data = data[
        ~(
            data["actor_list"].apply(lambda x: isinstance(x, list) and len(x) == 0)
            & data["combined_rating"].isnull()
        )
    ]
    return data
