"""Text cleaning, numerical scaling, and categorical encoding for movie data."""

from __future__ import annotations

import ast
import re
import string

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, RobustScaler

# ---------------------------------------------------------------------------
# NLTK helpers — lazy download so the package works without a manual setup step
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    import nltk

    resources = {
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
    }
    for path, name in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(name, quiet=True)


def clean_text(text: str) -> str:
    """Lowercase, remove punctuation/digits/stopwords, and lemmatize a string."""
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    _ensure_nltk()

    text = text.strip().lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop = set(stopwords.words("english"))
    tokens = [w for w in tokens if w not in stop]
    lemma = WordNetLemmatizer()
    tokens = [lemma.lemmatize(lemma.lemmatize(w, pos="v"), pos="n") for w in tokens]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Numerical scaling
# ---------------------------------------------------------------------------

def scale_year(series: pd.Series) -> np.ndarray:
    """RobustScaler on release year (handles outliers gracefully)."""
    return RobustScaler().fit_transform(series.to_numpy().reshape(-1, 1)).flatten()


def scale_runtime(series: pd.Series) -> np.ndarray:
    """MinMaxScaler on runtime in minutes."""
    return MinMaxScaler().fit_transform(series.to_numpy().reshape(-1, 1)).flatten()


def scale_rating(series: pd.Series) -> np.ndarray:
    """MinMaxScaler on combined rating; NaN values are imputed with the median."""
    s = series.copy().astype(float)
    s = s.fillna(s.median())
    return MinMaxScaler().fit_transform(s.to_numpy().reshape(-1, 1)).flatten()


# ---------------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------------

def encode_genres(df: pd.DataFrame, column: str = "genre_list") -> pd.DataFrame:
    """One-hot encode a space-separated genre string column via MultiLabelBinarizer."""
    df = df.copy()
    df[column] = df[column].fillna("").astype(str)
    mlb = MultiLabelBinarizer()
    genre_dummies = pd.DataFrame(
        mlb.fit_transform(df[column].str.split()),
        columns=mlb.classes_,
        index=df.index,
    )
    # Drop the empty-string class that appears when genre_list is blank
    genre_dummies = genre_dummies.drop(columns=[""], errors="ignore")
    return pd.concat([df, genre_dummies], axis=1)


def encode_language(df: pd.DataFrame, column: str = "language") -> pd.DataFrame:
    """Label-encode the primary language, keeping only the first value before delimiters."""
    df = df.copy()
    df[column] = (
        df[column]
        .fillna("unknown")
        .astype(str)
        .str.split(r"[,/;|]", regex=True)
        .str[0]
        .str.strip()
    )
    df[f"{column}_encoded"] = LabelEncoder().fit_transform(df[column])
    return df


# ---------------------------------------------------------------------------
# Crew parsing
# ---------------------------------------------------------------------------

def parse_crew_dict(df: pd.DataFrame, column: str = "crew_dict") -> pd.DataFrame:
    """Convert stringified dict in *column* to an actual Python dict."""
    def _safe_eval(val: object) -> dict:
        if isinstance(val, dict):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                return {}
        return {}

    df = df.copy()
    df[column] = df[column].apply(_safe_eval)
    return df


def extract_crew_roles(
    df: pd.DataFrame,
    column: str = "crew_dict",
    roles: list[str] | None = None,
) -> pd.DataFrame:
    """Create one column per role (Director, Writer, …) from a crew dict column."""
    if roles is None:
        roles = ["Director", "Writer", "Cinematography", "Composer"]
    df = df.copy()
    for role in roles:
        df[role.lower()] = df[column].apply(
            lambda x: x.get(role, []) if isinstance(x, dict) else []
        )
    return df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all preprocessing steps to a raw Letterboxd DataFrame.

    Returns a new DataFrame with:
    - cleaned description text
    - scaled year, minute, combined_rating columns (in-place replacement)
    - genre dummy columns appended
    - language_encoded column appended
    - crew roles extracted into director / writer / cinematography / composer columns
    """
    df = df.copy()

    # Text
    df["description"] = df["description"].fillna("").apply(
        lambda t: clean_text(t) if t else ""
    )

    # Numerical
    df["year"] = scale_year(df["year"].astype(float))
    df["minute"] = scale_runtime(df["minute"].astype(float))
    df["combined_rating"] = scale_rating(df["combined_rating"])

    # Categorical
    df = encode_genres(df, "genre_list")
    df = encode_language(df, "language")

    # Crew
    df = parse_crew_dict(df, "crew_dict")
    df = extract_crew_roles(df, "crew_dict")

    return df
