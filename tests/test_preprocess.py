"""Tests for data preprocessing functions.

All tests use small in-memory DataFrames — no Kaggle data required.
"""

import numpy as np
import pandas as pd
import pytest

from movie_recommender.data.preprocess import (
    clean_text,
    encode_genres,
    encode_language,
    extract_crew_roles,
    parse_crew_dict,
    preprocess,
    scale_rating,
    scale_runtime,
    scale_year,
)

# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

def test_clean_text_lowercases():
    assert clean_text("HELLO WORLD") == clean_text("hello world")


def test_clean_text_removes_punctuation():
    result = clean_text("hello, world!")
    assert "," not in result
    assert "!" not in result


def test_clean_text_removes_stopwords():
    result = clean_text("this is a very good film")
    assert "this" not in result.split()
    assert "is" not in result.split()


def test_clean_text_empty_string():
    assert clean_text("") == ""


def test_clean_text_returns_string():
    assert isinstance(clean_text("A great film about time."), str)


# ---------------------------------------------------------------------------
# Numerical scalers
# ---------------------------------------------------------------------------

def test_scale_year_output_shape():
    s = pd.Series([1990, 2000, 2010, 2020])
    result = scale_year(s)
    assert result.shape == (4,)


def test_scale_runtime_range():
    s = pd.Series([60, 90, 120, 150])
    result = scale_runtime(s)
    assert result.min() == pytest.approx(0.0)
    assert result.max() == pytest.approx(1.0)


def test_scale_rating_imputes_nan():
    s = pd.Series([3.0, 4.0, np.nan, 5.0])
    result = scale_rating(s)
    assert not np.isnan(result).any()
    assert result.shape == (4,)


# ---------------------------------------------------------------------------
# encode_genres
# ---------------------------------------------------------------------------

def _genre_df():
    return pd.DataFrame({"genre_list": ["action thriller", "drama comedy", "action drama", ""]})


def test_encode_genres_adds_columns():
    df = encode_genres(_genre_df())
    assert "action" in df.columns
    assert "thriller" in df.columns
    assert "drama" in df.columns
    assert "comedy" in df.columns


def test_encode_genres_binary_values():
    df = encode_genres(_genre_df())
    assert set(df["action"].unique()).issubset({0, 1})


def test_encode_genres_handles_empty_genre():
    df = encode_genres(_genre_df())
    # Empty genre row should have all zeros
    row = df.iloc[3]
    genre_cols = [c for c in df.columns if c != "genre_list"]
    assert row[genre_cols].sum() == 0


def test_encode_genres_does_not_mutate_input():
    original = _genre_df()
    encode_genres(original)
    assert list(original.columns) == ["genre_list"]


# ---------------------------------------------------------------------------
# encode_language
# ---------------------------------------------------------------------------

def _lang_df():
    return pd.DataFrame({"language": ["English", "French, Spanish", None, "German"]})


def test_encode_language_adds_encoded_column():
    df = encode_language(_lang_df())
    assert "language_encoded" in df.columns


def test_encode_language_keeps_first_value():
    df = encode_language(_lang_df())
    assert df.loc[1, "language"] == "French"


def test_encode_language_handles_null():
    df = encode_language(_lang_df())
    assert df.loc[2, "language"] == "unknown"


# ---------------------------------------------------------------------------
# parse_crew_dict / extract_crew_roles
# ---------------------------------------------------------------------------

def _crew_df():
    return pd.DataFrame({
        "crew_dict": [
            "{'Director': ['Alice'], 'Writer': ['Bob']}",
            {},
            None,
        ]
    })


def test_parse_crew_dict_converts_string():
    df = parse_crew_dict(_crew_df())
    assert isinstance(df.loc[0, "crew_dict"], dict)


def test_parse_crew_dict_handles_empty_dict():
    df = parse_crew_dict(_crew_df())
    assert df.loc[1, "crew_dict"] == {}


def test_parse_crew_dict_handles_none():
    df = parse_crew_dict(_crew_df())
    assert df.loc[2, "crew_dict"] == {}


def test_extract_crew_roles_creates_columns():
    df = parse_crew_dict(_crew_df())
    df = extract_crew_roles(df)
    assert "director" in df.columns
    assert "writer" in df.columns


def test_extract_crew_roles_values():
    df = parse_crew_dict(_crew_df())
    df = extract_crew_roles(df)
    assert df.loc[0, "director"] == ["Alice"]
    assert df.loc[1, "director"] == []


# ---------------------------------------------------------------------------
# preprocess (integration)
# ---------------------------------------------------------------------------

def _sample_df():
    return pd.DataFrame({
        "description": ["A great film about war.", "Comedy about love.", None],
        "year": [1990.0, 2005.0, 2015.0],
        "minute": [90.0, 100.0, 110.0],
        "combined_rating": [4.0, 3.5, np.nan],
        "genre_list": ["action war", "comedy romance", ""],
        "language": ["English", "French", None],
        "crew_dict": [
            "{'Director': ['Alice']}",
            {},
            None,
        ],
    })


def test_preprocess_returns_dataframe():
    result = preprocess(_sample_df())
    assert isinstance(result, pd.DataFrame)


def test_preprocess_no_nan_in_description():
    result = preprocess(_sample_df())
    assert result["description"].notna().all()


def test_preprocess_no_nan_in_rating():
    result = preprocess(_sample_df())
    assert result["combined_rating"].notna().all()


def test_preprocess_does_not_mutate_input():
    original = _sample_df()
    preprocess(original)
    assert original["description"].iloc[0] == "A great film about war."


def test_preprocess_row_count_preserved():
    df = _sample_df()
    result = preprocess(df)
    assert len(result) == len(df)
