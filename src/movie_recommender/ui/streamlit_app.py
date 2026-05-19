"""Streamlit frontend for the movie recommender system."""

from __future__ import annotations

import requests
import streamlit as st

from movie_recommender.ui import api_client

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

MODEL_LABELS = {
    "popularity": "Popularity baseline",
    "tfidf": "TF-IDF content-based",
    "hybrid": "Hybrid semantic (recommended)",
}

MODEL_DESCRIPTIONS = {
    "popularity": "Returns the highest-rated movies in the catalogue, "
                  "optionally filtered by genre or language. No personalisation.",
    "tfidf": "Finds movies with similar descriptions using TF-IDF bag-of-words "
             "and cosine similarity. Fast and explainable.",
    "hybrid": "Combines semantic plot embeddings (sentence-transformers) with "
              "structured metadata — genre, language, year, runtime. "
              "Captures thematic similarity better than TF-IDF.",
}


def _api_ready() -> bool:
    try:
        return api_client.health().get("ready", False)
    except requests.exceptions.ConnectionError:
        return False


def _star_rating(rating: float | None) -> str:
    if rating is None:
        return "—"
    filled = round(rating * 2) // 2
    full = int(filled)
    half = 1 if filled != int(filled) else 0
    return "★" * full + ("½" if half else "") + f"  {rating:.1f}/5"


def _render_rec_card(rec: dict, rank: int) -> None:
    with st.container(border=True):
        col_rank, col_info, col_score = st.columns([0.5, 7, 2])
        with col_rank:
            st.markdown(f"### {rank}")
        with col_info:
            year = f" ({rec['year']})" if rec.get("year") else ""
            st.markdown(f"**{rec['title']}{year}**")
            genres = rec.get("genres", "").replace("_", " ").title()
            lang = rec.get("language", "")
            meta_parts = []
            if genres:
                meta_parts.append(f"🎭 {genres}")
            if lang:
                meta_parts.append(f"🌐 {lang}")
            if meta_parts:
                st.caption(" · ".join(meta_parts))
            desc = rec.get("description", "")
            if desc:
                st.markdown(f"_{desc[:200]}{'…' if len(desc) > 200 else ''}_")
        with col_score:
            st.metric("Rating", _star_rating(rec.get("rating")))
            st.metric("Match", f"{rec.get('score', 0):.0%}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🎬 Movie Recommender")
    st.caption("Letterboxd dataset · three recommender approaches compared")

    st.divider()

    model_key = st.radio(
        "Recommender model",
        options=list(MODEL_LABELS.keys()),
        format_func=lambda k: MODEL_LABELS[k],
        index=2,
    )
    st.info(MODEL_DESCRIPTIONS[model_key])

    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=8, step=1)

    st.divider()
    st.caption("Run `scripts/train.py` then `scripts/run_api.py` to start the API.")

# ---------------------------------------------------------------------------
# API health check
# ---------------------------------------------------------------------------

if not _api_ready():
    st.error(
        "**API not reachable.**  \n"
        "Start it with `python scripts/run_api.py` or `docker compose up`."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Movie search
# ---------------------------------------------------------------------------

st.header("Find a movie")

search_query = st.text_input(
    "Search by title",
    placeholder="e.g. Parasite, Inception, The Godfather …",
    label_visibility="collapsed",
)

selected_key: str | None = None

if search_query:
    with st.spinner("Searching …"):
        results = api_client.search_movies(search_query, limit=20)

    if not results:
        st.warning("No movies found. Try a different title.")
    else:
        options = {r["key"]: r for r in results}
        selected_key = st.selectbox(
            "Select a movie",
            options=list(options.keys()),
            format_func=lambda k: (
                f"{options[k]['title']} ({options[k]['year'] or '?'}) "
                f"— {options[k].get('genres', '').replace('_', ' ').title()}"
            ),
        )

        if selected_key:
            movie = options[selected_key]
            with st.container(border=True):
                col_title, col_rating = st.columns([5, 1])
                with col_title:
                    st.subheader(f"{movie['title']} ({movie.get('year', '?')})")
                    st.caption(
                        f"🎭 {movie.get('genres', '').replace('_', ' ').title()} "
                        f"· 🌐 {movie.get('language', '—')}"
                    )
                with col_rating:
                    st.metric("Rating", _star_rating(movie.get("rating")))

# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

if selected_key:
    st.divider()
    st.header(f"Recommendations — {MODEL_LABELS[model_key]}")

    with st.spinner("Fetching recommendations …"):
        try:
            result = api_client.recommend(selected_key, model=model_key, top_k=top_k)
            recs = result.get("recommendations", [])
        except requests.exceptions.HTTPError as exc:
            st.error(f"API error: {exc}")
            recs = []

    if recs:
        for i, rec in enumerate(recs, start=1):
            _render_rec_card(rec, i)
    else:
        st.info("No recommendations returned.")
