from pathlib import Path
import re

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --tmdb-blue: #01b4e4;
        --tmdb-navy: #032541;
        --tmdb-green: #90cea1;
        --tmdb-text: #f2f8fc;
        --tmdb-muted: #b3c9d8;
    }

    .stApp {
        background: linear-gradient(135deg, #021b2e 0%, #032541 52%, #0b304b 100%);
        color: var(--tmdb-text);
    }

    [data-testid="stHeader"] {
        background: rgba(3, 37, 65, 0.92);
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4, h5, h6,
    p, li, label,
    [data-testid="stMarkdownContainer"] {
        color: var(--tmdb-text);
    }

    h1 {
        letter-spacing: -0.035em;
    }

    a, a:visited {
        color: var(--tmdb-blue) !important;
        font-weight: 600;
        text-decoration: none;
    }

    a:hover {
        color: var(--tmdb-green) !important;
        text-decoration: underline;
    }

    [data-testid="stCaptionContainer"] p,
    .stCaption {
        color: var(--tmdb-muted) !important;
    }

    hr {
        border-color: rgba(1, 180, 228, 0.28);
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        color: var(--tmdb-muted) !important;
        font-weight: 650;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--tmdb-blue) !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: var(--tmdb-blue) !important;
    }

    /* Buttons */
    div[data-testid="stButton"] > button {
        border: 1px solid var(--tmdb-blue);
        border-radius: 0.65rem;
        background: rgba(1, 180, 228, 0.10);
        color: var(--tmdb-text);
        font-weight: 650;
        transition: background 0.15s ease, transform 0.15s ease;
    }

    div[data-testid="stButton"] > button:hover {
        border-color: var(--tmdb-green);
        background: rgba(1, 180, 228, 0.22);
        color: #ffffff;
        transform: translateY(-1px);
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        border: 0;
        background: var(--tmdb-blue);
        color: var(--tmdb-navy) !important;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        background: var(--tmdb-green);
        color: var(--tmdb-navy) !important;
    }

    /* Text inputs, dropdowns and multiselects */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="textarea"] > div {
        background-color: #092e49 !important;
        border-color: #23617c !important;
        color: var(--tmdb-text) !important;
        border-radius: 0.6rem;
    }

    div[data-baseweb="select"] input,
    div[data-baseweb="input"] input,
    div[data-baseweb="textarea"] textarea {
        color: var(--tmdb-text) !important;
    }

    div[data-baseweb="select"] svg {
        fill: var(--tmdb-blue) !important;
    }

    [data-baseweb="tag"] {
        background-color: var(--tmdb-blue) !important;
        color: var(--tmdb-navy) !important;
        border-radius: 0.45rem;
    }

    [data-baseweb="tag"] span {
        color: var(--tmdb-navy) !important;
    }

    /* Sliders and selection controls */
    div[data-testid="stSlider"] [role="slider"] {
        background-color: var(--tmdb-blue) !important;
        border-color: var(--tmdb-blue) !important;
    }

    input[type="radio"], input[type="checkbox"] {
        accent-color: var(--tmdb-blue);
    }

    /* Movie posters */
    [data-testid="stImage"] img {
        border: 1px solid rgba(1, 180, 228, 0.25);
        border-radius: 0.7rem;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.28);
    }

    /* Feedback panels */
    div[data-testid="stAlert"] {
        border-radius: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🎬 Movie Recommendation System")
st.write("Find movies similar to a favourite or discover movies based on your preferences.")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent / "movie_recommender.joblib"
    return joblib.load(model_path)


try:
    model_bundle = load_model()
    movies_ml = model_bundle["movies_ml"]
    tfidf_matrix = model_bundle["tfidf_matrix"]
    knn_model = model_bundle["knn_model"]
except FileNotFoundError:
    st.error(
        "I couldn't find movie_recommender.joblib. Put it in the same folder as app.py."
    )
    st.stop()
except (KeyError, TypeError) as error:
    st.error(f"The saved model bundle is missing required data: {error}")
    st.stop()
except Exception as error:
    st.error(f"The saved model could not be loaded: {error}")
    st.stop()


required_columns = [
    "title",
    "genres_text",
    "overview",
    "user_score",
    "runtime_minutes",
    "poster_url",
    "movie_url",
]
missing_columns = [column for column in required_columns if column not in movies_ml.columns]
if missing_columns:
    st.error("The movie data is missing these columns: " + ", ".join(missing_columns))
    st.stop()


movie_titles = sorted(
    movies_ml["title"].dropna().astype(str).unique().tolist(),
    key=str.casefold,
)


def has_value(value):
    if value is None:
        return False
    try:
        return not pd.isna(value)
    except (TypeError, ValueError):
        return True


def display_movie_details(movie):
    genres = movie.get("genres_text", "")
    overview = movie.get("overview", "")
    rating = movie.get("user_score")
    runtime = movie.get("runtime_minutes")

    if not has_value(genres) or not str(genres).strip():
        genres = "Not available"

    if not has_value(overview) or not str(overview).strip():
        overview = "Overview not available."
    else:
        overview = str(overview).strip()
        if len(overview) > 300:
            overview = overview[:300].rsplit(" ", 1)[0] + "..."

    if has_value(rating):
        try:
            st.write(f"⭐ Rating: {float(rating):.1f}/10")
        except (TypeError, ValueError):
            pass

    if has_value(runtime):
        try:
            st.write(f"⏱️ Runtime: {int(float(runtime))} minutes")
        except (TypeError, ValueError):
            pass

    st.write(f"🎭 Genres: {genres}")
    st.write(overview)


def display_poster(movie):
    poster_url = movie.get("poster_url")
    if isinstance(poster_url, str) and poster_url.startswith(("https://", "http://")):
        st.image(poster_url, use_container_width=True)
    else:
        st.caption("Poster unavailable")


def display_tmdb_link(movie):
    movie_url = movie.get("movie_url")
    if isinstance(movie_url, str) and movie_url.startswith("https://"):
        st.markdown(f"[View on TMDB]({movie_url})")


similar_tab, preferences_tab = st.tabs(
    ["🎞️ Find Similar Movies", "🎯 Discover by Preferences"]
)


with similar_tab:
    st.subheader("Find movies similar to one you love")
    st.caption("Recommendations are generated by your trained TF-IDF and KNN model.")

    selected_title = st.selectbox("Search for a movie", movie_titles, key="similar_movie_title")
    number_of_recommendations = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=10,
        value=6,
        key="similar_movie_count",
    )

    if st.button("Recommend Similar Movies", type="primary", key="similar_movie_button"):
        matches = movies_ml["title"].fillna("").astype(str).str.casefold().eq(selected_title.casefold())
        match_positions = matches.to_numpy().nonzero()[0]

        if len(match_positions) == 0:
            st.error("Movie not found in the saved dataset.")
        else:
            movie_index = int(match_positions[0])
            selected_movie = movies_ml.iloc[movie_index]

            st.divider()
            st.subheader("Your Selected Movie")
            poster_column, details_column = st.columns([1, 3])

            with poster_column:
                display_poster(selected_movie)

            with details_column:
                st.header(str(selected_movie["title"]))
                display_movie_details(selected_movie)
                display_tmdb_link(selected_movie)

            neighbour_count = min(number_of_recommendations + 10, len(movies_ml))
            distances, indices = knn_model.kneighbors(
                tfidf_matrix[movie_index],
                n_neighbors=neighbour_count,
            )

            recommendations = []
            selected_id = selected_movie.get("movie_id")

            for neighbour_index, distance in zip(indices[0], distances[0]):
                neighbour_index = int(neighbour_index)
                if neighbour_index == movie_index:
                    continue

                candidate = movies_ml.iloc[neighbour_index]

                # Avoid returning a duplicate entry for the selected movie.
                if str(candidate.get("title", "")).casefold() == selected_title.casefold():
                    continue
                if has_value(selected_id) and has_value(candidate.get("movie_id")):
                    if candidate.get("movie_id") == selected_id:
                        continue

                recommendations.append(
                    {
                        "movie": candidate,
                        "similarity": max(0.0, 1.0 - float(distance)),
                    }
                )

                if len(recommendations) >= number_of_recommendations:
                    break

            st.divider()
            st.subheader("Movies You Might Like")

            if not recommendations:
                st.info("No recommendations were found for this movie.")
            else:
                for start in range(0, len(recommendations), 3):
                    columns = st.columns(3)

                    for column, recommendation in zip(
                        columns, recommendations[start : start + 3]
                    ):
                        movie = recommendation["movie"]
                        with column:
                            display_poster(movie)
                            st.markdown(f"### {movie['title']}")
                            display_movie_details(movie)
                            st.caption(
                                "Content similarity (cosine): "
                                f"{recommendation['similarity']:.1%}"
                            )
                            display_tmdb_link(movie)

            st.caption(
                "Similarity scores indicate text similarity, not the probability that you will like a movie."
            )


with preferences_tab:
    st.subheader("Choose what you feel like watching")
    st.caption("This option filters your movie dataset and ranks matches by user rating.")

    genre_options = [
        "Action",
        "Adventure",
        "Animation",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Family",
        "Fantasy",
        "History",
        "Horror",
        "Music",
        "Mystery",
        "Romance",
        "Science Fiction",
        "TV Movie",
        "Thriller",
        "War",
        "Western",
    ]

    preferred_genres = st.multiselect(
        "Choose your preferred genres",
        options=genre_options,
        placeholder="Select one or more genres",
        key="preferred_genres",
    )

    genre_match = st.radio(
        "How should selected genres match?",
        ["Any selected genre", "All selected genres"],
        horizontal=True,
        key="genre_match",
    )

    filter_column_one, filter_column_two = st.columns(2)

    with filter_column_one:
        minimum_rating = st.slider(
            "Minimum user rating",
            min_value=0.0,
            max_value=10.0,
            value=6.0,
            step=0.5,
            key="minimum_rating",
        )

    with filter_column_two:
        maximum_runtime = st.slider(
            "Maximum runtime (minutes)",
            min_value=60,
            max_value=300,
            value=180,
            step=15,
            key="maximum_runtime",
        )

    preference_count = st.slider(
        "Number of movies to show",
        min_value=3,
        max_value=15,
        value=9,
        key="preference_count",
    )

    if st.button("Find My Movies", type="primary", key="preference_button"):
        preference_movies = movies_ml.copy()
        preference_movies["score_filter"] = pd.to_numeric(
            preference_movies["user_score"], errors="coerce"
        )
        preference_movies["runtime_filter"] = pd.to_numeric(
            preference_movies["runtime_minutes"], errors="coerce"
        )

        preference_movies = preference_movies[
            (preference_movies["score_filter"] >= minimum_rating)
            & (preference_movies["runtime_filter"] <= maximum_runtime)
        ].copy()

        if preferred_genres:
            genre_text = preference_movies["genres_text"].fillna("").astype(str)
            genre_masks = [
                genre_text.str.contains(
                    rf"\b{re.escape(genre)}\b",
                    case=False,
                    regex=True,
                    na=False,
                )
                for genre in preferred_genres
            ]
            genre_mask_frame = pd.concat(genre_masks, axis=1)

            if genre_match == "All selected genres":
                genre_filter = genre_mask_frame.all(axis=1)
            else:
                genre_filter = genre_mask_frame.any(axis=1)

            preference_movies = preference_movies[genre_filter].copy()

        preference_movies = preference_movies.sort_values(
            by=["score_filter", "title"],
            ascending=[False, True],
        )

        if preference_movies.empty:
            st.warning(
                "No movies match these preferences. Try lowering the minimum rating, "
                "increasing the maximum runtime, or changing the genres."
            )
        else:
            results = preference_movies.head(preference_count)
            st.divider()
            st.subheader("Movies Selected for You")
            st.caption(
                f"Showing {len(results)} matches, ranked by user rating."
            )

            for start in range(0, len(results), 3):
                columns = st.columns(3)
                for column, (_, movie) in zip(
                    columns, results.iloc[start : start + 3].iterrows()
                ):
                    with column:
                        display_poster(movie)
                        st.markdown(f"### {movie['title']}")
                        display_movie_details(movie)
                        display_tmdb_link(movie)
