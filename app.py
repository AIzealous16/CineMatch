import os
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.cluster import KMeans

# ============================================================
# PAGE CONFIG
# ============================================================

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-heading">🍿 CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div style="font-size:20px; font-weight:600; color:#344054; margin-bottom:10px;">'
    'Movie Recommendation and Genre Clustering Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'Discover movies that match your interests. '
    'Choose a movie from the catalogue and explore '
    'similar titles based on genre characteristics '
    'and audience preferences.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PROFESSIONAL + COLOURFUL UI
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       GLOBAL
       ======================================================== */

    .stApp {
        background-color: #f4f6f9;
        color: #172033;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 35px;
        padding-bottom: 50px;
    }

    header {
        background: transparent !important;
    }

    h1, h2, h3 {
        color: #172033 !important;
    }

    p {
        color: #667085;
    }


    /* ========================================================
       HERO
       ======================================================== */

    .hero-box {
        background: linear-gradient(
            135deg,
            #ffffff 0%,
            #eef5ff 55%,
            #f5f0ff 100%
        );

        border: 1px solid #d9e4f2;
        border-radius: 20px;

        padding: 34px;

        margin-bottom: 25px;

        box-shadow:
            0px 8px 25px rgba(35, 55, 85, 0.07);
    }

    .hero-heading {
        font-size: 34px;
        font-weight: 750;

        color: #172033;

        margin-bottom: 8px;
    }

    .hero-description {
        font-size: 16px;
        color: #5f6b7a;

        line-height: 1.65;

        max-width: 900px;
    }


    /* ========================================================
       KPI CARDS
       ======================================================== */

    .kpi-box {
        background: #ffffff;

        border-radius: 16px;

        padding: 21px;

        min-height: 108px;

        box-shadow:
            0px 5px 18px rgba(30, 45, 70, 0.06);

        position: relative;

        overflow: hidden;
    }

    .kpi-box:nth-child(1) {
        border-top: 4px solid #4f8cff;
    }

    .kpi-label {
        font-size: 12px;

        font-weight: 700;

        color: #7b8794;

        text-transform: uppercase;

        letter-spacing: 0.8px;
    }

    .kpi-number {
        font-size: 30px;

        font-weight: 750;

        color: #172033;

        margin-top: 8px;
    }


    /* ========================================================
       SECTION HEADINGS
       ======================================================== */

    .section-heading {
        font-size: 22px;

        font-weight: 750;

        color: #172033;

        margin-top: 32px;

        margin-bottom: 5px;
    }

    .section-description {
        font-size: 14px;

        color: #7b8794;

        margin-bottom: 16px;
    }


    /* ========================================================
       SELECTED MOVIE
       ======================================================== */

    .selected-box {
        background: linear-gradient(
            135deg,
            #ffffff,
            #f1f7ff
        );

        border: 1px solid #cfe0f7;

        border-left: 5px solid #4f8cff;

        border-radius: 16px;

        padding: 25px;

        margin-top: 12px;

        margin-bottom: 25px;

        box-shadow:
            0px 6px 20px rgba(50, 90, 140, 0.06);
    }

    .selected-name {
        font-size: 26px;

        font-weight: 750;

        color: #172033;

        margin-bottom: 8px;
    }

    .selected-info {
        color: #667085;

        font-size: 14px;
    }


    /* ========================================================
       RECOMMENDATION CARDS
       ======================================================== */

    .recommendation-card {
        background: #ffffff;

        border: 1px solid #e0e6ee;

        border-radius: 15px;

        padding: 18px;

        min-height: 190px;

        box-shadow:
            0px 4px 15px rgba(30, 40, 60, 0.05);
    }

    .recommendation-number {
        font-size: 11px;

        font-weight: 750;

        color: #8b5cf6;

        letter-spacing: 0.8px;

        margin-bottom: 9px;
    }

    .recommendation-title {
        font-size: 16px;

        font-weight: 700;

        color: #172033;

        line-height: 1.4;

        min-height: 45px;
    }

    .recommendation-genre {
        font-size: 12px;

        color: #7b8794;

        margin-top: 10px;

        line-height: 1.5;
    }

    .recommendation-rating {
        font-size: 14px;

        font-weight: 700;

        color: #e49a00;

        margin-top: 13px;
    }


    /* ========================================================
       INFO CARDS
       ======================================================== */

    .info-card {
        background: #ffffff;

        border: 1px solid #e0e6ee;

        border-radius: 16px;

        padding: 20px;

        margin-top: 10px;

        box-shadow:
            0px 4px 15px rgba(30, 40, 60, 0.04);
    }

    .info-card-title {
        font-size: 17px;

        font-weight: 750;

        color: #172033;

        margin-bottom: 12px;
    }


    /* ========================================================
       STREAMLIT INPUTS
       ======================================================== */

    div[data-baseweb="select"] > div {

        border-radius: 10px;

        border: 1px solid #cdd7e4;

        background-color: #ffffff;
    }

    .stTextInput input {

    border-radius: 10px;

    border: 1px solid #cdd7e4;

    background-color: #ffffff;

    color: #172033 !important;
}

.stTextInput input::placeholder {

    color: #98a2b3 !important;

    opacity: 1;
}

.stTextInput input:focus {

    border-color: #4f8cff;

    color: #172033 !important;

    background-color: #ffffff;
}

    .stTextInput input:focus {

        border-color: #4f8cff;
    }


    /* ========================================================
       METRIC CARDS
       ======================================================== */

    div[data-testid="stMetric"] {

        background: #ffffff;

        border: 1px solid #e0e6ee;

        border-radius: 15px;

        padding: 15px;

        box-shadow:
            0px 4px 14px rgba(30, 40, 60, 0.04);
    }

    div[data-testid="stMetricLabel"] {

        color: #667085 !important;
    }

    div[data-testid="stMetricValue"] {

        color: #172033 !important;
    }


    /* ========================================================
       CONTAINER CARDS
       ======================================================== */

    div[data-testid="stVerticalBlockBorderWrapper"] {

        background-color: #ffffff;

        border-radius: 15px;

        border: 1px solid #e0e6ee;

        box-shadow:
            0px 4px 15px rgba(30, 40, 60, 0.05);

        transition: 0.2s ease;
    }

    div[data-testid="stVerticalBlockBorderWrapper"]:hover {

        border-color: #b9cdf0;

        box-shadow:
            0px 7px 22px rgba(55, 90, 140, 0.10);
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton button {

        border-radius: 9px;

        border: 1px solid #cdd7e4;

        background-color: #ffffff;

        color: #344054;

        font-weight: 650;
    }

    .stButton button:hover {

        border-color: #4f8cff;

        color: #3269c7;
    }


    /* ========================================================
       FOOTER
       ======================================================== */

    .footer-line {

        margin-top: 45px;

        border-top: 1px solid #dfe4ea;

        padding-top: 18px;

        text-align: center;

        color: #98a2b3;

        font-size: 13px;
    }

</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

movies_file = os.path.join(BASE_DIR, "movies.csv")
ratings_file = os.path.join(BASE_DIR, "ratings.csv")


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    movies = pd.read_csv(movies_file)
    ratings = pd.read_csv(ratings_file)

    rating_summary = (
        ratings
        .groupby("movieId")["rating"]
        .agg(
            Average_Rating="mean",
            Rating_Count="count"
        )
        .reset_index()
    )

    df = movies.merge(
        rating_summary,
        on="movieId",
        how="left"
    )

    df["Average_Rating"] = df["Average_Rating"].fillna(0)
    df["Rating_Count"] = df["Rating_Count"].fillna(0)

    df["title"] = df["title"].fillna("Unknown Movie")
    df["genres"] = df["genres"].fillna("Unknown")

    return df, ratings


df, ratings = load_data()


# ============================================================
# GENRE FEATURES
# ============================================================

df["Genre_List"] = df["genres"].str.split("|")

mlb = MultiLabelBinarizer()

genre_matrix = mlb.fit_transform(df["Genre_List"])

genre_data = pd.DataFrame(
    genre_matrix,
    columns=mlb.classes_,
    index=df.index
)

if "(no genres listed)" in genre_data.columns:
    genre_data = genre_data.drop(
        columns=["(no genres listed)"]
    )


# ============================================================
# K-MEANS
# ============================================================

NUMBER_OF_CLUSTERS = 5

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(genre_data)


# ============================================================
# RECOMMENDATION FUNCTION
# ============================================================

def get_recommendations(movie_row, number=5):

    selected_cluster = movie_row["Cluster"]

    candidates = df[
        (df["Cluster"] == selected_cluster) &
        (df["movieId"] != movie_row["movieId"])
    ].copy()

    selected_genres = set(
        str(movie_row["genres"]).split("|")
    )

    candidates["Genre_Similarity"] = candidates[
        "genres"
    ].apply(
        lambda x: len(
            selected_genres.intersection(
                set(str(x).split("|"))
            )
        )
    )

    candidates = candidates.sort_values(
        by=[
            "Genre_Similarity",
            "Average_Rating",
            "Rating_Count"
        ],
        ascending=False
    )

    return candidates.head(number)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-heading">CineMatch</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-description">'
    'Discover movies that match your interests. '
    'Choose a movie from the catalogue and explore '
    'similar titles based on genre characteristics '
    'and audience preferences.'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# KPI SECTION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Total Movies</div>
            <div class="kpi-number">{len(df):,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:

    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Movie Clusters</div>
            <div class="kpi-number">{NUMBER_OF_CLUSTERS}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:

    st.markdown(
        f"""
        <div class="kpi-box">
            <div class="kpi-label">Total Ratings</div>
            <div class="kpi-number">{len(ratings):,}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MOVIE SEARCH
# ============================================================

st.markdown(
    '<div class="section-heading">Choose a movie</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Search the catalogue and select a movie to receive recommendations.'
    '</div>',
    unsafe_allow_html=True
)

search = st.text_input(
    "Search movie",
    placeholder="Example: Interstellar",
    label_visibility="collapsed"
)


# Filter movies

if search.strip():

    matching_movies = df[
        df["title"]
        .str.contains(
            search,
            case=False,
            na=False
        )
    ].copy()

else:

    matching_movies = df.copy()


matching_movies = matching_movies.sort_values("title")


if len(matching_movies) == 0:

    st.warning(
        "No matching movie found. Try another title."
    )

    selected_title = None

else:

    selected_title = st.selectbox(
        "Movie",
        matching_movies["title"].tolist(),
        label_visibility="collapsed"
    )


# ============================================================
# SELECTED MOVIE
# ============================================================

if selected_title is not None:

    selected_movie = df[
        df["title"] == selected_title
    ].iloc[0]

    st.markdown(
        '<div class="section-heading">Selected movie</div>',
        unsafe_allow_html=True
    )

    # Native Streamlit columns — NO HTML CONTENT HERE

    info1, info2, info3 = st.columns([2.5, 1, 1])

    with info1:

        st.markdown(
            '<div class="selected-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="selected-name">'
            f'{selected_movie["title"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            "Genres: "
            + str(selected_movie["genres"]).replace("|", " • ")
        )

        st.write(
            f"Audience rating: "
            f"{selected_movie['Average_Rating']:.1f} / 5"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

    with info2:

        st.metric(
            "Rating",
            f"{selected_movie['Average_Rating']:.1f}"
        )

    with info3:

        st.metric(
            "Ratings",
            f"{int(selected_movie['Rating_Count']):,}"
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown(
        '<div class="section-heading">'
        'Recommended for you'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'These movies share similar genre characteristics '
        'with your selected movie.'
        '</div>',
        unsafe_allow_html=True
    )

    recommendations = get_recommendations(
        selected_movie,
        number=5
    )


    # Five columns

    recommendation_columns = st.columns(5)

    for position, (_, movie) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        with recommendation_columns[position - 1]:

            # Native Streamlit card
            st.container(border=True)

            st.caption(
                f"RECOMMENDATION {position}"
            )

            st.markdown(
                f"**{movie['title']}**"
            )

            st.caption(
                str(movie["genres"]).replace(
                    "|",
                    " • "
                )
            )

            st.markdown(
                f"### ⭐ {movie['Average_Rating']:.1f}"
            )


# ============================================================
# EXPLORE SECTION
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Explore the catalogue'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Browse popular movies and understand the composition of the catalogue.'
    '</div>',
    unsafe_allow_html=True
)


left_column, right_column = st.columns(2)


# ============================================================
# POPULAR MOVIES
# ============================================================

with left_column:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-card-title">'
        'Popular Movies'
        '</div>',
        unsafe_allow_html=True
    )

    popular = (
        df
        .sort_values(
            by=[
                "Rating_Count",
                "Average_Rating"
            ],
            ascending=False
        )
        .head(8)
    )

    for _, movie in popular.iterrows():

        st.write(
            f"**{movie['title']}**"
        )

        st.caption(
            f"★ {movie['Average_Rating']:.1f} "
            f"· {int(movie['Rating_Count']):,} ratings"
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# GENRE DISTRIBUTION
# ============================================================

with right_column:

    st.markdown(
        '<div class="info-card">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="info-card-title">'
        'Genre Distribution'
        '</div>',
        unsafe_allow_html=True
    )

    genre_count = {}

    for genres in df["Genre_List"]:

        for genre in genres:

            if genre != "(no genres listed)":

                genre_count[genre] = (
                    genre_count.get(genre, 0) + 1
                )

    genre_df = pd.DataFrame(
        list(genre_count.items()),
        columns=["Genre", "Movies"]
    )

    genre_df = genre_df.sort_values(
        "Movies",
        ascending=False
    ).head(8)

    st.bar_chart(
        genre_df.set_index("Genre"),
        height=300
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


# ============================================================
# CLUSTER OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'Movie groups'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'The movies are organised into five groups according to their genre characteristics.'
    '</div>',
    unsafe_allow_html=True
)

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_df = pd.DataFrame({
    "Cluster": [
        f"Group {i + 1}"
        for i in cluster_counts.index
    ],
    "Movies": cluster_counts.values
})

st.bar_chart(
    cluster_df.set_index("Cluster"),
    height=300
)


# ============================================================
# ABOUT CINE MATCH
# ============================================================

st.markdown(
    '<div class="section-heading">'
    'About CineMatch'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">

    <div class="info-card-title">
    How the recommendation system works
    </div>

    """,
    unsafe_allow_html=True
)

st.write(
    "CineMatch analyses the genre characteristics of movies "
    "and organises similar movies into behavioural groups."
)

st.write(
    "When a movie is selected, the system searches within "
    "its group and ranks candidate movies according to "
    "genre similarity, audience rating and rating activity."
)

st.write(
    "This creates a simple personalised movie-discovery "
    "experience using machine learning."
)

st.markdown(
    "</div>",
    unsafe_allow_html=True
)


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer-line">'
    'CineMatch · Movie Recommendation System · Machine Learning'
    '</div>',
    unsafe_allow_html=True
)