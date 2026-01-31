import sys
from pathlib import Path

# make src importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd

from src.data_loader import load_ratings, load_movies
from src.preprocessing import train_test_split_by_user, create_user_item_matrix
from src.model_itemcf import compute_item_similarity
from src.model_popularity import compute_popularity
from src.recommend import recommend_for_user


@st.cache_data
def load_data():
    ratings = load_ratings()
    movies = load_movies()
    return ratings, movies


@st.cache_data
def prepare_artifacts():
    ratings, movies = load_data()
    train_df, test_df = train_test_split_by_user(ratings, test_ratio=0.2)

    user_item_train = create_user_item_matrix(train_df)
    sim_matrix = compute_item_similarity(user_item_train)

    pop_df = compute_popularity(train_df)
    popularity_list = pop_df["item_id"].tolist()

    return movies, user_item_train, sim_matrix, popularity_list , train_df


st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 Movie Recommendation System")
st.write("Item-based Collaborative Filtering + Popularity fallback (MovieLens 100K)")

movies, user_item_train, sim_matrix, popularity_list, train_df = prepare_artifacts()

with st.sidebar:
    st.header("👤 Viewer Profiles")
    user_id = st.number_input("Select a Viewer Profile (anonymized ID)", min_value=1, max_value=943, value=10, step=1)
    k = st.slider("Top-K recommendations", min_value=5, max_value=20, value=10, step=5)
    use_fallback = st.checkbox("Use popularity fallback for sparse profiles", value=True)
    min_history = st.slider("Min history (fallback threshold)", min_value=1, max_value=20, value=5, step=1)

history = int((user_item_train.loc[user_id].values > 0).sum())
st.info(
    f"👤 **Viewer Profile #{user_id}**  \n"
    f"📊 **History size:** {history} rated movies  \n"
    f"🧠 **Model:** Item-based Collaborative Filtering (cosine similarity)"
)

if st.button("🎯 Recommend"):
    try:
        recs = recommend_for_user(
            user_id=user_id,
            user_item_matrix=user_item_train,
            sim_matrix=sim_matrix,
            top_k=k,
            popularity_list=popularity_list if use_fallback else None,
            min_history=min_history
        )
        history = int((user_item_train.loc[user_id].values > 0).sum())

        col1, col2, col3 = st.columns(3)
        col1.metric("Viewer Profile", f"#{user_id}")
        col2.metric("History size", f"{history} ratings")
        col3.metric("Top-K", f"{k}")
        st.subheader("⭐ Loved Movies (5-star ratings)")

        user_train = train_df[train_df["user_id"] == user_id]
        liked = user_train[user_train["rating"] == 5].head(8)

        if liked.empty:
            st.write("No 5-star ratings found for this profile.")
        else:
            liked_titles = movies.set_index("item_id").loc[liked["item_id"]]["title"].tolist()
            for t in liked_titles:
                st.markdown(f"- **{t}**")

        rec_ids = list(recs)
        out = movies.set_index("item_id").loc[rec_ids].reset_index()
        out = out.rename(columns={"item_id": "movie_id"})
        st.success(f"✅ Top {k} recommendations for user {user_id}")
        st.subheader(f"🎯 Top {k} Recommendations")

        for i, row in out.iterrows():
            st.markdown(
                f"""
                <div style="padding:12px; border-radius:12px; background-color: rgba(255,255,255,0.04); margin-bottom:10px;">
                    <div style="font-size:14px; opacity:0.8;">#{i+1} • Movie ID: {row['movie_id']}</div>
                    <div style="font-size:18px; font-weight:700;">{row['title']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as e:
        st.error(str(e))

st.caption("🔒 MovieLens is an anonymized dataset — profiles use IDs instead of real names.")
st.caption("Built with Streamlit • ItemCF (cosine similarity) • MovieLens 100K")
