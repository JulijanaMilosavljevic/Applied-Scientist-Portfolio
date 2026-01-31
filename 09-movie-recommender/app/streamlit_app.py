import sys
from pathlib import Path
import matplotlib.pyplot as plt

# make src importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
import numpy as np
import pickle

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SIM_PATH = PROCESSED_DIR / "sim_matrix.npy"
UIM_PATH = PROCESSED_DIR / "user_item_train.pkl"   # dataframe
TRAIN_PATH = PROCESSED_DIR / "train_df.parquet"
POPULARITY_PATH = PROCESSED_DIR / "popularity.pkl"
MOVIES_PATH = PROCESSED_DIR / "movies.parquet"

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


@st.cache_resource
def prepare_artifacts():
    # 1) If artifacts exist -> load
    if SIM_PATH.exists() and UIM_PATH.exists() and TRAIN_PATH.exists() and POPULARITY_PATH.exists() and MOVIES_PATH.exists():
        movies = pd.read_parquet(MOVIES_PATH)
        train_df = pd.read_parquet(TRAIN_PATH)

        sim_matrix = np.load(SIM_PATH, allow_pickle=False)

        with open(UIM_PATH, "rb") as f:
            user_item_train = pickle.load(f)

        with open(POPULARITY_PATH, "rb") as f:
            popularity_list = pickle.load(f)

        return movies, user_item_train, sim_matrix, popularity_list, train_df

    # 2) Else compute + save
    ratings, movies = load_data()
    train_df, test_df = train_test_split_by_user(ratings, test_ratio=0.2)

    user_item_train = create_user_item_matrix(train_df)
    sim_matrix = compute_item_similarity(user_item_train)

    pop_df = compute_popularity(train_df)
    popularity_list = pop_df["item_id"].tolist()

    # Save artifacts
    movies.to_parquet(MOVIES_PATH, index=False)
    train_df.to_parquet(TRAIN_PATH, index=False)
    np.save(SIM_PATH, sim_matrix)

    with open(UIM_PATH, "wb") as f:
        pickle.dump(user_item_train, f)

    with open(POPULARITY_PATH, "wb") as f:
        pickle.dump(popularity_list, f)

    return movies, user_item_train, sim_matrix, popularity_list, train_df



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
def profile_label(history_size: int) -> str:
    if history_size < min_history:
        return "🆕 New"
    if history_size < 20:
        return "🎟️ Casual"
    return "🍿 Power"

col1, col2, col3 = st.columns(3)
col1.metric("Viewer Profile", profile_label(history))
col2.metric("History size", f"{history} ratings")
col3.metric("Top-K", f"{k}")
st.caption("🧠 ItemCF (cosine similarity) + popularity fallback")
with st.expander("📈 Quick stats"):
    fig, ax = plt.subplots()
    ax.hist(train_df["rating"], bins=5)
    ax.set_title("Ratings distribution (train)")
    st.pyplot(fig)
st.caption("MovieLens profiles are anonymized (IDs).")
if "recs" not in st.session_state:
    st.session_state["recs"] = None

if st.button("🎯 Recommend"):
    st.session_state["recs"] = recommend_for_user(
            user_id=user_id,
            user_item_matrix=user_item_train,
            sim_matrix=sim_matrix,
            top_k=k,
            popularity_list=popularity_list if use_fallback else None,
            min_history=min_history
        )
    recs = st.session_state["recs"]
    if recs is not None:
    # prikazi loved + recommendations
        history = int((user_item_train.loc[user_id].values > 0).sum())
        st.subheader("⭐ Loved Movies (5-star ratings)")

        user_train = train_df[train_df["user_id"] == user_id]
        liked = user_train[user_train["rating"] == 5].head(8)
        st.subheader("🕒 Recent rating history")
        recent = user_train.sort_values("timestamp", ascending=False).head(10)
        recent_titles = movies.set_index("item_id").loc[recent["item_id"]]["title"].tolist()
        for t, r in zip(recent_titles, recent["rating"].tolist()):
            st.markdown(f"- {t} — **{int(r)}★**")

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

st.caption("🔒 MovieLens is an anonymized dataset — profiles use IDs instead of real names.")
st.caption("Built with Streamlit • ItemCF (cosine similarity) • MovieLens 100K")
