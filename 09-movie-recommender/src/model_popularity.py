import pandas as pd


def compute_popularity(train_df: pd.DataFrame):
    """
    Popularity based on:
    - number of interactions (count)
    - average rating
    Returns a DataFrame sorted by a simple score.
    """
    agg = (
        train_df.groupby("item_id")
        .agg(count=("rating", "size"), avg_rating=("rating", "mean"))
        .reset_index()
    )

    # simple score: count * avg_rating (good enough baseline)
    agg["score"] = agg["count"] * agg["avg_rating"]
    agg = agg.sort_values("score", ascending=False).reset_index(drop=True)

    return agg


def recommend_popular(user_id, user_item_train, popularity_df, top_k=10):
    """
    Recommend top popular items excluding already rated.
    """
    user_ratings = user_item_train.loc[user_id]
    already = set(user_ratings[user_ratings > 0].index.tolist())

    recs = []
    for item_id in popularity_df["item_id"].tolist():
        if item_id not in already and item_id in user_item_train.columns:
            recs.append(item_id)
        if len(recs) == top_k:
            break
    return recs
