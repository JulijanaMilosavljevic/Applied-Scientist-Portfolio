import numpy as np


def recommend_for_user(
    user_id,
    user_item_matrix,
    sim_matrix,
    top_k=10,
    popularity_list=None,
    min_history=5
):
    """
    Item-based CF recommendations with fallback to popularity for cold-start / thin history users.

    popularity_list: list of item_ids sorted by popularity (optional but recommended)
    min_history: minimum number of rated items required to use CF
    """
    # user not in training matrix -> fallback
    if user_id not in user_item_matrix.index:
        if popularity_list is None:
            raise ValueError("User not found and no popularity_list provided for fallback.")
        return np.array(popularity_list[:top_k])

    user_ratings = user_item_matrix.loc[user_id].values
    history = int((user_ratings > 0).sum())

    # thin history -> fallback
    if history < min_history:
        if popularity_list is None:
            # still return something, but best practice is to provide popularity_list
            return user_item_matrix.columns.to_numpy()[:top_k]
        # filter already-rated items from popularity list
        already = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist())
        recs = [i for i in popularity_list if i not in already and i in user_item_matrix.columns]
        return np.array(recs[:top_k])

    scores = sim_matrix.dot(user_ratings)

    # remove already-rated items
    already_rated_mask = user_ratings > 0
    scores[already_rated_mask] = -np.inf

    # if all -inf (rare), fallback
    if np.all(np.isneginf(scores)):
        if popularity_list is None:
            return user_item_matrix.columns.to_numpy()[:top_k]
        already = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index.tolist())
        recs = [i for i in popularity_list if i not in already and i in user_item_matrix.columns]
        return np.array(recs[:top_k])

    top_indices = np.argsort(scores)[::-1][:top_k]
    item_ids = user_item_matrix.columns.to_numpy()[top_indices]
    return item_ids
