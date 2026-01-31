import numpy as np


def get_relevant_items(test_df, user_id, min_rating=4):
    """
    Relevant items: items user rated >= min_rating in test set.
    """
    user_test = test_df[test_df["user_id"] == user_id]
    relevant = user_test[user_test["rating"] >= min_rating]["item_id"].tolist()
    return relevant


def precision_at_k(recommended, relevant, k):
    if k == 0:
        return 0.0
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / k


def recall_at_k(recommended, relevant, k):
    if len(relevant) == 0:
        return 0.0
    rec_k = recommended[:k]
    return len(set(rec_k) & set(relevant)) / len(relevant)


def evaluate_model(user_item_train, sim_matrix, test_df, k=10, min_rating=4, popularity_list=None, min_history=5):
    precisions = []
    recalls = []

    users = test_df["user_id"].unique()

    for user_id in users:
        relevant = get_relevant_items(test_df, user_id, min_rating=min_rating)
        if len(relevant) == 0:
            continue

        from src.recommend import recommend_for_user
        recommended = recommend_for_user(
            user_id,
            user_item_train,
            sim_matrix,
            top_k=k,
            popularity_list=popularity_list,
            min_history=min_history
        )

        precisions.append(precision_at_k(recommended, relevant, k))
        recalls.append(recall_at_k(recommended, relevant, k))

    return float(np.mean(precisions)), float(np.mean(recalls))