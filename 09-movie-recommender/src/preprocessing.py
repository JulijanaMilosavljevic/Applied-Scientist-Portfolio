import numpy as np
import pandas as pd


def train_test_split_by_user(ratings: pd.DataFrame, test_ratio: float = 0.2, seed: int = 42):
    """
    Split interactions per user into train/test.
    We shuffle each user's interactions, then take last portion as test.
    """
    rng = np.random.default_rng(seed)

    train_parts = []
    test_parts = []

    for user_id, group in ratings.groupby("user_id"):
        idx = group.index.to_numpy()
        rng.shuffle(idx)

        n_test = max(1, int(len(idx) * test_ratio))
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]

        test_parts.append(ratings.loc[test_idx])
        train_parts.append(ratings.loc[train_idx])

    train_df = pd.concat(train_parts).reset_index(drop=True)
    test_df = pd.concat(test_parts).reset_index(drop=True)

    return train_df, test_df


def create_user_item_matrix(ratings: pd.DataFrame):
    """
    Returns user-item matrix with users as rows, items as columns.
    Missing values filled with 0.
    """
    user_item = ratings.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    return user_item
