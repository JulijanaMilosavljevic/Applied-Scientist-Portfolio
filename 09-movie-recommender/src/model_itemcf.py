import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_item_similarity(user_item_matrix):
    """
    Computes item-item cosine similarity using the user-item rating matrix (train).
    user_item_matrix: pandas DataFrame (users x items)
    Returns:
      sim_matrix: np.ndarray (items x items)
    """
    # items x users
    item_user = user_item_matrix.T.values
    sim_matrix = cosine_similarity(item_user)
    return sim_matrix
