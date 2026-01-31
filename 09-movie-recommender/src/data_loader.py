import pandas as pd
from src.config import PATHS


def load_ratings():
    """
    MovieLens 100K: u.data (tab-separated)
    columns: user_id, item_id, rating, timestamp
    """
    path = PATHS.data_raw / "u.data"
    ratings = pd.read_csv(
        path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    return ratings


def load_movies():
    """
    MovieLens 100K: u.item (pipe-separated, latin-1)
    We load only: item_id, title
    """
    path = PATHS.data_raw / "u.item"
    movies = pd.read_csv(
        path,
        sep="|",
        encoding="latin-1",
        header=None,
        usecols=[0, 1],
        names=["item_id", "title"],
        engine="python",
    )
    return movies
