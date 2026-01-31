import pandas as pd
from src.config import PATHS
from pathlib import Path
import zipfile
import urllib.request

MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

def ensure_movielens_100k(data_dir: Path):
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ratings_path = raw_dir / "u.data"
    movies_path = raw_dir / "u.item"

    # Already present
    if ratings_path.exists() and movies_path.exists():
        return

    zip_path = data_dir / "ml-100k.zip"
    extract_dir = data_dir / "ml-100k"

    # Download
    urllib.request.urlretrieve(MOVIELENS_URL, zip_path)

    # Extract
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(data_dir)

    # Move needed files into data/raw
    (extract_dir / "u.data").replace(ratings_path)
    (extract_dir / "u.item").replace(movies_path)

    # Optional cleanup (not required)
    # zip_path.unlink(missing_ok=True)


def load_ratings():
    """
    MovieLens 100K: u.data (tab-separated)
    columns: user_id, item_id, rating, timestamp
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ensure_movielens_100k(data_dir)
    path = data_dir / "raw" / "u.data"

    #path = PATHS.data_raw / "u.data"
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
    data_dir = Path(__file__).resolve().parents[1] / "data"
    ensure_movielens_100k(data_dir)
    path = data_dir / "raw" / "u.item"

    #path = PATHS.data_raw / "u.item"
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
