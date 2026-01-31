import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_np(path: Path, arr: np.ndarray):
    ensure_dir(path.parent)
    np.save(path, arr)

def load_np(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)

def save_pkl(path: Path, obj):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pkl(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def save_csv(path: Path, df: pd.DataFrame):
    ensure_dir(path.parent)
    df.to_csv(path, index=False)

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
