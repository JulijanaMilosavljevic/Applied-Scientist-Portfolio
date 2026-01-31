from pathlib import Path
from src.config import PATHS

REQUIRED = ["u.data", "u.item"]

def main():
    PATHS.data_raw.mkdir(parents=True, exist_ok=True)
    missing = [f for f in REQUIRED if not (PATHS.data_raw / f).exists()]

    if missing:
        print("❌ Missing files in data/raw/:", missing)
        print("➡️ Download MovieLens 100K and place u.data and u.item into data/raw/")
    else:
        print("✅ Data files found in data/raw/")

if __name__ == "__main__":
    main()
