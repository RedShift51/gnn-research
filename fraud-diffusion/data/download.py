"""Download the PaySim dataset from Kaggle into data/raw/paysim.csv."""

import shutil
import sys
import zipfile
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_CSV = RAW_DIR / "paysim.csv"
KAGGLE_DATASET = "ealaxi/paysim1"

MANUAL_INSTRUCTIONS = f"""
Kaggle credentials not found (~/.kaggle/kaggle.json).

To download automatically:
  1. Kaggle account -> Settings -> API -> Create New Token
  2. Save the downloaded kaggle.json to ~/.kaggle/kaggle.json
  3. chmod 600 ~/.kaggle/kaggle.json
  4. Re-run: python -m data.download

Or download manually:
  1. Visit https://www.kaggle.com/datasets/{KAGGLE_DATASET}
  2. Download and unzip
  3. Place the CSV at {TARGET_CSV}
"""


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_CSV.exists():
        print(f"Already present: {TARGET_CSV}")
        return

    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        print(MANUAL_INSTRUCTIONS)
        sys.exit(1)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("`kaggle` package not installed. Run: pip install kaggle")
        sys.exit(1)

    api = KaggleApi()
    api.authenticate()
    print(f"Downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)

    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        print("Download finished but no CSV found in data/raw/.")
        sys.exit(1)

    if csvs[0] != TARGET_CSV:
        shutil.move(str(csvs[0]), str(TARGET_CSV))

    print(f"Saved: {TARGET_CSV}")


if __name__ == "__main__":
    main()
