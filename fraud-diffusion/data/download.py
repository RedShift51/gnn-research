"""Download the PaySim and Elliptic Bitcoin datasets from Kaggle."""

import logging
import shutil
import sys
import zipfile
from pathlib import Path

logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
TARGET_CSV = RAW_DIR / "paysim.csv"
KAGGLE_DATASET = "ealaxi/paysim1"

ELLIPTIC_KAGGLE_DATASET = "ellipticco/elliptic-data-set"
ELLIPTIC_RAW_DIR = RAW_DIR / "elliptic"
# Confirmed by actually downloading it once: Kaggle unzips this dataset into a nested
# elliptic_bitcoin_dataset/ subfolder, not directly into ELLIPTIC_RAW_DIR.
ELLIPTIC_NESTED_DIR = ELLIPTIC_RAW_DIR / "elliptic_bitcoin_dataset"
ELLIPTIC_FEATURES_CSV = ELLIPTIC_NESTED_DIR / "elliptic_txs_features.csv"

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


def ensure_downloaded() -> Path:
    """Download PaySim if not already present, return the CSV path. Raises RuntimeError on
    failure (never sys.exit) — reused by serverless/handler.py, where killing the whole worker
    process on a per-job error would be wrong. See main() below for the CLI entrypoint."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_CSV.exists():
        logger.info(f"Already present: {TARGET_CSV}")
        return TARGET_CSV

    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        raise RuntimeError(MANUAL_INSTRUCTIONS)

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logger.info(f"Downloading {KAGGLE_DATASET} ...")
    api.dataset_download_files(KAGGLE_DATASET, path=str(RAW_DIR), unzip=True)

    csvs = list(RAW_DIR.glob("*.csv"))
    if not csvs:
        raise RuntimeError("Download finished but no CSV found in data/raw/.")

    if csvs[0] != TARGET_CSV:
        shutil.move(str(csvs[0]), str(TARGET_CSV))

    logger.info(f"Saved: {TARGET_CSV}")
    return TARGET_CSV


def ensure_downloaded_elliptic() -> Path:
    """Download the Elliptic Bitcoin dataset if not already present, return the directory
    containing the three CSVs (elliptic_txs_features/classes/edgelist.csv). Same never-sys.exit
    contract as ensure_downloaded() above."""
    ELLIPTIC_RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ELLIPTIC_FEATURES_CSV.exists():
        logger.info(f"Already present: {ELLIPTIC_NESTED_DIR}")
        return ELLIPTIC_NESTED_DIR

    kaggle_creds = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_creds.exists():
        raise RuntimeError(
            f"Kaggle credentials not found (~/.kaggle/kaggle.json). See ensure_downloaded()'s "
            f"MANUAL_INSTRUCTIONS for how to get one, or download "
            f"https://www.kaggle.com/datasets/{ELLIPTIC_KAGGLE_DATASET} manually into "
            f"{ELLIPTIC_NESTED_DIR}."
        )

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    logger.info(f"Downloading {ELLIPTIC_KAGGLE_DATASET} ...")
    api.dataset_download_files(ELLIPTIC_KAGGLE_DATASET, path=str(ELLIPTIC_RAW_DIR), unzip=True)

    if not ELLIPTIC_FEATURES_CSV.exists():
        # Layout changed since we last checked — search for it rather than failing blind.
        found = list(ELLIPTIC_RAW_DIR.rglob("elliptic_txs_features.csv"))
        if not found:
            raise RuntimeError("Download finished but elliptic_txs_features.csv not found anywhere.")
        return found[0].parent

    logger.info(f"Saved to: {ELLIPTIC_NESTED_DIR}")
    return ELLIPTIC_NESTED_DIR


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        ensure_downloaded()
    except RuntimeError as e:
        logger.error(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
