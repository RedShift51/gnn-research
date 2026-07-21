"""Elliptic Bitcoin dataset -> PyG graph (Data) for node classification.

Unlike PaySim, this dataset already comes as a graph — no construction from shared
accounts/entities needed. Nodes = Bitcoin transactions (203,769 total), edges = payment flows
(elliptic_txs_edgelist.csv, 234,355 total). Labels (elliptic_txs_classes.csv): class "1" =
illicit (fraud), class "2" = licit, "unknown" = unlabeled (~77% of nodes) — unknown nodes are
masked out of train/val/test but stay in the graph for message passing.

Split is temporal by the dataset's own 49 time steps (steps 1-34 train, 35-41 val, 42-49 test),
matching the original Elliptic paper's temporal-generalization setup — not a random split.

Verified directly against the real downloaded files before writing this (column counts, header
presence, class distribution, step range) rather than assumed from memory.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent
N_FEATURES = 165
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_from_config(config: dict) -> Path:
    logger.info(f"elliptic_preprocess config: {config}")
    data_cfg = config["data"]
    raw_dir = ROOT / data_cfg["raw_dir"]

    # No header: col 0 = txId, col 1 = time step (1-49), cols 2-166 = 165 features.
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "step"] + [f"f{i}" for i in range(N_FEATURES)]

    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")  # txId, class
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")   # txId1, txId2

    df = features.merge(classes, on="txId", how="left")
    # "1"=illicit(fraud)->1, "2"=licit->0, "unknown"->-1 (excluded from masks below, never
    # contributes to loss/eval; placeholder 0 substituted into the actual y tensor since masked-
    # out entries are never read).
    df["y"] = df["class"].map({"1": 1, "2": 0})
    known = df["y"].notna().to_numpy()
    df["y"] = df["y"].fillna(-1).astype(int)

    n = len(df)
    step = df["step"].to_numpy()
    train_end = data_cfg["train_end_step"]
    val_end = data_cfg["val_end_step"]

    train_mask = (step <= train_end) & known
    val_mask = (step > train_end) & (step <= val_end) & known
    test_mask = (step > val_end) & known

    logger.info(f"Nodes: {n} ({known.sum()} with known label, {n - known.sum()} unknown)")
    logger.info(f"Split sizes: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    logger.info(f"Fraud per split: train={df['y'].to_numpy()[train_mask].sum()} "
                f"val={df['y'].to_numpy()[val_mask].sum()} test={df['y'].to_numpy()[test_mask].sum()}")

    feature_cols = [f"f{i}" for i in range(N_FEATURES)]
    x = df[feature_cols].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(x[train_mask])
    x = scaler.transform(x).astype(np.float32)

    txid_to_idx = {tx: i for i, tx in enumerate(df["txId"])}
    src = edges["txId1"].map(txid_to_idx)
    dst = edges["txId2"].map(txid_to_idx)
    valid = src.notna() & dst.notna()
    dropped = len(edges) - valid.sum()
    if dropped:
        logger.info(f"Dropped {dropped} edges referencing unknown txIds")
    edge_index = np.stack([src[valid].to_numpy(dtype=np.int64), dst[valid].to_numpy(dtype=np.int64)])
    edge_index = np.concatenate([edge_index, edge_index[::-1]], axis=1)  # undirected
    # .T/slicing tricks like this leave a non-contiguous array — pyg-lib's NeighborLoader sampler
    # (used for GAT mini-batch training) fails outright on that. See LAB_JOURNAL.md's PaySim bug.
    edge_index = np.ascontiguousarray(edge_index)

    y = df["y"].to_numpy(dtype=np.int64)
    y = np.where(y == -1, 0, y)

    data = Data(
        x=torch.from_numpy(x),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )

    out_path = ROOT / data_cfg["processed_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    logger.info(f"Saved graph to {out_path} ({data.num_nodes} nodes, {edge_index.shape[1]} directed edges)")
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_from_config(config)


if __name__ == "__main__":
    main()
