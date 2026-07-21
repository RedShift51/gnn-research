"""PaySim CSV -> PyG graph (Data) for transaction-node fraud classification.

Nodes = transactions (TRANSFER/CASH_OUT only, stratified-subsampled).
Edges  = pairs of transactions sharing an account (nameOrig or nameDest),
         connected to nearby-in-time transactions on the same account, capped
         at `max_node_degree` per account to keep the graph sparse.
Split  = temporal (sorted by `step`), not random.
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

from data.temporal_edges import split_edge_index_by_time

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def stratified_subsample(df: pd.DataFrame, target_size: int, seed: int) -> pd.DataFrame:
    fraud = df[df["isFraud"] == 1]
    legit = df[df["isFraud"] == 0]

    n_legit_keep = max(target_size - len(fraud), 0)
    if n_legit_keep < len(legit):
        legit = legit.sample(n=n_legit_keep, random_state=seed)

    out = pd.concat([fraud, legit]).sort_values("step").reset_index(drop=True)
    return out


def build_node_features(df: pd.DataFrame) -> np.ndarray:
    amount = np.log1p(df["amount"].to_numpy())
    old_orig = np.log1p(df["oldbalanceOrg"].to_numpy())
    new_orig = np.log1p(df["newbalanceOrig"].to_numpy())
    old_dest = np.log1p(df["oldbalanceDest"].to_numpy())
    new_dest = np.log1p(df["newbalanceDest"].to_numpy())

    is_transfer = (df["type"] == "TRANSFER").to_numpy(dtype=np.float32)

    # Deviation from expected balance bookkeeping: 0 normally, nonzero flags tampering.
    delta_orig = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
    delta_dest = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]

    orig_zeroed = ((df["oldbalanceOrg"] > 0) & (df["newbalanceOrig"] == 0)).to_numpy(dtype=np.float32)
    dest_was_zero = (df["oldbalanceDest"] == 0).to_numpy(dtype=np.float32)

    # NOTE: deliberately NOT including an exact "oldbalanceOrg == amount AND newbalanceOrig == 0"
    # feature here. It correlates ~0.99 with isFraud in PaySim (a known simulator artifact, not a
    # generalizable fraud pattern — see LAB_JOURNAL.md Run 5) and makes any model trivially "solve"
    # this dataset without learning anything transferable. If you want to study that artifact
    # deliberately, add it in a separately-named ablation config, never in this default path.

    hour_of_day = (df["step"] % 24).to_numpy(dtype=np.float32)
    is_night = ((hour_of_day < 6) | (hour_of_day > 22)).astype(np.float32)

    features = np.stack(
        [
            amount,
            old_orig,
            new_orig,
            old_dest,
            new_dest,
            is_transfer,
            delta_orig.to_numpy(dtype=np.float32),
            delta_dest.to_numpy(dtype=np.float32),
            orig_zeroed,
            dest_was_zero,
            hour_of_day,
            is_night,
        ],
        axis=1,
    ).astype(np.float32)

    return features


def build_edges(df: pd.DataFrame, max_node_degree: int) -> np.ndarray:
    """Connect transactions sharing an account to their temporal neighbors on that account."""
    edges = set()

    for account_col in ("nameOrig", "nameDest"):
        for _, group in df.groupby(account_col).groups.items():
            idxs = sorted(group)
            if len(idxs) < 2:
                continue
            for pos, i in enumerate(idxs):
                for j in idxs[pos + 1 : pos + 1 + max_node_degree]:
                    edges.add((i, j))

    if not edges:
        return np.empty((2, 0), dtype=np.int64)

    edge_arr = np.array(sorted(edges), dtype=np.int64).T  # [2, E], directed forward-in-time
    # Make undirected for message passing.
    edge_arr = np.concatenate([edge_arr, edge_arr[::-1]], axis=1)
    # .T followed by concatenate leaves this non-contiguous (confirmed: both steps individually
    # produce non-contiguous arrays here) — fine for full-batch training, but pyg-lib's CSC
    # sampler for NeighborLoader requires a contiguous edge_index and fails with a bare
    # "Input should be contiguous" otherwise.
    return np.ascontiguousarray(edge_arr)


def temporal_split_masks(n: int, train_frac: float, val_frac: float) -> tuple:
    """Returns (train_mask, val_mask, test_mask, train_end, val_end) — the row-index breakpoints
    are returned too so callers can also filter edges by the same temporal boundary (see
    data/temporal_edges.py), since row position IS temporal position here (df is sorted by `step`
    before this runs)."""
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return train_mask, val_mask, test_mask, train_end, val_end


def run_from_config(config: dict) -> Path:
    """Run the full preprocess pipeline for an already-loaded config, return the output path.
    Reused by the CLI entrypoint (main, below) and by runpod/handler.py for serverless invocation."""
    # See training/train_gnn.py's matching log line — visibility into what's actually running at
    # every stage, not just at train time, in case preprocessing itself picks up a stale/wrong
    # config (e.g. a stale worker's git checkout missing a newly-added data.subsample_size value).
    logger.info(f"paysim_preprocess config: {config}")

    data_cfg = config["data"]
    seed = config["seed"]

    raw_csv = ROOT / data_cfg["raw_csv"]
    df = pd.read_csv(raw_csv)
    df = df[df["type"].isin(data_cfg["fraud_types"])].reset_index(drop=True)

    df = stratified_subsample(df, data_cfg["subsample_size"], seed)
    logger.info(f"Subsampled to {len(df)} rows ({df['isFraud'].sum()} fraud, "
                f"{df['isFraud'].mean():.4%} rate)")

    n = len(df)
    train_mask, val_mask, test_mask, train_end, val_end = temporal_split_masks(
        n, data_cfg["train_frac"], data_cfg["val_frac"]
    )
    logger.info(f"Split sizes: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    logger.info(f"Fraud per split: train={df['isFraud'][train_mask].sum()} "
                f"val={df['isFraud'][val_mask].sum()} test={df['isFraud'][test_mask].sum()}")

    features = build_node_features(df)

    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    features = scaler.transform(features).astype(np.float32)

    edge_index = build_edges(df, data_cfg["max_node_degree"])
    logger.info(f"Built {edge_index.shape[1]} directed edges over {n} nodes")

    # Leakage-free temporal edge splits (see data/temporal_edges.py's docstring — arXiv 2604.19514
    # found standard transductive GNN setups, where every training forward pass sees the WHOLE
    # graph including val/test-period edges, produce a large F1 gap vs strict inductive eval).
    # Row position IS temporal position here (df sorted by step before edges were built), so
    # node_time is just the row index. split_edge_index_by_time uses INCLUSIVE "<=" boundaries
    # (matching Elliptic's step<=train_end_step convention), but temporal_split_masks above uses
    # EXCLUSIVE Python-slice boundaries (train_mask[:train_end] -> indices 0..train_end-1 are
    # train, so index train_end itself is the FIRST val row) -- passing train_end/val_end directly
    # would misclassify that one boundary row as train-visible, a one-row reintroduction of the
    # exact leakage this split exists to prevent. -1 aligns the two conventions.
    node_time = np.arange(n)
    train_edge_index, val_edge_index = split_edge_index_by_time(edge_index, node_time, train_end - 1, val_end - 1)
    logger.info(f"Temporal edge splits: train={train_edge_index.shape[1]} "
                f"val={val_edge_index.shape[1]} full={edge_index.shape[1]} directed edges")

    y = df["isFraud"].to_numpy(dtype=np.int64)

    data = Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_index),
        train_edge_index=torch.from_numpy(train_edge_index),
        val_edge_index=torch.from_numpy(val_edge_index),
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )

    out_path = ROOT / data_cfg["processed_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    logger.info(f"Saved graph to {out_path}")
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
