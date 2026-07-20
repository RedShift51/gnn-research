"""PaySim CSV -> PyG graph (Data) for transaction-node fraud classification.

Nodes = transactions (TRANSFER/CASH_OUT only, stratified-subsampled).
Edges  = pairs of transactions sharing an account (nameOrig or nameDest),
         connected to nearby-in-time transactions on the same account, capped
         at `max_node_degree` per account to keep the graph sparse.
Split  = temporal (sorted by `step`), not random.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent


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
    return edge_arr


def temporal_split_masks(n: int, train_frac: float, val_frac: float) -> tuple:
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train_mask = np.zeros(n, dtype=bool)
    val_mask = np.zeros(n, dtype=bool)
    test_mask = np.zeros(n, dtype=bool)

    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True

    return train_mask, val_mask, test_mask


def run_from_config(config: dict) -> Path:
    """Run the full preprocess pipeline for an already-loaded config, return the output path.
    Reused by the CLI entrypoint (main, below) and by runpod/handler.py for serverless invocation."""
    data_cfg = config["data"]
    seed = config["seed"]

    raw_csv = ROOT / data_cfg["raw_csv"]
    df = pd.read_csv(raw_csv)
    df = df[df["type"].isin(data_cfg["fraud_types"])].reset_index(drop=True)

    df = stratified_subsample(df, data_cfg["subsample_size"], seed)
    print(f"Subsampled to {len(df)} rows ({df['isFraud'].sum()} fraud, "
          f"{df['isFraud'].mean():.4%} rate)")

    n = len(df)
    train_mask, val_mask, test_mask = temporal_split_masks(
        n, data_cfg["train_frac"], data_cfg["val_frac"]
    )
    print(f"Split sizes: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    print(f"Fraud per split: train={df['isFraud'][train_mask].sum()} "
          f"val={df['isFraud'][val_mask].sum()} test={df['isFraud'][test_mask].sum()}")

    features = build_node_features(df)

    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    features = scaler.transform(features).astype(np.float32)

    edge_index = build_edges(df, data_cfg["max_node_degree"])
    print(f"Built {edge_index.shape[1]} directed edges over {n} nodes")

    y = df["isFraud"].to_numpy(dtype=np.int64)

    data = Data(
        x=torch.from_numpy(features),
        edge_index=torch.from_numpy(edge_index),
        y=torch.from_numpy(y),
        train_mask=torch.from_numpy(train_mask),
        val_mask=torch.from_numpy(val_mask),
        test_mask=torch.from_numpy(test_mask),
    )

    out_path = ROOT / data_cfg["processed_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"Saved graph to {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_from_config(config)


if __name__ == "__main__":
    main()
