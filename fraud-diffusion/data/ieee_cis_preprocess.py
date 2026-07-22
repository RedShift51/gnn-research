"""IEEE-CIS Fraud Detection (train_transaction.csv + train_identity.csv) -> PyG graph (Data) for
transaction-node fraud classification -- the third dataset (real data, per project priority),
after PaySim (simulated) and Elliptic (real but graph-native/Bitcoin, not a typical fraud-tabular
setting). IEEE-CIS is real e-commerce transaction data with NO native graph structure -- edges have
to be constructed from shared entity-identifier columns, same philosophy as PaySim's account-
sharing edges (see data/paysim_preprocess.py's build_edges).

Nodes = transactions (labeled: isFraud in {0,1}).
Edges  = pairs of transactions sharing an entity key (card1 or addr1 -- common proxies for "same
         underlying account/user" in public IEEE-CIS solutions, since there's no direct user ID),
         connected to nearby-in-time transactions sharing that key, capped at `max_node_degree`
         per key to keep the graph sparse (same rationale as PaySim -- a shared card1/addr1 can be
         held by very many transactions; a full clique would blow up the graph).
Split  = temporal (sorted by TransactionDT, seconds since an arbitrary reference point), not random.

Feature scope (deliberate MVP-slice decision, not a limitation of the format): TransactionAmt,
ProductCD, card1-card6, addr1/addr2, dist1/dist2, P_emaildomain/R_emaildomain, C1-C14, D1-D15,
M1-M9, plus DeviceType/DeviceInfo from the identity table. The 339 anonymized V-columns are
SKIPPED for this first slice (heavy missingness, mostly-correlated blocks, would roughly triple
feature count) -- can be added later as an enrichment pass, consistent with this project's
established "MVP slice first, expand later" pattern (see PaySim's original scope decision).
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

NUMERIC_COLS = (
    ["TransactionAmt"]
    + [f"C{i}" for i in range(1, 15)]
    + [f"D{i}" for i in range(1, 16)]
)
# Treated as frequency-encoded IDs, not one-hot -- card1 alone has >10k unique values, one-hot
# would explode dimensionality; frequency (count in TRAIN) is a standard, leakage-safe way to turn
# a high-cardinality categorical/ID column into a single informative numeric feature.
CATEGORICAL_COLS = (
    ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "dist1", "dist2", "P_emaildomain", "R_emaildomain"]
    + [f"M{i}" for i in range(1, 10)]
    + ["DeviceType", "DeviceInfo"]
)
ENTITY_EDGE_COLS = ["card1", "addr1"]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _frequency_encode(df: pd.DataFrame, col: str, train_mask: np.ndarray) -> np.ndarray:
    """Replace each category with its occurrence COUNT in the train split only (unseen/NaN -> 0)
    -- avoids leaking val/test category frequencies into the encoding, and turns an arbitrary
    high-cardinality categorical/ID column into a single numeric feature without one-hot blowup."""
    counts = df[col][train_mask].value_counts()
    return df[col].map(counts).fillna(0).to_numpy(dtype=np.float32)


def build_node_features(df: pd.DataFrame, train_mask: np.ndarray) -> np.ndarray:
    numeric = np.stack(
        [np.log1p(df["TransactionAmt"].to_numpy(dtype=np.float64)).astype(np.float32)]
        + [df[c].fillna(-999).to_numpy(dtype=np.float32) for c in NUMERIC_COLS[1:]],
        axis=1,
    )
    categorical = np.stack(
        [_frequency_encode(df, c, train_mask) for c in CATEGORICAL_COLS], axis=1,
    )
    return np.concatenate([numeric, categorical], axis=1).astype(np.float32)


def build_edges(df: pd.DataFrame, max_node_degree: int) -> np.ndarray:
    """Connect transactions sharing an entity key (card1 or addr1) to their temporal neighbors on
    that key -- same construction as data/paysim_preprocess.py's build_edges, adapted to IEEE-CIS's
    entity columns. NaN entity values are skipped (a shared NaN does NOT mean a shared entity)."""
    edges = set()

    for entity_col in ENTITY_EDGE_COLS:
        for key, group in df.groupby(entity_col).groups.items():
            if pd.isna(key):
                continue
            idxs = sorted(group)
            if len(idxs) < 2:
                continue
            for pos, i in enumerate(idxs):
                for j in idxs[pos + 1 : pos + 1 + max_node_degree]:
                    edges.add((i, j))

    if not edges:
        return np.empty((2, 0), dtype=np.int64)

    edge_arr = np.array(sorted(edges), dtype=np.int64).T  # [2, E], directed forward-in-time
    edge_arr = np.concatenate([edge_arr, edge_arr[::-1]], axis=1)  # undirected
    return np.ascontiguousarray(edge_arr)  # NeighborLoader's CSC sampler needs contiguous memory


def build_entity_sequences(df: pd.DataFrame, entity_col: str, seq_len: int) -> np.ndarray:
    """For each transaction, the indices of its own entity's (card1's) previous `seq_len`
    transactions, oldest-first, RIGHT-aligned (most recent immediately before it in the last
    column; -1 padding at the START where fewer than seq_len exist). Feeds an RNN branch that
    explicitly encodes each transaction's own entity's ORDERED recent history -- distinct from
    build_edges' GNN aggregation, which treats same-entity neighbors as an unordered set (mean/
    diff aggregation) and never models temporal order among them. NaN entity values get no
    sequence at all (an all-padding row) -- a shared NaN isn't a shared entity, same rationale as
    build_edges."""
    n = len(df)
    seq_idx = np.full((n, seq_len), -1, dtype=np.int64)
    for key, group in df.groupby(entity_col).groups.items():
        if pd.isna(key):
            continue
        idxs = sorted(group)
        for pos, i in enumerate(idxs):
            start = max(0, pos - seq_len)
            prev = idxs[start:pos]
            seq_idx[i, seq_len - len(prev):] = prev
    return seq_idx


def temporal_split_masks(n: int, train_frac: float, val_frac: float) -> tuple:
    """Same convention as data/paysim_preprocess.py's temporal_split_masks -- row position IS
    temporal position here (df sorted by TransactionDT before this runs)."""
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
    logger.info(f"ieee_cis_preprocess config: {config}")
    data_cfg = config["data"]

    transaction_csv = ROOT / data_cfg["transaction_csv"]
    identity_csv = ROOT / data_cfg["identity_csv"]

    txn = pd.read_csv(transaction_csv)
    identity = pd.read_csv(identity_csv)
    df = txn.merge(identity, on="TransactionID", how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    subsample_size = data_cfg.get("subsample_size")
    if subsample_size is not None and subsample_size < len(df):
        seed = config["seed"]
        fraud = df[df["isFraud"] == 1]
        legit = df[df["isFraud"] == 0]
        # subsample_size smaller than the real fraud count would silently zero out n_legit_keep
        # below, producing an all-fraud graph with no legit nodes at all -- caught directly via a
        # 5000-row smoke test on IEEE-CIS's ~20663 real fraud rows (100% fraud rate came back).
        # A misconfiguration this silent deserves a loud failure, not a quietly broken dataset.
        if subsample_size < len(fraud):
            raise ValueError(
                f"data.subsample_size ({subsample_size}) is smaller than the real fraud count "
                f"({len(fraud)}) -- would silently drop ALL legit rows. Use a larger subsample_size "
                f"or null (full dataset)."
            )
        n_legit_keep = subsample_size - len(fraud)
        if n_legit_keep < len(legit):
            legit = legit.sample(n=n_legit_keep, random_state=seed)
        df = pd.concat([fraud, legit]).sort_values("TransactionDT").reset_index(drop=True)

    logger.info(f"Rows: {len(df)} ({df['isFraud'].sum()} fraud, {df['isFraud'].mean():.4%} rate)")

    n = len(df)
    train_mask, val_mask, test_mask, train_end, val_end = temporal_split_masks(
        n, data_cfg["train_frac"], data_cfg["val_frac"]
    )
    logger.info(f"Split sizes: train={train_mask.sum()} val={val_mask.sum()} test={test_mask.sum()}")
    logger.info(f"Fraud per split: train={df['isFraud'][train_mask].sum()} "
                f"val={df['isFraud'][val_mask].sum()} test={df['isFraud'][test_mask].sum()}")

    features = build_node_features(df, train_mask)
    scaler = StandardScaler()
    scaler.fit(features[train_mask])
    features = scaler.transform(features).astype(np.float32)

    edge_index = build_edges(df, data_cfg["max_node_degree"])
    logger.info(f"Built {edge_index.shape[1]} directed edges over {n} nodes")

    seq_len = data_cfg.get("entity_seq_len")
    seq_indices = None
    if seq_len:
        # Causal by construction: for any node, build_entity_sequences only ever looks at STRICTLY
        # earlier same-entity rows in the globally time-sorted df, regardless of train/val/test
        # split -- a test-period node's sequence may legitimately include train/val-period rows
        # (that's just real, earlier history, not leakage), and no node's sequence can ever include
        # a later row. No extra leakage-guard needed, unlike edge_index above.
        seq_indices = build_entity_sequences(df, data_cfg.get("entity_seq_col", "card1"), seq_len)
        logger.info(f"Built entity sequences (col={data_cfg.get('entity_seq_col', 'card1')}, "
                    f"seq_len={seq_len}): {(seq_indices >= 0).any(axis=1).sum()}/{n} nodes have >=1 prior transaction")

    # Leakage-free temporal edge splits (see data/temporal_edges.py; arXiv 2604.19514) -- same
    # -1 boundary-alignment rationale as data/paysim_preprocess.py (temporal_split_masks uses
    # exclusive Python-slice boundaries, split_edge_index_by_time uses inclusive "<=").
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
    if seq_indices is not None:
        data.seq_indices = torch.from_numpy(seq_indices)

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
