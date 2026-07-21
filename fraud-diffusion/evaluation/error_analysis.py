"""Error analysis for the RF baseline on Elliptic test predictions -- answers "which samples do we
misclassify, and does the graph structure RF ignores explain WHY?" Since RF (raw features only, no
structure) is our best classifier so far (LAB_JOURNAL.md Run 47/55), the interesting question isn't
just RF's confusion matrix but whether its errors correlate with structural properties a GNN could
exploit: if RF's false negatives are specifically fraud nodes with unusually HIGH same-class
neighbor density (structure would have caught them) that's a concrete case for a structure-aware
approach; if errors show no structural pattern, structure isn't the missing piece for those nodes.

Runs entirely locally (RF trains in ~1s, no GPU) -- no RunPod dispatch needed.

Usage:
    python -m evaluation.error_analysis --processed-path data/processed/elliptic_graph.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _neighbor_stats(edge_index: np.ndarray, y: np.ndarray, known_mask: np.ndarray, n: int) -> dict:
    """Per-node degree and fraction-of-known-neighbors-that-are-fraud, restricted to KNOWN-labeled
    neighbors (unknown-labeled nodes are mapped to y=0 upstream but aren't real "legit" evidence --
    counting them as legit neighbors would understate true fraud-neighbor density, see Run 46)."""
    src, dst = edge_index[0], edge_index[1]
    degree = np.zeros(n, dtype=np.int64)
    known_neighbors = np.zeros(n, dtype=np.int64)
    fraud_neighbors = np.zeros(n, dtype=np.int64)
    np.add.at(degree, dst, 1)
    known_src = known_mask[src]
    np.add.at(known_neighbors, dst[known_src], 1)
    fraud_src = known_src & (y[src] == 1)
    np.add.at(fraud_neighbors, dst[fraud_src], 1)
    frac_fraud = np.divide(fraud_neighbors, known_neighbors,
                           out=np.zeros(n, dtype=np.float64), where=known_neighbors > 0)
    return {"degree": degree, "known_neighbors": known_neighbors, "frac_fraud_neighbors": frac_fraud}


def run(processed_path: str, n_estimators: int = 300, seed: int = 42) -> dict:
    data = torch.load(ROOT / processed_path, weights_only=False)
    x = data.x.numpy()
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    test_mask = data.test_mask.numpy()
    known_mask = train_mask | data.val_mask.numpy() | test_mask
    n = x.shape[0]

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced",
                                  n_jobs=-1, random_state=seed)
    clf.fit(x[train_mask], y[train_mask])
    test_pred = clf.predict(x[test_mask])
    test_true = y[test_mask]

    stats = _neighbor_stats(data.edge_index.numpy(), y, known_mask, n)
    test_degree = stats["degree"][test_mask]
    test_frac_fraud = stats["frac_fraud_neighbors"][test_mask]
    test_known_neighbors = stats["known_neighbors"][test_mask]

    groups = {
        "TP (fraud, caught)": (test_true == 1) & (test_pred == 1),
        "FN (fraud, MISSED)": (test_true == 1) & (test_pred == 0),
        "TN (legit, cleared)": (test_true == 0) & (test_pred == 0),
        "FP (legit, falsely flagged)": (test_true == 0) & (test_pred == 1),
    }

    report = {}
    for name, mask in groups.items():
        if mask.sum() == 0:
            report[name] = {"n": 0}
            continue
        report[name] = {
            "n": int(mask.sum()),
            "mean_degree": float(test_degree[mask].mean()),
            "median_degree": float(np.median(test_degree[mask])),
            "mean_frac_fraud_neighbors": float(test_frac_fraud[mask].mean()),
            "mean_known_neighbors": float(test_known_neighbors[mask].mean()),
            "isolated_frac (0 known neighbors)": float((test_known_neighbors[mask] == 0).mean()),
        }

    logger.info(f"RF error analysis on {processed_path}:")
    for name, stats_dict in report.items():
        logger.info(f"  {name}: {stats_dict}")
    return report


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", required=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.processed_path, args.n_estimators, args.seed)
    print(result)


if __name__ == "__main__":
    main()
