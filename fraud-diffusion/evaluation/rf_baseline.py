"""Random Forest baseline on raw node features (no graph structure at all) — the calibration
point our whole leakage-fix investigation was triggered by: arXiv 2604.19514 found RF on raw
features (F1=0.821) beats every GNN tested under a strict inductive protocol on Elliptic. Meant to
run as a standing background reference alongside every real experiment, not a one-off — trains in
seconds locally, no GPU needed, so there's no reason not to always have this number on hand.

Uses the SAME processed graph .pt files and the SAME compute_metrics() as the GNN pipeline, so
results are directly comparable — this is exactly the honest baseline/GNN comparison, just without
the graph.

Usage:
    python -m evaluation.rf_baseline --processed-path data/processed/elliptic_graph.pt
"""

import argparse
import logging
from pathlib import Path

import torch
from sklearn.ensemble import RandomForestClassifier

from evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def run(processed_path: str, n_estimators: int = 300, max_depth: int | None = None, seed: int = 42) -> dict:
    data = torch.load(ROOT / processed_path, weights_only=False)
    x = data.x.numpy()
    y = data.y.numpy()

    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced",
        n_jobs=-1, random_state=seed,
    )
    clf.fit(x[data.train_mask.numpy()], y[data.train_mask.numpy()])

    val_probs = clf.predict_proba(x[data.val_mask.numpy()])[:, 1]
    test_probs = clf.predict_proba(x[data.test_mask.numpy()])[:, 1]
    val_metrics = compute_metrics(y[data.val_mask.numpy()], val_probs)
    test_metrics = compute_metrics(y[data.test_mask.numpy()], test_probs)

    logger.info(f"RF baseline ({processed_path}): Val={val_metrics} Test={test_metrics}")
    return {"val": val_metrics, "test": test_metrics}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", required=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.processed_path, args.n_estimators, args.max_depth, args.seed)
    print(result)


if __name__ == "__main__":
    main()
