"""GNN encoder trained via metric learning (triplet loss on embeddings) instead of a
classification head -- a direct empirical test of whether this fares better than everything else
tried against LAB_JOURNAL.md's regime-break "hard core" (Runs 74-77), rather than just arguing
from theory that it should fail the same way. Metric learning is naturally imbalance-robust:
triplets are constructed explicitly (anchor/positive/negative sampled in a controlled ratio), so
class frequency in the raw dataset doesn't directly set each example's gradient weight the way
plain cross-entropy does.

Classification is done post-hoc via nearest-CENTROID distance in the learned embedding space
(train-fraud centroid vs train-legit centroid), not a learned classifier head -- this is the
metric-learning-native way to classify and lets us directly compare "does the hard core sit
closer to train fraud in the LEARNED space than in raw feature space" against Run 76's raw-feature
nearest-neighbor numbers.

Deliberately duplicates (not reuses) training/train_gnn.py's training loop, matching
evaluation/gnn_rf_hybrid.py's own documented tradeoff -- this trains a totally different
objective (triplet loss on embeddings, no classification logits at all) so there's little to share
beyond build_model/pick_device/set_seed.

Usage:
    python -m evaluation.metric_learning --config configs/elliptic_full_graphsage_diff.yaml
"""

import argparse
import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb

from evaluation.metrics import compute_metrics
from training.train_gnn import ROOT, build_model, init_wandb, load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


def _sample_triplets(fraud_idx: torch.Tensor, legit_idx: torch.Tensor, n_triplets: int,
                      generator: torch.Generator) -> tuple:
    """Anchor=fraud, positive=another fraud, negative=legit -- explicitly balanced regardless of
    the true ~2.3% fraud rate, the core imbalance-robustness property of metric learning that
    plain cross-entropy (even with class weighting) doesn't share at the sampling level."""
    anchor = fraud_idx[torch.randint(0, len(fraud_idx), (n_triplets,), generator=generator)]
    positive = fraud_idx[torch.randint(0, len(fraud_idx), (n_triplets,), generator=generator)]
    negative = legit_idx[torch.randint(0, len(legit_idx), (n_triplets,), generator=generator)]
    return anchor, positive, negative


def _centroid_scores(embeddings: torch.Tensor, fraud_centroid: torch.Tensor,
                      legit_centroid: torch.Tensor) -> np.ndarray:
    """Fraud-likeness score for nearest-centroid classification: positive = closer to the fraud
    centroid than the legit one. Sigmoid maps this to (0,1) with 0.5 exactly at the equidistant
    boundary, so evaluation/metrics.py's compute_metrics (which thresholds at 0.5) does the right
    thing without needing a separately-fit calibration."""
    dist_fraud = (embeddings - fraud_centroid).pow(2).sum(-1).sqrt()
    dist_legit = (embeddings - legit_centroid).pow(2).sum(-1).sqrt()
    return torch.sigmoid(dist_legit - dist_fraud).cpu().numpy()


def run(config: dict, n_triplets_per_epoch: int = 2000, margin: float = 1.0) -> dict:
    wandb_run = init_wandb(config, "metric_learning")
    set_seed(config["seed"])
    device = pick_device(config["train"]["device"])
    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)

    data.x = data.x.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)
    data.train_edge_index = data.train_edge_index.to(device)
    data.val_edge_index = data.val_edge_index.to(device)

    model = build_model(config, in_dim=data.x.shape[1]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"],
    )
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, p=2)
    generator = torch.Generator(device="cpu").manual_seed(config["seed"])

    train_fraud_idx = data.train_mask.nonzero(as_tuple=True)[0][data.y[data.train_mask] == 1].cpu()
    train_legit_idx = data.train_mask.nonzero(as_tuple=True)[0][data.y[data.train_mask] == 0].cpu()
    logger.info(f"Train fraud={len(train_fraud_idx)}, train legit={len(train_legit_idx)}")

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    patience = config["train"]["patience"]

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        embeddings = model.embed(data.x, data.train_edge_index)
        embeddings = F.normalize(embeddings, dim=-1)  # unit-norm, standard metric-learning practice

        anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
        loss = triplet_loss_fn(embeddings[anchor_idx.to(device)], embeddings[pos_idx.to(device)], embeddings[neg_idx.to(device)])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_embeddings = F.normalize(model.embed(data.x, data.val_edge_index), dim=-1)
            fraud_centroid = val_embeddings[train_fraud_idx.to(device)].mean(dim=0)
            legit_centroid = val_embeddings[train_legit_idx.to(device)].mean(dim=0)
            val_scores = _centroid_scores(val_embeddings[data.val_mask], fraud_centroid, legit_centroid)
            val_y = data.y[data.val_mask].cpu().numpy()
            val_metrics = compute_metrics(val_y, val_scores)

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 25 == 0 or epoch == 1:
            logger.info(f"epoch {epoch} triplet_loss={loss.item():.4f} val_auc={val_metrics['auc_roc']:.4f}")
        wandb.log({"epoch": epoch, "triplet_loss": loss.item(), "val_auc": val_metrics["auc_roc"]}, step=epoch)

        if epochs_since_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_edge_index = data.edge_index.to(device)
        test_embeddings = F.normalize(model.embed(data.x, test_edge_index), dim=-1)
        fraud_centroid = test_embeddings[train_fraud_idx.to(device)].mean(dim=0)
        legit_centroid = test_embeddings[train_legit_idx.to(device)].mean(dim=0)
        test_scores = _centroid_scores(test_embeddings[data.test_mask], fraud_centroid, legit_centroid)

    test_y = data.y[data.test_mask].cpu().numpy()
    test_metrics = compute_metrics(test_y, test_scores)

    logger.info(f"Metric learning (best_epoch={best_epoch}): Test={test_metrics}")
    wandb_run.summary["best_epoch"] = best_epoch
    for k, v in test_metrics.items():
        if not isinstance(v, dict):
            wandb_run.summary[f"test_{k}"] = v
    wandb.finish()

    return {
        "test": test_metrics, "best_epoch": best_epoch,
        "test_scores": test_scores.tolist(), "test_y": test_y.tolist(),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-triplets", type=int, default=2000)
    parser.add_argument("--margin", type=float, default=1.0)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config, args.n_triplets, args.margin)
    print({k: v for k, v in result.items() if k not in ("test_scores", "test_y")})


if __name__ == "__main__":
    main()
