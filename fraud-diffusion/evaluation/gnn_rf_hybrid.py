"""Hybrid: train a GNN (GraphSAGE/GraphSAGEDiff), extract its learned node embeddings, concatenate
with raw features, and feed that into Random Forest instead of the GNN's own linear classifier
head. Motivated by LAB_JOURNAL.md Run 52's feature ablation: RF on Elliptic's raw features alone
already beats every GNN variant tried, and local/aggregated (graph-derived) feature subsets are
COMPLEMENTARY (neither alone matches the combined score) -- suggesting RF's advantage is less
"the graph doesn't matter" and more "RF models nonlinear feature interactions far better than a
single linear layer on top of GNN embeddings does." This hands RF the GNN's structural encoding
directly instead of treating the two as competing approaches.

Deliberately duplicates (rather than reuses) training/train_gnn.py's full-batch training loop --
a documented, intentional tradeoff: reusing it would need a larger refactor of run_from_config()'s
return contract (which handler.py depends on returning a JSON-serializable dict, not a live
nn.Module), and this script only needs to support the full-batch (non-mini-batch) path, which is
all Elliptic (the current focus) ever uses. Reuses the well-tested pieces that don't need
duplicating: build_model, evaluate, EMA, FocalLoss, pick_device, set_seed.

Usage (as a RunPod job, via handler.py's config.hybrid.enabled flag, or standalone):
    python -m evaluation.gnn_rf_hybrid --config configs/elliptic_full.yaml
"""

import argparse
import logging

import numpy as np
import torch
import wandb
from sklearn.ensemble import RandomForestClassifier

from evaluation.metrics import compute_metrics
from training.ema import EMA
from training.losses import FocalLoss
from training.train_gnn import ROOT, build_model, evaluate, init_wandb, load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


def _train_gnn_for_embeddings(config: dict) -> tuple:
    """Mirrors train_gnn.py's full-batch training loop through early stopping, then returns
    (model, data, device) with the BEST checkpoint loaded -- for embedding extraction, not for
    reporting the GNN's own classifier-head metrics (see run() below, which ignores those)."""
    set_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
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
    ema_decay = config["train"].get("ema_decay", 0.0)
    ema_start_epoch = config["train"].get("ema_start_epoch", 20)
    ema, eval_model = None, None
    criterion = FocalLoss(alpha=config["loss"]["alpha"], gamma=config["loss"]["gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"],
    )

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    patience = config["train"]["patience"]

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.train_edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if ema_decay > 0 and epoch == ema_start_epoch:
            ema = EMA(model, decay=ema_decay)
            eval_model = build_model(config, in_dim=data.x.shape[1]).to(device)
        elif ema is not None:
            ema.update(model)

        eval_source = eval_model if ema is not None else model
        if ema is not None:
            eval_model.load_state_dict(ema.state_dict())
        val_metrics = evaluate(eval_source, data, data.val_mask, device, edge_index=data.val_edge_index)

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            source = ema.state_dict() if ema is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in source.items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 50 == 0 or epoch == 1:
            logger.info(f"[hybrid pretrain] epoch {epoch} val_auc={val_metrics['auc_roc']:.4f}")

        if epochs_since_improve >= patience and (ema_decay <= 0 or epoch > ema_start_epoch):
            logger.info(f"[hybrid pretrain] early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    return model, data, device, best_epoch


def run(config: dict, n_estimators: int = 300) -> dict:
    wandb_run = init_wandb(config, "gnn_rf_hybrid")
    model, data, device, best_epoch = _train_gnn_for_embeddings(config)

    model.eval()
    with torch.no_grad():
        # Full edge_index (not train_edge_index) -- fixed weights, no gradient, so this is
        # legitimate inference-time use of the complete graph, same rationale as test-time eval.
        # Moved to device just-in-time, matching train_gnn.py's own pattern (it's never resident
        # on GPU during training, only needed once here for embedding extraction).
        embeddings = model.embed(data.x, data.edge_index.to(device)).cpu().numpy()
    raw_x = data.x.cpu().numpy()
    combined_x = np.concatenate([raw_x, embeddings], axis=1)
    y = data.y.cpu().numpy()
    train_mask = data.train_mask.cpu().numpy()
    val_mask = data.val_mask.cpu().numpy()
    test_mask = data.test_mask.cpu().numpy()

    clf = RandomForestClassifier(
        n_estimators=n_estimators, class_weight="balanced", n_jobs=-1, random_state=config["seed"],
    )
    clf.fit(combined_x[train_mask], y[train_mask])

    val_probs = clf.predict_proba(combined_x[val_mask])[:, 1]
    test_probs = clf.predict_proba(combined_x[test_mask])[:, 1]
    val_metrics = compute_metrics(y[val_mask], val_probs)
    test_metrics = compute_metrics(y[test_mask], test_probs)

    logger.info(f"GNN+RF hybrid (gnn_best_epoch={best_epoch}): Val={val_metrics} Test={test_metrics}")
    wandb_run.summary["gnn_best_epoch"] = best_epoch
    for split_name, metrics in (("val", val_metrics), ("test", test_metrics)):
        for k, v in metrics.items():
            if not isinstance(v, dict):
                wandb_run.summary[f"{split_name}_{k}"] = v
    wandb.finish()

    return {"val": val_metrics, "test": test_metrics, "gnn_best_epoch": best_epoch}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-estimators", type=int, default=300)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config, args.n_estimators)
    print(result)


if __name__ == "__main__":
    main()
