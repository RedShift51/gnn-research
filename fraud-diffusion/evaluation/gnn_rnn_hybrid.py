"""GraphSAGEDiff + GRU hybrid: the GNN aggregates card1/addr1-sharing neighbors as an unordered
set (mean/diff), never modeling temporal ORDER within that group; the GRU branch explicitly
encodes each transaction's own entity's (card1's) prior transactions in time order. Motivated by
2026-07-22's discussion of temporal-sequence techniques (diffusion forcing / self-forcing /
framepack) -- none of those transplant directly (they're generative-video-specific mechanisms
solving problems this discriminative, mostly-static-graph task doesn't have), but the one
legitimate idea underneath them -- a per-entity temporal sequence encoder -- is real and testable
on its own merits. IEEE-CIS only: Elliptic's nodes are one-off transactions with no recurring
entity, so there's no sequence to model there at all (see LAB_JOURNAL.md's temporal-technique
discussion).

Deliberately a separate training loop (not a train_gnn.py extension) -- same rationale as
evaluation/metric_learning.py and evaluation/gnn_rf_hybrid.py: this model's forward signature
(x, edge_index, seq_x) doesn't fit train_gnn.py's generic model-agnostic (x, edge_index) dispatch,
and retrofitting that dispatch for every existing model class isn't worth the risk to the
well-tested main pipeline for one architecture.

Mini-batch only: IEEE-CIS's ~590K-node/41M-edge graph requires it (see
configs/ieee_cis_graphsage_diff_minibatch.yaml's comment) -- full-batch isn't implemented here.

Usage:
    python -m evaluation.gnn_rnn_hybrid --config configs/ieee_cis_gnn_rnn.yaml
"""

import argparse
import logging

import torch
import wandb
from torch_geometric.loader import NeighborLoader

from evaluation.metrics import compute_metrics
from models.gnn.graphsage_rnn import GraphSAGERNN
from training.losses import FocalLoss
from training.train_gnn import ROOT, _graph_view, init_wandb, load_config, pick_device, set_seed

logger = logging.getLogger(__name__)


def _gather_seq_x(x_cpu: torch.Tensor, seq_indices_cpu: torch.Tensor, seed_global_ids: torch.Tensor,
                   device: torch.device) -> torch.Tensor:
    """Builds the [batch, seq_len, in_dim] sequence tensor for a batch's seed nodes, gathering
    directly from the FULL (CPU-resident) feature matrix by precomputed global indices --
    independent of whatever subgraph NeighborLoader happened to sample for the GNN branch, since a
    node's own entity history is a separate structure from its graph neighborhood. -1 padding
    positions get zeroed out after the gather (their padding index is clamped to 0 first only to
    make the gather itself valid; the real effect is entirely from the subsequent masking)."""
    seq_idx = seq_indices_cpu[seed_global_ids.cpu()]  # [batch, seq_len], -1 = padding
    mask = (seq_idx >= 0).unsqueeze(-1)  # [batch, seq_len, 1]
    gathered = x_cpu[seq_idx.clamp(min=0)]  # [batch, seq_len, in_dim]
    seq_x = torch.where(mask, gathered, torch.zeros_like(gathered))
    return seq_x.to(device)


def run(config: dict, rnn_hidden_dim: int = 64) -> dict:
    wandb_run = init_wandb(config, "gnn_rnn_hybrid")
    set_seed(config["seed"])
    device = pick_device(config["train"]["device"])
    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)
    assert hasattr(data, "seq_indices"), (
        "config.data.entity_seq_len must be set during preprocessing to populate data.seq_indices "
        "-- see data/ieee_cis_preprocess.py's build_entity_sequences"
    )
    x_cpu = data.x  # stays on CPU; NeighborLoader moves only sampled subgraphs to device per batch
    seq_indices_cpu = data.seq_indices

    model = GraphSAGERNN(
        in_dim=data.x.shape[1], hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"], dropout=config["model"]["dropout"],
        rnn_hidden_dim=rnn_hidden_dim,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"],
    )
    lcfg = config["loss"]
    criterion = FocalLoss(alpha=lcfg["alpha"], gamma=lcfg["gamma"])

    num_neighbors = config["train"]["num_neighbors"]
    batch_size = config["train"]["batch_size"]
    train_loader = NeighborLoader(
        _graph_view(data, data.train_edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.train_mask, shuffle=True,
    )
    val_loader = NeighborLoader(
        _graph_view(data, data.val_edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.val_mask, shuffle=False,
    )
    test_loader = NeighborLoader(
        _graph_view(data, data.edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=data.test_mask, shuffle=False,
    )

    def _evaluate(loader, return_raw=False):
        model.eval()
        all_probs, all_y = [], []
        with torch.no_grad():
            for batch in loader:
                seed_global_ids = batch.n_id[: batch.batch_size]
                seq_x = _gather_seq_x(x_cpu, seq_indices_cpu, seed_global_ids, device)
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, seq_x)
                seed_logits = logits[: batch.batch_size]
                seed_y = batch.y[: batch.batch_size]
                all_probs.append(torch.sigmoid(seed_logits).cpu())
                all_y.append(seed_y.cpu())
        probs = torch.cat(all_probs).numpy()
        y_true = torch.cat(all_y).numpy()
        metrics = compute_metrics(y_true, probs)
        if return_raw:
            return metrics, probs, y_true
        return metrics

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    patience = config["train"]["patience"]

    for epoch in range(1, config["train"]["epochs"] + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            seed_global_ids = batch.n_id[: batch.batch_size]
            seq_x = _gather_seq_x(x_cpu, seq_indices_cpu, seed_global_ids, device)
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch.x, batch.edge_index, seq_x)
            seed_logits = logits[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]
            loss = criterion(seed_logits, seed_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_metrics = _evaluate(val_loader)
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        logger.info(f"Epoch {epoch:3d} | loss={train_loss:.4f} | val_auc_roc={val_metrics['auc_roc']:.4f} "
                    f"| val_f1_macro={val_metrics['f1_macro']:.4f}")
        wandb.log({"epoch": epoch, "train_loss": train_loss, "val": val_metrics}, step=epoch)

        if epochs_since_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_metrics, test_probs, test_y = _evaluate(test_loader, return_raw=True)

    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Test: {test_metrics}")
    wandb_run.summary["best_epoch"] = best_epoch
    for k, v in test_metrics.items():
        if not isinstance(v, dict):
            wandb_run.summary[f"test_{k}"] = v
    wandb.finish()

    return {
        "test": test_metrics, "best_epoch": best_epoch,
        "test_predictions": {"probs": test_probs.tolist(), "y_true": test_y.tolist()},
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config)
    print({k: v for k, v in result.items() if k != "test_predictions"})


if __name__ == "__main__":
    main()
