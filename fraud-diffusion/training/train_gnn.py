import argparse
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch_geometric.loader import NeighborLoader

from evaluation.metrics import compute_metrics
from models.gnn.gat import GAT
from models.gnn.graphsage import GraphSAGE
from training.ema import EMA
from training.losses import FocalLoss

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(config: dict, in_dim: int) -> nn.Module:
    mcfg = config["model"]
    name = mcfg["name"]

    if name == "graphsage":
        return GraphSAGE(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
        )
    if name == "gat":
        return GAT(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            heads=mcfg.get("heads", 8),
            dropout=mcfg["dropout"],
        )
    raise ValueError(f"Unknown model.name in config: {name!r} (expected 'graphsage' or 'gat')")


def evaluate(model, data, mask, device) -> dict:
    """Full-batch evaluation: one forward pass over the whole graph."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
    return compute_metrics(y_true, probs)


def evaluate_batched(model, loader, device) -> dict:
    """Mini-batch evaluation via NeighborLoader — only the seed nodes in each batch (the first
    `batch.batch_size` nodes) are scored; the rest are sampled neighbors providing context."""
    model.eval()
    all_probs, all_y = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index)
            seed_logits = logits[: batch.batch_size]
            seed_y = batch.y[: batch.batch_size]
            all_probs.append(torch.sigmoid(seed_logits).cpu())
            all_y.append(seed_y.cpu())
    probs = torch.cat(all_probs).numpy()
    y_true = torch.cat(all_y).numpy()
    return compute_metrics(y_true, probs)


def train_epoch_batched(model, loader, criterion, optimizer, device) -> float:
    """One epoch of mini-batch training via NeighborLoader — used when the full graph's
    attention/message-passing tensors don't fit in GPU memory as a single full-batch forward
    pass (see GAT on the full PaySim graph: even hidden_dim=32/heads=4 OOM'd on a 48GB GPU).
    Only the seed nodes' loss is backpropagated; sampled neighbors provide context only."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x, batch.edge_index)
        seed_logits = logits[: batch.batch_size]
        seed_y = batch.y[: batch.batch_size]
        loss = criterion(seed_logits, seed_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


def append_journal_entry(config: dict, config_path: str, device: torch.device,
                          n_nodes: int, n_edges: int, epoch_stopped: int,
                          val_metrics: dict, test_metrics: dict) -> None:
    journal_path = ROOT / config["journal"]["path"]
    run_name = config["journal"]["run_name"]

    existing_runs = 0
    if journal_path.exists():
        existing_runs = journal_path.read_text().count("\n## [")
    run_id = existing_runs + 1

    dcfg, mcfg, lcfg, tcfg = config["data"], config["model"], config["loss"], config["train"]

    entry = f"""
## [{date.today().isoformat()}] Run {run_id} — {run_name}
- Dataset / split: PaySim ({'+'.join(dcfg['fraud_types'])}, subsample={dcfg['subsample_size']}), \
temporal {dcfg['train_frac']:.0%}/{dcfg['val_frac']:.0%}/{dcfg['test_frac']:.0%}, config={config_path}
- Graph: {n_nodes} nodes, {n_edges} directed edges, max_node_degree={dcfg['max_node_degree']}
- Model / config: {mcfg['name']} {mcfg['num_layers']}-layer, hidden={mcfg['hidden_dim']}, \
dropout={mcfg['dropout']}, {lcfg['name']}Loss(alpha={lcfg['alpha']}, gamma={lcfg['gamma']}), \
lr={tcfg['lr']}, stopped_epoch={epoch_stopped}
- Compute: {device}
- Val results: F1-macro={val_metrics['f1_macro']:.4f}, AUC-ROC={val_metrics['auc_roc']:.4f}, \
AUPRC={val_metrics['auprc']:.4f}, G-mean={val_metrics['g_mean']:.4f}
- Test results: F1-macro={test_metrics['f1_macro']:.4f}, AUC-ROC={test_metrics['auc_roc']:.4f}, \
AUPRC={test_metrics['auprc']:.4f}, G-mean={test_metrics['g_mean']:.4f}
- Observations: (fill in manually)
- Next: (fill in manually)
"""
    with open(journal_path, "a") as f:
        f.write(entry)
    print(f"Appended run {run_id} to {journal_path}")


def run_from_config(config: dict, config_path: str) -> dict:
    """Run the full train+eval pipeline for an already-loaded config. Reused by the CLI
    entrypoint (main, below) and by runpod/handler.py for serverless invocation."""
    set_seed(config["seed"])

    device = pick_device(config["train"]["device"])
    print(f"Using device: {device}")

    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)

    mini_batch = config["train"].get("mini_batch", False)

    if mini_batch:
        # Full graph stays on CPU; NeighborLoader samples small subgraphs and moves only those
        # to the GPU per batch — this is what lets a bigger model fit at all (see GAT OOM'ing
        # even at hidden_dim=32/heads=4 in full-batch mode on both 24GB and 48GB GPUs).
        num_neighbors = config["train"]["num_neighbors"]
        batch_size = config["train"]["batch_size"]
        train_loader = NeighborLoader(
            data, num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=data.train_mask, shuffle=True,
        )
        val_loader = NeighborLoader(
            data, num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=data.val_mask, shuffle=False,
        )
        test_loader = NeighborLoader(
            data, num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=data.test_mask, shuffle=False,
        )
    else:
        data = data.to(device)

    model = build_model(config, in_dim=data.x.shape[1]).to(device)

    ema_decay = config["train"].get("ema_decay", 0.0)
    ema_start_epoch = config["train"].get("ema_start_epoch", 20)
    ema = None            # created lazily at ema_start_epoch, snapshotting the model then
    eval_model = None      # reused each epoch to evaluate EMA weights without touching `model`

    criterion = FocalLoss(alpha=config["loss"]["alpha"], gamma=config["loss"]["gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # AUC-ROC (threshold-independent) is a much less noisy early-stopping signal than F1-macro
    # at a fixed 0.5 threshold: under heavy class imbalance, F1@0.5 can stay flat for many epochs
    # while the model is still learning to rank fraud higher, then jump once probabilities cross
    # the threshold — monitoring F1 there triggers early stopping on noise, not convergence.
    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    patience = config["train"]["patience"]

    for epoch in range(1, config["train"]["epochs"] + 1):
        if mini_batch:
            loss = train_epoch_batched(model, train_loader, criterion, optimizer, device)
        else:
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            loss = loss.item()

        if ema_decay > 0 and epoch == ema_start_epoch:
            ema = EMA(model, decay=ema_decay)               # snapshot current (warmed-up) weights
            eval_model = build_model(config, in_dim=data.x.shape[1]).to(device)
        elif ema is not None:
            ema.update(model)

        eval_source = eval_model if ema is not None else model
        if ema is not None:
            eval_model.load_state_dict(ema.state_dict())
        if mini_batch:
            val_metrics = evaluate_batched(eval_source, val_loader, device)
        else:
            val_metrics = evaluate(eval_source, data, data.val_mask, device)

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            source = ema.state_dict() if ema is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in source.items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss:.4f} | val_auc_roc={val_metrics['auc_roc']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}")

        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    if mini_batch:
        val_metrics = evaluate_batched(model, val_loader, device)
        test_metrics = evaluate_batched(model, test_loader, device)
    else:
        val_metrics = evaluate(model, data, data.val_mask, device)
        test_metrics = evaluate(model, data, data.test_mask, device)

    print(f"Best epoch: {best_epoch}")
    print(f"Val:  {val_metrics}")
    print(f"Test: {test_metrics}")

    append_journal_entry(
        config, config_path, device,
        n_nodes=data.num_nodes, n_edges=data.edge_index.shape[1],
        epoch_stopped=best_epoch, val_metrics=val_metrics, test_metrics=test_metrics,
    )

    return {"val": val_metrics, "test": test_metrics, "best_epoch": best_epoch, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_from_config(config, args.config)


if __name__ == "__main__":
    main()
