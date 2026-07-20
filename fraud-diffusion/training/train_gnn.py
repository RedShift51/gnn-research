import argparse
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

from evaluation.metrics import compute_metrics
from models.gnn.graphsage import GraphSAGE
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


def evaluate(model, data, mask, device) -> dict:
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = torch.sigmoid(logits[mask]).cpu().numpy()
        y_true = data.y[mask].cpu().numpy()
    return compute_metrics(y_true, probs)


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
    data = data.to(device)

    model = GraphSAGE(
        in_dim=data.x.shape[1],
        hidden_dim=config["model"]["hidden_dim"],
        num_layers=config["model"]["num_layers"],
        dropout=config["model"]["dropout"],
    ).to(device)

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
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = criterion(logits[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        val_metrics = evaluate(model, data, data.val_mask, device)
        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | loss={loss.item():.4f} | val_auc_roc={val_metrics['auc_roc']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}")

        if epochs_since_improve >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
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
