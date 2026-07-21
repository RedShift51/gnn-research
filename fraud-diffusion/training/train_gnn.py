import argparse
import logging
import os
import random
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from evaluation.metrics import compute_metrics
from models.gnn.gat import GAT
from models.gnn.graphsage import GraphSAGE, GraphSAGEDiff, GraphSAGEGated
from training.ema import EMA
from training.losses import FocalLoss

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


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
            classifier_hidden_dim=mcfg.get("classifier_hidden_dim"),
            feature_encoder_hidden_dim=mcfg.get("feature_encoder_hidden_dim"),
        )
    if name == "graphsage_diff":
        return GraphSAGEDiff(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            classifier_hidden_dim=mcfg.get("classifier_hidden_dim"),
            feature_encoder_hidden_dim=mcfg.get("feature_encoder_hidden_dim"),
        )
    if name == "graphsage_gated":
        return GraphSAGEGated(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            dropout=mcfg["dropout"],
            classifier_hidden_dim=mcfg.get("classifier_hidden_dim"),
            feature_encoder_hidden_dim=mcfg.get("feature_encoder_hidden_dim"),
        )
    if name == "gat":
        return GAT(
            in_dim=in_dim,
            hidden_dim=mcfg["hidden_dim"],
            num_layers=mcfg["num_layers"],
            heads=mcfg.get("heads", 8),
            dropout=mcfg["dropout"],
        )
    raise ValueError(
        f"Unknown model.name in config: {name!r} "
        "(expected 'graphsage', 'graphsage_diff', 'graphsage_gated', or 'gat')"
    )


def build_oversampled_input_nodes(data, mask, target_fraud_frac: float, seed: int) -> torch.Tensor:
    """Repeat fraud node indices so they make up `target_fraud_frac` of the training seed-node
    pool NeighborLoader draws batches from. Mini-batch training sees far sparser fraud signal per
    step than full-batch (e.g. ~2 fraud nodes per batch of 1024 out of 3643 total train-fraud on
    the full PaySim graph) — oversampling the seeds directly, rather than only tuning EMA/patience
    around it, addresses that at the source. Only used for the TRAIN loader; val/test keep the
    true distribution so evaluation stays honest."""
    idx = mask.nonzero(as_tuple=True)[0]
    y = data.y[idx]
    fraud_idx = idx[y == 1]
    legit_idx = idx[y == 0]
    if fraud_idx.numel() == 0:
        return idx

    repeats = max(1, round((target_fraud_frac * legit_idx.numel())
                           / ((1 - target_fraud_frac) * fraud_idx.numel())))
    oversampled_fraud = fraud_idx.repeat(repeats)
    combined = torch.cat([oversampled_fraud, legit_idx])

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(combined.numel(), generator=generator)
    return combined[perm]


def _graph_view(data, edge_index):
    """Same node features/labels, different edge_index — used to build separate NeighborLoaders
    per split so mini-batch training never samples neighbors across val/test-period edges either
    (the mini-batch analogue of the full-batch fix in evaluate(); see data/temporal_edges.py).
    Deliberately NOT data.clone() + override: NeighborLoader only reads x/y/edge_index from the
    Data it's given, and cloning the whole object would carry all three edge_index variants
    (train/val/full) into every one of the three loaders' underlying Data — tripling memory on
    PaySim's large graph (which has a real OOM history, see LAB_JOURNAL.md's GAT runs) for
    tensors that loader will never touch."""
    return Data(x=data.x, y=data.y, edge_index=edge_index)


def evaluate(model, data, mask, device, edge_index=None) -> dict:
    """Full-batch evaluation: one forward pass over the given edge_index (defaults to the full
    graph). Pass data.val_edge_index/data.edge_index explicitly — see run_from_config's callers —
    never the raw attribute name, so it's obvious at each call site which temporal edge view is
    in use (train forward passes must NEVER default to the full graph; see
    data/temporal_edges.py)."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index if edge_index is None else edge_index)
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
                          n_nodes: int, n_edges: int, n_train_edges: int, n_val_edges: int,
                          epoch_stopped: int, val_metrics: dict, test_metrics: dict) -> None:
    journal_path = ROOT / config["journal"]["path"]
    run_name = config["journal"]["run_name"]

    existing_runs = 0
    if journal_path.exists():
        existing_runs = journal_path.read_text().count("\n## [")
    run_id = existing_runs + 1

    dcfg, mcfg, lcfg, tcfg = config["data"], config["model"], config["loss"], config["train"]

    dataset_name = dcfg.get("dataset", "paysim")
    if dataset_name == "paysim":
        dataset_desc = (f"PaySim ({'+'.join(dcfg['fraud_types'])}, subsample={dcfg['subsample_size']}), "
                        f"temporal {dcfg['train_frac']:.0%}/{dcfg['val_frac']:.0%}/{dcfg['test_frac']:.0%}")
    elif dataset_name == "elliptic":
        dataset_desc = (f"Elliptic Bitcoin, temporal by step (train<={dcfg['train_end_step']}, "
                        f"val<={dcfg['val_end_step']}, test after)")
    else:
        dataset_desc = f"{dataset_name} (config={dcfg})"

    entry = f"""
## [{date.today().isoformat()}] Run {run_id} — {run_name}
- Dataset / split: {dataset_desc}, config={config_path}
- Graph: {n_nodes} nodes, {n_edges} directed edges (train-visible: {n_train_edges}, \
val-visible: {n_val_edges}) — leakage-free temporal edge split, see data/temporal_edges.py
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
    logger.info(f"Appended run {run_id} to {journal_path}")


def init_wandb(config: dict, config_path: str):
    """Opt-out-by-default: logs to wandb whenever WANDB_API_KEY is present in the environment
    (local shell via infra/load_secrets.sh, or the serverless container's template env), falls
    back to a disabled no-op run otherwise so training never hangs waiting on an interactive
    login prompt (there's no stdin inside a RunPod worker). config["wandb"]["enabled"]=False
    overrides either way, for quick local smoke tests you don't want cluttering the project."""
    wcfg = config.get("wandb", {})
    if not wcfg.get("enabled", True):
        return wandb.init(mode="disabled")
    mode = wcfg.get("mode") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")
    run_name = config.get("journal", {}).get("run_name", config_path)
    return wandb.init(
        project=wcfg.get("project", "fraud-diffusion"),
        name=run_name,
        group=config["data"].get("dataset", "paysim"),
        config=config,
        mode=mode,
        reinit=True,
    )


def run_from_config(config: dict, config_path: str, git_commit: str | None = None) -> dict:
    """Run the full train+eval pipeline for an already-loaded config. Reused by the CLI
    entrypoint (main, below) and by runpod/handler.py for serverless invocation. git_commit, when
    passed by handler.py, records which exact worker checkout produced this run in wandb too (not
    just the job's returned output) — the same "detect a stale worker" insurance as config_label,
    now on both sides."""
    set_seed(config["seed"])

    # RunPod reuses the same long-lived worker process across many jobs (see entrypoint.sh's own
    # comment on this, and LAB_JOURNAL.md's stale-worker incidents) — within that one process,
    # PyTorch's CUDA caching allocator can retain "reserved but unallocated" memory from a PREVIOUS
    # job's now-freed tensors, never returning it to the OS. Confirmed directly: a PaySim job
    # OOM'd with "15.42 GiB in use" within ~84s of starting, at the same backward() call that
    # succeeded fine on a fresh worker before. Clearing at the start of every job is cheap
    # insurance (a no-op if the cache is already clean) against a fresh job inheriting a dirty
    # cache from whatever ran here before it.
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Print the actual config being used, in full, before anything else — a stale/warm serverless
    # worker silently running the wrong config (e.g. defaulting away from an unrecognized
    # config_dict key) is otherwise invisible until you notice the results look off. Cheap
    # insurance against exactly that (see LAB_JOURNAL.md's caught 40%-oversample incident).
    logger.info(f"Config path/label: {config_path}")
    logger.info(f"Full config: {config}")

    wandb_run = init_wandb(config, config_path)
    if git_commit is not None:
        wandb_run.summary["worker_git_commit"] = git_commit

    device = pick_device(config["train"]["device"])
    logger.info(f"Using device: {device}")

    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)
    wandb_run.summary["n_nodes"] = data.num_nodes
    wandb_run.summary["n_edges"] = data.edge_index.shape[1]
    wandb_run.summary["n_train_edges"] = data.train_edge_index.shape[1]
    wandb_run.summary["n_val_edges"] = data.val_edge_index.shape[1]

    mini_batch = config["train"].get("mini_batch", False)

    if mini_batch:
        # Full graph stays on CPU; NeighborLoader samples small subgraphs and moves only those
        # to the GPU per batch — this is what lets a bigger model fit at all (see GAT OOM'ing
        # even at hidden_dim=32/heads=4 in full-batch mode on both 24GB and 48GB GPUs).
        num_neighbors = config["train"]["num_neighbors"]
        batch_size = config["train"]["batch_size"]

        oversample_frac = config["train"].get("oversample_fraud_frac", 0.0)
        if oversample_frac > 0:
            train_input_nodes = build_oversampled_input_nodes(
                data, data.train_mask, oversample_frac, config["seed"]
            )
        else:
            train_input_nodes = data.train_mask

        # Each loader samples from a DIFFERENT edge view — train must never sample across
        # val/test-period edges (see data/temporal_edges.py); val/test inference legitimately can,
        # since that's fixed-weight inference, not training.
        train_loader = NeighborLoader(
            _graph_view(data, data.train_edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=train_input_nodes, shuffle=True,
        )
        val_loader = NeighborLoader(
            _graph_view(data, data.val_edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=data.val_mask, shuffle=False,
        )
        test_loader = NeighborLoader(
            _graph_view(data, data.edge_index), num_neighbors=num_neighbors, batch_size=batch_size,
            input_nodes=data.test_mask, shuffle=False,
        )
    else:
        # Deliberately NOT data.to(device) (which would move ALL THREE edge_index variants to
        # GPU at once): data.edge_index (the full graph) is only needed once, for the final
        # test-time eval after training completes. Since the leakage fix added train_edge_index/
        # val_edge_index as separate tensors, a blanket .to(device) here now holds all three
        # resident on GPU throughout training — on PaySim's large graph this regressed a real
        # CUDA OOM on configs that fit fine before (see LAB_JOURNAL.md). train_edge_index and
        # val_edge_index ARE used every epoch, so those move now; edge_index moves just-in-time
        # at the test-eval call site instead.
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
    ema = None            # created lazily at ema_start_epoch, snapshotting the model then
    eval_model = None      # reused each epoch to evaluate EMA weights without touching `model`

    criterion = FocalLoss(alpha=config["loss"]["alpha"], gamma=config["loss"]["gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    # Opt-in only (train.lr_schedule, default "none") — existing configs are unaffected. Added
    # because several high-alpha PaySim runs hit the epoch cap without ever plateauing (Runs
    # 24-31 predate wandb, so this was only visible in printed logs, not a full curve) — a decaying
    # LR is the standard fix for a model that's still improving late in a fixed epoch budget,
    # worth testing directly now that wandb can actually confirm whether it helps.
    lr_schedule = config["train"].get("lr_schedule", "none")
    if lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["train"]["epochs"])
    elif lr_schedule == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown train.lr_schedule: {lr_schedule!r} (expected 'none' or 'cosine')")

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
            # train_edge_index, NOT edge_index — the training forward pass must never see
            # val/test-period edges (see data/temporal_edges.py; arXiv 2604.19514).
            logits = model(data.x, data.train_edge_index)
            loss = criterion(logits[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            loss = loss.item()

        if scheduler is not None:
            scheduler.step()

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
            val_metrics = evaluate(eval_source, data, data.val_mask, device, edge_index=data.val_edge_index)

        if val_metrics["auc_roc"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc"]
            source = ema.state_dict() if ema is not None else model.state_dict()
            best_state = {k: v.detach().clone() for k, v in source.items()}
            best_epoch = epoch
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch:3d} | loss={loss:.4f} | lr={current_lr:.6f} | val_auc_roc={val_metrics['auc_roc']:.4f} | val_f1_macro={val_metrics['f1_macro']:.4f}")
        wandb.log({"epoch": epoch, "train_loss": loss, "lr": current_lr, "val": val_metrics}, step=epoch)

        # Guard against stopping before EMA ever gets a chance to run: without this, a plateau in
        # the raw model right around ema_start_epoch (exactly what happened in LAB_JOURNAL Run 17
        # — best_epoch=7, ema_start_epoch=15, patience=8) fires early stopping before EMA engages
        # even once, so every "evaluated" checkpoint that whole run was the raw model.
        if epochs_since_improve >= patience and (ema_decay <= 0 or epoch > ema_start_epoch):
            logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    model.load_state_dict(best_state)
    test_predictions = None
    if mini_batch:
        val_metrics = evaluate_batched(model, val_loader, device)
        test_metrics = evaluate_batched(model, test_loader, device)
    else:
        val_metrics = evaluate(model, data, data.val_mask, device, edge_index=data.val_edge_index)
        # Moved to GPU just-in-time — see the setup comment above for why this isn't kept
        # resident throughout training.
        test_edge_index = data.edge_index.to(device)
        test_metrics = evaluate(model, data, data.test_mask, device, edge_index=test_edge_index)
        if config.get("debug", {}).get("return_test_predictions", False):
            # Per-node test predictions, in the SAME boolean-mask order as
            # evaluation/error_analysis.py's x[test_mask]/y[test_mask] arrays, so a caller can
            # cross-reference which specific test nodes a GNN gets right/wrong against another
            # model's (e.g. RF's) error breakdown without needing matching node IDs — just the
            # same mask ordering. Off by default: JSON output size, not needed for normal runs.
            model.eval()
            with torch.no_grad():
                test_probs = torch.sigmoid(model(data.x, test_edge_index)[data.test_mask]).cpu().numpy()
            test_predictions = {
                "probs": test_probs.tolist(),
                "y_true": data.y[data.test_mask].cpu().numpy().tolist(),
            }

    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Val:  {val_metrics}")
    logger.info(f"Test: {test_metrics}")

    wandb.log({"best_epoch": best_epoch, "final_val": val_metrics, "final_test": test_metrics})
    wandb_run.summary["best_epoch"] = best_epoch
    for split_name, metrics in (("val", val_metrics), ("test", test_metrics)):
        for k, v in metrics.items():
            if isinstance(v, dict):
                # confusion matrix: flatten tp/fp/tn/fn too, not just the scalar metrics derived
                # from it — this exact breakdown is what root-caused the Elliptic diffusion
                # regression (LAB_JOURNAL.md Run 32), skipping it here would lose the one thing
                # that made that diagnosis possible from the run comparison table.
                for sub_k, sub_v in v.items():
                    wandb_run.summary[f"{split_name}_{k}_{sub_k}"] = sub_v
            else:
                wandb_run.summary[f"{split_name}_{k}"] = v

    append_journal_entry(
        config, config_path, device,
        n_nodes=data.num_nodes, n_edges=data.edge_index.shape[1],
        n_train_edges=data.train_edge_index.shape[1], n_val_edges=data.val_edge_index.shape[1],
        epoch_stopped=best_epoch, val_metrics=val_metrics, test_metrics=test_metrics,
    )

    # Included so a job's result can be sanity-checked from the returned output alone (e.g. via
    # the RunPod status API after the fact) without needing to have watched live logs — a stale
    # worker silently running the wrong config is otherwise invisible in unattended/overnight runs.
    config_summary = {
        "config_label": config_path,
        "dataset": config["data"].get("dataset", "paysim"),
        "model_name": config["model"]["name"],
        "mini_batch": mini_batch,
        "oversample_fraud_frac": config["train"].get("oversample_fraud_frac", 0.0),
        "epochs_cap": config["train"]["epochs"],
        "subsample_size": config["data"].get("subsample_size"),  # None for non-PaySim datasets
    }
    wandb_url = None if wandb_run.disabled else wandb_run.url
    wandb.finish()

    return {
        "val": val_metrics, "test": test_metrics, "best_epoch": best_epoch, "device": str(device),
        "config_summary": config_summary, "wandb_url": wandb_url, "test_predictions": test_predictions,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    run_from_config(config, args.config)


if __name__ == "__main__":
    main()
