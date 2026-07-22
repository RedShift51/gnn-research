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
from torch_geometric.loader import NeighborLoader

from evaluation.metrics import compute_metrics
from training.train_gnn import ROOT, _graph_view, build_model, init_wandb, load_config, pick_device, set_seed

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


def _hard_triplets(embeddings: torch.Tensor, fraud_idx: torch.Tensor, legit_idx: torch.Tensor) -> tuple:
    """Batch-hard mining (Hermans et al. 2017, "In Defense of the Triplet Loss for Person
    Re-Identification") over the FULL train fraud/legit population (small enough here -- ~3462
    fraud, ~5500 legit -- to do exactly, not just within a mini-batch): for every fraud anchor,
    the hardest positive is the OTHER fraud point currently farthest away (hardest to pull
    together), and the hardest negative is the legit point currently closest (hardest to push
    apart). Directly targets 2026-07-21's finding: post-break fraud is exactly the "hard positive"
    case random triplet sampling under-weights, since it's rare and far from most other fraud --
    hard mining forces every epoch to confront it instead of hoping random sampling includes it."""
    fraud_emb = embeddings[fraud_idx]
    legit_emb = embeddings[legit_idx]
    fraud_dist = torch.cdist(fraud_emb, fraud_emb)
    fraud_dist.fill_diagonal_(-1.0)  # exclude self as its own "positive"
    hardest_pos = fraud_dist.argmax(dim=1)
    fraud_legit_dist = torch.cdist(fraud_emb, legit_emb)
    hardest_neg = fraud_legit_dist.argmin(dim=1)
    return fraud_emb, fraud_emb[hardest_pos], legit_emb[hardest_neg]


def _embed_nodes(model, data_view, node_ids: torch.Tensor, num_neighbors: list, batch_size: int,
                  device) -> torch.Tensor:
    """Compute embeddings for an arbitrary set of global node ids via mini-batch NeighborLoader --
    the metric-learning analogue of train_gnn.py's evaluate_batched, needed because IEEE-CIS's
    ~590K-node/41M-edge graph makes a full-graph model.embed() call (what the non-mini_batch path
    below does every epoch) OOM/time out the same way full-batch GraphSAGEDiff classification did
    (see configs/ieee_cis_graphsage_diff_minibatch.yaml's comment).

    Aligns results by batch.n_id, NOT by assuming the loader preserves input_nodes' order --
    confirmed empirically (2026-07-22, IEEE-CIS hard-core investigation) that NeighborLoader can
    locally swap two ADJACENT seed nodes' relative order (27 swapped pairs out of ~88.6K test
    nodes), almost certainly from its internal node-dedup when one seed node is also sampled as a
    neighbor of another seed node in the same batch. That's a tolerable ~0.06% corruption for a
    final eval readout cross-referenced against an independent model, but silently wrong gradients
    during training (mismatched anchor/positive/negative embeddings) would be far worse and much
    harder to notice. n_id-based alignment is correct regardless of internal reordering.
    Differentiable: torch.cat and advanced indexing both preserve autograd, so this is safe to call
    inside a training step, not just under torch.no_grad()."""
    loader = NeighborLoader(
        data_view, num_neighbors=num_neighbors, batch_size=batch_size,
        input_nodes=node_ids.cpu(), shuffle=False,
    )
    all_emb, all_ids = [], []
    for batch in loader:
        batch = batch.to(device)
        emb = model.embed(batch.x, batch.edge_index)[: batch.batch_size]
        all_emb.append(emb)
        all_ids.append(batch.n_id[: batch.batch_size])
    all_emb = torch.cat(all_emb, dim=0)
    all_ids = torch.cat(all_ids, dim=0).cpu()

    pos = torch.full((data_view.num_nodes,), -1, dtype=torch.long)
    pos[all_ids] = torch.arange(len(all_ids))
    order = pos[node_ids.cpu()]
    assert (order >= 0).all(), "some requested node_ids never came back as a loader seed node"
    return all_emb[order]


def _camo_weighted_loss(embeddings: torch.Tensor, anchor_idx: torch.Tensor, pos_idx: torch.Tensor,
                         neg_idx: torch.Tensor, legit_centroid: torch.Tensor, margin: float) -> torch.Tensor:
    """Soft importance-weighted triplet loss -- a gentler alternative to _hard_triplets' all-or-
    nothing hardest-example selection (Run 79: that destabilized training badly, F1 crashed to
    0.325). Same RANDOM triplets as _sample_triplets, but each triplet's loss contribution is
    up-weighted by how close its fraud ANCHOR currently is to the legit centroid -- i.e., how
    "camouflaged" it already looks. Directly motivated by Run 81: post-break fraud aligns 95% with
    a camouflaged sub-cluster that already exists (at ~40% prevalence) within the easy, well-
    classified period, but gets diluted by averaging with the "obvious" majority in the current
    (unweighted) triplet loss. Softmax weighting (not hard top-1 selection) keeps every triplet
    contributing something, avoiding the non-stationary "hardest changes every epoch" instability."""
    anchor = embeddings[anchor_idx]
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    d_pos = (anchor - pos).pow(2).sum(-1)
    d_neg = (anchor - neg).pow(2).sum(-1)
    per_triplet_loss = F.relu(d_pos - d_neg + margin)

    dist_to_legit = (anchor - legit_centroid).pow(2).sum(-1).sqrt()
    weight = torch.softmax(-dist_to_legit, dim=0) * len(dist_to_legit)  # mean weight ~= 1
    return (per_triplet_loss * weight.detach()).mean()  # detach: weight is a sampling emphasis,
    # not something the anchor's position should be optimized to change


def _centroid_scores(embeddings: torch.Tensor, fraud_centroid: torch.Tensor,
                      legit_centroid: torch.Tensor) -> np.ndarray:
    """Fraud-likeness score for nearest-centroid classification: positive = closer to the fraud
    centroid than the legit one. Sigmoid maps this to (0,1) with 0.5 exactly at the equidistant
    boundary, so evaluation/metrics.py's compute_metrics (which thresholds at 0.5) does the right
    thing without needing a separately-fit calibration."""
    dist_fraud = (embeddings - fraud_centroid).pow(2).sum(-1).sqrt()
    dist_legit = (embeddings - legit_centroid).pow(2).sum(-1).sqrt()
    return torch.sigmoid(dist_legit - dist_fraud).cpu().numpy()


def _compression_loss(embeddings: torch.Tensor, fraud_idx: torch.Tensor, legit_idx: torch.Tensor) -> torch.Tensor:
    """Center Loss (Wen et al. 2016): explicit penalty pulling each class's embeddings toward
    their OWN class centroid (computed fresh from the current forward pass, gradient flows
    through both the points and the centroid -- a simpler variant than the original paper's
    EMA-buffered centroid, equivalent to directly minimizing within-class variance). Triplet loss
    only pulls anchor/positive together relatively; this adds an absolute pull toward each class's
    center, directly targeting LAB_JOURNAL.md's finding that legit's spread (0.344) was ~2.7x
    fraud's (0.126) in the unsupervised (plain triplet) embedding -- legit had nothing pulling it
    together the way fraud-fraud positive pairs did."""
    fraud_emb = embeddings[fraud_idx]
    legit_emb = embeddings[legit_idx]
    fraud_centroid = fraud_emb.mean(dim=0)
    legit_centroid = legit_emb.mean(dim=0)
    fraud_var = (fraud_emb - fraud_centroid).pow(2).sum(-1).mean()
    legit_var = (legit_emb - legit_centroid).pow(2).sum(-1).mean()
    return fraud_var + legit_var


def run(config: dict, n_triplets_per_epoch: int = 2000, margin: float = 1.0,
        compression_weight: float = 0.0, return_embeddings: bool = False, mining: str = "random") -> dict:
    wandb_run = init_wandb(config, "metric_learning")
    set_seed(config["seed"])
    device = pick_device(config["train"]["device"])
    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)
    mini_batch = config["train"].get("mini_batch", False)

    if not mini_batch:
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

    train_mask_cpu = data.train_mask if mini_batch else data.train_mask.cpu()
    y_cpu = data.y if mini_batch else data.y.cpu()
    train_fraud_idx = train_mask_cpu.nonzero(as_tuple=True)[0][y_cpu[train_mask_cpu] == 1]
    train_legit_idx = train_mask_cpu.nonzero(as_tuple=True)[0][y_cpu[train_mask_cpu] == 0]
    logger.info(f"Train fraud={len(train_fraud_idx)}, train legit={len(train_legit_idx)}")

    best_val_auc = -1.0
    best_state = None
    best_epoch = 0
    epochs_since_improve = 0
    patience = config["train"]["patience"]

    if mining == "hard" and mini_batch:
        raise NotImplementedError(
            "hard mining needs the full train fraud/legit population embedded every epoch -- "
            "infeasible in mini-batch mode at IEEE-CIS's scale, and Run 79 already found it "
            "destabilizes training badly even where it WAS feasible (Elliptic, full-batch)."
        )

    if mini_batch:
        num_neighbors = config["train"]["num_neighbors"]
        batch_size = config["train"]["batch_size"]
        # Fixed once, not resampled per epoch -- IEEE-CIS's train-legit pool is ~400K, far too
        # large to embed in full every epoch (that's the whole reason mini-batch mode exists).
        # A single representative subsample serves as BOTH the camo_weighted live-centroid
        # reference during training and the legit reference for eval-time nearest-centroid scoring.
        legit_ref_size = min(len(train_legit_idx), config.get("metric_learning", {}).get("legit_ref_size", 3000))
        legit_ref_idx = train_legit_idx[torch.randperm(len(train_legit_idx), generator=generator)[:legit_ref_size]]
        logger.info(f"Mini-batch mode: legit reference pool = {legit_ref_size} (fixed for the whole run)")

        train_view = _graph_view(data, data.train_edge_index)
        val_view = _graph_view(data, data.val_edge_index)

        for epoch in range(1, config["train"]["epochs"] + 1):
            model.train()
            optimizer.zero_grad()

            anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
            needed = torch.unique(torch.cat(
                [anchor_idx, pos_idx, neg_idx, legit_ref_idx] if mining == "camo_weighted" else [anchor_idx, pos_idx, neg_idx]
            ))
            embeddings_subset = F.normalize(_embed_nodes(model, train_view, needed, num_neighbors, batch_size, device), dim=-1)

            pos_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
            pos_map[needed] = torch.arange(len(needed))
            anchor_local = pos_map[anchor_idx].to(device)
            pos_local = pos_map[pos_idx].to(device)
            neg_local = pos_map[neg_idx].to(device)

            if mining == "random":
                loss = triplet_loss_fn(embeddings_subset[anchor_local], embeddings_subset[pos_local], embeddings_subset[neg_local])
            elif mining == "camo_weighted":
                legit_ref_local = pos_map[legit_ref_idx].to(device)
                legit_centroid_live = embeddings_subset[legit_ref_local].mean(dim=0)
                loss = _camo_weighted_loss(embeddings_subset, anchor_local, pos_local, neg_local, legit_centroid_live, margin)
            else:
                raise ValueError(f"Unknown mining: {mining!r} (expected 'random' or 'camo_weighted' in mini-batch mode)")
            if compression_weight > 0:
                fraud_local = torch.unique(torch.cat([anchor_local, pos_local]))
                legit_local = torch.unique(torch.cat([neg_local, legit_ref_local]) if mining == "camo_weighted" else neg_local)
                comp_loss = _compression_loss(embeddings_subset, fraud_local, legit_local)
                loss = loss + compression_weight * comp_loss
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                ref_needed = torch.unique(torch.cat([train_fraud_idx, legit_ref_idx]))
                ref_emb = F.normalize(_embed_nodes(model, val_view, ref_needed, num_neighbors, batch_size, device), dim=-1)
                ref_pos = torch.full((data.num_nodes,), -1, dtype=torch.long)
                ref_pos[ref_needed] = torch.arange(len(ref_needed))
                fraud_centroid = ref_emb[ref_pos[train_fraud_idx]].mean(dim=0)
                legit_centroid = ref_emb[ref_pos[legit_ref_idx]].mean(dim=0)

                val_idx = data.val_mask.nonzero(as_tuple=True)[0]
                val_embeddings = F.normalize(_embed_nodes(model, val_view, val_idx, num_neighbors, batch_size, device), dim=-1)
                val_scores = _centroid_scores(val_embeddings, fraud_centroid, legit_centroid)
                val_y = data.y[val_idx].cpu().numpy()
                val_metrics = compute_metrics(val_y, val_scores)

            if val_metrics["auc_roc"] > best_val_auc:
                best_val_auc = val_metrics["auc_roc"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1

            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"epoch {epoch} triplet_loss={loss.item():.4f} val_auc={val_metrics['auc_roc']:.4f}")
            wandb.log({"epoch": epoch, "triplet_loss": loss.item(), "val_auc": val_metrics["auc_roc"]}, step=epoch)

            if epochs_since_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            test_view = _graph_view(data, data.edge_index)
            ref_needed = torch.unique(torch.cat([train_fraud_idx, legit_ref_idx]))
            ref_emb = F.normalize(_embed_nodes(model, test_view, ref_needed, num_neighbors, batch_size, device), dim=-1)
            ref_pos = torch.full((data.num_nodes,), -1, dtype=torch.long)
            ref_pos[ref_needed] = torch.arange(len(ref_needed))
            fraud_centroid = ref_emb[ref_pos[train_fraud_idx]].mean(dim=0)
            legit_centroid = ref_emb[ref_pos[legit_ref_idx]].mean(dim=0)
            train_fraud_emb = ref_emb[ref_pos[train_fraud_idx]]
            train_legit_emb = ref_emb[ref_pos[legit_ref_idx]]

            test_idx = data.test_mask.nonzero(as_tuple=True)[0]
            test_emb_masked = F.normalize(_embed_nodes(model, test_view, test_idx, num_neighbors, batch_size, device), dim=-1)
            test_scores = _centroid_scores(test_emb_masked, fraud_centroid, legit_centroid)
            dist_to_fraud = (test_emb_masked - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            dist_to_legit = (test_emb_masked - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            fraud_spread = (train_fraud_emb - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            legit_spread = (train_legit_emb - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

        test_y = data.y[test_idx].cpu().numpy()

    else:
        for epoch in range(1, config["train"]["epochs"] + 1):
            model.train()
            optimizer.zero_grad()
            embeddings = model.embed(data.x, data.train_edge_index)
            embeddings = F.normalize(embeddings, dim=-1)  # unit-norm, standard metric-learning practice

            if mining == "hard":
                anchor_emb, pos_emb, neg_emb = _hard_triplets(embeddings, train_fraud_idx.to(device), train_legit_idx.to(device))
                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            elif mining == "random":
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                anchor_emb, pos_emb, neg_emb = embeddings[anchor_idx.to(device)], embeddings[pos_idx.to(device)], embeddings[neg_idx.to(device)]
                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            elif mining == "camo_weighted":
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                loss = _camo_weighted_loss(embeddings, anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device),
                                            legit_centroid_live, margin)
            else:
                raise ValueError(f"Unknown mining: {mining!r} (expected 'random', 'hard', or 'camo_weighted')")
            if compression_weight > 0:
                comp_loss = _compression_loss(embeddings, train_fraud_idx.to(device), train_legit_idx.to(device))
                loss = loss + compression_weight * comp_loss
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

            # Raw per-centroid distances (not just the combined sigmoid score) -- lets a caller test
            # legit-centroid-only scoring (2026-07-21 discussion: fraud may not cluster coherently
            # enough for its own centroid to be a meaningful reference point, unlike legit) without
            # retraining.
            test_emb_masked = test_embeddings[data.test_mask]
            dist_to_fraud = (test_emb_masked - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            dist_to_legit = (test_emb_masked - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

            # Within-class spread in the LEARNED embedding space -- is train fraud actually a
            # coherent cluster (tight spread around its own centroid) or diffuse (large spread,
            # meaning the fraud centroid itself is a weak/uninformative reference point)?
            train_fraud_emb = test_embeddings[train_fraud_idx.to(device)]
            train_legit_emb = test_embeddings[train_legit_idx.to(device)]
            fraud_spread = (train_fraud_emb - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            legit_spread = (train_legit_emb - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

        test_y = data.y[data.test_mask].cpu().numpy()

    test_metrics = compute_metrics(test_y, test_scores)

    logger.info(f"Metric learning (best_epoch={best_epoch}): Test={test_metrics}")
    logger.info(f"Train fraud spread around its centroid: mean={fraud_spread.mean():.4f} std={fraud_spread.std():.4f}")
    logger.info(f"Train legit spread around its centroid: mean={legit_spread.mean():.4f} std={legit_spread.std():.4f}")
    wandb_run.summary["best_epoch"] = best_epoch
    for k, v in test_metrics.items():
        if not isinstance(v, dict):
            wandb_run.summary[f"test_{k}"] = v
    wandb.finish()

    result = {
        "test": test_metrics, "best_epoch": best_epoch,
        "test_scores": test_scores.tolist(), "test_y": test_y.tolist(),
        "dist_to_fraud": dist_to_fraud.tolist(), "dist_to_legit": dist_to_legit.tolist(),
        "fraud_spread_mean": float(fraud_spread.mean()), "fraud_spread_std": float(fraud_spread.std()),
        "legit_spread_mean": float(legit_spread.mean()), "legit_spread_std": float(legit_spread.std()),
    }
    if return_embeddings:
        # Off by default -- only needed for the embedding-dimension / linear-probe analysis
        # (2026-07-21 discussion), which doesn't need retraining once these are captured.
        # BUG FOUND (2026-07-21): returning ALL embeddings (test ~8841 + train fraud ~3462 +
        # train legit ~5500, each 128-dim, as JSON) silently produced a job with status=COMPLETED
        # but output=None -- almost certainly a RunPod output-payload size limit (~30-40MB
        # estimated), which fails SILENTLY rather than raising a clear error. Fixed by keeping ALL
        # fraud (rare, most important, small in absolute count) but subsampling legit, plus
        # rounding to 5 decimals -- cuts payload by roughly an order of magnitude.
        rng = np.random.default_rng(config["seed"])
        test_y_np = test_y
        test_fraud_pos = np.where(test_y_np == 1)[0]
        test_legit_pos = np.where(test_y_np == 0)[0]
        test_legit_sample = rng.choice(test_legit_pos, size=min(1500, len(test_legit_pos)), replace=False)
        test_keep = np.sort(np.concatenate([test_fraud_pos, test_legit_sample]))

        legit_idx_sample = rng.choice(len(train_legit_emb), size=min(1500, len(train_legit_emb)), replace=False)

        def _round(arr):
            return np.round(arr, 5).tolist()

        result["test_embeddings"] = _round(test_emb_masked[test_keep].cpu().numpy())
        result["test_embeddings_y"] = test_y_np[test_keep].tolist()
        result["train_fraud_embeddings"] = _round(train_fraud_emb.cpu().numpy())  # small (~3462), kept in full
        result["train_legit_embeddings"] = _round(train_legit_emb[legit_idx_sample].cpu().numpy())
    return result


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-triplets", type=int, default=2000)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--compression-weight", type=float, default=0.0)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run(config, args.n_triplets, args.margin, args.compression_weight)
    print({k: v for k, v in result.items() if k not in ("test_scores", "test_y")})


if __name__ == "__main__":
    main()
