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
import torch.nn as nn
import torch.nn.functional as F
import wandb
from sklearn.mixture import GaussianMixture
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
                         neg_idx: torch.Tensor, legit_centroid: torch.Tensor, margin: float,
                         temperature: float = 1.0) -> torch.Tensor:
    """Soft importance-weighted triplet loss -- a gentler alternative to _hard_triplets' all-or-
    nothing hardest-example selection (Run 79: that destabilized training badly, F1 crashed to
    0.325). Same RANDOM triplets as _sample_triplets, but each triplet's loss contribution is
    up-weighted by how close its fraud ANCHOR currently is to the legit centroid -- i.e., how
    "camouflaged" it already looks. Directly motivated by Run 81: post-break fraud aligns 95% with
    a camouflaged sub-cluster that already exists (at ~40% prevalence) within the easy, well-
    classified period, but gets diluted by averaging with the "obvious" majority in the current
    (unweighted) triplet loss. Softmax weighting (not hard top-1 selection) keeps every triplet
    contributing something, avoiding the non-stationary "hardest changes every epoch" instability.
    temperature=1.0 reproduces the original recipe exactly; <1 sharpens the weight distribution
    toward the single most-camouflaged anchors (approaching hard selection as T->0), >1 flattens it
    toward uniform (approaching plain unweighted triplet loss as T->inf)."""
    anchor = embeddings[anchor_idx]
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    d_pos = (anchor - pos).pow(2).sum(-1)
    d_neg = (anchor - neg).pow(2).sum(-1)
    per_triplet_loss = F.relu(d_pos - d_neg + margin)

    dist_to_legit = (anchor - legit_centroid).pow(2).sum(-1).sqrt()
    weight = torch.softmax(-dist_to_legit / temperature, dim=0) * len(dist_to_legit)  # mean weight ~= 1
    return (per_triplet_loss * weight.detach()).mean()  # detach: weight is a sampling emphasis,
    # not something the anchor's position should be optimized to change


def _camo_weighted_dual_loss(embeddings: torch.Tensor, anchor_idx: torch.Tensor, pos_idx: torch.Tensor,
                              neg_idx: torch.Tensor, legit_centroid: torch.Tensor, fraud_centroid: torch.Tensor,
                              margin: float, temperature: float = 1.0) -> torch.Tensor:
    """Dual-sided extension of _camo_weighted_loss: the original recipe only up-weights triplets
    by how camouflaged the fraud ANCHOR looks (close to the legit centroid). This adds a second,
    symmetric signal -- how camouflaged the randomly-sampled legit NEGATIVE looks (close to the
    fraud centroid), i.e. a naturally confusable legit example that random sampling would otherwise
    treat the same as an obviously-legit one. A triplet hard on EITHER end gets emphasized. Cheaper
    than _semi_hard_negatives' explicit closest-legit-candidate search (no per-anchor scan over a
    legit pool) and avoids reintroducing _hard_triplets' full top-1 instability -- still a softmax
    over the whole sampled batch, just with two additive terms instead of one."""
    anchor = embeddings[anchor_idx]
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    d_pos = (anchor - pos).pow(2).sum(-1)
    d_neg = (anchor - neg).pow(2).sum(-1)
    per_triplet_loss = F.relu(d_pos - d_neg + margin)

    dist_anchor_to_legit = (anchor - legit_centroid).pow(2).sum(-1).sqrt()
    dist_neg_to_fraud = (neg - fraud_centroid).pow(2).sum(-1).sqrt()
    combined_score = -dist_anchor_to_legit - dist_neg_to_fraud
    weight = torch.softmax(combined_score / temperature, dim=0) * len(combined_score)  # mean weight ~= 1
    return (per_triplet_loss * weight.detach()).mean()


def _camo_weighted_adaptive_margin_loss(embeddings: torch.Tensor, anchor_idx: torch.Tensor, pos_idx: torch.Tensor,
                                         neg_idx: torch.Tensor, legit_centroid: torch.Tensor, margin: float,
                                         margin_scale: float = 1.0) -> torch.Tensor:
    """Alternative mechanism to _camo_weighted_loss: instead of reweighting the LOSS VALUE by
    camouflage severity (leaving the fixed margin's gradient direction untouched), scales the
    MARGIN itself per triplet -- a camouflaged anchor is required to achieve a strictly larger
    enforced separation from its negative than an obvious one, rather than just contributing a
    bigger gradient at a fixed target. margin_scale controls how much the margin can grow above/
    shrink below the base margin (1.0 = the most camouflaged anchor in the batch gets up to ~2x
    margin, the mean multiplier stays ~=1 since camo_weight's softmax already averages to 1)."""
    anchor = embeddings[anchor_idx]
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    d_pos = (anchor - pos).pow(2).sum(-1)
    d_neg = (anchor - neg).pow(2).sum(-1)

    dist_to_legit = (anchor - legit_centroid).pow(2).sum(-1).sqrt()
    camo_weight = torch.softmax(-dist_to_legit, dim=0) * len(dist_to_legit)  # mean weight ~= 1
    adaptive_margin = margin * (1.0 + margin_scale * (camo_weight.detach() - 1.0))
    per_triplet_loss = F.relu(d_pos - d_neg + adaptive_margin)
    return per_triplet_loss.mean()


class _CamoWeightMLP(nn.Module):
    """Learned replacement for _camo_weighted_loss's hand-designed `softmax(-dist_to_legit)`
    scoring function -- takes a RICHER per-anchor feature set (its own embedding plus both
    centroid distances, not just the single scalar dist_to_legit) and outputs one importance
    logit per anchor, softmax-normalized the same way as every other camo_weighted variant.
    Deliberately tiny (one hidden layer) -- ~150-200 camouflaged training examples per epoch is
    not enough signal to justify anything bigger, and this is already a small-sample-overfitting
    risk on top of the collapse risk below.

    KNOWN RISK, not fully resolved by this implementation (2026-07-23 discussion): the MLP is
    trained by a term that lets it look at each triplet's OWN per_triplet_loss, giving it a direct
    incentive to DECREASE weight on high-loss (hard/camouflaged) triplets and INCREASE it on
    low-loss (easy) ones -- exactly the opposite of what camo_weighted's hand-designed prior does,
    and the textbook degenerate-collapse failure mode of naive (non-bi-level) learned example
    reweighting (Ren et al. 2018, "Learning to Reweight Examples"). Shipped anyway per explicit
    instruction to test empirically rather than reason it away in advance -- the run must be
    checked for whether the learned weight ends up POSITIVELY correlated with dist_to_legit
    (collapsed: up-weights obvious/easy fraud) or NEGATIVELY correlated (didn't collapse: still
    up-weights camouflaged fraud like the hand-designed formula does), not just read off the
    headline recovery number.

    Two mitigations _camo_weighted_mlp_loss layers on top of the bare mechanism (both requested
    directly, not things this MLP class enforces itself): (1) an EMA shadow copy is what actually
    reweights the ENCODER's loss, not the raw live MLP -- see _update_ema -- so the encoder isn't
    chasing a new, noisy scoring function every single epoch, the same non-stationarity concern
    already on record for why Run 79's hard-triplet mining destabilized training; (2) a warmup
    period where the encoder trains on plain unweighted triplet loss while the MLP trains quietly
    in the background, so the EMA has already averaged over a less-random MLP before it ever
    touches the encoder's gradient."""

    def __init__(self, input_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _update_ema(ema_module: "_CamoWeightMLP", live_module: "_CamoWeightMLP", decay: float) -> None:
    """Polyak/EMA parameter update, called once per epoch (not per mini-batch step -- there are no
    mini-batches in the full-batch training loop this is used from), so decay should be read as
    an epoch-level smoothing constant, not the ~0.999 per-step values common in mini-batch-SGD
    EMA/mean-teacher setups -- with only ~O(100-300) epochs total here, 0.999 would barely move at
    all before training ends."""
    with torch.no_grad():
        for ema_p, live_p in zip(ema_module.parameters(), live_module.parameters()):
            ema_p.mul_(decay).add_(live_p, alpha=1 - decay)


def _camo_weighted_mlp_loss(embeddings: torch.Tensor, anchor_idx: torch.Tensor, pos_idx: torch.Tensor,
                             neg_idx: torch.Tensor, legit_centroid: torch.Tensor, fraud_centroid: torch.Tensor,
                             margin: float, weight_mlp: "_CamoWeightMLP", ema_weight_mlp: "_CamoWeightMLP",
                             use_ema_weight: bool, anchor_weight_coeff: float = 0.1) -> tuple:
    """camo_weighted_mlp mining: same triplet loss shape as _camo_weighted_loss, but the scoring
    function deciding each anchor's importance weight is _CamoWeightMLP instead of a fixed
    `-dist_to_legit` formula. The MLP's input is DETACHED (anchor embedding + both centroid
    distances) so the main encoder can't receive a gradient through the weighting branch that
    would let it "game" its own embedding position to win a favorable weight -- the only gradient
    path back into embeddings is the ordinary per_triplet_loss term, same as every other variant
    here.

    Two SEPARATE loss terms, deliberately not one shared computation:
    - `mlp_loss` uses the LIVE weight_mlp against per_triplet_loss.detach(), PLUS an anchor
      regularizer pulling live_weight toward _camo_weighted_loss's own hand-designed
      `softmax(-dist_to_legit)` weight. This is the only thing that ever updates weight_mlp's own
      parameters, and it runs every epoch (even during warmup), so the EMA shadow already reflects
      a partially-trained MLP once warmup ends. The anchor term exists because the bare
      per_triplet_loss-driven signal has a KNOWN degenerate direction (down-weight high-loss/
      camouflaged triplets -- the opposite of the domain prior); pulling toward the known-good
      hand-designed weight makes that degenerate solution costly instead of free, while still
      leaving the MLP room to use its richer features (embedding, dist_to_fraud) to deviate from
      the pure-distance formula where that's actually informative. Not a hard constraint --
      anchor_weight_coeff trades off "stay close to the trusted prior" against "learn something
      the prior can't express"; 0 recovers the unconstrained (collapse-prone) version.
    - `encoder_loss` uses the EMA-smoothed ema_weight_mlp (use_ema_weight=True, post-warmup) or
      plain unweighted per_triplet_loss (use_ema_weight=False, warmup) -- this is what actually
      reweights the ENCODER's gradient. ema_weight_mlp runs under no_grad, so this term can never
      backprop into weight_mlp -- by construction the two objectives can't fight over the same
      parameters in the same step.
    Returns (loss, live_weight, ema_weight_or_None) for the caller's collapse-diagnostic logging."""
    anchor = embeddings[anchor_idx]
    pos = embeddings[pos_idx]
    neg = embeddings[neg_idx]
    d_pos = (anchor - pos).pow(2).sum(-1)
    d_neg = (anchor - neg).pow(2).sum(-1)
    per_triplet_loss = F.relu(d_pos - d_neg + margin)

    dist_to_legit = (anchor - legit_centroid).pow(2).sum(-1).sqrt()
    dist_to_fraud = (anchor - fraud_centroid).pow(2).sum(-1).sqrt()
    mlp_input = torch.cat(
        [anchor.detach(), dist_to_legit.detach().unsqueeze(-1), dist_to_fraud.detach().unsqueeze(-1)], dim=-1
    )

    live_logits = weight_mlp(mlp_input)
    live_weight = torch.softmax(live_logits, dim=0) * len(live_logits)  # mean weight ~= 1, same convention throughout
    with torch.no_grad():
        anchor_weight = torch.softmax(-dist_to_legit, dim=0) * len(dist_to_legit)  # the hand-designed prior, fixed reference
    anchor_reg = (live_weight - anchor_weight).pow(2).mean()
    mlp_loss = (per_triplet_loss.detach() * live_weight).mean() + anchor_weight_coeff * anchor_reg

    if use_ema_weight:
        with torch.no_grad():
            ema_logits = ema_weight_mlp(mlp_input)
        ema_weight = torch.softmax(ema_logits, dim=0) * len(ema_logits)
        encoder_loss = (per_triplet_loss * ema_weight).mean()
    else:
        encoder_loss = per_triplet_loss.mean()
        ema_weight = None

    loss = encoder_loss + mlp_loss
    return loss, live_weight.detach(), (ema_weight.detach() if ema_weight is not None else None)


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


def _semi_hard_negatives(anchor_emb: torch.Tensor, pos_emb: torch.Tensor, legit_emb: torch.Tensor) -> torch.Tensor:
    """Semi-hard negative SELECTION (Schroff et al. 2015, FaceNet), pure tensor op -- for each
    (anchor, positive) pair, picks the CLOSEST legit candidate that's still farther than the
    positive (d(a,n) > d(a,p)): violates the margin only slightly, giving a small but informative
    gradient. This is the literature-standard middle ground between random sampling (_sample_
    triplets: too easy, most triplets already satisfy the margin, weak gradient) and pure
    batch-hard mining (_hard_triplets, Run 79: destabilized training badly, F1 crashed to 0.325,
    from the non-stationary "hardest changes every epoch" dynamic) -- a gap this codebase never
    actually tested despite having both extremes. Falls back to the single hardest (closest)
    candidate when none is farther than the positive (rare, but must be handled). Separated from
    sampling so mini-batch mode can reuse it against a fixed legit reference pool instead of the
    full (there, infeasible-to-embed-every-epoch) train-legit population _semi_hard_triplets uses."""
    d_pos = (anchor_emb - pos_emb).pow(2).sum(-1).sqrt()
    d_an = torch.cdist(anchor_emb, legit_emb)

    semi_hard_mask = d_an > d_pos.unsqueeze(1)
    d_an_masked = d_an.masked_fill(~semi_hard_mask, float("inf"))
    semi_hard_local = d_an_masked.argmin(dim=1)
    no_semi_hard = ~semi_hard_mask.any(dim=1)
    if no_semi_hard.any():
        hardest_local = d_an.argmin(dim=1)
        semi_hard_local = torch.where(no_semi_hard, hardest_local, semi_hard_local)
    return legit_emb[semi_hard_local]


def _semi_hard_triplets(embeddings: torch.Tensor, fraud_idx: torch.Tensor, legit_idx: torch.Tensor,
                         n_triplets: int, generator: torch.Generator) -> tuple:
    """Full-batch entry point: samples random anchor/positive fraud pairs, then applies
    _semi_hard_negatives against the FULL legit population (feasible here since embeddings covers
    every node already)."""
    anchor_idx = fraud_idx[torch.randint(0, len(fraud_idx), (n_triplets,), generator=generator)]
    pos_idx = fraud_idx[torch.randint(0, len(fraud_idx), (n_triplets,), generator=generator)]
    anchor_emb = embeddings[anchor_idx]
    pos_emb = embeddings[pos_idx]
    legit_emb = embeddings[legit_idx]
    neg_emb = _semi_hard_negatives(anchor_emb, pos_emb, legit_emb)
    return anchor_emb, pos_emb, neg_emb


def _fraud_prototypes(fraud_emb: torch.Tensor, legit_centroid: torch.Tensor) -> tuple:
    """Fits a 2-component GMM on CURRENT fraud embeddings' distance-to-legit-centroid and returns
    the two sub-cluster centroids (in embedding space) plus a label array aligned to fraud_emb's
    own row order -- the obvious/camouflaged split from Run 81, recomputed live from whatever
    embeddings are passed in (live training embeddings, or a val/test-view snapshot for eval, so
    the split always reflects the CURRENT model rather than a stale one-time fit)."""
    dist = (fraud_emb - legit_centroid).pow(2).sum(-1).sqrt().detach().cpu().numpy()
    gmm = GaussianMixture(n_components=2, random_state=0, n_init=1).fit(dist.reshape(-1, 1))
    labels = torch.from_numpy(gmm.predict(dist.reshape(-1, 1))).to(fraud_emb.device)
    centroid_0 = fraud_emb[labels == 0].mean(dim=0)
    centroid_1 = fraud_emb[labels == 1].mean(dim=0)
    return centroid_0, centroid_1, labels


def _sample_multi_prototype_triplets(fraud_idx: torch.Tensor, legit_idx: torch.Tensor, sub_labels: torch.Tensor,
                                      n_triplets: int, generator: torch.Generator) -> tuple:
    """Like _sample_triplets, but the positive is drawn from the SAME GMM sub-cluster as the
    anchor (obvious-with-obvious, camo-with-camo) instead of any random fraud point -- directly
    targets the multi-prototype idea motivated by Run 82's ArcFace catch: a single fraud centroid
    forces one prototype onto a genuinely bimodal class, diluting the rare camouflaged archetype
    into an average dominated by the obvious majority. Keeping each archetype's own positive pairs
    within-cluster should keep both sub-clusters internally cohesive instead."""
    sub_labels_cpu = sub_labels.cpu()
    anchor_pos_in_fraud = torch.randint(0, len(fraud_idx), (n_triplets,), generator=generator)
    anchor_idx = fraud_idx[anchor_pos_in_fraud]
    anchor_sub_labels = sub_labels_cpu[anchor_pos_in_fraud]

    pos_idx = torch.empty(n_triplets, dtype=torch.long)
    for lbl in (0, 1):
        mask = anchor_sub_labels == lbl
        if not mask.any():
            continue
        pool = (sub_labels_cpu == lbl).nonzero(as_tuple=True)[0]
        if len(pool) == 0:
            pool = torch.arange(len(fraud_idx))  # degenerate fallback: GMM collapsed to one side
        choice = pool[torch.randint(0, len(pool), (int(mask.sum()),), generator=generator)]
        pos_idx[mask] = fraud_idx[choice]

    neg_idx = legit_idx[torch.randint(0, len(legit_idx), (n_triplets,), generator=generator)]
    return anchor_idx, pos_idx, neg_idx


def _multi_centroid_scores(embeddings: torch.Tensor, obvious_centroid: torch.Tensor,
                            camo_centroid: torch.Tensor, legit_centroid: torch.Tensor) -> np.ndarray:
    """Nearest-of-3 classification for the multi-prototype variant: fraud-likeness is distance to
    legit vs. distance to the CLOSER of the two fraud sub-centroids, not one blended fraud
    centroid."""
    dist_legit = (embeddings - legit_centroid).pow(2).sum(-1).sqrt()
    dist_obvious = (embeddings - obvious_centroid).pow(2).sum(-1).sqrt()
    dist_camo = (embeddings - camo_centroid).pow(2).sum(-1).sqrt()
    dist_fraud = torch.minimum(dist_obvious, dist_camo)
    return torch.sigmoid(dist_legit - dist_fraud).cpu().numpy()


def _supcon_loss(embeddings: torch.Tensor, batch_idx: torch.Tensor, batch_labels: torch.Tensor,
                  temperature: float = 0.1) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020). Pulls every anchor toward ALL same-label
    points in the batch and pushes it from ALL different-label points simultaneously, unlike
    triplet loss's one-positive-one-negative-per-step. Doesn't presuppose a single class-
    conditional shape the way pulling toward one fraud centroid does (Run 82's ArcFace catch), so
    it may handle the camo/obvious bimodality more gracefully without an explicit multi-prototype
    structure -- the class is defined by whichever points happen to be in the batch, not by a
    single running average."""
    z = embeddings[batch_idx]  # already unit-normalized
    sim = (z @ z.T) / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability, doesn't change softmax
    exp_sim = torch.exp(sim)
    self_mask = torch.eye(len(batch_idx), device=z.device, dtype=torch.bool)
    exp_sim = exp_sim.masked_fill(self_mask, 0.0)

    label_eq = batch_labels.unsqueeze(0) == batch_labels.unsqueeze(1)
    pos_mask = label_eq & ~self_mask

    denom = exp_sim.sum(dim=1)
    log_prob = sim - torch.log(denom.unsqueeze(1) + 1e-12)
    pos_count = pos_mask.sum(dim=1).clamp(min=1)
    loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / pos_count

    valid = pos_mask.sum(dim=1) > 0  # anchors with zero same-label peers in this batch contribute nothing
    return loss_per_anchor[valid].mean()


def _align_uniform_loss(embeddings: torch.Tensor, fraud_idx: torch.Tensor, legit_idx: torch.Tensor,
                         generator: torch.Generator, n_pairs: int, align_weight: float = 1.0,
                         uniform_weight: float = 1.0, t: float = 2.0) -> torch.Tensor:
    """Alignment + uniformity decomposition (Wang & Isola 2020, "Understanding Contrastive
    Representation Learning through Alignment and Uniformity on the Hypersphere"). Alignment pulls
    same-class pairs together (same spirit as triplet loss's positive term); uniformity spreads
    ALL points evenly across the hypersphere instead of actively compressing toward a class
    centroid. Directly motivated by Run 82's Center/Compression Loss finding: explicit compression
    toward a centroid improved aggregate F1 but HURT hard-core recovery specifically (over-
    tightened the already-thin camo sub-cluster, weight=0.5: fraud spread 0.126->0.068). Uniformity
    only prevents wholesale collapse -- it doesn't actively shrink spread the way Center Loss's
    variance penalty does, so it shouldn't repeat that trade-off."""
    a1 = fraud_idx[torch.randint(0, len(fraud_idx), (n_pairs,), generator=generator)]
    a2 = fraud_idx[torch.randint(0, len(fraud_idx), (n_pairs,), generator=generator)]
    l1 = legit_idx[torch.randint(0, len(legit_idx), (n_pairs,), generator=generator)]
    l2 = legit_idx[torch.randint(0, len(legit_idx), (n_pairs,), generator=generator)]
    align = ((embeddings[a1] - embeddings[a2]).pow(2).sum(-1).mean()
             + (embeddings[l1] - embeddings[l2]).pow(2).sum(-1).mean()) / 2

    all_idx = torch.cat([fraud_idx, legit_idx])
    sample = all_idx[torch.randperm(len(all_idx), generator=generator)[: min(len(all_idx), 2 * n_pairs)]]
    z = embeddings[sample]
    sq_dist = torch.cdist(z, z).pow(2)
    uniform = torch.log(torch.exp(-t * sq_dist).mean() + 1e-12)

    return align_weight * align + uniform_weight * uniform


def run(config: dict, n_triplets_per_epoch: int = 2000, margin: float = 1.0,
        compression_weight: float = 0.0, return_embeddings: bool = False, mining: str = "random",
        temperature: float = 0.1, align_weight: float = 1.0, uniform_weight: float = 1.0,
        gate_aux_weight: float = 0.0, camo_temperature: float = 1.0, margin_scale: float = 1.0,
        camo_mlp_hidden_dim: int = 32, camo_mlp_ema_decay: float = 0.9,
        camo_mlp_warmup_epochs: int = 40, camo_mlp_anchor_weight: float = 0.1) -> dict:
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
    # Only instantiated for camo_weighted_mlp -- input_dim = hidden_dim (anchor embedding) + 2
    # (dist_to_legit, dist_to_fraud). weight_mlp's parameters join the SAME optimizer as the main
    # encoder so it trains jointly, end-to-end, off the main triplet objective (see
    # _camo_weighted_mlp_loss). ema_weight_mlp is a Polyak-averaged shadow copy, never touched by
    # the optimizer directly (updated only via _update_ema after each step) -- it's what actually
    # reweights the encoder's loss, not the noisier live weight_mlp (see _CamoWeightMLP's docstring).
    weight_mlp = ema_weight_mlp = None
    if mining == "camo_weighted_mlp":
        weight_mlp = _CamoWeightMLP(config["model"]["hidden_dim"] + 2, camo_mlp_hidden_dim).to(device)
        ema_weight_mlp = _CamoWeightMLP(config["model"]["hidden_dim"] + 2, camo_mlp_hidden_dim).to(device)
        for p in ema_weight_mlp.parameters():
            p.requires_grad_(False)
        # NOT seeded from weight_mlp's random init here -- seeded instead at the warmup boundary
        # (see the training loop, "epoch == camo_mlp_warmup_epochs + 1"), from whatever weight_mlp
        # has actually learned over the warmup epochs. Averaging in the near-random pre-warmup
        # state would just be noise the EMA has to work its way back out of.
    optimizer = torch.optim.Adam(
        list(model.parameters()) + (list(weight_mlp.parameters()) if weight_mlp is not None else []),
        lr=config["train"]["lr"], weight_decay=config["train"]["weight_decay"],
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
            uses_legit_ref = mining in ("camo_weighted", "semi_hard")
            needed = torch.unique(torch.cat(
                [anchor_idx, pos_idx, legit_ref_idx] if mining == "semi_hard"
                else [anchor_idx, pos_idx, neg_idx, legit_ref_idx] if mining == "camo_weighted"
                else [anchor_idx, pos_idx, neg_idx]
            ))
            embeddings_subset = F.normalize(_embed_nodes(model, train_view, needed, num_neighbors, batch_size, device), dim=-1)

            pos_map = torch.full((data.num_nodes,), -1, dtype=torch.long)
            pos_map[needed] = torch.arange(len(needed))
            anchor_local = pos_map[anchor_idx].to(device)
            pos_local = pos_map[pos_idx].to(device)
            legit_ref_local = pos_map[legit_ref_idx].to(device) if uses_legit_ref else None
            neg_local = pos_map[neg_idx].to(device) if mining != "semi_hard" else None

            if mining == "random":
                loss = triplet_loss_fn(embeddings_subset[anchor_local], embeddings_subset[pos_local], embeddings_subset[neg_local])
            elif mining == "camo_weighted":
                legit_centroid_live = embeddings_subset[legit_ref_local].mean(dim=0)
                loss = _camo_weighted_loss(embeddings_subset, anchor_local, pos_local, neg_local, legit_centroid_live, margin)
            elif mining == "semi_hard":
                # Candidate negative pool is the fixed legit_ref subsample, not the full ~400K-node
                # train-legit population _semi_hard_triplets uses in full-batch mode -- same
                # infeasible-to-embed-every-epoch reason camo_weighted's live centroid uses it too.
                anchor_emb = embeddings_subset[anchor_local]
                pos_emb = embeddings_subset[pos_local]
                neg_emb = _semi_hard_negatives(anchor_emb, pos_emb, embeddings_subset[legit_ref_local])
                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            else:
                raise ValueError(f"Unknown mining: {mining!r} (expected 'random', 'camo_weighted', or 'semi_hard' in mini-batch mode)")
            if compression_weight > 0:
                fraud_local = torch.unique(torch.cat([anchor_local, pos_local]))
                legit_local = torch.unique(torch.cat([neg_local, legit_ref_local]) if mining == "camo_weighted"
                                            else legit_ref_local if mining == "semi_hard" else neg_local)
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
            # dist-to-legit-centroid for TRAIN fraud, not just test -- needed for a time-independent
            # bimodality check (2026-07-22: does a camo/obvious split exist in fraud generally, not
            # tied to any temporal break). A scalar-per-node array, so far cheaper to return than
            # raw embeddings would be -- no need to fight the payload-size limit for this.
            train_fraud_dist_to_legit = (train_fraud_emb - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

        test_y = data.y[test_idx].cpu().numpy()

    else:
        if hasattr(model, "set_legit_centroid"):
            model.set_legit_centroid(data.x, train_legit_idx.to(device))

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
            elif mining == "camo_weighted_temp":
                # Same recipe as camo_weighted, generalized with an explicit softmax temperature --
                # sharper (<1) concentrates weight on the single most-camouflaged anchors in the
                # batch (approaching hard selection); flatter (>1) approaches unweighted random.
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                loss = _camo_weighted_loss(embeddings, anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device),
                                            legit_centroid_live, margin, temperature=camo_temperature)
            elif mining == "camo_weighted_dual":
                # Extends camo_weighted's anchor-side camouflage signal with a symmetric
                # negative-side one -- also up-weights triplets whose randomly-sampled legit
                # negative already looks fraud-like (close to the fraud centroid), not just
                # triplets whose fraud anchor looks legit-like.
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                fraud_centroid_live = embeddings[train_fraud_idx.to(device)].mean(dim=0)
                loss = _camo_weighted_dual_loss(embeddings, anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device),
                                                 legit_centroid_live, fraud_centroid_live, margin, temperature=camo_temperature)
            elif mining == "camo_weighted_margin_scale":
                # Alternative mechanism to camo_weighted's loss-reweighting: scales the MARGIN
                # itself by camouflage severity instead, so camouflaged anchors are optimized
                # toward a strictly larger enforced separation, not just a bigger gradient at a
                # fixed target.
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                loss = _camo_weighted_adaptive_margin_loss(embeddings, anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device),
                                                            legit_centroid_live, margin, margin_scale=margin_scale)
            elif mining == "camo_weighted_mlp":
                anchor_idx, pos_idx, neg_idx = _sample_triplets(train_fraud_idx, train_legit_idx, n_triplets_per_epoch, generator)
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                fraud_centroid_live = embeddings[train_fraud_idx.to(device)].mean(dim=0)
                use_ema_weight = epoch > camo_mlp_warmup_epochs
                if epoch == camo_mlp_warmup_epochs + 1:
                    # EMA starts tracking NOW, seeded from whatever weight_mlp has learned over the
                    # warmup epochs -- not a decayed average that includes its near-random epoch-1 state.
                    ema_weight_mlp.load_state_dict(weight_mlp.state_dict())
                    logger.info(f"epoch {epoch}: seeding ema_weight_mlp from live weight_mlp, EMA weighting begins")
                loss, live_weight, ema_weight_used = _camo_weighted_mlp_loss(
                    embeddings, anchor_idx.to(device), pos_idx.to(device), neg_idx.to(device),
                    legit_centroid_live, fraud_centroid_live, margin, weight_mlp, ema_weight_mlp, use_ema_weight,
                    anchor_weight_coeff=camo_mlp_anchor_weight)
                if epoch % 25 == 0 or epoch == 1:
                    # Diagnostic for the collapse risk flagged in _CamoWeightMLP's docstring: does
                    # the learned weight end up negatively correlated with dist_to_legit (still
                    # up-weighting camouflaged anchors, like the hand-designed formula) or
                    # positively (collapsed to up-weighting easy/obvious ones instead)? Checked on
                    # the live weight always (it's training from epoch 1 regardless of warmup) and
                    # on the EMA weight once it's actually in use.
                    dist_to_legit_live = (embeddings[anchor_idx.to(device)] - legit_centroid_live).pow(2).sum(-1).sqrt()
                    live_corr = torch.corrcoef(torch.stack([live_weight, dist_to_legit_live]))[0, 1].item()
                    log_line = f"epoch {epoch} camo_weighted_mlp: live weight-vs-dist_to_legit corr={live_corr:.4f}"
                    wandb_payload = {"camo_mlp_live_weight_dist_corr": live_corr}
                    if ema_weight_used is not None:
                        ema_corr = torch.corrcoef(torch.stack([ema_weight_used, dist_to_legit_live]))[0, 1].item()
                        log_line += f" ema corr={ema_corr:.4f} (this is what's actually reweighting the encoder)"
                        wandb_payload["camo_mlp_ema_weight_dist_corr"] = ema_corr
                    else:
                        log_line += " (warmup: encoder still on plain unweighted loss)"
                    log_line += " -- negative=up-weights camo like the hand-designed formula, positive=collapsed"
                    logger.info(log_line)
                    wandb.log(wandb_payload, step=epoch)
            elif mining == "semi_hard":
                anchor_emb, pos_emb, neg_emb = _semi_hard_triplets(embeddings, train_fraud_idx.to(device), train_legit_idx.to(device),
                                                                     n_triplets_per_epoch, generator)
                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            elif mining == "multi_prototype":
                legit_centroid_live = embeddings[train_legit_idx.to(device)].mean(dim=0)
                fraud_emb_live = embeddings[train_fraud_idx.to(device)]
                _, _, sub_labels = _fraud_prototypes(fraud_emb_live, legit_centroid_live)
                anchor_idx, pos_idx, neg_idx = _sample_multi_prototype_triplets(
                    train_fraud_idx, train_legit_idx, sub_labels, n_triplets_per_epoch, generator)
                anchor_emb, pos_emb, neg_emb = embeddings[anchor_idx.to(device)], embeddings[pos_idx.to(device)], embeddings[neg_idx.to(device)]
                loss = triplet_loss_fn(anchor_emb, pos_emb, neg_emb)
            elif mining == "supcon":
                batch_fraud = train_fraud_idx[torch.randperm(len(train_fraud_idx), generator=generator)[:n_triplets_per_epoch]]
                batch_legit = train_legit_idx[torch.randperm(len(train_legit_idx), generator=generator)[:n_triplets_per_epoch]]
                batch_idx = torch.cat([batch_fraud, batch_legit]).to(device)
                batch_labels = torch.cat([torch.ones(len(batch_fraud)), torch.zeros(len(batch_legit))]).to(device)
                loss = _supcon_loss(embeddings, batch_idx, batch_labels, temperature)
            elif mining == "align_uniform":
                loss = _align_uniform_loss(embeddings, train_fraud_idx.to(device), train_legit_idx.to(device),
                                            generator, n_triplets_per_epoch, align_weight, uniform_weight)
            else:
                raise ValueError(f"Unknown mining: {mining!r} (expected 'random', 'hard', 'camo_weighted', "
                                  f"'camo_weighted_temp', 'camo_weighted_dual', 'camo_weighted_margin_scale', "
                                  f"'camo_weighted_mlp', 'semi_hard', 'multi_prototype', 'supcon', or 'align_uniform')")
            if compression_weight > 0:
                comp_loss = _compression_loss(embeddings, train_fraud_idx.to(device), train_legit_idx.to(device))
                loss = loss + compression_weight * comp_loss
            if gate_aux_weight > 0 and hasattr(model, "gate_logits"):
                # Directly supervises GraphSAGEGated's per-edge gate against y_src==y_dst on
                # known-labeled training edges -- same mechanism as train_gnn.py's
                # loss.gate_auxiliary_weight, never actually run as an experiment before (the gate
                # previously only got an indirect signal through the main classification loss,
                # Runs 59-61, and showed no real effect). Combined here with a metric-learning main
                # objective instead of BCE/focal classification, since metric learning is this
                # investigation's only confirmed lever that beats plain classification on Elliptic.
                gate_logits = model.gate_logits(data.x, data.train_edge_index)
                src, dst = data.train_edge_index[0], data.train_edge_index[1]
                both_train = data.train_mask[src] & data.train_mask[dst]
                if both_train.any():
                    same_class = (data.y[src] == data.y[dst]).float()
                    gate_loss = F.binary_cross_entropy_with_logits(gate_logits[both_train], same_class[both_train])
                    loss = loss + gate_aux_weight * gate_loss
            loss.backward()
            optimizer.step()
            if mining == "camo_weighted_mlp" and epoch > camo_mlp_warmup_epochs + 1:
                # epoch == warmup+1 itself already seeded ema_weight_mlp directly above (exact
                # copy, decay update there would be a no-op anyway since ema==live at that instant)
                _update_ema(ema_weight_mlp, weight_mlp, camo_mlp_ema_decay)

            model.eval()
            with torch.no_grad():
                val_embeddings = F.normalize(model.embed(data.x, data.val_edge_index), dim=-1)
                legit_centroid = val_embeddings[train_legit_idx.to(device)].mean(dim=0)
                if mining == "multi_prototype":
                    fraud_emb_val = val_embeddings[train_fraud_idx.to(device)]
                    c0, c1, _ = _fraud_prototypes(fraud_emb_val, legit_centroid)
                    val_scores = _multi_centroid_scores(val_embeddings[data.val_mask], c0, c1, legit_centroid)
                else:
                    fraud_centroid = val_embeddings[train_fraud_idx.to(device)].mean(dim=0)
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
            legit_centroid = test_embeddings[train_legit_idx.to(device)].mean(dim=0)
            train_fraud_emb = test_embeddings[train_fraud_idx.to(device)]
            train_legit_emb = test_embeddings[train_legit_idx.to(device)]
            test_emb_masked = test_embeddings[data.test_mask]

            if mining == "multi_prototype":
                obvious_centroid, camo_centroid, _ = _fraud_prototypes(train_fraud_emb, legit_centroid)
                test_scores = _multi_centroid_scores(test_emb_masked, obvious_centroid, camo_centroid, legit_centroid)
                dist_to_obvious = (test_emb_masked - obvious_centroid).pow(2).sum(-1).sqrt()
                dist_to_camo = (test_emb_masked - camo_centroid).pow(2).sum(-1).sqrt()
                dist_to_fraud = torch.minimum(dist_to_obvious, dist_to_camo).cpu().numpy()
                dist_to_legit = (test_emb_masked - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
                fraud_dist_obvious = (train_fraud_emb - obvious_centroid).pow(2).sum(-1).sqrt()
                fraud_dist_camo = (train_fraud_emb - camo_centroid).pow(2).sum(-1).sqrt()
                fraud_spread = torch.minimum(fraud_dist_obvious, fraud_dist_camo).cpu().numpy()
            else:
                fraud_centroid = train_fraud_emb.mean(dim=0)
                test_scores = _centroid_scores(test_emb_masked, fraud_centroid, legit_centroid)
                # Raw per-centroid distances (not just the combined sigmoid score) -- lets a caller
                # test legit-centroid-only scoring (2026-07-21 discussion: fraud may not cluster
                # coherently enough for its own centroid to be a meaningful reference point, unlike
                # legit) without retraining.
                dist_to_fraud = (test_emb_masked - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
                dist_to_legit = (test_emb_masked - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
                # Within-class spread in the LEARNED embedding space -- is train fraud actually a
                # coherent cluster (tight spread around its own centroid) or diffuse (large spread,
                # meaning the fraud centroid itself is a weak/uninformative reference point)?
                fraud_spread = (train_fraud_emb - fraud_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

            legit_spread = (train_legit_emb - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()
            train_fraud_dist_to_legit = (train_fraud_emb - legit_centroid).pow(2).sum(-1).sqrt().cpu().numpy()

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

    # Rounded to 5 decimals -- full float64 precision buys nothing here and IEEE-CIS's test set
    # (~88K, vs. Elliptic's ~8.8K) makes these arrays 10x bigger; unrounded, that's most of what
    # pushed a return_embeddings=True IEEE-CIS run into the same silent RunPod payload-size
    # failure (COMPLETED status, output=None) the 2026-07-21 fix addressed for Elliptic.
    result = {
        "test": test_metrics, "best_epoch": best_epoch,
        "test_scores": np.round(test_scores, 5).tolist(), "test_y": test_y.tolist(),
        "dist_to_fraud": np.round(dist_to_fraud, 5).tolist(), "dist_to_legit": np.round(dist_to_legit, 5).tolist(),
        "fraud_spread_mean": float(fraud_spread.mean()), "fraud_spread_std": float(fraud_spread.std()),
        "legit_spread_mean": float(legit_spread.mean()), "legit_spread_std": float(legit_spread.std()),
        "train_fraud_dist_to_legit": np.round(train_fraud_dist_to_legit, 5).tolist(),
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
        # REOPENED (2026-07-22): "small" was an Elliptic-specific assumption (~3462 train fraud).
        # IEEE-CIS has ~14538 train fraud -- reproduced the exact same silent COMPLETED-with-
        # output=None failure. Cap train fraud the same way legit already is, instead of assuming
        # any dataset's fraud count is inherently small.
        rng = np.random.default_rng(config["seed"])
        test_y_np = test_y
        test_fraud_pos = np.where(test_y_np == 1)[0]
        test_legit_pos = np.where(test_y_np == 0)[0]
        test_legit_sample = rng.choice(test_legit_pos, size=min(1500, len(test_legit_pos)), replace=False)
        test_keep = np.sort(np.concatenate([test_fraud_pos, test_legit_sample]))

        legit_idx_sample = rng.choice(len(train_legit_emb), size=min(1500, len(train_legit_emb)), replace=False)
        fraud_idx_sample = rng.choice(len(train_fraud_emb), size=min(4000, len(train_fraud_emb)), replace=False)

        def _round(arr):
            return np.round(arr, 5).tolist()

        result["test_embeddings"] = _round(test_emb_masked[test_keep].cpu().numpy())
        result["test_embeddings_y"] = test_y_np[test_keep].tolist()
        result["train_fraud_embeddings"] = _round(train_fraud_emb[fraud_idx_sample].cpu().numpy())
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
