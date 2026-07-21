import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


def _build_encoder(in_dim: int, feature_encoder_hidden_dim: int | None, dropout: float) -> tuple[nn.Module | None, int]:
    """Optional MLP applied to raw node features BEFORE message passing (input side) -- distinct
    from _build_classifier's MLP (output side, tested in LAB_JOURNAL.md Run 56, no effect). Lets
    the network learn nonlinear raw-feature combinations upstream of structural aggregation,
    rather than hoping a single linear SAGEConv projection captures them before they get diluted
    by mean-aggregation over a mostly-legit neighborhood (2026-07-21 discussion)."""
    if feature_encoder_hidden_dim is None:
        return None, in_dim
    encoder = nn.Sequential(
        nn.Linear(in_dim, feature_encoder_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
    )
    return encoder, feature_encoder_hidden_dim


def _build_classifier(hidden_dim: int, classifier_hidden_dim: int | None, dropout: float) -> nn.Module:
    """Plain nn.Linear (default, backward compatible) or a small MLP head if classifier_hidden_dim
    is set -- a neural (still purely GNN-family) way of adding the same nonlinear feature-
    combination power the GNN+RF hybrid gets from swapping in RandomForestClassifier (Run 53/55),
    to test whether the RF gap is specifically about tree-based inductive bias or just about
    classifier expressiveness in general (2026-07-21 discussion)."""
    if classifier_hidden_dim is None:
        return nn.Linear(hidden_dim, 1)
    return nn.Sequential(
        nn.Linear(hidden_dim, classifier_hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(classifier_hidden_dim, 1),
    )


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE node classifier (binary fraud logit)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 classifier_hidden_dim: int | None = None, feature_encoder_hidden_dim: int | None = None):
        super().__init__()
        assert num_layers >= 1

        self.encoder, conv_in_dim = _build_encoder(in_dim, feature_encoder_hidden_dim, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(conv_in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Node embeddings (last hidden layer, pre-classifier) — for the GNN-embeddings+RF hybrid
        (see evaluation/gnn_rf_hybrid.py): RF is far better than a single linear layer at modeling
        nonlinear interactions between raw and graph-derived features (LAB_JOURNAL.md Run 52), so
        this hands RF the GNN's structural encoding instead of the GNN's own classifier head."""
        if self.encoder is not None:
            x = self.encoder(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)  # raw logits, shape [N]


class GraphSAGEGated(nn.Module):
    """GraphSAGEDiff + a learned per-edge similarity gate on the neighbor-aggregation step --
    a lightweight, fully end-to-end-differentiable take on the CARE-GNN (Dou et al. 2020) / PC-GNN
    (Liu et al. 2021) idea: down-weight likely-camouflaged/heterophilic neighbors instead of
    aggregating every neighbor equally. Those papers use RL-based neighbor selection or label-aware
    samplers; here a small MLP scores each edge from [x_src, x_dst, x_src - x_dst] and a sigmoid
    gate reweights that neighbor's contribution to the mean, so structurally-dissimilar (likely
    legit-camouflage) neighbors contribute less than similar ones.

    Directly targets the root cause identified in LAB_JOURNAL.md Run 46 (fraud-touching edges only
    41% homophilic) rather than adding generic capacity -- Run 56 already showed more layers/a
    bigger classifier head don't help, because they don't change WHAT gets aggregated, only how
    much capacity processes the (still-diluted) result afterward. This changes what gets
    aggregated."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 classifier_hidden_dim: int | None = None, feature_encoder_hidden_dim: int | None = None):
        super().__init__()
        assert num_layers >= 1

        self.encoder, conv_in_dim = _build_encoder(in_dim, feature_encoder_hidden_dim, dropout)
        self.gate_mlp = nn.Sequential(
            nn.Linear(conv_in_dim * 3, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(conv_in_dim * 3, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def gate_logits(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Raw (pre-sigmoid) per-edge gate score, exposed separately from embed() so the training
        loop can add an auxiliary same-class-supervision loss (LAB_JOURNAL.md Run 59/60: the gate
        showed no real effect when trained only indirectly, through the final classification loss
        — this lets it be supervised directly against y_src==y_dst on known-labeled training
        edges instead) without changing embed()'s tensor-only contract (relied on by
        evaluation/gnn_rf_hybrid.py)."""
        if self.encoder is not None:
            x = self.encoder(x)
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([x[src], x[dst], x[src] - x[dst]], dim=-1)
        return self.gate_mlp(edge_feat).squeeze(-1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """See GraphSAGE.embed's docstring — same rationale, for the GNN-embeddings+RF hybrid."""
        if self.encoder is not None:
            x = self.encoder(x)
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([x[src], x[dst], x[src] - x[dst]], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(edge_feat)).squeeze(-1)  # [E] learned neighbor-relevance weight

        weighted_sum = scatter(gate.unsqueeze(-1) * x[src], dst, dim=0, dim_size=x.shape[0], reduce="sum")
        gate_sum = scatter(gate, dst, dim=0, dim_size=x.shape[0], reduce="sum").clamp(min=1e-6).unsqueeze(-1)
        neighbor_weighted = weighted_sum / gate_sum  # gated (weighted) mean, replaces plain mean

        x = torch.cat([x, neighbor_weighted, x - neighbor_weighted], dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)  # raw logits, shape [N]


class GraphSAGESpectral(nn.Module):
    """Multi-scale spectral-style band-pass filter bank -- a simplified, from-scratch take on
    BWGNN/GHRN's core idea (Tang et al. 2022, "Rethinking Graph Neural Networks for Anomaly
    Detection"): anomalous/fraud nodes carry more HIGH-frequency graph-signal energy (they differ
    sharply from their neighborhood) than normal nodes, so decomposing node signals into several
    frequency bands and letting the network weigh them should surface camouflaged fraud signal that
    a single-scale deviation feature (GraphSAGEDiff, Run 49) or a learned per-edge gate (Run 59/60/
    61, all failed -- likely because a differentiable proxy for same-class similarity isn't
    reliable when fraud is camouflaged in feature space too) couldn't reliably extract.

    Real BWGNN builds its filter bank from actual Beta-distribution wavelet coefficients over the
    graph Laplacian's spectrum (requires either eigendecomposition or a polynomial approximation of
    it). This is a lighter, still principled generalization of GraphSAGEDiff's own trick: h_0 = x,
    h_{k+1} = mean_aggregate(h_k) (repeated 1-hop unweighted neighbor averaging -- same operator
    GraphSAGEDiff already uses once, applied K times here for K increasingly-smoothed/lower-
    frequency views). Consecutive differences b_k = h_k - h_{k+1} isolate the signal specific to
    each scale (b_0 IS GraphSAGEDiff's original single-hop deviation feature); the final h_K is the
    purely low-frequency (smoothest) band. A learnable per-band scalar gain lets training decide how
    much to weigh each frequency band rather than hard-coding equal weight."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 classifier_hidden_dim: int | None = None, feature_encoder_hidden_dim: int | None = None,
                 num_bands: int = 3):
        super().__init__()
        assert num_layers >= 1
        assert num_bands >= 1
        self.num_bands = num_bands

        self.encoder, conv_in_dim = _build_encoder(in_dim, feature_encoder_hidden_dim, dropout)
        self.band_weights = nn.Parameter(torch.ones(num_bands + 1))  # +1 for the final low-freq band
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(conv_in_dim * (num_bands + 1), hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    @staticmethod
    def _propagate_mean(h: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        return scatter(h[src], dst, dim=0, dim_size=num_nodes, reduce="mean")

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """See GraphSAGE.embed's docstring — same rationale, for the GNN-embeddings+RF hybrid."""
        if self.encoder is not None:
            x = self.encoder(x)
        n = x.shape[0]
        h = x
        bands = []
        for k in range(self.num_bands):
            h_next = self._propagate_mean(h, edge_index, n)
            bands.append(self.band_weights[k] * (h - h_next))
            h = h_next
        bands.append(self.band_weights[-1] * h)  # final purely-low-frequency band
        x = torch.cat(bands, dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)  # raw logits, shape [N]


class GraphSAGEDiff(nn.Module):
    """GraphSAGE with an explicit self-vs-neighborhood deviation feature at the input:
    concat [x, neighbor_mean(x), x - neighbor_mean(x)] before the first SAGEConv layer.

    SAGEConv already uses separate weight matrices for self vs neighbor-aggregate
    (root_weight=True by default), so it's not purely "mean-pooling that discards self
    information" — but it never explicitly computes or emphasizes their DIFFERENCE; the network
    would have to discover that on its own. This makes the residual signal explicit at the input,
    directly targeting fraud's own signature (deviating from an otherwise mostly-legit
    neighborhood) rather than hoping the network learns it implicitly. Motivated by Elliptic's
    homophily analysis (LAB_JOURNAL.md Run 46): fraud-touching edges are only 41% homophilic
    (vs an ~11.6% random-chance baseline) -- real fraud-clustering signal exists but is a minority
    of a typical fraud node's edges, exactly what naive mean-aggregation dilutes away."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 classifier_hidden_dim: int | None = None, feature_encoder_hidden_dim: int | None = None):
        super().__init__()
        assert num_layers >= 1

        self.encoder, conv_in_dim = _build_encoder(in_dim, feature_encoder_hidden_dim, dropout)
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(conv_in_dim * 3, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """See GraphSAGE.embed's docstring — same rationale, for the GNN-embeddings+RF hybrid.
        The (optional) feature encoder runs BEFORE the self-vs-neighbor deviation is computed, so
        the deviation feature itself is taken in the encoder's learned (potentially nonlinear)
        feature space rather than raw feature space."""
        if self.encoder is not None:
            x = self.encoder(x)
        src, dst = edge_index[0], edge_index[1]
        neighbor_mean = scatter(x[src], dst, dim=0, dim_size=x.shape[0], reduce="mean")
        x = torch.cat([x, neighbor_mean, x - neighbor_mean], dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)  # raw logits, shape [N]
