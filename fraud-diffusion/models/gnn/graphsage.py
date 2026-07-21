import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


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
                 classifier_hidden_dim: int | None = None):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Node embeddings (last hidden layer, pre-classifier) — for the GNN-embeddings+RF hybrid
        (see evaluation/gnn_rf_hybrid.py): RF is far better than a single linear layer at modeling
        nonlinear interactions between raw and graph-derived features (LAB_JOURNAL.md Run 52), so
        this hands RF the GNN's structural encoding instead of the GNN's own classifier head."""
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
                 classifier_hidden_dim: int | None = None):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim * 3, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """See GraphSAGE.embed's docstring — same rationale, for the GNN-embeddings+RF hybrid."""
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
