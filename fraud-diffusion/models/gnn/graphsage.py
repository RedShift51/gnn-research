import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import scatter


class GraphSAGE(nn.Module):
    """2-layer GraphSAGE node classifier (binary fraud logit)."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)  # raw logits, shape [N]


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

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        assert num_layers >= 1

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim * 3, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))

        self.dropout = dropout
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        neighbor_mean = scatter(x[src], dst, dim=0, dim_size=x.shape[0], reduce="mean")
        x = torch.cat([x, neighbor_mean, x - neighbor_mean], dim=-1)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x).squeeze(-1)  # raw logits, shape [N]
