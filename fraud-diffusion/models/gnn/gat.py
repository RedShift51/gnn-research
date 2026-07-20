import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class GAT(nn.Module):
    """Multi-head GAT node classifier (binary fraud logit).

    Attention over neighbors matters more than plain mean/max aggregation (GraphSAGE) when
    fraud camouflages itself among mostly-legitimate neighbors (heterophily) — GAT can learn
    to downweight the legitimate majority of a node's neighborhood.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 heads: int = 8, dropout: float = 0.6):
        super().__init__()
        assert num_layers >= 1

        self.dropout = dropout
        self.convs = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=1, dropout=dropout))
        else:
            self.convs.append(GATv2Conv(in_dim, hidden_dim, heads=heads, dropout=dropout))
            for _ in range(num_layers - 2):
                self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout))
            # Final GAT layer: average heads instead of concatenating, so output dim == hidden_dim.
            self.convs.append(GATv2Conv(hidden_dim * heads, hidden_dim, heads=heads, concat=False, dropout=dropout))

        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(x)

        return self.classifier(x).squeeze(-1)
