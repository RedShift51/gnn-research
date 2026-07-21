import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from models.gnn.graphsage import _build_classifier, _build_encoder


class FALayer(nn.Module):
    """One Frequency-Adaptive propagation step (Bo et al. 2021, "Beyond Low-frequency Information
    in Graph Convolutional Networks"). Computes a per-edge SIGNED gate alpha_ij = tanh(g([h_i,
    h_j])) in [-1, 1] -- the key difference from GraphSAGEGated's sigmoid gate (Run 59/60/61,
    confirmed failed): that gate could only ever down-weight a neighbor toward zero, never actively
    oppose/subtract its contribution. A negative alpha here lets the model genuinely cancel out a
    dissimilar (likely heterophilic/camouflaged) neighbor's signal instead of merely diluting it,
    which is closer to a true high-pass filter than a magnitude-only attention gate."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, 1)
        self.dropout = dropout

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, deg_inv_sqrt: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        edge_feat = torch.cat([h[src], h[dst]], dim=-1)
        alpha = torch.tanh(self.gate(edge_feat)).squeeze(-1)  # signed, in [-1, 1]
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        weighted = (alpha * norm).unsqueeze(-1) * h[src]
        return scatter(weighted, dst, dim=0, dim_size=h.shape[0], reduce="sum")


class FAGCN(nn.Module):
    """Frequency-Adaptive GCN -- an established, validated heterophily-specific architecture
    (Bo et al. 2021), not a homemade mechanism like GraphSAGEGated (learned sigmoid gate, failed,
    Run 59/60/61) or GraphSAGESpectral (hand-rolled multi-band filter, failed but confounded with
    a real implementation bug, Run 62). Each FALayer's signed gate can genuinely subtract a
    dissimilar neighbor's contribution rather than only ever down-weighting toward zero.

    h_0 = dropout(relu(input_proj(x))); each layer computes h_{l+1} = eps*h_0 + FALayer(h_l) --
    the eps*h_0 term is an initial-residual connection (same anti-oversmoothing idea as APPNP's
    personalized PageRank restart), keeping the original signal available at every depth instead of
    letting repeated signed aggregation wash it out."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 classifier_hidden_dim: int | None = None, feature_encoder_hidden_dim: int | None = None,
                 eps: float = 0.2):
        super().__init__()
        assert num_layers >= 1
        self.eps = eps
        self.dropout = dropout

        self.encoder, conv_in_dim = _build_encoder(in_dim, feature_encoder_hidden_dim, dropout)
        self.input_proj = nn.Linear(conv_in_dim, hidden_dim)
        self.layers = nn.ModuleList([FALayer(hidden_dim, dropout) for _ in range(num_layers)])
        self.classifier = _build_classifier(hidden_dim, classifier_hidden_dim, dropout)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """See GraphSAGE.embed's docstring — same rationale, for the GNN-embeddings+RF hybrid."""
        if self.encoder is not None:
            x = self.encoder(x)
        n = x.shape[0]
        src, dst = edge_index[0], edge_index[1]
        degree = scatter(torch.ones_like(dst, dtype=x.dtype), dst, dim=0, dim_size=n, reduce="sum").clamp(min=1)
        deg_inv_sqrt = degree.pow(-0.5)

        h0 = F.dropout(F.relu(self.input_proj(x)), p=self.dropout, training=self.training)
        h = h0
        for layer in self.layers:
            h = self.eps * h0 + layer(h, edge_index, deg_inv_sqrt)
        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index)).squeeze(-1)  # raw logits, shape [N]
