import torch
import torch.nn as nn

from models.gnn.graphsage import GraphSAGEDiff


class GraphSAGERNN(nn.Module):
    """GraphSAGEDiff branch (graph-structural embedding over card1/addr1-sharing edges) + a GRU
    branch (each transaction's own entity's — card1's — PRIOR transactions, in time order).

    The GNN aggregates same-entity neighbors as an unordered set (mean/diff over whichever
    same-card1 transactions happen to be within max_node_degree hops); it never models temporal
    ORDER within that group. The GRU branch is the explicit sequence-modeling complement: a small
    RNN over each entity's own transaction history, oldest to newest, so the model can learn
    genuinely sequential patterns (e.g. escalating amounts, a burst of transactions) that a
    permutation-invariant aggregator structurally cannot represent. seq_x is expected already
    gathered and zero-padded by the caller (see evaluation/gnn_rnn_hybrid.py) — right-aligned, so
    the GRU's last output always corresponds to the most recent real transaction regardless of how
    many leading positions are padding.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 2, dropout: float = 0.3,
                 rnn_hidden_dim: int = 64):
        super().__init__()
        self.gnn = GraphSAGEDiff(in_dim, hidden_dim, num_layers, dropout)
        self.rnn = nn.GRU(input_size=in_dim, hidden_size=rnn_hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim + rnn_hidden_dim, 1)

    def embed(self, x: torch.Tensor, edge_index: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        # x/edge_index cover the WHOLE mini-batch-sampled subgraph (seed nodes + their sampled
        # neighbors, per NeighborLoader convention); seq_x covers only the seed nodes (the first
        # seq_x.shape[0] rows, same "seed nodes come first" convention train_gnn.py's own
        # evaluate_batched relies on for logits[:batch.batch_size]). Slice the GNN branch down to
        # match before concatenating -- neighbor-node embeddings were only ever needed to CONTRIBUTE
        # to the seed nodes' own aggregation, not to be classified themselves.
        gnn_emb = self.gnn.embed(x, edge_index)[: seq_x.shape[0]]
        _, h_n = self.rnn(seq_x)  # h_n: [1, batch, rnn_hidden_dim]
        rnn_emb = h_n.squeeze(0)
        return torch.cat([gnn_emb, rnn_emb], dim=-1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, seq_x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.embed(x, edge_index, seq_x)).squeeze(-1)  # raw logits, shape [batch]
