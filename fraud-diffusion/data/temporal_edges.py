"""Shared helper for building leakage-free temporal edge splits, used by both
paysim_preprocess.py and elliptic_preprocess.py.

Per arXiv 2604.19514 ("When Graph Structure Becomes a Liability", Apr 2026): standard transductive
GNN node-classification setups build ONE graph with every edge regardless of time period, then only
mask which node LABELS contribute to the loss — but every training forward pass still does message
passing over the full graph, so a train node's representation can be informed by its val/test-period
neighbors' features via aggregation. The paper found this "training-time exposure to test-period
adjacency" accounts for a 39.5-point F1 gap on Elliptic specifically; under a strict inductive
protocol, Random Forest on raw features beat every GNN tested.

This returns three edge sets instead of one:
  - train_edge_index: both endpoints <= train_end -- the ONLY graph the model may see during
    training forward passes (used for the loss-computing pass, every epoch).
  - val_edge_index: both endpoints <= val_end (train+val period) -- legitimate to use at
    validation-time inference (fixed weights, no gradient), since a deployed inductive model can
    always see the graph as it exists up to the current evaluation point.
  - the full edge_index (all edges, unfiltered) -- used at test-time inference, for the same reason.
"""

import numpy as np


def split_edge_index_by_time(edge_index: np.ndarray, node_time: np.ndarray,
                              train_end, val_end) -> tuple:
    """edge_index: [2, E] array (already made bidirectional). node_time: per-node time/step/position
    value, comparable to train_end/val_end via <=. Returns (train_edge_index, val_edge_index)."""
    src, dst = edge_index[0], edge_index[1]
    train_edge_mask = (node_time[src] <= train_end) & (node_time[dst] <= train_end)
    val_edge_mask = (node_time[src] <= val_end) & (node_time[dst] <= val_end)
    train_edge_index = np.ascontiguousarray(edge_index[:, train_edge_mask])
    val_edge_index = np.ascontiguousarray(edge_index[:, val_edge_mask])
    return train_edge_index, val_edge_index
