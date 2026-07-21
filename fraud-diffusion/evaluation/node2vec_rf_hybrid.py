"""Node2Vec + RF hybrid: unsupervised, LABEL-FREE structural embeddings (random-walk skip-gram)
concatenated with raw features, fed into RandomForestClassifier -- a different mechanism than
evaluation/gnn_rf_hybrid.py's supervised GNN embeddings. That hybrid's embeddings are shaped by the
classification objective; these capture purely structural position (community membership, degree
pattern, role in the graph) independent of any label, so the two are complementary rather than
redundant (2026-07-21 discussion). Also much cheaper: no GPU needed.

Hand-rolled random walk + skip-gram (NOT torch_geometric.nn.Node2Vec) -- that class requires
pyg-lib, which has no macOS wheel (confirmed: `uv pip install pyg-lib` fails with "not found in
the package registry" on this platform). This is a plain DeepWalk-style walk (uniform neighbor
sampling, p=q=1, no 2nd-order bias) + skip-gram with negative sampling, fully vectorized with numpy
for the walk generation (CSR adjacency, batched random neighbor lookups) so it stays fast even on
Elliptic's ~203k nodes.

Embeddings are trained on train_edge_index ONLY (never val/test-period edges), mirroring the same
leakage-free convention as GNN training (data/temporal_edges.py) -- later-arriving val/test nodes
with sparse/no train-period connectivity get lower-quality (near-random-init) embeddings, an
accepted, documented tradeoff rather than a leakage risk.

Usage:
    python -m evaluation.node2vec_rf_hybrid --processed-path data/processed/elliptic_graph.pt
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier

from evaluation.metrics import compute_metrics

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def _build_csr(edge_index: np.ndarray, num_nodes: int) -> tuple:
    """(row_ptr, col_idx, degree) for O(1) random-neighbor lookup per node."""
    src, dst = edge_index[0], edge_index[1]
    order = np.argsort(src, kind="stable")
    src_sorted, dst_sorted = src[order], dst[order]
    degree = np.bincount(src, minlength=num_nodes)
    row_ptr = np.zeros(num_nodes + 1, dtype=np.int64)
    np.cumsum(degree, out=row_ptr[1:])
    return row_ptr, dst_sorted, degree


def _generate_walks(edge_index: np.ndarray, num_nodes: int, walk_length: int, walks_per_node: int,
                     seed: int) -> np.ndarray:
    """Vectorized uniform random walks (DeepWalk-style) -- one big array op per step across ALL
    walks simultaneously, rather than a per-node Python loop, to stay fast on ~200k+ node graphs."""
    rng = np.random.default_rng(seed)
    row_ptr, col_idx, degree = _build_csr(edge_index, num_nodes)

    starts = np.repeat(np.arange(num_nodes), walks_per_node)
    walks = np.empty((len(starts), walk_length), dtype=np.int64)
    walks[:, 0] = starts
    current = starts.copy()

    for step in range(1, walk_length):
        deg_cur = degree[current]
        has_neighbors = deg_cur > 0
        rand_offset = (rng.random(len(current)) * np.maximum(deg_cur, 1)).astype(np.int64)
        next_idx = row_ptr[current] + rand_offset
        next_node = np.where(has_neighbors, col_idx[np.clip(next_idx, 0, len(col_idx) - 1)], current)
        walks[:, step] = next_node
        current = next_node

    return walks


def _skipgram_pairs(walks: np.ndarray, window: int) -> np.ndarray:
    """(center, context) positive pairs from a sliding window over each walk, vectorized via
    strided slicing rather than a Python loop over positions."""
    pairs = []
    walk_length = walks.shape[1]
    for offset in range(1, window + 1):
        if offset >= walk_length:
            break
        pairs.append(np.stack([walks[:, :-offset].reshape(-1), walks[:, offset:].reshape(-1)], axis=1))
        pairs.append(np.stack([walks[:, offset:].reshape(-1), walks[:, :-offset].reshape(-1)], axis=1))
    return np.concatenate(pairs, axis=0)


def train_node2vec(edge_index: torch.Tensor, num_nodes: int, embedding_dim: int = 64,
                    walk_length: int = 6, walks_per_node: int = 2, window: int = 1,
                    num_negative: int = 5, epochs: int = 2, batch_size: int = 16384,
                    seed: int = 42) -> np.ndarray:
    torch.manual_seed(seed)
    edge_np = edge_index.numpy()

    walks = _generate_walks(edge_np, num_nodes, walk_length, walks_per_node, seed)
    pairs = _skipgram_pairs(walks, window)
    logger.info(f"[node2vec] {len(walks)} walks, {len(pairs)} skip-gram pairs")

    center_emb = nn.Embedding(num_nodes, embedding_dim)
    context_emb = nn.Embedding(num_nodes, embedding_dim)
    nn.init.normal_(center_emb.weight, std=0.1)
    nn.init.normal_(context_emb.weight, std=0.1)
    optimizer = torch.optim.Adam(list(center_emb.parameters()) + list(context_emb.parameters()), lr=0.01)

    rng = np.random.default_rng(seed)
    pairs_t = torch.from_numpy(pairs)
    n_pairs = len(pairs_t)

    for epoch in range(1, epochs + 1):
        perm = torch.from_numpy(rng.permutation(n_pairs))
        total_loss, n_batches = 0.0, 0
        for start in range(0, n_pairs, batch_size):
            batch_idx = perm[start:start + batch_size]
            batch = pairs_t[batch_idx]
            centers, contexts = batch[:, 0], batch[:, 1]
            negatives = torch.from_numpy(rng.integers(0, num_nodes, size=(len(centers), num_negative)))

            optimizer.zero_grad()
            c_vec = center_emb(centers)
            pos_score = (c_vec * context_emb(contexts)).sum(-1)
            neg_score = torch.bmm(context_emb(negatives), c_vec.unsqueeze(-1)).squeeze(-1)

            loss = -F.logsigmoid(pos_score).mean() - F.logsigmoid(-neg_score).mean()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        logger.info(f"[node2vec] epoch {epoch} loss={total_loss / n_batches:.4f}")

    return center_emb.weight.data.numpy()


def run(processed_path: str, embedding_dim: int = 64, n_estimators: int = 300, seed: int = 42) -> dict:
    data = torch.load(ROOT / processed_path, weights_only=False)
    n = data.x.shape[0]

    embeddings = train_node2vec(data.train_edge_index, num_nodes=n, embedding_dim=embedding_dim, seed=seed)
    combined_x = np.concatenate([data.x.numpy(), embeddings], axis=1)
    y = data.y.numpy()
    train_mask = data.train_mask.numpy()
    val_mask = data.val_mask.numpy()
    test_mask = data.test_mask.numpy()

    clf = RandomForestClassifier(n_estimators=n_estimators, class_weight="balanced", n_jobs=-1, random_state=seed)
    clf.fit(combined_x[train_mask], y[train_mask])

    val_probs = clf.predict_proba(combined_x[val_mask])[:, 1]
    test_probs = clf.predict_proba(combined_x[test_mask])[:, 1]
    val_metrics = compute_metrics(y[val_mask], val_probs)
    test_metrics = compute_metrics(y[test_mask], test_probs)

    logger.info(f"Node2Vec+RF hybrid ({processed_path}): Val={val_metrics} Test={test_metrics}")
    return {"val": val_metrics, "test": test_metrics}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-path", required=True)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.processed_path, args.embedding_dim, args.n_estimators, args.seed)
    print(result)


if __name__ == "__main__":
    main()
