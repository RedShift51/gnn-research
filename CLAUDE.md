# Diffusion-Augmented Graph Fraud Detection

## Goal
Build a GNN-based fraud detection system on transaction graphs, augmented with diffusion models for synthetic fraud generation. Show that diffusion augmentation improves fraud detection metrics over vanilla GNN baselines on REAL datasets (not just synthetic PaySim).

## Why This Matters
- Fraud = 0.1-3.5% of transactions → extreme class imbalance
- GNNs struggle with few fraud examples + camouflage (heterophily)
- Diffusion models can generate realistic synthetic fraud patterns
- This combination (diffusion + GNN + fraud) is cutting-edge (2025-2026), few papers exist

## Target Metrics — Benchmarks to Beat

### PaySim (synthetic, 6.3M transactions, 0.13% fraud)
| Method | F1-macro | AUC-ROC | Type | Source |
|---|---|---|---|---|
| XGBoost + feat eng | 0.945 | 0.99 | Tabular | Standard baseline |
| HOT-GNN (SOTA) | 0.890 | 0.993 | GNN | Expert Systems 2026 |
| GRAD (graph diffusion) | 0.820 | 0.948 | GNN + Diffusion | WWW 2025 |
| GNN-XAI | 0.857 | 0.874 | GNN + Explainability | 2025 |
| QTGNN (quantum-inspired) | 0.990 | 0.998 | Hybrid | 2025 (suspicious) |

### IEEE-CIS (REAL e-commerce, 590K transactions, 3.5% fraud)
| Method | F1 | AUC-ROC | Type |
|---|---|---|---|
| LightGBM | 0.726 | 0.962 | Tabular |
| XGBoost | 0.616 | 0.932 | Tabular |
| Random Forest | 0.550 | 0.898 | Tabular |
| SCAFDS (GNN) | 0.508 | 0.802 | GNN (institution-level) |
| IMHA | 0.862 | 0.978 | Attention-based |

### Elliptic Bitcoin (REAL, 203K nodes, 2% illicit)
| Method | F1 | AUC-ROC | Type |
|---|---|---|---|
| GCN baseline | ~0.65 | ~0.85 | GNN |
| GraphSAGE | ~0.70 | ~0.88 | GNN |
| EvolveGCN | ~0.75 | ~0.90 | Temporal GNN |

### ⚠️ Benchmark validity warning: transductive vs inductive evaluation (2026-07-21)

arXiv 2604.19514 ("When Graph Structure Becomes a Liability: A Critical Re-Evaluation of GNNs for
Bitcoin Fraud Detection under Temporal Distribution Shift", Apr 2026) found that most published
Elliptic GNN results are evaluated transductively, and the consensus that GNNs beat feature-only
baselines does NOT hold once that leakage is removed:

```
Transductive (as everyone has been doing):
  Build the graph from ALL transactions (train + val + test)
  Train the GNN on this complete graph
  -> the GNN sees test-node neighbors during training
  -> leakage: the model "peeks" into the future
  -> F1 = 0.77+ (looks good, but is not honest)

Inductive (the correct way):
  Build the graph from ONLY train transactions
  Remove test edges entirely
  -> the GNN never sees future neighbors
  -> F1 = 0.689 (GraphSAGE) -- worse than Random Forest
```

Concretely, under their strict inductive protocol: Random Forest on raw features (F1=0.821) beats
every GNN tested; their GraphSAGE only reaches F1=0.689; a paired controlled experiment attributes
a 39.5-point F1 gap to training-time exposure to test-period adjacency; and edge-shuffle ablations
show randomly-wired graphs outperforming the real transaction graph under temporal shift.

**We had exactly this bug** in `training/train_gnn.py` (full-batch training forward pass used the
complete, temporally unfiltered graph every epoch) — found and fixed 2026-07-21, see
`fraud-diffusion/LAB_JOURNAL.md` Run 38 for the fix and its (surprising: much smaller than the
paper's reported gap, for reasons not yet fully understood) empirical effect on our own numbers.

**Other benchmarks in the literature to treat with suspicion** until their eval protocol is
confirmed (flagging rather than asserting — we have not verified these ourselves):

| Method | Reported F1 | Leakage risk |
|---|---|---|
| EvolveGCN | ~0.75-0.77 | Transductive? ⚠️ |
| CNN-GNN-LSTM | 0.902 | Unclear ⚠️⚠️ |
| ChronoWave-GNN | 0.979 | Likely transductive 🚩 (a 0.979 F1 on a dataset this hard is itself a red flag) |
| MDST-GNN | (best-in-class claimed) | Claims a temporal/inductive setup — verify before trusting |

Any time we cite a literature number as a target to beat on Elliptic, check its eval protocol
first — a transductive number isn't a fair comparison point for our own strictly inductive results.

### Our own honest Elliptic numbers (2026-07-21, see LAB_JOURNAL.md Runs 38-55 for full detail)
All under the corrected, leakage-free pipeline. Single-seed numbers turned out to be unreliable
this session (several "best results" were seed=42 favorable draws, not the true expected
performance) — multi-seed means (5 seeds) are the ones to trust:

| Method | Multi-seed mean F1 | Notes |
|---|---|---|
| Plain GraphSAGE (honest baseline, no diffusion) | 0.7335 | — |
| GraphSAGEDiff (explicit self-vs-neighborhood deviation feature) | 0.7483 | not yet conventionally significant (p=0.125, n=5) vs plain GraphSAGE |
| **GraphSAGEDiff embeddings + Random Forest hybrid** | **0.7670** | closes ~60% of the gap to RF; best-supported GNN-based result |
| **Random Forest, raw features, NO graph at all** | **0.7890** | the number to beat |

**We independently reproduced the paper's core finding in our own pipeline**: plain RF beats every
pure-GNN variant tried. But the GNN-embeddings+RF hybrid (train GraphSAGEDiff normally, feed its
learned node embeddings + raw features into RF instead of the GNN's own linear classifier head)
closes most of that gap — evidence that RF's advantage is substantially about modeling nonlinear
feature interactions (which tree ensembles do and a single linear layer doesn't), not simply
"graph structure is useless here." Diffusion augmentation did NOT stack additively with any other
intervention tried this session (alpha, adversarial loss, GraphSAGEDiff, the hybrid) — a
repeatable pattern (4 independent instances), not a fluke; test new ideas without diffusion first.

### The real bottleneck: a temporal regime break, not architecture (2026-07-21, Runs 56-86)
An exhaustive architecture search (learned neighbor gates, spectral filters, FAGCN, DropEdge,
degree-awareness — every one tested and found flat or negative, several confirmed via multi-seed)
led to a much bigger finding: **RF/GNN's shared "hard core" of ~52% unrecoverable fraud is a sharp,
absolute-timeline regime break**, not a generic difficulty. Recall collapses from ~80% to ~0-5%
exactly at step 42/43 of Elliptic's 49-step timeline, and stays collapsed for the rest of the test
period — confirmed via three separate falsification tests (shifted split boundaries, a pre-break-
only window, and training WITH post-break labels), all reproducing the identical break regardless
of where the train/test boundary sits. Post-break fraud sits ~2x outside the training-fraud
distribution (nearest-neighbor test) and its embedding profile moves sharply toward legit — mean
distance-to-legit-centroid drops from 1.54 (pre-break fraud) to 0.91 (post-break fraud), vs.
legit's own mean of 0.65 — but it does NOT fully collapse onto legit: post-break fraud's
distribution keeps a distinct, right-shifted mode peaking around 0.67 vs. legit's peak at 0.42.
Real, exploitable separation remains, just much weaker and more overlapping than pre-break fraud's
signal (whose mode peaks near 1.68, almost fully separate from legit). This looks like real
camouflage/evasion evolving over time, not an unrelated new pattern. Domain adaptation (CORAL) and
resampling were both tested and both fail for principled reasons (see LAB_JOURNAL.md Runs 74-77).

**What "camo-weighted" is** (the one lever that DID work, confirmed across 5 seeds — Run 82/84):
a GraphSAGEDiff encoder trained with **metric learning** (triplet loss on embeddings, classified
by nearest centroid — fraud centroid vs. legit centroid — rather than a classification head) plus
one specific twist. Investigating the pre-break ("easy") period's own fraud population revealed it
is NOT one uniform cluster — ~40% of it is already a **camouflaged sub-archetype** that looks much
closer to legit than the "obvious" 60% majority does (a 2-component GMM on distance-to-legit
cleanly separates them). Post-break fraud aligns 95% with this pre-existing camouflaged
sub-archetype, not with the obvious one. The problem with plain triplet loss: random anchor/
positive sampling treats every fraud example equally, so the rare camouflaged 40% gets diluted
into one aggregate fraud centroid dominated by the obvious majority. **Camo-weighted mining fixes
this with a soft importance weight**: every random triplet's contribution to the loss is
up-weighted by how close its fraud anchor currently is to the legit centroid (softmax-normalized,
not a hard top-1 selection — a gentler alternative to standard batch-hard mining, which was tried
first and badly destabilized training, F1 crashing to 0.32). This forces the model to keep
learning from the camouflaged/hard cases throughout training instead of letting them be
outnumbered by easy examples every epoch.

**Result, confirmed across 5 seeds**: recovers 12-27% (mean ~18%) of the previously fully-
unrecoverable hard core, with **zero cases lost** among everything already correctly classified,
in every single seed. This is the first result in the entire Elliptic investigation that both
meaningfully moves the hard-core number and survives proper multi-seed scrutiny without
evaporating (contrast: DropEdge and a degree-aware variant both looked promising at n=5 seeds and
fully evaporated at n=10 — a repeated lesson this session about not trusting small-n wins).
A follow-up subspace-restriction idea (score using only the embedding dimensions that show the
most per-node anomaly signal) also helps, but the first version of that test leaked (dimensions
were selected using the very test labels being scored) — the corrected, leakage-free version still
shows a real but smaller gain (AUC 0.782 → 0.793 on 15-20 dims chosen from held-out train data
only). See LAB_JOURNAL.md Runs 78, 81, 82, 84, 85, 86 for full detail and `evaluation/
metric_learning.py` for the implementation.

### Our Targets
- **Minimum:** Beat GNN baseline by 5%+ F1 with diffusion augmentation
- **Good:** Beat HOT-GNN on PaySim (>0.89 F1) OR beat LightGBM on IEEE-CIS (>0.726 F1)
- **Publishable:** Consistent improvement across 3+ datasets with ablation study
- **Exceptional:** Beat IMHA on IEEE-CIS (>0.862 F1) with graph-based approach

## Datasets

### Primary (MUST USE — real data):

**1. IEEE-CIS Fraud Detection (Vesta Corporation)**
- 590,540 real e-commerce card-not-present transactions
- 393 features (transaction + identity tables)
- 3.5% fraud rate
- Temporal ordering preserved
- Source: Kaggle (requires competition join)
- Amazon FDB benchmark: github.com/amazon-science/fraud-dataset-benchmark
- Graph construction: user → card → device → IP → merchant

**2. Elliptic Bitcoin**
- 203,769 Bitcoin transaction nodes
- 234,355 edges (payment flows)
- 166 features per node
- 2% illicit labels, 21% unknown
- 49 time steps → temporal evaluation
- Already a graph — no construction needed
- Source: kaggle.com/datasets/ellipticco/elliptic-data-set

### Secondary (for completeness):

**3. PaySim (synthetic baseline)**
- 6.3M synthetic mobile money transactions
- 0.13% fraud rate
- Simple features but standard benchmark
- Source: kaggle.com/datasets/ealaxi/paysim1

**4. T-Finance**
- 39,357 nodes, 21.2M edges
- ~4.6% fraud rate
- Available in PyG datasets
- Large-scale graph

**5. DGraph (Finvolution)**
- 3.7M nodes, 4.3M edges
- 1.3% fraud, largest public financial graph
- Source: dgraph.xinye.com

**6. YelpChi (review fraud)**
- 45,954 nodes, 3 relation types
- ~14.5% spam reviews
- Heterogeneous graph
- Source: DGL library

**7. Amazon (review fraud)**
- 11,944 nodes, 2 relation types
- ~9.5% fake reviews
- Source: DGL library

## HOT-GNN: Current SOTA to Beat

### Key innovations:
1. **Heterophily-aware**: Fraudsters befriend legitimate accounts (camouflage)
   - Decoupled multi-view message passing
   - Separate aggregation for homophilic vs heterophilic neighbors
   - Multi-view fusion via attention

2. **Outlier-aware (HOS measure)**:
   - Hybrid Outlier-aware Similarity for each edge
   - Structural similarity + Semantic similarity + Anomaly alignment
   - Distinguishes genuine from deceptive relations

3. **Temporal-aware**:
   - Temporal positional encoding in node features
   - No explicit graph snapshots needed (efficient)
   - Captures behavioral dynamics

### Why we can beat it:
- HOT-GNN does NOT use data augmentation → class imbalance hurts
- Adding diffusion-generated fraud samples should boost recall significantly
- HOT-GNN + diffusion augmentation = our approach (additive improvement)

## Architecture

### Stage 1: Graph Construction (for IEEE-CIS)
```python
# IEEE-CIS doesn't come as a graph — we build one
# Nodes: transactions (or accounts)
# Edges: shared card, shared device, shared IP, shared email domain

# Option A: Transaction-centric graph
# Each transaction = node, edges = shared entity
import torch_geometric
from torch_geometric.data import HeteroData

# Option B: Bipartite graph  
# Account nodes + Merchant nodes, transaction edges
# Edge features: amount, time, type
```

### Stage 2: Feature Engineering
```python
# Transaction features (from IEEE-CIS 393 raw features):
# - Amount, time delta, card info, device info
# - V1-V339 (anonymized Vesta features)

# Derived graph features:
# - Node degree (how many transactions)
# - PageRank / betweenness centrality
# - Temporal velocity (transactions per hour)
# - Amount deviation from account mean
# - Cross-border flag
# - New merchant flag
# - Night transaction flag
```

### Stage 3: GNN Baselines (implement all 3)
```python
# 1. GraphSAGE — inductive, works on unseen nodes
# Best for production (new transactions at inference)
from torch_geometric.nn import SAGEConv

# 2. GAT — attention over neighbors
# Best for heterophily (attends to important neighbors)  
from torch_geometric.nn import GATConv

# 3. GCN — simplest baseline
from torch_geometric.nn import GCNConv

# Loss: Focal Loss (handles extreme imbalance)
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.25, gamma=2.0):

# Eval: F1-macro, AUC-ROC, G-mean, AUPRC
```

### Stage 4: Diffusion Augmentation (3 approaches)
```python
# Approach 1: Tabular Diffusion (TabDDPM)
# Generate synthetic fraud TRANSACTIONS (tabular features)
# Then add to graph as new nodes
# Paper: TabDDPM (ICML 2023)

# Approach 2: Node Feature Diffusion
# Train DDPM on fraud node EMBEDDINGS (from GNN encoder)
# Sample new fraud embeddings → add synthetic fraud nodes
# More graph-aware than pure tabular

# Approach 3: Graph Structure Diffusion (GRAD-style)
# Generate new fraud EDGES/RELATIONS
# Augment graph topology to expose hidden fraud patterns
# Paper: GRAD (WWW 2025)

# Approach 4: Combined
# TabDDPM for features + Graph Diffusion for structure
# Both augmentations together
```

**Implemented and validated (2026-07-21, see LAB_JOURNAL.md Run 40)**: Approach 1 (TabDDPM) is
built and working (models/diffusion/tabddpm.py, training/train_diffusion.py), with two optional
auxiliary losses on top, both off by default and gated to start partway through training (not
epoch 1 — the denoiser needs to already be decent before either signal is useful):
- **Adversarial fine-tuning** (`diffusion.adversarial.enabled`): a discriminator scores real vs
  synthetic (the denoiser's one-step x0 estimate, not a full multi-step sample — too expensive per
  batch). First attempt regressed (discriminator learned faster than the denoiser could adapt, a
  classic GAN pathology, diagnosed via wandb curves); fixed with a 10x lower discriminator LR,
  which then beat the non-adversarial baseline.
- **Spectral matching** (`diffusion.spectral.enabled`): MSE between the synthetic and real fraud
  sets' feature-covariance eigenvalues (same idea as FID's covariance term). Stable out of the box,
  no GAN-style imbalance to diagnose.

Approaches 2/3 (embedding diffusion, graph-structure diffusion) are not yet started.

### Stage 5: Full Pipeline
```
Raw transactions 
→ Graph construction (shared entities)
→ Feature engineering (393 + derived)
→ Diffusion augmentation (synthetic fraud)
→ GNN training (GraphSAGE/GAT with Focal Loss)
→ Evaluation (F1, AUC, G-mean on temporal test split)
→ Ablation study
```

### Stage 6: Ablation Study (critical for paper)
```
1. XGBoost only (tabular baseline, no graph)
2. GNN only (graph baseline, no augmentation)
3. GNN + SMOTE (classical oversampling)
4. GNN + Random oversampling
5. GNN + TabDDPM (diffusion on tabular features)
6. GNN + Node Diffusion (diffusion on embeddings)
7. GNN + GRAD (diffusion on graph structure)
8. GNN + Combined (tabular + graph diffusion) ← expected best
9. HOT-GNN (reproduce their results as reference)
10. HOT-GNN + our diffusion augmentation ← stretch goal
```

## Project Structure
```
fraud-diffusion/
├── data/
│   ├── download.py              # Download IEEE-CIS, Elliptic, PaySim
│   ├── ieee_cis_preprocess.py   # IEEE-CIS → graph
│   ├── elliptic_preprocess.py   # Elliptic graph loading
│   ├── paysim_preprocess.py     # PaySim → graph
│   └── graph_builder.py         # Shared graph construction utils
├── models/
│   ├── gnn/
│   │   ├── graphsage.py
│   │   ├── gat.py
│   │   ├── gcn.py
│   │   └── hot_gnn.py           # HOT-GNN reproduction
│   ├── diffusion/
│   │   ├── tabddpm.py           # Tabular diffusion (TabDDPM)
│   │   ├── node_diffusion.py    # Node embedding diffusion
│   │   ├── graph_diffusion.py   # GRAD-style graph diffusion
│   │   └── scheduler.py         # Noise schedulers
│   └── baselines/
│       ├── xgboost_baseline.py
│       ├── smote_gnn.py
│       └── random_oversample.py
├── training/
│   ├── train_gnn.py
│   ├── train_diffusion.py
│   ├── train_augmented.py       # GNN + diffusion pipeline
│   └── losses.py                # Focal loss, class-weighted CE
├── evaluation/
│   ├── metrics.py               # F1-macro, AUC-ROC, G-mean, AUPRC
│   ├── ablation.py              # Run all ablation experiments
│   ├── visualize.py             # t-SNE, attention maps, ROC curves
│   └── statistical_test.py      # Wilcoxon signed-rank (10 seeds)
├── configs/
│   ├── ieee_cis.yaml
│   ├── elliptic.yaml
│   └── paysim.yaml
├── notebooks/
│   ├── 01_eda_ieee_cis.ipynb
│   ├── 02_eda_elliptic.ipynb
│   ├── 03_results.ipynb
│   └── 04_visualizations.ipynb
├── scripts/
│   ├── run_all_experiments.sh
│   └── generate_tables.py
├── CLAUDE.md
└── README.md
```

## Tech Stack
- Python 3.11+
- PyTorch 2.x
- PyTorch Geometric (PyG) — GNN, graph data
- torch-geometric-temporal — temporal graphs
- diffusers or custom DDPM — diffusion models
- tab-ddpm — tabular diffusion (github.com/yandex-research/tab-ddpm)
- scikit-learn — baselines, metrics
- XGBoost / LightGBM — tabular baselines
- DGL (optional) — for loading YelpChi/Amazon
- wandb — experiment tracking
- pandas, numpy, matplotlib, seaborn

## GPU Compute Infrastructure

### Recommended: RunPod (best balance of price/API/reliability)

**Serverless (for inference & short experiments):**
```python
import runpod

# Deploy serverless endpoint
endpoint = runpod.Endpoint("YOUR_ENDPOINT_ID")
run = endpoint.run_sync({
    "input": {
        "experiment": "graphsage_ieee_cis",
        "config": {"lr": 0.001, "epochs": 100, "seed": 42}
    }
})
# Pay-per-second, scale-to-zero when idle
```

**Pods (for training & long experiments):**
```python
# Programmatic pod creation for agent workflows
pod = runpod.create_pod(
    name="fraud-exp-42",
    image_name="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel",
    gpu_type_id="NVIDIA RTX A6000",  # $0.76/hr
    gpu_count=1,
    volume_in_gb=50,
    ports="8888/http,6006/http",  # Jupyter + TensorBoard
)
# SSH in, run experiments, destroy when done
```

**Agent integration:**
```python
# Agent can programmatically:
# 1. Create pod with specific GPU
# 2. Upload code/data
# 3. Run experiment
# 4. Download results
# 5. Destroy pod
# All via RunPod Python SDK or GraphQL API
```

### Budget option: Vast.ai (cheapest, less reliable)
```bash
# CLI-driven — agent launches via subprocess
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 inet_down>200 reliability>0.95'
vastai create instance $OFFER_ID \
    --image pytorch/pytorch:2.1.0-cuda12.1 \
    --disk 50 \
    --onstart-cmd "pip install torch-geometric wandb xgboost"
vastai execute $INSTANCE_ID "cd /workspace && python train.py --config exp42.yaml"
vastai logs $INSTANCE_ID
vastai destroy instance $INSTANCE_ID
```

### Price Comparison (on-demand, as of mid-2026)
| GPU | Vast.ai | RunPod | Lambda | Spheron | Modal |
|---|---|---|---|---|---|
| RTX 4090 | $0.33/hr | $0.39/hr | — | $0.58/hr | ~$1.00/hr |
| A100 80GB | $0.67/hr spot | $1.64/hr | $1.29/hr | $1.07/hr | ~$2.80/hr |
| H100 SXM | $2.00/hr | $2.69/hr | $2.49/hr | $2.01/hr | ~$3.95/hr |

### Recommended Setup for This Project
```
Phase 1 (EDA + baselines):     Local / free Colab (no GPU needed for XGBoost)
Phase 2 (GNN training):        RunPod RTX 4090 ($0.39/hr) — 2-4 hours per experiment
Phase 3 (Diffusion training):  RunPod A100 ($1.64/hr) — TabDDPM needs more VRAM
Phase 4 (Full ablation):       Vast.ai spot A100 ($0.67/hr) — 10 seeds × 10 configs = 100 runs
Phase 5 (Final results):       RunPod A100 — clean reproducible runs for paper

Estimated total compute cost: $50-100 for full project
```

### Experiment Tracking
```python
# wandb for all experiments
import wandb
wandb.init(project="fraud-diffusion", config=config)
wandb.log({"f1_macro": f1, "auc_roc": auc, "g_mean": gmean})

# Agent can query wandb API for best configs
api = wandb.Api()
runs = api.runs("fraud-diffusion")
best_run = max(runs, key=lambda r: r.summary.get("f1_macro", 0))
```

### Project Structure Addition
```
fraud-diffusion/
├── ...existing structure...
├── infra/
│   ├── runpod_launcher.py       # Programmatic pod creation
│   ├── vastai_launcher.py       # Vast.ai CLI wrapper
│   ├── experiment_agent.py      # Agent that manages GPU experiments
│   └── sync_results.py          # Download results from remote GPU
```

## Implementation Plan

### Week 1: Data + Baselines
- Day 1: Download IEEE-CIS, Elliptic, PaySim. EDA notebooks
- Day 2: IEEE-CIS graph construction (shared card/device/IP)
- Day 3: XGBoost/LightGBM baselines on IEEE-CIS (reproduce 0.96 AUC)
- Day 4: GraphSAGE baseline on IEEE-CIS graph
- Day 5: GAT baseline + GCN baseline. Compare all on 3 datasets

### Week 2: Diffusion + Augmentation
- Day 6: Implement TabDDPM for fraud transactions
- Day 7: Generate synthetic fraud, evaluate quality (column distributions, pairwise correlations)
- Day 8: Train GNN on augmented dataset (TabDDPM)
- Day 9: Implement node embedding diffusion + GRAD-style graph diffusion
- Day 10: Full ablation study — all 10 combinations on IEEE-CIS

### Week 3: Multi-dataset + Polish
- Day 11: Run winning approach on Elliptic + PaySim + T-Finance
- Day 12: Statistical significance (10 seeds, Wilcoxon test)
- Day 13: Visualizations: t-SNE of embeddings, ROC curves, attention maps
- Day 14: Write README, clean code, GitHub publish

## Key Papers to Reference
1. **HOT-GNN** (Expert Systems with Applications, 2026) — Current SOTA, heterophily-aware GNN
2. **GRAD** (WWW 2025) — Guided Relation Diffusion for Graph Fraud Detection
3. **FraudDiffuse** (ICAIF 2024) — Diffusion-aided synthetic fraud augmentation
4. **TabDDPM** (ICML 2023) — Diffusion for tabular data generation
5. **PRAGMA** (arXiv 2026) — Foundation model for financial transactions
6. **PMP** (2024) — Partitioning Message Passing for graph fraud
7. **RGTAN** (2023) — Relational Graph Transformer for fraud
8. **TF-GNN** (2026) — Temporal Feedback GNN with discriminative regularization
9. **DiffScene** (AAAI 2025) — diffusion for safety-critical scenarios
10. **SCAFDS** (2025) — Edge-feature GAT for interbank fraud
11. **FDB** (Amazon Science) — Fraud Dataset Benchmark
12. **DGraph** (NeurIPS 2022) — Large-scale financial graph dataset
13. **"When Graph Structure Becomes a Liability"** (arXiv 2604.19514, Apr 2026) — critical
    re-evaluation of GNNs on Elliptic under strict inductive (leakage-free) evaluation; see the
    benchmark validity warning above

## Key GitHub Repos
- safe-graph/graph-fraud-detection-papers — curated paper list
- amazon-science/fraud-dataset-benchmark — standardized benchmark
- yandex-research/tab-ddpm — tabular diffusion
- PYG datasets — T-Finance, DGraph built-in

## Success Criteria
- [ ] XGBoost baseline reproduced on IEEE-CIS (AUC ~0.96)
- [ ] GNN baseline working on IEEE-CIS graph (AUC ~0.85+)
- [ ] Diffusion augmentation improves GNN by 5%+ F1
- [ ] Beat HOT-GNN (0.89 F1 on PaySim) OR beat LightGBM (0.726 F1 on IEEE-CIS)
- [ ] Consistent improvement on 2+ real datasets (IEEE-CIS + Elliptic)
- [ ] 10-seed statistical significance
- [ ] Clean GitHub repo with reproducible results
- [ ] (Stretch) Short paper submitted to ICAIF/KDD workshop
