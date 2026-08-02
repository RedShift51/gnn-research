# Research Roadmap: QMC-Enhanced Diffusion for Graph Fraud Detection & AI Safety

## Status: Active research, July 2026

---

## Research Track 1: QMC-Enhanced Diffusion for Fraud Detection

### 1.1 Core Insight
Standard diffusion models use Monte Carlo noise sampling → high variance, slow convergence.
QMC (Quasi-Monte Carlo) provides low-discrepancy sequences → better coverage, faster convergence.
Applied to fraud detection: better synthetic fraud generation + more accurate anomaly scoring.

### 1.2 Three Applications of QMC in Fraud Diffusion Pipeline

**A. QMC Noise in Diffusion Training**
```python
# Standard: random timesteps, random noise
t = torch.randint(0, T, (batch_size,))
noise = torch.randn_like(x)

# QMC: Sobol-distributed timesteps, stratified noise
sobol = torch.quasirandom.SobolEngine(dimension=1)
t = (sobol.draw(batch_size) * T).long()
# → Better coverage of noise levels
# → Faster training convergence (fewer epochs needed)
```

**B. QMC Sampling in Synthetic Fraud Generation**
```python
# Standard: sample from learned distribution with MC
z = torch.randn(n_synthetic, latent_dim)
synthetic_fraud = diffusion_model.decode(z)

# QMC: Sobol points in latent space
sobol = torch.quasirandom.SobolEngine(dimension=latent_dim, scramble=True)
z = sobol.draw(n_synthetic)
z = torch.erfinv(2 * z - 1) * sqrt(2)  # inverse CDF transform
synthetic_fraud = diffusion_model.decode(z)
# → More diverse synthetic samples
# → Better coverage of fraud pattern space
# → Less mode collapse
```

**C. QMC for Anomaly Score Estimation**
```python
# Anomaly score = E[||x - denoise(x + noise)||]
# Standard: average over K random noise samples
scores = [compute_score(x, random_noise()) for _ in range(K)]
anomaly_score = mean(scores)  # high variance

# QMC: average over K QMC noise samples
sobol_noise = generate_sobol_noise(K, dim)
scores = [compute_score(x, sobol_noise[i]) for i in range(K)]
anomaly_score = mean(scores)  # lower variance, same K
# → More accurate fraud/not-fraud decision boundary
# → Fewer false positives at same recall
```

### 1.3.1 Progress update (2026-07-21) — see fraud-diffusion/LAB_JOURNAL.md for full detail
The non-QMC baseline (plain TabDDPM + focal-loss GraphSAGE) is now solidly established on both
datasets, with an honest, leakage-audited evaluation protocol — the actual QMC integration (1.2's
A/B/C below) has NOT started yet; everything below is the conventional baseline this track's QMC
work will build on top of.
- **Diffusion augmentation is a genuine, if modest, win once done correctly**: on Elliptic,
  +0.018 F1 over the honest baseline (best recipe: alpha=0.75, n_synthetic=1731, k_connections=5,
  clamp_std=3 on the reverse-process x0); on PaySim, +0.040 F1 at low alpha, though a wash once
  combined with PaySim's own best focal-alpha (0.95) — the two techniques address overlapping
  ground on the easier dataset.
- **Unplanned but significant finding, possibly its own contribution**: found and fixed a
  transductive edge-leakage bug (training saw val/test-period graph structure) affecting our own
  pipeline and, per arXiv 2604.19514 (Apr 2026), most published Elliptic GNN results industry-wide.
  Our own before/after delta was surprisingly small (F1 0.727->0.745, not the 39.5-point collapse
  the paper found for a generic transductive setup) — an open, honestly-unresolved question worth
  discussing in any eventual writeup, not yet explained.
- Extensions attempted: adversarial fine-tuning of the diffusion denoiser (discriminator-based,
  starting partway through training) — result pending as of this update.
- Statistical rigor: multi-seed + paired Wilcoxon infra built (infra/multi_seed.py) but not yet
  applied at scale — single-seed comparisons so far should be treated as preliminary, not final,
  results for anything going into a paper.

### 1.3 Expected Results
- Training speedup: 1.5-3x fewer epochs for same quality (QMC methods typically show ~2x speedup over Monte Carlo in related work)
- Synthetic diversity: higher coverage of fraud patterns (measurable via FID-like metrics)
- Anomaly detection: lower FPR at same TPR (measurable via partial AUC)

### 1.4 Datasets
- IEEE-CIS (primary, real e-commerce, 590K transactions)
- Elliptic Bitcoin (temporal graph, 203K nodes)
- PaySim (synthetic baseline, 6.3M transactions)
- T-Finance (large-scale, 39K nodes, 21M edges)

---

## Research Track 2: Diffusion Forcing for Temporal Graph Fraud Detection

### 2.1 Core Insight
Diffusion forcing (Chen & Goldstein, MIT 2024): different noise levels for different timesteps.
In fraud context: old transactions = more noise (less certain), recent = less noise (more important).
This is a NATURAL fit for temporal fraud graphs — no one has done this.

### 2.2 Architecture
```
Transaction history:
t=1: Cafe $5        ── high noise (30 days ago)
t=2: Taxi $15       ── medium noise (15 days ago)
t=3: Cafe $8        ── low noise (yesterday)
t=4: ???            ── predict (now)

Diffusion Forcing Temporal GNN:
1. Inject variable noise into node embeddings at each timestep
2. Temporal GNN propagates through noisy graph snapshots
3. Denoise jointly across all timesteps
4. Predict next transaction + anomaly score
5. High reconstruction error at t=4 → fraud alert
```

### 2.3 Why This is Novel
- SDG (arXiv Jan 2026): diffusion for temporal link prediction ✓ but NOT fraud
- DiffSTG: diffusion for spatio-temporal graphs ✓ but NOT fraud
- Diffusion forcing: variable noise ✓ but NOT graphs
- **Diffusion forcing + temporal GNN + fraud = ZERO papers**

---

## Research Track 3: Adversarial Robustness for Graph Fraud Models

### 3.1 Core Insight
Fraudsters actively attack detection models. GNN-based fraud detection is vulnerable to:
- Camouflage: create legitimate connections to hide fraud signal
- Injection: fake accounts pollute graph structure
- Temporal evasion: spread fraud over time to avoid velocity triggers

### 3.2 Attack Taxonomy for Financial Graphs
```
A. Structural attacks:
   - Add edges to legitimate accounts (camouflage)
   - Create fake account clusters (injection)
   - Remove suspicious edges (evasion)

B. Feature attacks:
   - Slightly modify transaction amounts ($100.00 → $99.97)
   - Change transaction timing to avoid patterns
   - Spoof device/IP features

C. Temporal attacks:
   - Space out fraudulent transactions over weeks
   - Interleave fraud with legitimate transactions
   - Mimic legitimate temporal patterns

D. Adaptive attacks:
   - Attacker knows the GNN architecture
   - Gradient-based attacks on graph structure
   - Reinforcement learning attacker
```

### 3.3 Defense: Diffusion-Based Graph Purification
```
Attacked graph (with adversarial edges/features)
→ Add noise (diffusion forward process)
→ Denoise (diffusion reverse process)
→ Purified graph (adversarial perturbations removed)
→ GNN fraud detection on clean graph
```

DiffScene (AAAI 2025) applies diffusion to safety-critical scenarios in autonomous driving — same principle applied here to financial graphs.

---

## Research Track 4: AI Safety via Statistical Physics

### 4.1 Core Insight
Predicting neural network behavior from weights alone, without running the model.
This IS AI Safety: if we can predict what a model will do without running it, we can verify safety.

### 4.2 Extending to deeper and richer architectures
```
Current baseline:
  Random MLP → predict mean activations → QMC + analytical methods

Next steps:
  A. Deeper networks (Phase 2: 64+ layers)
  B. Non-ReLU activations (GELU, SiLU — used in LLMs)
  C. Attention layers (Transformers, not just MLPs)
  D. Convolutional layers (Vision models)
  E. Recurrent connections (temporal models)
```

### 4.3 Random Screens → Transformer Interpretability
```
Rytov: wave propagation through random screens
  → predict average field at output

Transformer: input propagation through attention + FFN layers
  → predict average activation patterns from weights

Key insight: each transformer layer = one "random screen"
  Attention = anisotropic scattering (direction-dependent)
  FFN = isotropic scattering (ReLU truncation)
  LayerNorm = renormalization
```

### 4.4 Potential Breakthroughs
- Predict transformer behavior from weights → interpretability without probing
- Identify dangerous weight configurations → safety verification
- Quantify model uncertainty from architecture → better calibration

---

## Research Track 5: Quantum + AI (Long-term, 3-5 years)

### 5.1 Quantum Chemistry via GNN
```
Current baseline:
  GNN predicts thermal conductivity of graphene/BN composites
  → Classical simulation

Future:
  Quantum computer simulates molecular interactions
  → GNN learns from quantum data
  → Hybrid quantum-classical materials discovery
```

### 5.2 Quantum Error Correction via ML
```
Statistical physics: noise propagation in random media
Related problem: predict activations despite noisy layers
Quantum: predict qubit states despite decoherence

Same problem: signal recovery in noisy system
Same tools: statistical physics + ML
```

### 5.3 Quantum-Enhanced Fraud Detection
```
Transaction graph → quantum walk on graph
→ quantum interference highlights anomalies
→ exponentially faster than classical random walk
(theoretical, needs 1000+ logical qubits)
```

---

## Tools & Infrastructure

### Compute
- RunPod Serverless: experiments ($0.39-2.69/hr)
- Vast.ai: large ablation runs ($0.33-2.00/hr)
- Budget: $50-100/paper

### Frameworks
- PyTorch + PyTorch Geometric (GNN)
- tab-ddpm (tabular diffusion)
- diffusers (standard diffusion)
- PennyLane (quantum, future)
- wandb (tracking)

### Datasets
- IEEE-CIS, Elliptic, PaySim, T-Finance, DGraph
- Composites simulation data

### Agent Infrastructure
```
CPU Pod ($0.10/hr, 24/7)
├── Research agent loop
├── Hypothesis generation (Claude API)
├── Code generation + patching
├── Experiment dispatch → GPU Serverless
├── Results analysis + wandb
└── Paper drafting assistance
```
