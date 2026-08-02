# Lab Journal — Diffusion-Augmented Graph Fraud Detection

Append-only log of every experiment run: dataset/split, model/config, compute, results,
observations, next steps. `training/train_gnn.py` appends a run section automatically after
each training run — observations/next-steps lines are filled in manually afterward.

## Planned Next Experiments (updated 2026-07-21 after Run 50's GraphSAGEDiff signal)
Status:
- **Elliptic, best single architecture lead**: GraphSAGEDiff (explicit self-vs-neighborhood
  deviation feature) — multi-seed mean F1=0.7483 vs plain GraphSAGE's 0.7335 (Run 50), p=0.125
  (not yet conventionally significant, n=5, but the strongest combined signal — mean+variance+
  consistent direction — of any comparison this session). Combined with the best diffusion recipe:
  result pending.
- **Elliptic, best plain-diffusion single-seed result**: alpha=0.75/n_synthetic=1731/
  ddim_steps=100, F1=0.7749 (Run 40) — but Run 45's multi-seed check revealed this exact recipe's
  TRUE mean across 5 fresh seeds is only ~0.754, meaningfully below the single seed=42 number.
  **Treat every single-seed "best result" this session as a provisional lead, not a settled
  number, until multi-seeded.**
- **Elliptic, honest no-diffusion baseline**: single-seed F1=0.7453 (Run 38); multi-seed mean
  0.7335 (Run 50, seeds 0-4 — different seed set than Run 38's seed=42, consistent with the
  "single seeds vary" lesson above).
- **Elliptic, Random Forest on raw features (no graph)**: mean F1=0.7890, std=0.0025 across 5
  seeds (Run 47) — REMAINS the number to beat, and is now the most rigorously confirmed result in
  the whole project (RF's own ensembling makes it far more stable than any single GNN run).
- **GAT on Elliptic: confirmed negative, twice** (Runs 46/48, F1=0.674 and 0.723 respectively,
  even after fairer hyperparameters) — early convergence to a poor optimum both times. H100/more
  compute would NOT fix this (it's an optimization/convergence issue on a graph that already
  trains fine, not a scale issue, per 2026-07-21 discussion).
- **Elliptic homophily analysis** (Run 46): fraud-touching edges are 41% homophilic vs an ~11.6%
  random-chance baseline — real fraud-clustering signal exists (~3.5x enrichment) but is a
  minority of a typical fraud node's edges. This is why naive mean-aggregation dilutes the signal,
  and motivates both GraphSAGEDiff (tried) and neighbor-selection architectures (not yet tried).
- **PaySim**: focal-alpha sweep done through 0.95 (Run 28, F1=0.946). Leakage fix has ~no
  measurable effect (Run 41, consistent with Elliptic). Cosine scheduler consistently HURTS at
  both alpha levels tested (Run 41) — not recommended. RF baseline on PaySim's full graph still
  not run (only the small local MVP fixture available locally, not the ~2.77M-row full graph).
- Third dataset (IEEE-CIS) not yet started. Benchmark-validity warning (CLAUDE.md) flags
  EvolveGCN/CNN-GNN-LSTM/ChronoWave-GNN/MDST-GNN as needing their own protocol verified.

Next (in order):
1. GraphSAGEDiff + best diffusion recipe combined test — dispatched, result pending.
2. **Feature enrichment idea** (per user, 2026-07-21, explicitly queued for "after all current
   experiments"): compute explicit structural/invariant features (degree, centrality, clustering
   coefficient, possibly spectral coordinates) as ENGINEERED INPUT columns available to BOTH RF
   and the GNN, rather than relying on message-passing to discover graph-derived signal
   implicitly. This generalizes GraphSAGEDiff's own idea (an explicit self-vs-neighbor feature)
   and gives a cleaner test of whether it's specifically *aggregation* that's the bottleneck or
   graph-derived signal in general — if RF-with-structural-features also improves, that's evidence
   for the latter; if only the GNN improves, that's evidence for the former.
3. Feature-ablation experiment (still not done): train RF/GNN on ONLY Elliptic's 93 "local"
   feature columns (excluding the ~72 pre-aggregated ones) to test whether those pre-aggregated
   columns already capture most of the graph-derived signal our own GNN would otherwise add.
4. If GraphSAGEDiff (with or without the feature enrichment above) still doesn't close the RF gap,
   the next real architecture lever is a proper neighbor-selection mechanism (CARE-GNN/PC-GNN-
   style) rather than more tuning of what's already been tried.
5. Proper 10-seed (or more) consolidation pass across ALL the leads found this session
   (GraphSAGEDiff, template-attach, best diffusion recipe, adversarial/spectral losses) — several
   comparisons so far used only 5 seeds and landed short of conventional significance; a larger
   pass would settle which leads are real before writing any of this up.
6. PaySim: run the RF baseline on the full graph (not just the small local fixture); re-check
   diffusion+alpha under the corrected pipeline (only the non-diffusion alpha sweep has been
   re-measured so far).
7. Lower priority: PaySim's alpha ceiling (>0.95); classical baselines (oversampling x10/x20,
   SMOTE); QMC-enhanced TabDDPM (Research Track 1, RESEARCH_ROADMAP.md).
8. **Add a third dataset, prioritizing real data** (per user: PaySim is synthetic, real evidence
   matters more) — IEEE-CIS, deprioritized earlier only because it needs graph construction from
   scratch, unlike Elliptic. Do this once the Elliptic architecture-search arc above concludes.

## [2026-07-20] Run 1 — mvp_graphsage_paysim (SMOKE TEST, NOT REAL DATA)
- Dataset / split: **synthetic fixture** shaped like PaySim (20K random rows, hand-generated, NOT the
  real Kaggle dataset — no kaggle.json configured yet), subsample=150000, temporal 70%/15%/15%, config=configs/paysim.yaml
- Graph: 9938 nodes, 65510 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.3, focalLoss(alpha=0.25, gamma=2.0), lr=0.001, stopped_epoch=2
- Compute: mps (local)
- Val results: F1-macro=0.4966, AUC-ROC=0.4063, AUPRC=0.0230, G-mean=0.0000
- Test results: F1-macro=0.4971, AUC-ROC=0.4377, AUPRC=0.0111, G-mean=0.0000
- Observations: Purpose was purely mechanical — confirm preprocess → graph → train → eval → journal
  runs without errors. Metrics are near-random (G-mean=0, AUC~0.4) as expected on random fixture data
  with no real fraud signal. **These numbers are not meaningful and should not be compared to any
  benchmark in CLAUDE.md.**
- Next: Get real Kaggle credentials (~/.kaggle/kaggle.json) → `python -m data.download` → rerun
  preprocess + train on real PaySim → this will be the first real Run entry.

## [2026-07-20] Run 2 — mvp_graphsage_paysim — FIRST REAL DATA, BUGGY EARLY STOPPING
- Dataset / split: real PaySim (TRANSFER+CASH_OUT, subsample=150000, 8213/150000=5.48% fraud kept),
  temporal 70%/15%/15%, config=configs/paysim.yaml
- Graph: 150000 nodes, 81150 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.3, focalLoss(alpha=0.25, gamma=2.0), lr=0.001, stopped_epoch=3
- Compute: mps
- Val results: F1-macro=0.4936, AUC-ROC=0.1776, AUPRC=0.0142, G-mean=0.0000
- Test results: F1-macro=0.4515, AUC-ROC=0.1746, AUPRC=0.1042, G-mean=0.0000
- Observations: AUC well below 0.5 (worse than random) + G-mean=0 flagged a real bug. Diagnosed:
  early stopping / best-checkpoint selection monitored val F1-macro at a fixed 0.5 threshold, which
  stays flat (~0.49, all-negative predictions) for ~25 epochs while the model is still learning to
  rank fraud higher (val AUC climbs 0.16→0.78 over those same epochs) — F1@0.5 is a noisy, late-firing
  signal under 5.5% base rate, so early stopping (patience=15) triggered on epoch-3 noise, long before
  the model had actually converged. Feature sanity check (amount/is_transfer/hour means, fraud vs
  legit) was consistent between train and test, ruling out a data/graph/leakage bug.
- Next: switch the monitored metric to val AUC-ROC (threshold-independent, smoother signal) → Run 3.

## [2026-07-20] Run 3 — mvp_graphsage_paysim — FIXED, FIRST GOOD REAL RESULT
- Dataset / split: real PaySim (TRANSFER+CASH_OUT, subsample=150000, 5.48% fraud), temporal
  70%/15%/15%, config=configs/paysim.yaml
- Graph: 150000 nodes, 81150 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.3, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=200 (hit cap, never early-stopped — val AUC still rising at ep200), checkpoint/
  early-stop now monitors val AUC-ROC instead of F1-macro (see Run 2)
- Compute: mps (local, subsample — not comparable 1:1 to full-dataset CLAUDE.md benchmarks)
- Val results: F1-macro=0.8842, AUC-ROC=0.9846, AUPRC=0.8953, G-mean=0.7939
- Test results: F1-macro=0.8783, AUC-ROC=0.9824, AUPRC=0.9580, G-mean=0.8103
- Observations: Confirms the pipeline (graph construction, features, GraphSAGE, focal loss) is sound
  — test F1-macro=0.878 on a 150K subsample is already close to the HOT-GNN SOTA benchmark (0.890 F1
  on full PaySim) in CLAUDE.md, though not a fair comparison yet (subsample vs full 6.3M dataset).
  Metric was still improving at the epoch cap — model likely under-trained, not over-trained.
- Next: move training off local CPU/MPS onto RunPod (RTX 4090 Pod) for (a) full PaySim (not just
  150K subsample) and (b) enough epochs to let val AUC actually plateau before stopping. This is the
  last local-subsample run for the MVP slice; subsequent runs should log Compute as the RunPod GPU.

## [2026-07-20] Run 4 — full_graphsage_paysim_runpod_4090
- Dataset / split: real PaySim (TRANSFER+CASH_OUT, subsample_size=5000000 i.e. no downsampling —
  all 2,770,409 rows kept, 8213 fraud = 0.30% rate), temporal 70%/15%/15%, config=configs/paysim_full.yaml
- Graph: 2,770,409 nodes, 28,310,700 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.3, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=300 (hit cap again, never early-stopped — val AUC still rising at ep300)
- Compute: RunPod Pod, RTX 4090 (Secure Cloud, $0.69/hr), pod id 2hkmlwlbw1haqq
- Val results: F1-macro=0.7783, AUC-ROC=0.8577, AUPRC=0.4736, G-mean=0.6324
- Test results: F1-macro=0.8033, AUC-ROC=0.8649, AUPRC=0.5544, G-mean=0.6659
- Observations: First full-dataset (not subsample) real result. Test F1 (0.803) and AUC (0.865) are
  lower than the 150K-subsample Run 3 (F1=0.878, AUC=0.982) — expected, since the true fraud rate
  here (0.30%) is ~18x sparser than the artificially-boosted subsample (5.48%), a much harder
  imbalance to learn from with the same 2-layer GraphSAGE/focal-loss setup. AUC was still climbing
  linearly at epoch 300 (never plateaued, patience=30 never triggered) — this run is capped by
  epoch budget, not by convergence or overfitting. Not yet comparable to the HOT-GNN 0.890 F1
  benchmark in CLAUDE.md (that number is presumably on the same full/near-full dataset with a more
  sophisticated heterophily/temporal-aware architecture — ours is a plain baseline).
- Next: (a) let a future run train for more epochs / adjust LR schedule so val AUC actually plateaus
  before comparing to benchmarks properly; (b) this used a RunPod Pod — validate the serverless
  Docker build (serverless/Dockerfile) next, since we're moving off Pods once serverless works;
  (c) eventually add GAT/GCN baselines and start the diffusion-augmentation slice per CLAUDE.md.

## [2026-07-20] Run 5 — full_graphsage_paysim_runpod_4090 — DO NOT COMPARE TO BENCHMARKS (leaky feature)
- Dataset / split: same as Run 4 (real PaySim, full 2,770,409 rows, 0.30% fraud), temporal
  70%/15%/15%, config=configs/paysim_full.yaml. Only change: added an `is_exact_drain` node feature
  (`|oldbalanceOrg - amount| < 0.01 AND |newbalanceOrig| < 0.01 AND oldbalanceOrg > 0`) to
  data/paysim_preprocess.py, intended to help the model learn PaySim's origin-account-drained
  signature explicitly rather than only from continuous delta features.
- Graph: 2,770,409 nodes, 28,310,700 directed edges, max_node_degree=20 (13 node features now, was 12)
- Model / config: graphsage 2-layer, hidden=128, dropout=0.3, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=1000 (hit cap, val AUC saturated near-perfectly well before this)
- Compute: RunPod Pod, RTX 4090 (Secure Cloud, $0.69/hr), pod id 2hkmlwlbw1haqq
- Val results: F1-macro=0.9991, AUC-ROC=1.0000, AUPRC=0.9993, G-mean=0.9982
- Test results: F1-macro=0.9982, AUC-ROC=0.9965, AUPRC=0.9950, G-mean=0.9964
- Observations: **This is not a genuine modeling improvement — it's exploiting a known PaySim
  synthetic-data-generation artifact.** Checked directly: `is_exact_drain` has 0.987 correlation
  with `isFraud` on the full filtered dataset; crosstab shows it is TRUE for 8008/8213 fraud rows
  (97.5% recall) and for ZERO legit rows (100% precision) — i.e. this single boolean feature is
  almost a direct proxy for the label in PaySim, not a learned pattern. This is presumably also why
  the plain Decision Tree baseline in CLAUDE.md already gets AUC=0.985 — a tree can isolate this
  exact threshold rule trivially. Giving the GNN this feature directly just lets it memorize the
  same shortcut; the near-perfect F1/AUC here says nothing about whether GraphSAGE (or a future
  diffusion-augmented version) generalizes, and must NOT be reported against HOT-GNN/GNN-XAI/etc.
  Run 4 (F1=0.803, AUC=0.865, without this feature) remains the honest baseline for comparison.
  This is a well-documented limitation of PaySim as a fraud-detection benchmark, not specific to
  our pipeline.
- Next: (a) keep `is_exact_drain` out of the "fair benchmark" config path (or keep as a clearly
  labeled ablation, e.g. configs/paysim_full_with_leak.yaml, never the default) so future runs don't
  silently inherit this shortcut; (b) go back to Run 4's feature set as the baseline to improve on
  honestly; (c) if IEEE-CIS/Elliptic are added later, check them for similar near-deterministic
  artifacts before trusting any near-100% result on those either.

## [2026-07-20] Run 6 — full_graphsage_paysim_runpod_4090 — HONEST BASELINE
- Dataset / split: PaySim (TRANSFER+CASH_OUT, subsample=5000000, 0.30% fraud), temporal
  70%/15%/15%, config=configs/paysim_full.yaml. is_exact_drain removed (see Run 5); 12 honest
  node features. dropout raised 0.3 -> 0.5 (standard GraphSAGE/GAT regularization value).
- Graph: 2,770,409 nodes, 28,310,700 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=1000 (hit cap, val AUC still inching up at ep1000)
- Compute: RunPod Pod, RTX 4090 (Secure Cloud, $0.69/hr), pod id 2hkmlwlbw1haqq, tmux session
- Val results: F1-macro=0.8569, AUC-ROC=0.9763, AUPRC=0.7179, G-mean=0.7459
- Test results: F1-macro=0.8634, AUC-ROC=0.9653, AUPRC=0.7938, G-mean=0.7574
- Observations: First HONEST full-dataset result (no leaky feature). Test F1=0.863, AUC=0.965 —
  a real, large improvement over Run 4 (F1=0.803, AUC=0.865), from more epochs (1000 vs 300)
  reaching much further into convergence plus the higher dropout. This now beats the Decision
  Tree (0.802 F1) and GNN-XAI (0.857 F1) baselines in CLAUDE.md, and is closing in on HOT-GNN
  (0.890 F1) — with a plain 2-layer GraphSAGE, no diffusion augmentation yet. Val AUC was still
  creeping up at epoch 1000 (0.9758->0.9763 from ep990->1000) — likely near its ceiling for this
  architecture but not fully flat.
- Next: (a) GAT baseline running next in the same tmux session for comparison; (b) this is now
  the honest baseline to beat with diffusion augmentation; (c) add EMA of model weights to
  stabilize the noisy full-batch training signal; (d) HOT-GNN has no public code (confirmed) —
  use CARE-GNN or PC-GNN (both have open code, camouflage/heterophily-aware like HOT-GNN) as the
  "proper architecture" step instead of reimplementing HOT-GNN blind from the paper.

## [2026-07-20] Run 7 — mvp_graphsage_paysim — EMA DEBUG (BROKEN: no bias correction)
- Dataset / split: same 150K PaySim subsample as Run 3, config=configs/paysim.yaml. Local sanity
  test after adding EMA (naive version: shadow initialized to the model's random init weights,
  updated every step from step 1, no bias correction).
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, ema_decay=0.98, stopped_epoch=1 (!)
- Compute: mps (local sanity check, not a real experiment)
- Val results: F1-macro=0.1987, AUC-ROC=0.2702 (worse than random), G-mean=0.3437
- Observations: Early stopping fired after 20 epochs of "no improvement" with best_epoch=1 — a
  clear bug signal. Root cause: with decay=0.98, after 20 steps the shadow is still ~0.98^20≈67%
  the original random init, so the EMA "model" being evaluated is mostly untrained noise for a
  long time — its accidental epoch-1 AUC just happened to look (barely) better than later noisy
  epochs, well before patience=20 gave training any real chance to matter.
- Next: add Adam-style bias correction (divide by 1 - decay^t) — but see Run 8, got this wrong too.

## [2026-07-20] Run 8 — mvp_graphsage_paysim — EMA DEBUG (BROKEN: bias correction on wrong init)
- Same setup as Run 7, with bias correction added: state_dict() divides the shadow by
  (1 - decay^num_updates), Adam-style.
- Val results: F1-macro=0.3111, AUC-ROC=0.3321, still early-stopping at best_epoch=1.
- Observations: Bias correction alone didn't fix it — because the shadow was still initialized to
  the model's random init (not zero). Adam's 1/(1-decay^t) correction assumes the accumulator
  started at zero; applying it to a shadow that started at the (nonzero) initial weights just
  amplifies that arbitrary initial state instead of removing its influence. Two different fixes
  exist: (a) start shadow at zero + this correction (textbook Adam-style), or (b) skip correction
  entirely and instead delay EMA until a warm-up period has passed, initializing the shadow from
  the model's already-partially-trained weights at that point. Went with (b) — simpler, no
  bias-correction math to get wrong, and matches the intuition that pre-warm-up weights are noise
  not worth averaging in at all.
- Next: implement warm-up-start EMA (ema_start_epoch) → Run 9.

## [2026-07-20] Run 9 — mvp_graphsage_paysim — EMA FIXED (warm-up start)
- Same 150K subsample, config=configs/paysim.yaml. EMA now created at ema_start_epoch=20,
  snapshotting the model's actual (already-trained-for-20-epochs) weights as the shadow's
  starting point, then averaging normally from there — no bias correction needed.
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, ema_decay=0.98,
  ema_start_epoch=20, stopped_epoch=200 (hit cap, never early-stopped)
- Compute: mps
- Val results: F1-macro=0.8467, AUC-ROC=0.9698, AUPRC=0.8469, G-mean=0.7340
- Test results: F1-macro=0.8329, AUC-ROC=0.9674, AUPRC=0.9329, G-mean=0.7432
- Observations: Smooth, monotonically increasing val AUC from epoch 1 (0.16) to 200 (0.97) — the
  epoch-1-looks-best bug is gone. Confirms the EMA mechanism itself is now correct. Not a
  head-to-head vs Run 3 (F1=0.878 without EMA) — this run also has dropout=0.5 vs Run 3's 0.3, so
  two things changed at once; on this same 150K/200-epoch setup EMA doesn't obviously help or hurt
  here, but this was a mechanism check, not a proper ablation.
- Next: sync the fixed EMA code to the RunPod pod and use it in the full-dataset GraphSAGE/GAT
  runs (configs/paysim_full*.yaml, ema_start_epoch=30) — that's the real test of whether EMA helps
  on the harder full (0.30% fraud) setting.

## [2026-07-20] Run 10 — full_graphsage_paysim_runpod_4090 — EMA decay=0.999 TOO SLOW, DO NOT USE
- Dataset / split: PaySim full (2,770,409 rows, 0.30% fraud), temporal 70/15/15, config=configs/paysim_full.yaml
- Graph: 2,770,409 nodes, 28,310,700 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, ema_decay=0.999, ema_start_epoch=30,
  stopped_epoch=1000 (hit cap)
- Compute: RunPod Pod, RTX 4090 (Secure Cloud, $0.69/hr), pod id 2hkmlwlbw1haqq, tmux session
- Val results: F1-macro=0.4997, AUC-ROC=0.7963, AUPRC=0.1401, G-mean=0.0000
- Test results: F1-macro=0.4976, AUC-ROC=0.8262, AUPRC=0.3062, G-mean=0.0000
- Observations: **Confirms the decay=0.999 concern flagged mid-run — it's not just slower, it makes
  the final result substantially WORSE than no EMA at all.** G-mean=0 means the EMA-averaged model
  never confidently crosses the 0.5 threshold for fraud despite AUC=0.826 (moderate ranking) —
  test F1=0.498 is far below Run 6's honest no-EMA baseline (F1=0.863) and even below Run 4
  (F1=0.803). Root cause (calculated mid-run): with decay=0.999 and ema_start_epoch=30, even at
  epoch 1000 the epoch-30 snapshot still carries 0.999^970≈38% weight in the average — the shadow
  never converges close to the actually-trained model within this budget, so the "EMA" being
  evaluated and checkpointed is a poorly-calibrated blend, not a genuine smoothed-and-improved
  model. This is a config/calibration bug, not evidence that EMA itself doesn't work — see Run 9,
  where a shorter budget with better-matched decay=0.98 converged cleanly.
- Next: decay lowered to 0.98 for all full-scale configs (see configs/paysim_full*.yaml) before
  this was even finished — corrected re-runs of GraphSAGE and GAT dispatched in parallel on the new
  RunPod Serverless endpoint (2px0ahqve1orpm) rather than re-using the pod. Do not compare this
  Run 10 number against any benchmark; it measures a bug, not the architecture.

## [2026-07-21] Run 11 — full_graphsage_paysim (serverless) — EMA decay=0.98, BEST HONEST RESULT
- Dataset / split: PaySim full (2,770,409 rows, 0.30% fraud), temporal 70/15/15, config=configs/paysim_full.yaml
- Graph: 2,770,409 nodes, 28,310,700 directed edges, max_node_degree=20
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, ema_decay=0.98, ema_start_epoch=30,
  stopped_epoch=1000 (hit cap, no early stop)
- Compute: RunPod Serverless (endpoint 2px0ahqve1orpm, ADA_24/RTX 4090 class), job
  44677aaf-c567-46f6-be0f-e99d5cf941b3-e2, delayTime=27.3s, executionTime=624.5s (~10.4 min,
  download+preprocess+train all inside one worker invocation)
- Val results: F1-macro=0.8524, AUC-ROC=0.9734, AUPRC=0.7001, G-mean=0.7385
- Test results: F1-macro=0.8586, AUC-ROC=0.9618, AUPRC=0.7811, G-mean=0.7498
- Observations: Confirms serverless end-to-end (registry auth, private image pull, in-container
  Kaggle download, preprocess, train, EMA) works correctly and matches pod-quality results. Test
  F1=0.859 vs Run 6's honest no-EMA baseline (F1=0.863, AUC=0.965): essentially the same, EMA
  with a correctly-calibrated decay neither helped nor hurt materially here — unlike Run 10's
  broken decay=0.999 which actively wrecked the result (F1=0.498). This is now the reference
  honest GraphSAGE full-dataset number: F1=0.859, AUC=0.962. Beats Decision Tree (0.802 F1) and
  GNN-XAI (0.857 F1) from CLAUDE.md; still short of HOT-GNN (0.890 F1).
- Next: this is the baseline to beat with diffusion augmentation. GAT comparison failed (Run 12) —
  needs a real fix (mini-batch/neighbor sampling), not just further shrinking.

## [2026-07-21] Run 12 — full_gat_paysim (serverless) — FAILED, OOM even after shrinking
- Dataset / split: same full PaySim graph as Run 11, config=configs/paysim_full_gat.yaml
- Model / config: gat (GATv2Conv) 2-layer, hidden_dim=32, heads=4 (already reduced from the
  original 64/8 after the Pod's earlier 59GB OOM — see configs/paysim_full_gat.yaml comment),
  dropout=0.6, ema_decay=0.98
- Compute: RunPod Serverless, job 18a5e7af-3b49-4ce7-80be-c013ca6589d7-e2, delayTime=193.8s,
  executionTime=104.1s before crashing
- Result: **FAILED** — torch.OutOfMemoryError during the first forward pass (GATv2Conv edge
  attention on 28.3M edges): "Tried to allocate 14.82 GiB. GPU 0 has a total capacity of 23.64 GiB
  ... Process has 19.09 GiB memory in use." Traceback: train_gnn.py:144 -> models/gnn/gat.py:37 ->
  torch_geometric gatv2_conv.py edge_updater/edge_collect/_index_select.
- Observations: Even at 1/4 the original attention width (128 vs 512 effective concat dim), full-
  batch GAT still doesn't fit a 24GB GPU on this graph (28.3M directed edges) — per-edge-per-head
  attention tensors are the bottleneck, not model parameter count, and shrinking hidden_dim/heads
  further would likely make the model too weak to be a meaningful comparison point anyway. This
  confirms the concern raised earlier: the real fix is neighbor-sampled mini-batch training (PyG
  NeighborLoader) instead of full-batch, not continued shrinking. Deferred for now since we're
  still in the testing/exploration phase (per-user decision) — GraphSAGE (Run 11) is our working
  honest baseline in the meantime.
- Next: when GAT (or any larger-capacity model) is actually needed — e.g. for a fair HOT-GNN-style
  comparison, or once diffusion augmentation increases graph size further — implement mini-batch
  training via NeighborLoader rather than attempting further full-batch capacity cuts.

## [2026-07-21] Run 13 — full_gat_paysim (serverless, L40S-class) — FAILED, near-miss OOM + Blackwell trap
- Attempted full-batch GAT (heads=4/hidden=32, same as Run 12) on a bigger GPU instead of shrinking
  further. RunPod's "ADA_48_PRO" pool (48GB) is NOT Ada-only despite the name — it can schedule onto
  Blackwell RTX PRO 6000 MIG 2g.48gb slices (sm_120), which our image's CUDA 12.4 PyTorch (built for
  up to sm_90) can't run on at all: "no kernel image is available for execution on the device". Two
  wasted queue cycles landing on Blackwell before excluding it explicitly via gpu_ids=
  "ADA_48_PRO,-NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb,-NVIDIA RTX PRO 5000
  Blackwell,-NVIDIA RTX PRO 6000 Blackwell Server Edition" (the first exclusion attempt used the
  wrong exact GPU id — the MIG variant is a separate catalog entry from the non-MIG one).
- Once scheduled on a real 48GB Ada card: **still OOM'd** — "Tried to allocate 14.82 GiB. GPU 0 has
  a total capacity of 47.38 GiB of which 13.41 GiB is free. Process has 33.96 GiB memory in use" —
  needed ~48.8GB total vs 47.38GB available. Missed by only ~1.4GB.
- Observations: Confirms full-batch GAT attention doesn't fit this graph at hidden_dim=32/heads=4 on
  ANY GPU tier we have access to (24GB or 48GB) — it's not a "just get a bigger card" problem, the
  per-edge-per-head attention memory scales with the full 28.3M-edge graph regardless of GPU size.
  This is what motivated switching to mini-batch/neighbor-sampling for real (see Runs 14-17) instead
  of hunting bigger hardware.
- Next: implement NeighborLoader-based mini-batch training; document the ADA_48_PRO/Blackwell trap
  in infra/runpod_serverless.py so it isn't rediscovered the hard way again.

## [2026-07-21] Run 14 — gat_minibatch_smoke (serverless, 150K subsample) — SUCCESS (mechanism check)
- Dataset: small 150K PaySim subsample (config=configs/paysim_gat_smoke.yaml), purely to validate
  the NeighborLoader mini-batch code path before spending real money on the full graph.
- Model / config: gat, hidden_dim=64, heads=8 (FULL standard capacity — mini-batch no longer needs
  the shrink from Runs 12-13), mini_batch=true, batch_size=128, num_neighbors=[15,10], 10 epochs.
- First attempt failed immediately: `ImportError: 'NeighborSampler' requires either 'pyg-lib' or
  'torch-sparse'` — added pyg-lib to serverless/Dockerfile (pinned to the base image's exact
  torch 2.4.1+cu124 via PyG's wheel index, not on plain PyPI).
- Second attempt failed: `RuntimeError: Input should be contiguous` inside pyg_lib's index_sort
  (CSC construction for the sampler). Root cause (verified directly in a Python snippet): our
  edge_index construction (`.T` on a sorted-tuples array, then `np.concatenate` for the
  bidirectional edges) produces a non-contiguous array — harmless for full-batch training, fatal
  for pyg-lib's sampler. Fixed with `np.ascontiguousarray()` in data/paysim_preprocess.py.
- Third attempt: **SUCCESS**. Compute: RunPod Serverless, job 0fd3deb7-8220-4b5a-b3c1-14516ae55180-e1
  (a stale/warm worker briefly served a 404-ish FileNotFoundError for the config path on a retry
  in between — recycled after ~30s idle-timeout and worked on the next try).
  Val: F1-macro=0.539, AUC-ROC=0.953. Test: F1-macro=0.502, AUC-ROC=0.951, G-mean=0.225.
- Observations: Not a real result (only 10 epochs, ema_start_epoch=5 on a smoke config) — AUC~0.95
  already at 10 epochs just confirms the mini-batch mechanics (sampling, EMA, batched eval) are
  wired correctly end to end.
- Next: run a full-graph memory/timing check before committing to a full multi-epoch run.

## [2026-07-21] Run 15 — gat_minibatch_memcheck (serverless, full graph) — SUCCESS, no OOM
- Purpose-built 1-epoch, EMA-off check (config=configs/paysim_gat_memcheck.yaml) on the FULL
  2,770,409-node graph with FULL capacity (hidden_dim=64, heads=8), batch_size=1024,
  num_neighbors=[15,10] — specifically to verify batch_size=1024 (never tested above 128) doesn't
  OOM at full scale, since bumping it without checking would have been an unverified assumption.
- Compute: RunPod Serverless, executionTime=166.2s for 1 epoch (~1900 steps) — no OOM.
  Val: AUC-ROC=0.821, F1-macro=0.500. Test: AUC-ROC=0.845, F1-macro=0.498 (EMA disabled, meaningless
  beyond confirming the forward/backward pass runs).
- Observations: batch_size=1024 fits comfortably even at full hidden_dim=64/heads=8 — the mini-batch
  approach genuinely solves the memory problem, unlike shrinking the model (Runs 12-13). ~166s/epoch
  → a 40-epoch run should take roughly 1-2 hours if timing scales linearly.
- Next: launch the full 40-epoch run — but first had to fix the epoch/patience config, which was
  still calibrated for full-batch semantics (see Run 16).

## [2026-07-21] Run 16 — full_gat_paysim_minibatch (serverless) — CANCELLED, epoch count miscalibrated
- Launched the "full" run with the config left over from full-batch tuning: epochs=1000,
  batch_size=128. In mini-batch mode, one "epoch" = one full pass over ~1.94M train nodes in
  batches of `batch_size` — NOT one gradient step. At batch_size=128 that's ~15,158 steps/epoch,
  so epochs=1000 meant ~15 MILLION total steps, not the intended ~1M-ish.
- Caught within 28 seconds of execution and cancelled via the RunPod cancel API before meaningful
  cost was incurred. Recalibrated: batch_size 128->1024 (~1900 steps/epoch), epochs 1000->30,
  patience 50->5, ema_start_epoch 30->3.
- Observations: the ema_start_epoch=3 recalibration was itself immediately flagged (by the user) as
  too aggressive relative to a sensible amount of warm-up, and batch_size=1024 was at that point
  still unverified at full scale — both addressed before the real run (see Run 15 for the memcheck,
  and Run 17's config for the corrected ema_start_epoch=15/epochs=40/patience=8).
- Next: verify memory at batch_size=1024 (Run 15), then relaunch with corrected timing.

## [2026-07-21] Run 17 — full_gat_paysim_minibatch (serverless) — COMPLETED, BAD RESULT (EMA/patience
  interaction bug, not a capacity problem)
- Dataset / split: full PaySim graph (2,770,409 rows, 0.30% fraud), config=configs/paysim_full_gat.yaml
- Model / config: gat, hidden_dim=64, heads=8 (full standard capacity), mini_batch=true,
  batch_size=1024, num_neighbors=[15,10], epochs=40, patience=8, ema_decay=0.98, ema_start_epoch=15
- Compute: RunPod Serverless, job a1a1a3f2-bcbe-48a9-bd59-b26a079cedf1-e1, executionTime=1,266,825ms
  (~21.1 min total including preprocess) — much faster than expected, because early stopping fired
  early: best_epoch=7.
- Val: F1-macro=0.4997, AUC-ROC=0.8327, AUPRC=0.230, G-mean=0. Test: F1-macro=0.4976, AUC-ROC=0.8428,
  AUPRC=0.430, G-mean=0.
- Observations: **Root cause identified, not yet fixed.** `best_epoch=7` while `ema_start_epoch=15`
  means EMA never activated before patience (8 epochs without improvement past epoch 7 = stop at
  ~epoch 15) exhausted — training stopped almost exactly AT the epoch EMA was set to start, so every
  evaluated checkpoint this whole run was the raw (non-EMA) model. The raw model's AUC~0.84 at
  epoch 7 with F1/G-mean at the trivial-negative-prediction level matches the same "still confident-
  calibrating, hasn't crossed 0.5 threshold yet" pattern seen early in GraphSAGE's full-batch runs
  (e.g. Run 6 was still ~0.26 AUC around a comparable point in training) — this looks like genuine
  under-convergence, not a new bug, compounded by patience firing before real convergence had a
  chance to happen. Separately: per-batch class signal is much sparser here than full-batch — at
  batch_size=1024 out of 1.94M train nodes with 3643 total train-fraud, each batch sees only ~2
  fraud examples on average (vs all 3643 every single step in GraphSAGE's full-batch setup) — mini-
  batch SGD on this level of imbalance may genuinely need more steps/epochs to converge, not fewer.
- Next: (a) fix the patience/ema_start_epoch interaction (patience should not be able to fire before
  ema_start_epoch, or ema_start_epoch should be earlier relative to patience); (b) consider more
  epochs given the sparse per-batch fraud signal; (c) open decision point — keep iterating on GAT
  mini-batch tuning (cheap: ~21 min/attempt) vs. treat GraphSAGE (Run 11, F1=0.859) as the working
  baseline and move on to diffusion augmentation, which is the actual research goal per CLAUDE.md.

## [2026-07-21] Run 18 — full_gat_paysim_minibatch (serverless) — FIXED, big improvement
- Dataset / split: same full PaySim graph as Run 17, config=configs/paysim_full_gat.yaml
- Model / config: gat, hidden_dim=64, heads=8, mini_batch=true, batch_size=1024, num_neighbors=
  [15,10], epochs=40, patience=8, ema_decay=0.98, ema_start_epoch=15, **oversample_fraud_frac=0.15**
  (new — see build_oversampled_input_nodes() in training/train_gnn.py), plus a guard so early
  stopping can no longer fire before ema_start_epoch (both added in response to Run 17's diagnosis).
- Compute: RunPod Serverless, job 64958f89-21f6-4fc4-a800-a19bda6ca997-e1, executionTime=1,634,634ms
  (~27.2 min), best_epoch=9.
- Val: F1-macro=0.7125, AUC-ROC=0.8697, AUPRC=0.349, G-mean=0.581.
  Test: F1-macro=0.7789, AUC-ROC=0.8801, AUPRC=0.539, **G-mean=0.652** (vs 0 in Run 17).
- Observations: **Both fixes worked.** F1 jumped from 0.498 to 0.779, G-mean from 0 to 0.652 — the
  model is now confidently crossing the decision threshold instead of defaulting to all-negative
  predictions. Still below GraphSAGE's honest baseline (Run 11: F1=0.859, AUC=0.962), but this is
  now a legitimate, working GAT result rather than a broken one. The gap to GraphSAGE likely
  reflects (a) far more tuning iterations already spent on GraphSAGE's config vs GAT's first-ever
  working attempt, (b) inherent noise from neighbor-sampled mini-batch training vs GraphSAGE's exact
  full-batch gradient, and (c) the oversampling itself shifting the training distribution away from
  the true one (val/test stay honest, but the model trains on an artificially fraud-heavy mix) —
  none of these make GAT inherently worse for this task, just less mature here so far.
- Next: sweep oversample_fraud_frac (0.05, 0.30, 0.40) to see whether 0.15 was actually a good choice
  before spending more tuning effort elsewhere — dispatched as three parallel jobs using the new
  inline `config_dict` mechanism (see infra changes below), no image rebuild needed for the
  hyperparameter change itself.

## [2026-07-21] Infra: inline config dicts + git-pull code deployment (not an experiment run)
Two structural changes to remove recurring friction, prompted directly by how much time repeated
CI rebuilds were costing mid-session:
1. **serverless/handler.py now accepts `job_input["config_dict"]`** (a full config dict) as an
   alternative to `job_input["config"]` (a path baked into the image). `infra/runpod_serverless.
   sweep_config()` builds one by loading a base YAML and applying nested overrides client-side —
   new hyperparameter values no longer need a new committed YAML file + image rebuild.
2. **serverless/Dockerfile no longer `COPY . .`s the application code.** Only the base image, git,
   and Python deps (requirements.txt + pyg-lib) are baked in; `serverless/entrypoint.sh` clones (or
   pulls, if already present) the repo from GitHub at container start using `GITHUB_REPO_TOKEN`
   (reused from `gh auth token`, stored in Keychain). Code/config changes now take effect on the
   next worker spin-up with zero rebuild — only genuine requirements.txt changes still need one.
   CI workflow path triggers narrowed accordingly (Dockerfile/entrypoint.sh/requirements.txt only).
   Caveat: only re-syncs at container start, not mid-session for an already-warm worker.
- Currently dispatching new experiments (the oversample sweep above) against the SAME endpoint
  (2px0ahqve1orpm) — new workers spun up for those jobs get the fresh git-pull-based image since
  the existing 2 workers were busy with the low/high oversample jobs (not eligible for reuse).

## [2026-07-21] Incident: silent wrong-config run — oversample_fraud_frac=0.40 job, NOT a real result
- Dispatched the 0.40 oversampling point of the sweep via the new `config_dict` mechanism (job
  8ff9f365-c297-4e61-84cd-fb8733871cc7-e1, workerId 7mjspcg7h83q9u — the SAME worker that had just
  served the 0.05 job). Result looked like: best_epoch=200, executionTime=11.2s,
  test F1-macro=0.833, AUC-ROC=0.967 — numbers that exactly match Run 9 (the small 150K-subsample
  GraphSAGE MVP run), not a full-graph GAT run (which takes 20+ minutes and caps at epoch 40).
- Root cause: that worker was still warm from the 0.05 job, running an in-memory handler.py from
  BEFORE config_dict support was added to the image. entrypoint.sh's git pull only runs once at
  container boot, not per job — a warm/reused worker never re-syncs mid-session. The old
  handler.py doesn't recognize the "config_dict" key at all, so `job_input.get("config",
  "configs/paysim.yaml")` silently fell through to the small default MVP config and ran that
  instead — no error, no warning, a completely different (but superficially plausible-looking)
  result.
- Caught by: the result shape being inconsistent with what the requested experiment could produce
  (impossible epoch count, impossibly fast runtime) — not by any built-in safeguard. This was
  luck, not verification.
- Fix (immediate): print the full resolved config at the very start of run_from_config (see
  training/train_gnn.py) — a mismatch between what was requested and what's printed in the logs
  will now be visible in seconds instead of only inferable from suspicious final metrics.
- Fix (structural, still a real caveat): config_dict avoids rebuilds but is only as fresh as the
  worker that happens to serve the job — a warm worker from before the feature existed will
  silently ignore it. File-based `config` paths have the same staleness risk in principle (a
  worker whose git checkout predates a newly-added YAML file), just less likely to occur silently
  wrong versus visibly missing (FileNotFoundError, as actually happened once already in Run 14).
  No full fix implemented yet — current mitigation is the config print + treating early, cheap
  sanity-check dispatches as mandatory before trusting a new mechanism's first real result.
- Next: redispatch the actual 0.40 oversample experiment; treat its result as unverified until the
  logged config in a config-print check confirms it's really running oversample_fraud_frac=0.40
  on the full graph.

## [2026-07-21] oversample_fraud_frac sweep — results so far (0.05 / 0.15 / 0.30), full graph, GAT
All using the same fixed config (hidden_dim=64, heads=8, mini_batch, batch_size=1024,
num_neighbors=[15,10], epochs=40, patience=8, ema_decay=0.98, ema_start_epoch=15), varying only
oversample_fraud_frac. Test-set numbers:
| oversample_fraud_frac | F1-macro | AUC-ROC | G-mean | best_epoch |
|---|---|---|---|---|
| 0.05 | 0.7584 | 0.8753 | 0.6093 | 1 |
| 0.15 (Run 18) | 0.7789 | 0.8801 | 0.6521 | 9 |
| 0.30 | 0.7814 | 0.9169 | 0.7445 | 7 |
| 0.40 | pending — first two dispatch attempts failed (see incident below), redispatched |
Observations: monotonically improving with more oversampling so far, AUC especially (0.875 ->
0.880 -> 0.917) — 0.30 is currently the best point tried. Still all below GraphSAGE's honest
baseline (Run 11: F1=0.859, AUC=0.962). Worth trying even higher oversample fractions once 0.40
lands, though there's presumably a point where oversampling too aggressively hurts calibration on
the true (non-oversampled) val/test distribution.

## [2026-07-21] Incident: endpoint had a stuck/never-recycling worker, ignored 3 new files in a row
- After the 0.40-oversample config_dict incident (above), redispatched three NEW jobs as real
  committed YAML files instead (configs/paysim_full_gat_oversample_extreme.yaml,
  paysim_full_focal_alpha050.yaml, paysim_full_focal_alpha075.yaml) — expecting the "fail loud"
  property (FileNotFoundError) instead of the earlier silent-wrong-config failure mode.
- All three landed on the SAME worker (77h55hm5rgdtyx, the one that had just finished the 0.30
  job) and all three failed with FileNotFoundError for their respective new config files — that
  worker's git checkout never advanced past the commit before these files were added, despite
  idle_timeout=30 in the endpoint config. It kept picking up new queued jobs fast enough
  (RunPod's QUEUE_DELAY scaler saw low queue wait time) that it apparently never went idle long
  enough to recycle, and entrypoint.sh's git pull only runs once per container boot — so a
  worker that never restarts never re-syncs, no matter how many commits land in between.
- Fix applied: deleted the endpoint entirely (2px0ahqve1orpm) and created a fresh one
  (YOUR_ENDPOINT_ID) to force all-new workers with a guaranteed-current git checkout. Confirmed via
  GraphQL introspection that it's disabled on RunPod's API, so there's no query to list/inspect
  individual worker versions or force-recycle a specific one — endpoint delete+recreate is the
  only lever available found so far.
- Next: if this recurs, consider lowering idle_timeout further, or building an explicit
  version/commit-hash check into the config_summary output (already includes config content, could
  add `git rev-parse HEAD` at the top of entrypoint.sh as an env var) so a stale worker is
  detectable from the job output even when it DOESN'T hit a missing-file error (e.g. if it's only
  a few commits behind and everything it needs happens to already exist).

## [2026-07-20] Run 23 — SMOKE_TEST_diffusion_augmented_DELETE_ME — mechanism check only
- First end-to-end local test of the new TabDDPM pipeline (models/diffusion/tabddpm.py,
  training/train_diffusion.py, data/augment_graph.py): trained a tiny diffusion model (30 epochs,
  200 timesteps) on the 150K subsample's 3669 TRAIN-fraud node features, sampled 100 synthetic
  fraud nodes via DDIM (20 steps), attached each to 5 random real TRAIN nodes, then ran GraphSAGE
  for just 10 epochs on the augmented graph (150000 -> 150100 nodes).
- Val/Test results are meaningless here on purpose (10 epochs is nowhere near converged, same
  "still calibrating" pattern seen in every other under-trained run this session) — this run
  existed only to confirm the three new stages (diffusion train -> sample+augment -> GNN train)
  chain together without crashing, wiring/shape bugs, before spending real GPU time overnight.
  Diffusion training loss did fall cleanly (0.976 -> 0.367 over 30 epochs) and DDIM sampling
  produced finite, reasonably-scaled features (mean=0.101, std=1.115 vs the real standardized
  features' mean=0/std=1) — nothing points to a broken generator either.
- Next: full-scale overnight run — configs/paysim_full_diffusion_augmented.yaml, real epoch
  budgets for both the diffusion model and GraphSAGE, dispatched via serverless.

## [2026-07-21] Run 24 — full_graphsage_focal_alpha050 (serverless) — BEST RESULT SO FAR
- Dataset / split: full PaySim graph (2,770,409 rows, 0.30% fraud), config=
  configs/paysim_full_focal_alpha050.yaml — identical to Run 11 (honest GraphSAGE baseline)
  except focal loss alpha 0.25 -> 0.50.
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, focalLoss(alpha=0.50, gamma=2.0),
  lr=0.001, epochs=1000 (hit cap, best_epoch=1000 — still improving at the cap), ema_decay=0.98,
  ema_start_epoch=30
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), job
  9282c3d6-717b-43ec-bbe8-77f6699dd1ae-e2, executionTime=638.1s (~10.6 min)
- Val: F1-macro=0.8964, AUC-ROC=0.9927, AUPRC=0.8022, G-mean=0.8135
  Test: **F1-macro=0.9122, AUC-ROC=0.9870, AUPRC=0.8636, G-mean=0.8401**
- Verified genuine: `config_summary` in the returned output confirms config_label=
  configs/paysim_full_focal_alpha050.yaml, model_name=graphsage, epochs_cap=1000 — matches what
  was requested, not a repeat of the stale-worker incidents from earlier tonight.
- Observations: **Beats HOT-GNN (0.890 F1) from CLAUDE.md** — the current best result in this
  project, from a single-line change (focal loss alpha 0.25->0.50) on top of the existing honest
  GraphSAGE baseline. Still improving at epoch 1000 (never plateaued) — likely has more room left
  if run longer. Next obvious question: is 0.75 (Run 11's originally planned second sweep point,
  currently in flight) even better, or is 0.50 near-optimal with 0.75 overcorrecting?
- Next: check the in-flight alpha=0.75 and GAT oversample=0.40 results; if alpha=0.50 holds up as
  the new best, it becomes the baseline the diffusion-augmentation run should be compared against
  (though the diffusion run was dispatched using alpha=0.25 for direct comparability with Run 11 —
  worth rerunning at alpha=0.50 too once that's confirmed better).

## [2026-07-21] Run 25 — full_graphsage_focal_alpha075 (serverless) — NEW BEST, beats Run 24 too
- Dataset / split: full PaySim graph, config=configs/paysim_full_focal_alpha075.yaml — same as
  Run 24 except focal loss alpha 0.50 -> 0.75.
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, focalLoss(alpha=0.75, gamma=2.0),
  lr=0.001, epochs=1000 (hit cap, best_epoch=1000 — still improving), ema_decay=0.98,
  ema_start_epoch=30
- Compute: RunPod Serverless, job 18b84b35-5167-446b-9e41-7cbd43f49d91-e1, executionTime=728.9s
  (~12.1 min)
- Val: F1-macro=0.9118, AUC-ROC=0.9976, AUPRC=0.8474, G-mean=0.8453
  Test: **F1-macro=0.9291, AUC-ROC=0.9947, AUPRC=0.9079, G-mean=0.8716**
- Verified genuine: config_summary confirms config_label=configs/paysim_full_focal_alpha075.yaml,
  matches what was requested.
- Observations: **Beats Run 24 (alpha=0.50, F1=0.912) again** — the alpha->higher trend hasn't
  reversed yet (0.25 baseline F1=0.859 -> 0.50: F1=0.912 -> 0.75: F1=0.929). Both runs still
  improving at the epoch cap (never plateaued), so neither is fully converged — the true ceiling
  for either alpha value might be even higher than what's reported here. Worth trying alpha closer
  to 1.0 (or checking gamma too) to see where this actually plateaus/reverses, since focal loss
  alpha=1.0 would essentially just be uniform full weight on the positive class.
- Next: this (F1=0.929, AUC=0.995) is now the reference GraphSAGE result to beat — including for
  the in-flight diffusion-augmentation run, which used alpha=0.25 for comparability with the
  now-superseded Run 11. Consider a follow-up diffusion run on top of alpha=0.50 or 0.75 once this
  sweep concludes, to see if augmentation adds anything ON TOP of the better loss weighting, not
  just compared to the weaker original baseline.

## [2026-07-21] Overnight batch — 4 jobs completed, all verified via config_summary
Four jobs dispatched overnight (user asleep), all confirmed via the config_summary/
worker_git_commit fields in their returned output — no repeat of the stale-worker incidents from
earlier in the night. Summarizing all four here; each gets its own subsection below.

### Run 26 — full_gat_paysim_minibatch_oversample_extreme (GAT, oversample_fraud_frac=0.40)
- Config: configs/paysim_full_gat_oversample_extreme.yaml — same GAT mini-batch setup as Runs
  18/low/high (hidden_dim=64, heads=8, batch_size=1024, num_neighbors=[15,10], epochs=40,
  patience=8, ema_decay=0.98, ema_start_epoch=15), oversample_fraud_frac=0.40 (the extreme point
  of the sweep, redone as a real committed file after the earlier config_dict incident).
- Compute: job 68f817e0-e3de-4a17-8205-e3bcfbd13e7f-e2, executionTime=2,307,137ms (~38.5 min,
  much longer than the ~20-30 min of the 0.05/0.15/0.30 points — possibly queue/cold-start delay
  rather than actual compute time), best_epoch=12.
- Val: F1-macro=0.7202, AUC-ROC=0.9340, G-mean=0.6977.
  Test: F1-macro=0.8093, AUC-ROC=0.9354, G-mean=0.7388.
- **Completes the oversample_fraud_frac sweep for GAT:**
  | oversample_fraud_frac | Test F1 | Test AUC | Test G-mean |
  |---|---|---|---|
  | 0.05 | 0.758 | 0.875 | 0.609 |
  | 0.15 | 0.779 | 0.880 | 0.652 |
  | 0.30 | 0.781 | 0.917 | 0.745 |
  | 0.40 | 0.809 | 0.935 | 0.739 |
  Monotonically improving on F1/AUC across the whole sweep — 0.40 is the best GAT result so far,
  still below every GraphSAGE focal-alpha result in this batch. Untested whether >0.40 keeps
  helping or starts hurting calibration; not pursued further tonight in favor of the alpha sweep.

### Run 27 — full_graphsage_focal_alpha085
- Config: configs/paysim_full_focal_alpha085.yaml, alpha=0.85, otherwise identical to Runs 24/25.
- Compute: job a10220d4-475b-4dbc-a3bf-c59f663dbed4-e2, executionTime=737.6s (~12.3 min),
  best_epoch=1000 (hit cap again).
- Val: F1-macro=0.9205, AUC-ROC=0.9986, G-mean=0.8780.
  Test: **F1-macro=0.9366, AUC-ROC=0.9965, G-mean=0.8885.**
- Beats Run 25 (alpha=0.75, F1=0.929) again — trend still climbing.

### Run 28 — full_graphsage_focal_alpha095 — BEST RESULT OVERALL SO FAR
- Config: configs/paysim_full_focal_alpha095.yaml, alpha=0.95 (near the theoretical ceiling —
  alpha=1.0 would put zero loss weight on the negative/legit class).
- Compute: job 5c85dbc9-b089-4926-a749-0609eb0921e7-e1, executionTime=922.5s (~15.4 min),
  best_epoch=1000 (hit cap again — STILL never plateaued across the entire alpha sweep).
- Val: F1-macro=0.8936, AUC-ROC=0.9993, G-mean=0.9173 (note: val F1 dips slightly below Run 27's
  val F1 even though test F1 is higher — some val/test variance at this extreme alpha, worth
  keeping an eye on rather than over-reading any single split).
  Test: **F1-macro=0.9465, AUC-ROC=0.9979, G-mean=0.9217.**
- **Full alpha sweep, all on identical architecture/data, only loss.alpha varies:**
  | alpha | Test F1 | Test AUC | Test G-mean |
  |---|---|---|---|
  | 0.25 (Run 11, original baseline) | 0.859 | 0.962 | 0.750 |
  | 0.50 (Run 24) | 0.912 | 0.987 | 0.840 |
  | 0.75 (Run 25) | 0.929 | 0.995 | 0.872 |
  | 0.85 (Run 27) | 0.937 | 0.997 | 0.889 |
  | 0.95 (Run 28) | **0.946** | **0.998** | **0.922** |
  Monotonic improvement all the way to 0.95, never reversing, and EVERY run in this sweep hit the
  1000-epoch cap still improving (never early-stopped) — meaning none of these numbers are even
  fully converged yet. This single hyperparameter (already in the code, zero new engineering) has
  been by far the highest-leverage change of the whole project so far, beating HOT-GNN (0.890 F1)
  by a wide margin at every alpha point >= 0.75.
- Next: try alpha even closer to 1.0 (0.99?) to find where this actually stops improving or
  reverses; consider whether more epochs (raise the cap past 1000) would push these even higher
  given none have plateaued; re-run the diffusion augmentation (Run 29, below) and GAT sweep on
  top of alpha=0.85 or 0.95 instead of the now-far-outdated alpha=0.25, once a ceiling is found.

### Run 29 — full_graphsage_tabddpm_diffusion_augmented
- Config: configs/paysim_full_diffusion_augmented.yaml — TabDDPM (models/diffusion/tabddpm.py)
  trained on 3643 real TRAIN-fraud node features (1000 timesteps, hidden_dim=128, 1000 epochs),
  3643 synthetic fraud nodes sampled via DDIM (50 steps), each attached to 5 random real TRAIN
  nodes, added to the graph (2,770,409 -> 2,774,052 nodes), then GraphSAGE trained on the
  augmented graph with alpha=0.25 (Run 11's original value, chosen for direct comparability to
  the ORIGINAL baseline — this predates today's alpha sweep discovery).
- Compute: job 62b457b0-cf17-4f35-992e-b0d1786e4fad-e2, executionTime=829.1s (~13.8 min),
  best_epoch=1000. Required 3 dispatch attempts before succeeding — first two failed with
  FileNotFoundError on this exact config file because of the stale-worker issue (see the
  "endpoint had a stuck worker" incident above); this one succeeded after adding the per-job
  git-pull fix (worker_git_commit=39ef5dd, a real hash, confirming the fix worked).
- Val: F1-macro=0.8875, AUC-ROC=0.9882, G-mean=0.7977.
  Test: **F1-macro=0.8989, AUC-ROC=0.9831, G-mean=0.8169.**
- Observations: Beats Run 11 (the honest alpha=0.25 baseline this is actually comparable to:
  F1=0.859 -> F1=0.899, a genuine +0.040 F1 improvement from diffusion augmentation alone) —
  **this is the first real evidence the core research idea (diffusion-augmented fraud detection)
  does something.** However it's now well below the focal-alpha-only results discovered in
  parallel tonight (alpha=0.95 alone gets F1=0.946 with zero augmentation) — meaning at least on
  this dataset, simply reweighting the loss function so far outperforms the more complex diffusion
  pipeline. This doesn't mean diffusion augmentation is useless — it wasn't tested WITH the better
  alpha, and the two techniques address different things (loss weighting vs. actual training-data
  diversity) that could plausibly stack — but it's not yet a clean win over the simplest baseline
  fix available.
- Next: the real next step is diffusion augmentation ON TOP OF alpha=0.85-0.95 (once a ceiling is
  found), not alpha=0.25 — that's the comparison that actually tests whether TabDDPM adds anything
  once the easy win is already banked. Also worth checking: does the augmented graph's synthetic
  fraud actually look realistic (t-SNE/feature-distribution comparison vs real fraud), or is
  GraphSAGE just tolerating noise without the synthetic nodes actively helping?

## [2026-07-21] Run 26 — elliptic_graphsage (local) — first real Elliptic result (recovered after a bug ate the journal write)
- Dataset / split: Elliptic Bitcoin (real, downloaded from Kaggle, layout verified directly rather
  than assumed — see data/elliptic_preprocess.py), config=configs/elliptic_full.yaml. Temporal
  split by the dataset's own 49 steps: train<=34 (29,894 nodes, 3462 fraud, 11.6%), val 35-41
  (7,829 nodes, 675 fraud, 8.6%), test 42-49 (8,841 nodes, 408 fraud, 4.6%) — only the 46,564
  known-label (illicit/licit) nodes get masks; the ~77% unknown-label nodes stay in the graph for
  message passing only.
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=1000 (early-stopped at epoch 465, best_epoch=415), ema_decay=0.98,
  ema_start_epoch=30
- Compute: mps (local) — graph is small (203,769 nodes, 468,710 directed edges) so full-batch on
  a Mac is fine, unlike PaySim's full graph.
- Val: F1-macro=0.8374, AUC-ROC=0.9389, AUPRC=0.8049, G-mean=0.7395.
  Test: F1-macro=0.7381, AUC-ROC=0.8407, AUPRC=0.5064, G-mean=0.5811.
- Observations: Test noticeably lower than val (0.738 vs 0.837 F1) — expected given the fraud
  rate genuinely drops over time in this dataset (11.6% -> 8.6% -> 4.6%), a real distribution
  shift documented in the literature (a dark-market shutdown mid-dataset causes an illicit-
  activity spike followed by dropoff), not a bug. Against CLAUDE.md's Elliptic benchmarks —
  GCN~0.65 F1/~0.85 AUC, GraphSAGE~0.70 F1/~0.88 AUC, EvolveGCN~0.75 F1/~0.90 AUC — our plain
  (non-temporal-aware) GraphSAGE at F1=0.738/AUC=0.841 already beats the GraphSAGE reference point
  and is close to EvolveGCN's, despite EvolveGCN being purpose-built for this dataset's temporal
  structure and ours not using time information beyond the split itself.
- **This result was initially lost**: run_from_config crashed in append_journal_entry right after
  printing these metrics (see Run 27/28 below for the bug and fix) — recovering it here from the
  captured stdout rather than losing it, since the training itself completed successfully and
  the numbers are real.
- Next: dispatch the (now-fixed) pipeline on RunPod for a faster/repeatable run; check whether
  this dataset has any PaySim-style near-deterministic leakage before trusting it fully; consider
  whether the focal-alpha sweep that worked so well on PaySim transfers here too.

## [2026-07-21] Run 27/28 — elliptic_graphsage — BUGFIX VERIFICATION ONLY (5 and 3 epochs)
- First Elliptic RunPod dispatch (job 9bb42b52-a4f7-4c30-9e81-ae7aaba9421d-e2) FAILED with
  `KeyError: 'fraud_types'` inside append_journal_entry — it hardcoded PaySim-specific config
  fields (fraud_types, subsample_size, train_frac/val_frac/test_frac, max_node_degree) with no
  guard for other datasets. The SAME bug was actually hit by the earlier local Elliptic test run
  too (see the real Run 26 result below) — I just hadn't checked its exit code/journal write,
  only glanced at the printed Val/Test metrics and assumed success. Lesson: printed metrics
  appearing is not the same as the run actually completing cleanly.
- Fixed append_journal_entry to branch on `data.dataset` (paysim vs elliptic vs other) instead of
  assuming PaySim fields exist; fixed the same unguarded `config["data"]["subsample_size"]` access
  in config_summary (now `.get()`, None for non-PaySim datasets, plus an explicit `dataset` field).
- Runs 27 (5 epochs) and 28 (3 epochs) here are just mechanism checks confirming the fix — not
  real results (epochs far too low to mean anything, numbers not analyzed).
- Next: relaunch the real RunPod Elliptic run with the fix in place.

## [2026-07-21] Run 29 — elliptic_graphsage (serverless) — confirms Run 26 on real infra
- Dataset / split: same as Run 26 (Elliptic Bitcoin, temporal by step, config=configs/elliptic_full.yaml)
- Model / config: graphsage 2-layer, hidden=128, dropout=0.5, focalLoss(alpha=0.25, gamma=2.0),
  lr=0.001, epochs=1000 (early-stopped, best_epoch=344), ema_decay=0.98, ema_start_epoch=30
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), job
  fdcf3171-2a0e-4f66-9da9-fb686a7fc1bf-e1, executionTime=20.9s (small graph — 203,769 nodes,
  468,710 edges — full-batch is cheap even including download+preprocess)
- Val: F1-macro=0.8278, AUC-ROC=0.9378, AUPRC=0.7895, G-mean=0.7253.
  Test: F1-macro=0.7270, AUC-ROC=0.8367, AUPRC=0.5020, G-mean=0.5640.
- Verified genuine: config_summary confirms dataset=elliptic, config_label=configs/elliptic_full.yaml.
- Observations: Consistent with Run 26's local result (F1=0.738/AUC=0.841 vs this run's
  F1=0.727/AUC=0.837) — small run-to-run variance (different best_epoch: 344 here vs 415 locally,
  expected given MPS vs CUDA numerics aren't bit-identical even with the same seed), not a sign of
  a bug. Elliptic Bitcoin is now a second working real dataset alongside PaySim, with a sane
  first honest baseline in the same ballpark as CLAUDE.md's GraphSAGE reference (~0.70 F1) and
  approaching EvolveGCN's temporal-aware ~0.75 F1 despite not modeling time explicitly.
- Next: this is the honest Elliptic baseline going forward. Candidates: (a) try the focal-alpha
  sweep here too, given how much it helped on PaySim; (b) check for PaySim-style near-deterministic
  leakage before trusting any future too-good-to-be-true result; (c) eventually run the
  diffusion-augmentation pipeline on this dataset too, once the PaySim ablation ladder concludes.

## [2026-07-21] Run 30 — elliptic_graphsage_tabddpm_alpha095 — SMOKE TEST ONLY, not a real result
- Local mechanism check of TabDDPM diffusion augmentation on Elliptic's 165-dim features (much
  higher-dimensional than PaySim's 12) before committing to a real RunPod run: diffusion trained
  for only 20 epochs (vs the real config's 1000), 50 synthetic nodes (vs 3462), 5 GraphSAGE epochs.
  Val/Test F1/AUC here are meaningless — deliberately tiny budgets, not analyzed.
- Observation worth carrying forward: the sampled synthetic features had mean=2.02, std=96.0 —
  wildly off from the real standardized features' mean~0/std~1. With only 20/1000 diffusion
  epochs this is expected (the model hasn't learned to denoise properly at most noise levels
  yet), not necessarily a bug — but worth explicitly checking in the real (1000-epoch) run's
  logs too, since a diffusion model that hasn't converged would generate unrealistic synthetic
  fraud nodes that could hurt rather than help GraphSAGE.
- Next: dispatch the real run (configs/elliptic_diffusion_alpha095.yaml, full budgets) to RunPod;
  check the real run's sampled-feature mean/std against real fraud features before trusting the
  result, given this smoke test's red flag.

## [2026-07-21] Run 31 — full_graphsage_tabddpm_alpha095 (PaySim) — diffusion + best alpha combined
- Config: configs/paysim_diffusion_alpha095.yaml — Run 28's alpha=0.95 (best focal-alpha point,
  still climbing at the epoch cap there) combined with Run 29's TabDDPM diffusion augmentation
  (3643 synthetic fraud nodes via DDIM, 50 steps, k_connections=5), same architecture otherwise.
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), job
  df1fa45b-23f8-4a52-ae8f-0a711b4ad2d7-e1, executionTime=613.4s (~10.2 min), best_epoch=1000
  (hit cap again, consistent with every alpha>=0.75 run never plateauing).
- Val: F1-macro=0.8253, AUC-ROC=0.9994, G-mean=0.9434.
  Test: F1-macro=0.9392, AUC-ROC=0.9989, G-mean=0.9439.
- Observations: Essentially a wash against Run 28 (alpha=0.95 alone, no diffusion: Test
  F1=0.9465/AUC=0.9979/G-mean=0.9217) — F1 dips slightly (-0.0073, within run-to-run noise given
  neither run early-stopped), AUC ticks up marginally, G-mean improves more noticeably
  (0.9217->0.9439). Unlike Run 29 (diffusion at alpha=0.25, which gave a genuine +0.040 F1 win),
  diffusion adds nothing measurable once alpha=0.95 has already pushed PaySim close to its
  ceiling — there's little headroom left for a second class-imbalance intervention to move F1 on
  this dataset. See Run 32 below: the same combination behaves completely differently on Elliptic.
- Next: PaySim's alpha=0.95+diffusion combo is not worth pursuing further on its own; the real
  finding this session is the cross-dataset contrast with Run 32.

## [2026-07-21] Run 32 — elliptic_graphsage_tabddpm_alpha095 — diffusion + alpha=0.95 REGRESSES badly, root-caused
- Config: configs/elliptic_diffusion_alpha095.yaml — alpha=0.95 (PaySim's best point, transferred
  directly per 2026-07-21 discussion) + TabDDPM diffusion (3462 synthetic fraud nodes, ~doubling
  real train-fraud, DDIM 50 steps, k_connections=5), same GraphSAGE architecture as the Elliptic
  baseline (Run 26/29).
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), job
  f1fce62e-de71-4dd0-9564-dcae825f9024-e2 (first pass, aggregate metrics only), re-run as job
  (same config, worker_git_commit=8fcb20c) after adding confusion-matrix/per-class
  precision-recall instrumentation to evaluation/metrics.py — both passes agree on the aggregate
  numbers (Test F1=0.4446 vs 0.4443, noise-level difference), confirming the result is real, not
  a fluke.
- Val: F1-macro=0.5473, AUC-ROC=0.9067, AUPRC=0.5280, G-mean=0.7736.
  Test: F1-macro=0.4443, AUC-ROC=0.8224, AUPRC=0.3505, G-mean=0.7033.
  Test confusion matrix: TP=351, FP=3585, TN=4848, FN=57. precision_fraud=0.089,
  recall_fraud=0.860, specificity=0.575, predicted_positive_rate=44.5% (true fraud rate: 4.6%).
- **Root cause, confirmed by the confusion matrix (not just inferred from aggregates):** compare
  to the honest baseline (Run 26/29, alpha=0.25, no diffusion) re-run with the same
  instrumentation: TP=130, FP=16, TN=8417, FN=278, precision_fraud=0.890, recall_fraud=0.319,
  specificity=0.998, predicted_positive_rate=1.65%. Going from baseline to
  diffusion+alpha=0.95 makes the model flag **44.5% of all test nodes as fraud, a ~27x jump**
  against a true rate of 4.6%. Recall improves a lot (0.32->0.86, good) but precision collapses
  (0.890->0.089) because the false-positive *count* (3585) dwarfs the true-positive count (351)
  on a 95%-legit dataset — this is exactly why G-mean (a geometric mean of *rates*: sensitivity
  and specificity, both of which stay in a respectable range: 0.86 and 0.575) looks like an
  improvement while F1-macro (precision-sensitive, hurt directly by the false-positive count)
  craters. alpha=0.95 and diffusion augmentation are two overlapping corrections for the same
  problem (class imbalance) — stacked together they overshoot the decision threshold badly.
  This overshoot was tolerable on PaySim (Run 31) because that dataset's classes are already
  almost perfectly separable (baseline AUC ~0.96+); Elliptic's classes overlap far more
  (baseline AUC 0.837, baseline recall only 0.32 even without any extra push toward positive
  predictions), so the same overshoot in predicted-positive rate produces a much larger absolute
  flood of false positives. This is a genuine dataset-dependent finding, not a bug — the smoke
  test's flagged undertrained-diffusion-sample risk (Run 30, std=96 anomaly) does NOT appear to
  be the cause here, since the failure mode (threshold miscalibration via predicted_positive_rate)
  is fully explained by the alpha+diffusion stacking effect without needing bad synthetic samples
  as an additional explanation.
- Next: (a) do NOT blindly transfer alpha=0.95 across datasets — it needs per-dataset tuning,
  especially when stacked with diffusion augmentation; (b) try diffusion alone at Elliptic's own
  alpha=0.25 (isolate whether diffusion by itself helps here, the fair comparison to Run 29 on
  PaySim); (c) try a lower alpha (e.g. 0.5-0.75) combined with diffusion on Elliptic to see if a
  milder combination avoids the overshoot; (d) consider decoupling the eval threshold from 0.5
  (e.g. picking the val-optimal threshold) so extreme-alpha models aren't penalized purely for
  being uncalibrated at the default cutoff — F1@0.5 conflates "does it rank fraud well" (AUC/AUPRC
  say yes: 0.82/0.35, comparable to baseline's 0.84/0.50) with "is 0.5 the right cutoff" (clearly
  no, here).

## [2026-07-21] Run 33 — elliptic diffusion+alpha full sweep (0.25/0.40/0.50/0.60/0.75/0.85) — diffusion itself looks harmful, not an alpha problem
- Configs: configs/elliptic_diffusion_alpha{025,040,050,060,075,085}.yaml, identical to Run 32's
  alpha=0.95 config except loss.alpha — same TabDDPM diffusion (3462 synthetic fraud nodes via
  DDIM, 50 steps, k_connections=5), same GraphSAGE architecture. Dispatched per user request to
  find the optimal per-dataset alpha for diffusion on Elliptic, given Run 32 showed PaySim's best
  alpha (0.95) doesn't transfer.
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), 6 parallel jobs, all completed cleanly
  (worker_git_commit=cad0f07 on all).
- **Full sweep, Test split, vs. the honest no-diffusion baseline (Run 26/29, alpha=0.25, F1=0.727):**
  | alpha | Test F1 | Test AUC | precision_fraud | recall_fraud | predicted_positive_rate | best_epoch |
  |---|---|---|---|---|---|---|
  | baseline (no diffusion) | **0.727** | 0.837 | 0.890 | 0.319 | 1.65% | 346 |
  | 0.25 + diffusion | 0.488 | 0.545 | 0 | 0 | 0% | 1 (collapsed) |
  | 0.40 + diffusion | 0.539 | 0.720 | 0.342 | 0.061 | 0.83% | 23 |
  | 0.50 + diffusion | 0.564 | 0.694 | 0.156 | 0.194 | 5.71% | 19 |
  | 0.60 + diffusion | 0.566 | 0.833 | 0.146 | 0.591 | 18.7% | 1000 |
  | 0.75 + diffusion | 0.544 | 0.836 | 0.129 | 0.657 | 23.4% | 1000 |
  | 0.85 + diffusion | 0.516 | 0.823 | 0.115 | 0.735 | 29.6% | 1000 |
  | 0.95 + diffusion (Run 32) | 0.444 | 0.822 | 0.089 | 0.860 | 44.5% | 1000 |
- **Every single alpha point with diffusion is worse than the no-diffusion baseline.** Best
  diffusion point (alpha=0.60, F1=0.566) is still 0.16 F1 below just not using diffusion at all.
  This rules out "wrong alpha" as the explanation — the honest baseline already IS alpha=0.25, and
  adding diffusion at that exact same alpha collapses the model entirely (best_epoch=1, val_auc
  never improves past a near-random first-epoch value, ends up predicting 0% positive).
- Two distinct failure regimes visible in the sweep: (a) **low alpha (0.25-0.50) — unstable/
  collapsed training**, best_epoch stuck at 1/19/23 instead of hitting the cap, AUC well below
  baseline (0.55-0.72 vs 0.837) — the synthetic nodes seem to be actively disrupting optimization,
  not just adding noise the model tolerates; (b) **higher alpha (0.60-0.95) — stable training that
  converges to a bad decision boundary**, best_epoch=1000 every time (never plateaus, same pattern
  as PaySim's alpha sweep, but here that's a bad sign since it never reaches a good optimum either),
  precision_fraud monotonically collapsing (0.146->0.089) as recall_fraud climbs (0.591->0.860) —
  a smooth precision/recall trade-off along the sweep, but AUC plateaus around 0.82-0.84, i.e.
  *slightly below* the baseline's 0.837 at every single point, meaning the synthetic nodes aren't
  just shifting the threshold, they're mildly degrading the model's ranking ability too.
- This points at the diffusion-generated synthetic samples themselves rather than alpha/loss
  tuning: Elliptic's 165-dim feature space is far higher-dimensional than PaySim's 12-dim, same
  TabDDPM architecture/capacity/~3.4K real-fraud training examples — Run 30's smoke test flagged a
  std=96 anomaly for synthetic samples (attributed at the time to that test's tiny 20-epoch
  budget), but this full sweep's uniformly bad results across every alpha suggest the real
  (1000-epoch) TabDDPM model may also not be modeling this feature distribution well.
- Next: inspect the actual synthetic fraud node features directly (mean/std, feature-wise
  distribution vs real fraud, t-SNE) before spending more effort on alpha tuning here — if the
  samples are genuinely poor quality, the fix is on the diffusion side (more capacity/epochs for
  165-dim, or fewer/less-aggressively-attached synthetic nodes), not further alpha search.

## [2026-07-21] Run 35 — elliptic_graphsage_tabddpm_alpha025 — diffusion sampler root cause found and fixed, FIRST genuine win
- Investigated Run 33/34's uniformly-bad sweep directly rather than more alpha tuning. Ruled out
  graph-attachment topology first (attach_to=fraud_only + k_connections=1 test: F1=0.482, AUC=0.544
  — virtually unchanged from the original random-attachment alpha=0.25 result, so NOT the cause).
- **Root cause, confirmed locally**: trained the full 1000-epoch TabDDPM on Elliptic locally
  (models/diffusion/tabddpm.py, unchanged code) and sampled synthetic fraud features directly for
  inspection. Real fraud TRAIN features: mean=-0.112, std=0.563, mean feature-vector norm=6.66.
  Synthetic (DDIM, 50 steps): mean=0.335, **std=78.5**, mean norm=1002 — ~150x too large. DDPM
  ancestral sampling gave the same pathology (std=110.4). Neither ddpm_sample nor ddim_sample
  clamped the predicted x0 at each reverse step — a standard DDPM safeguard against reverse-process
  error compounding over many steps, absent from both samplers. Without it, a slightly-off
  prediction early in the 1000-step chain drifts further out-of-distribution each subsequent step
  instead of being corrected — plausible on this harder 165-dim/~3.4K-sample estimation problem
  (vs PaySim's easy 12-dim case, where diffusion augmentation genuinely worked — Run 29).
- **Fix**: added per-feature x0-clamping to both samplers (models/diffusion/tabddpm.py), bounded to
  real fraud TRAIN features' own mean +/- 3*std (data/augment_graph.py, configurable via
  augment.clamp_std/clamp_synthetic). Verified locally before re-running on RunPod: clamped
  synthetic mean=-0.102, std=0.798, norm=10.3 — much closer to real (still not exact, but no longer
  a 150x blowup).
- Config: configs/elliptic_diffusion_alpha025.yaml (alpha=0.25, matching the honest baseline
  exactly for the fairest comparison), clamp fix active. Compute: RunPod Serverless (endpoint
  YOUR_ENDPOINT_ID), worker_git_commit=17546c3. First run with wandb tracking active end-to-end:
  https://wandb.ai/[project]/fraud-diffusion/runs/oxqht2nj
- Val: F1-macro=0.8233, AUC-ROC=0.9454, AUPRC=0.7538, G-mean=0.7311.
  Test: **F1-macro=0.7447, AUC-ROC=0.8422, AUPRC=0.4664, G-mean=0.5974.**
  Test confusion: TP=146, FP=23, TN=8410, FN=262 (precision_fraud=0.864, recall_fraud=0.358).
- **First genuine win for diffusion augmentation on Elliptic**: beats the honest no-diffusion
  baseline (Run 26/29: F1=0.727, AUC=0.837) on both F1 (+0.018) and AUC (+0.005), and training
  converged normally (best_epoch=592, stable — not the epoch-1 collapse of every broken-sampler
  run in Run 33). Recall improves (0.319->0.358) at a small precision cost (0.890->0.864),
  consistent with what real additional fraud-training-signal should do, unlike Run 32/33's pattern
  of predicted_positive_rate exploding uncontrollably.
- Next: this is a modest win (+0.018 F1), not yet the "beats baseline by a wide margin" result
  PaySim's alpha tuning gave — worth a small sweep (alpha x clamp_std) via the new infra/sweep.py
  to see if this can be pushed further, now that the sampler itself is trustworthy. Also worth
  re-testing higher alpha (0.60, previously the best of the broken sweep at F1=0.566) with the
  clamp fix active, since the earlier alpha sweep's shape is no longer informative once the
  underlying synthetic data was garbage.

## [2026-07-21] Run 36 — elliptic diffusion alpha x clamp_std grid sweep (first real infra/sweep.py use)
- Config: 9-point grid on top of configs/elliptic_diffusion_alpha025.yaml — loss.alpha in
  {0.25, 0.4, 0.6} x augment.clamp_std in {2, 3, 4} (clamp fix from Run 35 active throughout).
  Dispatched via the new infra/sweep.py (grid dispatcher, threaded parallel RunPod invocations) —
  first real use of that tool, not just a smoke test. Caught and fixed a real bug on the first
  attempt: sweep.py/wandb_sweep.py's CLIs never set runpod.api_key before calling invoke()
  (unlike runpod_serverless.py's main(), which does) — every job failed instantly with "Expected
  run_pod.api_key to be initialized". Fixed, re-dispatched, all 9 completed cleanly the second time.
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), 9 parallel jobs across 4 workers,
  ~1-6 min each (diffusion training + GNN training).
- **Full grid, Test split:**
  | alpha \\ clamp_std | 2 | 3 | 4 |
  |---|---|---|---|
  | 0.25 | 0.7407 | 0.7503 | 0.7300 |
  | 0.40 | 0.7443 | 0.7482 | 0.7456 |
  | 0.60 | 0.7545 | **0.7601** | 0.7552 |
  (all AUC in a tight 0.833-0.849 band, no clear trend by alpha or clamp_std — this grid is mainly
  moving F1 via precision/recall trade-off at the fixed 0.5 threshold, not overall ranking quality)
- **alpha=0.6 beats alpha=0.4 beats alpha=0.25 at every single clamp_std value** — higher alpha
  keeps helping once the sampler is trustworthy, mirroring PaySim's own alpha-tuning pattern. This
  strongly suggests Run 32/33's original "alpha=0.95 overshoots on Elliptic" conclusion was an
  artifact of the broken sampler (unbounded synthetic feature scale), not a real property of
  alpha itself — alpha=0.6 alone previously scored F1=0.566 (Run 33, broken sampler) vs **0.760**
  now with the exact same alpha and a fixed sampler, a swing far too large to be alpha's own doing.
  clamp_std=3 (the default) is the best or near-best choice at every alpha tested.
- Best point (alpha=0.6, clamp_std=3): Test F1=0.7601, AUC=0.8330, G-mean=0.6325 — beats the
  honest no-diffusion baseline (Run 26/29: F1=0.727) by +0.033 F1, the best diffusion result on
  Elliptic so far.
- Next: alpha was still climbing at 0.6 (the top of this grid) — extend to 0.75/0.85/0.95 at
  clamp_std=3 to find where it actually plateaus or reverses, now that the confound is removed.

## [2026-07-21] Run 37 — elliptic diffusion alpha extension (0.75/0.85/0.95 at clamp_std=3) — found the real peak, BEST result overall
- Config: same as Run 36, extending loss.alpha to {0.75, 0.85, 0.95} at the winning clamp_std=3,
  dispatched via infra/sweep.py (second real use, no new bugs this time).
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), 3 parallel jobs, ~2-3 min each.
- **Full alpha curve at clamp_std=3 (combining Run 36 + this run), Test split:**
  | alpha | Test F1 | Test AUC |
  |---|---|---|
  | 0.25 | 0.7503 | 0.8361 |
  | 0.40 | 0.7482 | 0.8356 |
  | 0.60 | 0.7601 | 0.8330 |
  | **0.75** | **0.7671** | 0.8311 |
  | 0.85 | 0.7565 | 0.8349 |
  | 0.95 | 0.7444 | 0.8277 |
- **Non-monotonic with a clear interior peak at alpha=0.75** — rises 0.25->0.75, then falls
  0.75->0.95. This is qualitatively different from PaySim, where every alpha point up to 0.95
  still climbed monotonically without plateauing (Run 28). Elliptic's real fraud-detection
  difficulty (baseline AUC 0.837 vs PaySim's 0.96+) means there's an actual sweet spot beyond
  which extra positive-class loss weighting starts trading away too much precision for recall,
  rather than continuing to help — consistent with the mechanism identified in Run 32 (predicted-
  positive-rate inflation), just no longer catastrophic now that the sampler bug is fixed.
- **New best Elliptic result overall**: alpha=0.75, clamp_std=3 — Test F1=0.7671, AUC=0.8311,
  G-mean=0.6349. Beats the honest no-diffusion baseline (F1=0.727) by **+0.040 F1** — right in
  line with PaySim's own diffusion-alone win magnitude (Run 29: +0.040 F1 at alpha=0.25). This is
  the first Elliptic result that matches PaySim's evidence that diffusion augmentation itself adds
  real value, not just an artifact of loss-weighting.
- Next: this grid never tested values between the tested points (e.g. 0.65-0.70, either side of
  the 0.75 peak) — a continuous-range search (wandb Sweep, bayes method, via infra/wandb_sweep.py)
  would find the true optimum without needing another hand-picked round, and is the natural next
  step now that this exact "grid -> eyeball peak -> hand-launch a follow-up round" pattern has
  played out twice in a row (Run 36 -> Run 37) for the same reason bayesian search exists.

## [2026-07-21] Run 38 — elliptic_graphsage, corrected honest baseline after fixing transductive edge leakage
- **Context**: arXiv 2604.19514 ("When Graph Structure Becomes a Liability", Apr 2026) found
  standard transductive GNN setups on Elliptic — where every training forward pass sees the WHOLE
  graph, including val/test-period edges, and only node LABELS are masked from the loss — produce
  a 39.5-point F1 gap vs strict inductive evaluation (their numbers: Random Forest on raw features
  F1=0.821 beats every GNN tested; their GraphSAGE only reaches F1=0.689 under the honest protocol).
  Audited our own pipeline and found EXACTLY this bug: train_gnn.py's training loop called
  `model(data.x, data.edge_index)` using the complete, temporally unfiltered graph every epoch,
  for both datasets — every run so far (Runs 26/29/32/33/35/36/37, all our alpha+diffusion tuning)
  was done under this leaky protocol.
- **Fix** (data/temporal_edges.py, elliptic_preprocess.py, paysim_preprocess.py, train_gnn.py,
  augment_graph.py): training forward passes now use `train_edge_index` (edges where BOTH
  endpoints are in the train period) exclusively; val-time inference legitimately uses
  `val_edge_index` (train+val edges); test-time inference uses the full graph — this is standard,
  non-leaky inductive evaluation, not a second leak, since it's fixed-weight inference with no
  gradient flow. Code review (before trusting any new numbers) caught two further bugs: the
  mini-batch loader fix initially tripled memory by cloning all 3 edge variants into every
  loader's Data object (fixed to build a minimal Data(x,y,edge_index)); and PaySim's exclusive
  Python-slice split boundaries vs the helper's inclusive `<=` convention would have misclassified
  one boundary row as train-visible (fixed with a `-1` offset at the PaySim call site).
- Config: configs/elliptic_full.yaml (alpha=0.25, identical hyperparameters to Run 26/29's
  original baseline) — same config, only the underlying leakage-free edge splits changed.
- Compute: local (mps), stopped_epoch=448 (early-stopped at 498, patience=50).
- Val: F1-macro=0.8443, AUC-ROC=0.9388, AUPRC=0.8051, G-mean=0.7513.
  Test: **F1-macro=0.7453, AUC-ROC=0.8422**, AUPRC=0.5039, G-mean=0.5935.
  Test confusion: TP=144, FP=16, TN=8417, FN=264 (precision_fraud=0.900, recall_fraud=0.353).
- **Surprising result: this essentially MATCHES the old leaky baseline (Run 26/29: F1=0.727,
  AUC=0.837) — actually a hair better (+0.018 F1), not the dramatic collapse the paper reports for
  generic transductive setups.** Plausible explanations, not yet disambiguated: (a) our original
  "leaky" implementation, while technically transductive, may not have been exploiting test-period
  adjacency as strongly as a generic/naive transductive setup would — dropout=0.5, 2-layer depth,
  and EMA-smoothed weights may have limited how much the model could actually learn to lean on
  test-adjacent structure specifically; (b) the paper's own comparison involves other protocol/
  architecture differences from ours (their exact hyperparameters, whether they use focal loss,
  EMA, etc. are unknown to us) that could account for their reported gap independent of the
  transductive/inductive distinction alone; (c) simple variance — a single run each side isn't
  strong evidence of "no effect," just evidence the effect isn't the huge collapse the paper found
  in THEIR setup. This is being reported honestly as an open, unresolved question, not spun into
  either "the fix didn't matter" or "our old numbers were fine after all" — the fix is still the
  methodologically correct thing to have done regardless of its empirical impact here.
- Next: (a) this is now the honest baseline going forward, replacing Run 26/29's number for all
  future comparisons; (b) re-validate the diffusion+alpha tuning arc (best known: alpha=0.75,
  clamp_std=3, old-pipeline F1=0.767) under the corrected pipeline — diffusion's interaction with
  a train-only edge view may differ from how it behaved against the old full-graph view during
  training; (c) do NOT assume PaySim is unaffected just because Elliptic's fix showed a small
  empirical delta — PaySim has the same class of fix now applied but has not yet been re-run to
  check.

## [2026-07-21] Run 39 — Elliptic diffusion-hyperparameter sweep (alpha x n_synthetic x k_connections), first results under the corrected leakage-free pipeline
- Config: 12-point grid on configs/elliptic_diffusion_alpha025.yaml — loss.alpha in {0.25, 0.75}
  x augment.n_synthetic in {1731 (half real train-fraud), 3462 (default, ~1x), 6924 (2x)} x
  augment.k_connections in {1, 5}, clamp_std=3 throughout (established best in Run 36/37). This is
  the FIRST diffusion sweep run entirely under the corrected pipeline (data/temporal_edges.py fix,
  Run 38) — both re-validates whether the alpha finding survives the fix and covers the
  diffusion-specific hyperparameters (n_synthetic, k_connections) that had never been swept before.
- Compute: RunPod Serverless (endpoint YOUR_ENDPOINT_ID), 12 jobs via infra/sweep.py. One point
  (alpha=0.25/n=3462/k=5) hit a transient RunPod API ProxyError on the first attempt (network
  blip, not a code bug) — re-dispatched successfully.
- **Full grid, Test F1:**
  | | k_connections=1 | k_connections=5 |
  |---|---|---|
  | alpha=0.25, n=1731 | 0.7317 | 0.7399 |
  | alpha=0.25, n=3462 | 0.7442 | 0.7353 |
  | alpha=0.25, n=6924 | 0.7483 | 0.7241 |
  | alpha=0.75, n=1731 | 0.7600 | **0.7635** |
  | alpha=0.75, n=3462 | 0.7610 | 0.7555 |
  | alpha=0.75, n=6924 | 0.7601 | 0.7447 |
- **The alpha finding survives the leakage fix**: every alpha=0.75 point is close to or above
  every alpha=0.25 point (5 of 6 alpha=0.75 points beat all 6 alpha=0.25 points; the one
  exception, alpha=0.75/n=6924/k=5 at 0.7447, still nearly ties alpha=0.25's own best of 0.7483).
  alpha remains the dominant lever, consistent with Runs 36/37's pre-fix finding — reassuring,
  since that arc was entirely tuned against the old leaky pipeline and could plausibly have not
  transferred.
- **n_synthetic and k_connections have a real but much smaller, non-monotonic effect** — no
  consistent "more synthetic data is better" or "fewer/more connections is better" pattern at
  either alpha value. This is a genuine negative/null result for those two hyperparameters
  specifically (within the tested range) — the augmentation *amount* and *connectivity* matter
  far less than the loss-reweighting *alpha* does.
- **New best point**: alpha=0.75, n_synthetic=1731, k_connections=5 — Test F1=0.7635, AUC=0.8383.
  Against Run 38's corrected honest baseline (F1=0.7453): **+0.018 F1**, a real, validated
  improvement under the correct protocol — smaller than the +0.040 the old (leaky-pipeline)
  comparison suggested, since the honest baseline itself is now higher, but still a genuine win.
- **Convergence check (prompted by a direct question: "does it feel undertrained?")**: pulled full
  per-epoch wandb histories for two of these runs (best point, and one with a low best_epoch=316)
  rather than guessing from final numbers alone. Both show val AUC/F1 climbing steadily then
  going genuinely flat over the last 50-100+ epochs before patience triggers (e.g. best-point run:
  val_auc 0.9373->0.9381 over its final 80 logged epochs, essentially noise) — these runs are
  well-converged, not prematurely stopped. This is NOT evidence about PaySim's high-alpha runs
  (Runs 24-31), which predate wandb integration entirely and have no logged history to check —
  those runs' "still climbing, never plateaued" observations come from the printed epoch logs at
  the time, a real but less precise signal than a full wandb curve.
- Next: (a) no LR scheduler exists anywhere in train_gnn.py currently (plain constant-LR Adam) —
  worth considering (cosine annealing or similar) especially for PaySim's high-alpha configs that
  hit the 1000-epoch cap without plateauing, now that wandb curves can actually confirm whether it
  helps rather than guessing; (b) still open: whether Elliptic's own precomputed "aggregated"
  feature columns (~72 of 165, computed by the original dataset authors from one-hop neighbor
  info) could carry a dataset-inherent leakage independent of our edge_index fix, if any of that
  aggregation pulled in a transaction's *later* neighbors (e.g. a subsequent transaction spending
  its output) — not yet investigated, flagged honestly as unverified rather than assumed clean;
  (c) re-run PaySim's diffusion+alpha arc under the corrected pipeline (still not done).

## [2026-07-21] Run 40 — Elliptic: ddim_steps/sampler sweep, adversarial diffusion loss (2 attempts), and a Random Forest reality check
Three threads from the same session, reported together since they build on each other.

### ddim_steps x sampler sweep — new best result
- Config: 8-point grid on top of Run 39's best point (alpha=0.75, n_synthetic=1731,
  k_connections=5, clamp_std=3) — augment.ddim_steps in {20, 50, 100, 200} x augment.sampler in
  {ddim, ddpm}. Dispatched via infra/sweep.py.
- **Full grid, Test F1** (ddim_steps has no effect on ddpm sampling — those 4 rows are redundant
  duplicates around the same true ddpm value, ~0.756-0.758, confirming that parameter correctly
  does nothing there):
  | ddim_steps | ddim F1 | ddim AUC | ddpm F1 (reference, steps-invariant) |
  |---|---|---|---|
  | 20 | 0.7691 | 0.8418 | 0.7568 |
  | 50 | 0.7652 | 0.8387 | 0.7570 |
  | **100** | **0.7749** | **0.8465** | 0.7566 |
  | 200 | 0.7616 | 0.8442 | 0.7579 |
- DDIM beats DDPM at every step count. Non-monotonic in steps — a sweet spot around 100, not
  "more steps is always better." **New best: ddim_steps=100, Test F1=0.7749, AUC=0.8465** — beats
  Run 39's 0.7635 by another +0.011 F1.

### Adversarial diffusion loss — diagnosed a real training-dynamics bug, then fixed it
- Per discussion, added an optional adversarial fine-tuning phase to TabDDPM (models/diffusion/
  tabddpm.py's new Discriminator class; training/train_diffusion.py): starting partway through
  training (not epoch 1 — the denoiser needs to already be decent first), L_total = L_denoise +
  lambda*L_adversarial, where the "fake" sample fed to the discriminator is the denoiser's
  one-step x0 estimate (GaussianDiffusion.predict_x0), not a full multi-step sample (far too
  expensive per batch). Clamped to real fraud features' own mean+-3*std for numerical stability.
- **First attempt** (configs/elliptic_diffusion_adversarial.yaml: start_epoch=500,
  lambda_adv=0.1, discriminator_lr=denoiser's own lr=0.001): Test F1=0.7517, AUC=0.838 — WORSE
  than the non-adversarial baseline (F1=0.7635, AUC=0.8383) at the identical recipe.
- **Diagnosed via wandb** (added logging to train_diffusion.py specifically to answer "is this a
  calibration issue or a real bug" — previously that script had no tracking at all, only printed
  logs from a tiny smoke test): loss_disc steadily DECREASED while loss_adv steadily INCREASED
  and loss_denoise got WORSE once the adversarial phase started — the textbook "discriminator
  winning too fast" GAN pathology. A discriminator with the same LR as the denoiser learns the
  real/fake distinction much faster than the denoiser (whose job is much harder — accurate
  per-timestep noise prediction, not binary classification) can adapt, turning the adversarial
  term into noise that drags the denoiser off its primary objective rather than useful guidance.
  Confirmed the mechanism, not just the symptom, via a 40-epoch local smoke test before spending
  RunPod time on a fix.
- **Fix**: discriminator_lr set to 10x lower than the denoiser's (configs/
  elliptic_diffusion_adversarial_lowdisclr.yaml) — the standard remedy for this exact GAN
  pathology. Verified locally first (loss_denoise now keeps improving post-adversarial-start
  instead of reversing), then dispatched the real 1000-epoch run.
- **Result: Test F1=0.7691, AUC=0.8526** — beats BOTH the broken adversarial attempt (0.7517) AND
  the non-adversarial baseline (0.7635/0.8383). AUC=0.8526 is the best AUC of any Elliptic
  diffusion variant so far (though this run used the default ddim_steps=50, not yet combined with
  the ddim_steps=100 finding above — that combination hasn't been tested yet).
- This is a good example of the diagnose-before-abandoning principle: the first attempt's
  regression could easily have been read as "adversarial loss doesn't help here" and dropped, but
  it was a fixable training-dynamics miscalibration, not a fundamental mismatch.

### Random Forest reality check — a humbling, important result
- Per arXiv 2604.19514's own core finding (RF on raw features beats every GNN tested under strict
  inductive eval), built evaluation/rf_baseline.py: RandomForestClassifier(n_estimators=300,
  class_weight="balanced") on raw node features, NO graph structure at all, using the exact same
  compute_metrics() as the GNN pipeline for a direct comparison. Meant to run as a standing
  background reference alongside every future experiment (seconds, no GPU), not a one-off check.
- **Elliptic result: Test F1=0.7888, AUC=0.8671** — beats our best GNN result to date (diffusion +
  fixed adversarial loss: F1=0.7691, AUC=0.8526), and beats every other Elliptic config tried this
  session (honest baseline 0.7453, best plain-diffusion 0.7749).
- **This reproduces the paper's core finding directly in our own pipeline, on our own features.**
  Despite the leakage fix, extensive alpha/clamp_std/n_synthetic/k_connections/ddim_steps tuning,
  and a working adversarial loss, the graph structure is not currently adding value over a simple
  tabular baseline on Elliptic. This doesn't mean the GNN+diffusion approach is worthless — it's
  still the interesting research question (why doesn't structure help here, and can that be
  fixed?) — but it means "beat RF" is the real bar for this dataset, not just "beat the honest
  no-diffusion GraphSAGE baseline," and right now we haven't cleared it.
- Next: (a) run the RF baseline on PaySim too (only the small local MVP fixture is available
  locally, not the full 2.77M-row graph our real PaySim experiments use — needs the full graph,
  likely via RunPod or downloading the full CSV) for the same calibration point there; (b) combine
  ddim_steps=100 with the fixed adversarial loss (untested combination, plausibly the best of both
  findings); (c) investigate WHY the graph isn't helping on Elliptic specifically — worth revisiting
  the still-open "does Elliptic's own aggregated feature engineering already encode most of the
  graph-derived signal" question from Run 39, since that would directly explain why adding our own
  graph-based modeling on top adds so little.

## [2026-07-21] Run 41 — PaySim re-measurement: leakage fix, cosine scheduler, and an OOM saga
- Config: configs/paysim_full.yaml grid — loss.alpha in {0.25, 0.95} x train.lr_schedule in
  {none, cosine}, all under the corrected (leakage-free) pipeline.
- **Infra saga first, since it ate real time**: full-batch PaySim training (hidden_dim=128) OOM'd
  repeatedly on RunPod's 24GB GPU pool, intermittently rather than deterministically (the same
  config sometimes succeeded, sometimes failed). Root-caused two real, separate bugs before
  landing clean results: (1) invoke() used Endpoint.run_sync(), which silently returns None on
  jobs longer than a few minutes — fixed to async dispatch + poll (infra/runpod_serverless.py);
  (2) a worker that OOMs mid-training stays "healthy" from RunPod's perspective and keeps serving
  future jobs, but the failed job's exception traceback pins its tensors in GPU memory forever
  within that process — confirmed directly (two different jobs landed on the same worker PID
  back-to-back, both OOM'd with near-identical memory-in-use numbers even after adding
  torch.cuda.empty_cache() at job start). Fixed by killing the worker process on OOM
  (serverless/handler.py) so RunPod provisions a fresh one. Also added automatic retry
  (infra/sweep.py) and, pragmatically, reduced model.hidden_dim 128->96 for this specific batch to
  guarantee it fits rather than keep chasing an intermittent ceiling.
- **Full results (hidden_dim=96 throughout this batch)**:
  | alpha | scheduler | Test F1 | Test AUC |
  |---|---|---|---|
  | 0.95 | none | **0.9460** | **0.9979** |
  | 0.95 | cosine | 0.9389 | 0.9967 |
  | 0.25 | none | 0.8362 | 0.9746 |
  | 0.25 | cosine | 0.8150 | 0.9385 |
  (a fifth point, alpha=0.25/none/hidden_dim=128, from an earlier diagnostic job: F1=0.8536,
  AUC=0.9693 — the hidden_dim=96 equivalent above, 0.8362, is ~0.017 F1 lower, suggesting the
  capacity reduction costs a bit more at lower alpha than at 0.95, where hidden_dim=96's 0.9460
  essentially matches both the old leaky baseline (Run 28: F1=0.9465) and the earlier
  hidden_dim=128 leakage-fixed diagnostic (F1=0.9466) almost exactly.)
- **Leakage fix has ~no measurable effect on PaySim**, consistent with what Run 38 found on
  Elliptic — alpha=0.95 pre- and post-fix are within noise of each other (0.9465 vs 0.9460/0.9466
  across variants). This makes sense given PaySim's classes are already near-perfectly separable
  (baseline AUC 0.96+) — there's little room for training-time structural leakage to matter when
  the task is already nearly solved without it.
- **Cosine scheduler consistently HURTS on PaySim, at both alpha levels** (alpha=0.95:
  -0.0071 F1; alpha=0.25: -0.0212 F1, a proportionally bigger hit). This contradicts the original
  motivating hypothesis (that a decaying LR would help the high-alpha configs that hit the
  1000-epoch cap without plateauing) — plain constant-LR Adam is simply better here. Not
  recommended for PaySim going forward; Elliptic's own cosine-schedule interaction hasn't been
  tested (n/a — no PaySim-analogous "never plateaus" symptom seen on Elliptic).
- Next: run the RF baseline on PaySim's full graph too (only the small local MVP fixture is
  available locally); PaySim's diffusion+alpha arc still needs re-checking under the corrected
  pipeline (only the non-diffusion alpha sweep was re-measured here).

## [2026-07-21] Run 42 — Elliptic: combining ddim_steps=100 with the fixed adversarial loss doesn't stack additively
- Config: configs/elliptic_diffusion_adversarial_lowdisclr.yaml (Run 40's fixed-adversarial recipe)
  with augment.ddim_steps swept over {50, 100, 150} — testing whether Run 40's two independent
  wins (ddim_steps=100 alone: F1=0.7749; adversarial loss alone at default ddim_steps=50: F1=0.7691)
  compound when combined.
- **Result: they don't stack — if anything, combining is a very slight net negative**:
  | ddim_steps | Test F1 | Test AUC |
  |---|---|---|
  | 50 | 0.7657 | 0.8470 |
  | 100 | 0.7674 | 0.8442 |
  | 150 | 0.7678 | 0.8645 (best AUC of any Elliptic config so far) |
  All three cluster tightly (0.766-0.768), each individually a bit BELOW the better of the two
  standalone techniques (plain ddim_steps=100: 0.7749; adversarial-alone: 0.7691). Best AUC yet
  (0.8645 at ddim_steps=150) despite F1 not improving — another instance of AUC/F1 not moving
  together, consistent with earlier findings that these levers mostly trade precision/recall at
  the fixed 0.5 threshold rather than uniformly improving ranking quality.
- Plausible explanation (not confirmed): both techniques independently push the synthetic samples
  toward being more "realistic"/diverse in overlapping ways (more denoising steps = closer to the
  true reverse-process trajectory; adversarial loss = explicitly pushed toward matching the real
  distribution) — stacking them may be redundant rather than complementary, each correcting for
  something the other already substantially addresses.
- **Still no config beats the Random Forest baseline (F1=0.7888, Run 40)** — this remains the
  central open question, not further hyperparameter combination.
- Next: stop stacking individual levers on the same underlying technique; prioritize investigating
  WHY structure doesn't help (Run 40's feature-ablation idea: local-only vs local+aggregated
  columns) over further diffusion-recipe refinement, given diminishing/negative returns from
  combining wins.

## [2026-07-21] Run 43 — Elliptic: spectral-matching loss lambda sweep
- Config: configs/elliptic_diffusion_spectral.yaml (best-known recipe: alpha=0.75,
  n_synthetic=1731, k_connections=5, ddim_steps=100) with diffusion.spectral.lambda_spectral in
  {0.01, 0.1, 0.5}, start_epoch=500.
- **Results**:
  | lambda_spectral | Test F1 | Test AUC |
  |---|---|---|
  | 0.01 | 0.7421 | 0.8266 |
  | **0.1** | **0.7676** | 0.8581 |
  | 0.5 | 0.7618 | 0.8495 |
  Non-monotonic, sweet spot at 0.1 (too weak a signal at 0.01, presumably too strong/distorting at
  0.5). Still below the standalone ddim_steps=100 result (0.7749) and the fixed-adversarial result
  (0.7691), and well below Random Forest (0.7888, Run 40).
- Like the adversarial+ddim_steps combination (Run 42), this doesn't establish a new best — the
  auxiliary losses explored so far (adversarial, spectral, and combining them) all land in the
  same 0.74-0.77 F1 band without beating either RF or the single best plain-diffusion config.
- Next: per the 2026-07-21 discussion, the more promising direction now is likely NOT more
  auxiliary losses on the same per-node-independent diffusion formulation, but addressing
  structural coherence directly — see the "template" attach mode (below/next entry) that clones a
  real fraud node's actual local neighborhood instead of inventing random edges, testing the
  hypothesis that our current random-heuristic wiring is itself vandalizing the graph's structural
  invariants and that's part of why diffusion augmentation isn't a bigger, more consistent win.

## [2026-07-21] Run 44 — Elliptic: structure-preserving ("template") attachment — counter-intuitive result, didn't help
- Config: configs/elliptic_diffusion_template.yaml — best-known recipe (alpha=0.75,
  n_synthetic=1731, ddim_steps=100) with augment.attach_to=template (data/augment_graph.py's new
  mode: each synthetic node clones a real fraud TRAIN node's actual local neighborhood — real
  edges, not k random ones) instead of the default "all" (k=5 random real TRAIN nodes). Verified
  locally first: 100 synthetic nodes -> 314 new edges (~1.57 real neighbors/node on average,
  varying per template) vs the old fixed k=5 for every node regardless of realism.
- **Result: Test F1=0.7566, AUC=0.8531** — WORSE than the same recipe's "all"-attach result
  (ddim_steps=100 alone: F1=0.7749, AUC=0.8465), not better. Directly contradicts the hypothesis
  as stated (that random-heuristic wiring vandalizes structure and fixing it would unlock a bigger
  diffusion win) — at least in this single run.
- **Plausible explanation**: Elliptic's real fraud nodes tend to have LOW degree (illicit actors
  plausibly minimize connections to avoid detection — a real, documented property of this
  dataset). Cloning their actual neighborhoods gave each synthetic node only ~1.6 real neighbors on
  average vs the old fixed k=5 — we may have traded "arbitrary but well-connected" for "realistic
  but under-connected," and the connectivity REDUCTION may matter more than the structural realism
  helps. If so, a middle ground (template-based neighbor TYPE selection but topped up to a fixed
  minimum degree) might be worth trying rather than abandoning the idea outright.
- **Honest caveat, not brushed aside**: this is a single run. The 2-seed smoke test earlier this
  session already showed ~0.02-0.03 F1 spread from seed alone on one config — the 0.018 F1 gap
  here (0.7749 vs 0.7566) is within that same range, so this result should be read as
  "inconclusive, no clear win, possibly a real regression" rather than a confident refutation.
  This is now the second single-seed comparison this session close enough to plausible noise to
  be genuinely ambiguous (see also Run 42/43's auxiliary-loss results, all clustered within a
  similarly narrow band) — the multi-seed pass (infra/multi_seed.py, built but not yet used at
  scale) is becoming less of a "nice to have for the eventual paper" and more of an immediate
  necessity before drawing further conclusions from single-point comparisons.
- Still true regardless: **no config has beaten Random Forest (F1=0.7888)**.
- Next: (a) run a proper multi-seed comparison (5-10 seeds) between the "all" and "template" attach
  modes specifically, since this single-run result is too close to call; (b) if template attach
  is confirmed genuinely worse, consider a hybrid (real neighbor types, topped up to a minimum
  degree) rather than discarding the structural-preservation idea entirely; (c) the RF-beating
  question remains unresolved and is arguably more important than further attachment-mechanism
  tuning at this point.

## [2026-07-21] Run 45 — Elliptic: 5-seed multi-seed comparison (all-attach vs template-attach) — Run 44's regression was noise, AND a bigger meta-finding
- Config: infra/multi_seed.py compare(), configs/elliptic_diffusion_best_all.yaml vs
  configs/elliptic_diffusion_template.yaml, seeds 0-4 (paired — same seed list drives both
  configs), Test F1-macro, paired Wilcoxon signed-rank test.
- **Results**:
  | seed | all-attach F1 | template-attach F1 |
  |---|---|---|
  | 0 | 0.7588 | 0.7494 |
  | 1 | 0.7637 | 0.7490 |
  | 2 | 0.7483 | 0.7545 |
  | 3 | 0.7520 | 0.7478 |
  | 4 | 0.7456 | 0.7479 |
  | **mean +- std** | **0.7537 +- 0.0075** | **0.7497 +- 0.0028** |
  Wilcoxon signed-rank: statistic=4.0, **p=0.4375** — nowhere near significant.
- **Run 44's apparent regression (template worse than all) is NOT a real effect** — well within
  noise. Good validation of why the multi-seed infra exists.
- **Bigger, more important meta-finding**: the "all-attach" recipe's mean F1 across 5 fresh seeds
  (0.7537) is meaningfully BELOW the single-seed=42 number we'd been citing as "the best Elliptic
  result" (ddim_steps=100, ~0.7749, Run 40) and even below the plain adversarial-loss single-seed
  result (0.7691, Run 40). **seed=42 was a favorably lucky draw for this recipe, not its true
  expected performance.** This recalibrates how several of this session's single-seed "wins"
  should be read — as promising leads, not confirmed results, until each gets the same multi-seed
  treatment.
- **Secondary, tentative observation** (n=5, treat cautiously): template-attach shows ~3x LOWER
  variance than all-attach (std 0.0028 vs 0.0075) despite a similar mean — structure-preserving
  attachment may make training more stable/reproducible even without improving the mean. Not
  strong enough evidence from 5 seeds to act on alone, but a reasonable hypothesis to keep in mind.
- **This also means the Random Forest comparison (Run 40, F1=0.7888) needs the same scrutiny** —
  it was evaluated at a single seed (implicitly, sklearn's RandomForestClassifier with
  random_state=42) too. The "RF beats every GNN" conclusion is very likely still directionally
  correct given the size of the gap (0.7888 vs a genuine mean of ~0.75), but should itself be
  multi-seeded before being treated as fully settled, especially now that we've seen how much a
  single favorable seed can inflate a headline number.
- Next: (a) multi-seed the RF baseline (evaluation/rf_baseline.py currently takes a seed param —
  just needs to be run across several and reported as mean+-std, cheap since it's seconds each);
  (b) treat all "best result" claims from this session as provisional until multi-seeded; (c) GAT
  on Elliptic dispatched (concrete evidence-based motivation: fraud-touching edges show 0.41
  homophily vs an ~0.116 random-chance baseline — real fraud-clustering signal exists but is a
  minority of a typical fraud node's edges, exactly the situation attention-based aggregation is
  meant to help with, unlike naive mean-pooling GraphSAGE) — result pending.

## [2026-07-21] Run 46 — GAT on Elliptic (first attempt): clear negative result
- Context: computed the real graph's homophily directly before this run — overall labeled-labeled
  homophily 0.947, but homophily on edges TOUCHING a fraud node only 0.410 (vs an ~0.116
  random-chance baseline given fraud is 11.6% of train nodes) — real fraud-clustering signal
  exists (~3.5x enrichment over random), but is still a minority of a typical fraud node's edges.
  Motivated trying attention-based aggregation (GAT), which can in principle learn to upweight the
  informative fraud-fraud edges instead of averaging them away with the majority-legit ones the
  way plain mean-pooling GraphSAGE does.
- Config: configs/elliptic_full_gat.yaml — full-batch GAT (Elliptic's graph, 468,710 directed
  edges, is ~60x smaller than PaySim's full graph, so no mini-batch/neighbor-sampling needed here
  unlike PaySim's GAT). Hyperparameters borrowed wholesale from PaySim's GAT config (hidden_dim=64,
  heads=8, dropout=0.6, lr=0.005) — NOT tuned for Elliptic specifically.
- **Result: Test F1=0.6736, AUC=0.8014** — worse than the honest GraphSAGE baseline (F1=0.7453,
  Run 38), worse than every diffusion variant tried, worse than RF (0.7888). best_epoch=25,
  notably early — training plateaued at a mediocre point before EMA even engaged (ema_start_epoch
  =30) and never improved afterward, a sign of getting stuck rather than genuinely fitting.
- **This is a clear negative result, but with a real caveat**: the hyperparameters were never
  tuned for Elliptic — they're PaySim's mini-batch GAT settings applied wholesale to a full-batch
  small-graph setting. Attention mechanisms are known to be harder to optimize than mean-pooling
  (more sensitive to lr, more prone to poor local optima) — this could easily be a bad
  hyperparameter draw rather than evidence that attention genuinely can't exploit the
  fraud-clustering signal the homophily analysis found.
- Next: retry with GraphSAGE-matched hyperparameters (lr=0.001, dropout=0.5, matching what's
  already well-tuned for this dataset) before concluding GAT doesn't help here — a fair comparison
  needs at least one attempt that isn't handicapped by an untuned learning rate.

## [2026-07-21] Run 47 — Multi-seed Random Forest baseline: the RF-vs-GNN gap is real, not a lucky seed
- Per Run 45's flag (RF had only been checked at a single seed, same concern that turned out to
  matter for several GNN "best results"), ran evaluation/rf_baseline.py across 5 seeds (0-4) on
  Elliptic. Fast enough (~1s/seed, sklearn, no GPU) to run locally without any of the hang risk
  GAT training has.
- **Results**: F1 = [0.7908, 0.7908, 0.7893, 0.7897, 0.7847], **mean=0.7890, std=0.0025**. AUC
  mean=0.8669, std=0.0037. Both remarkably stable — barely moved from the original single-seed=42
  number (F1=0.7888).
- Makes sense mechanistically: RandomForestClassifier(n_estimators=300) is already an ensemble
  averaging over 300 trees' worth of internal randomness, so a single sklearn "seed" has far less
  leverage over the final result than a single neural network training run does (contrast with
  Run 45's GNN comparison, where seed=42 alone swung headline results well above the 5-seed mean).
- **This settles the RF comparison**: with our best GNN mean (~0.75-0.76 across the diffusion
  variants tried, per Run 45) solidly below RF's low-variance ~0.789, the gap is real, not an
  artifact of favorable seed selection on either side. Random Forest genuinely and robustly beats
  every GNN approach tried on Elliptic so far.
- Next: this reinforces that the RF-beating question is the central one — GAT (Run 46, currently
  being retried with fairer hyperparameters) and any future architecture attempt should be
  multi-seeded before any "beats RF" claim is trusted, given how much single-seed variance has
  already been shown to matter this session.

## [2026-07-21] Run 48 — GAT on Elliptic, retry with GraphSAGE-matched hyperparameters — better, still a clear negative
- Config: configs/elliptic_full_gat.yaml with train.lr=0.001, model.dropout=0.5 (matching
  GraphSAGE's well-tuned settings) instead of Run 46's borrowed-from-PaySim lr=0.005/dropout=0.6.
- **Result: Test F1=0.7234, AUC=0.8180** — meaningfully better than Run 46's untuned attempt
  (0.6736), confirming hyperparameters mattered, but STILL clearly below the honest GraphSAGE
  baseline (0.7453, single-seed) or its ~0.75 multi-seed mean (Run 45), and well below RF's
  robust 0.789 (Run 47).
- **Overall verdict on GAT-on-Elliptic (two attempts, both negative)**: attention-based
  aggregation does not appear to be a straightforward fix for the fraud-camouflage/dilution
  problem identified via the homophily analysis (Run 46) — at least not without considerably more
  tuning than a "graft PaySim's or GraphSAGE's settings onto GAT" approach provides. This doesn't
  rule out attention-based approaches entirely (a properly-tuned GAT, or a purpose-built
  heterophily-aware architecture, might still work), but two straightforward attempts both
  underperforming is a real, honest data point against "just switch to GAT" as the fix.
- Next: given GAT's two negative results, the cheaper, more surgical self-vs-neighbor-diff feature
  modification to GraphSAGE (concat [self, neighbor_mean, self-neighbor_mean] instead of standard
  mean-aggregation) is now the more promising next experiment for directly addressing the
  homophily-dilution hypothesis, without committing to a harder-to-optimize attention architecture.

## [2026-07-21] Run 49 — GraphSAGEDiff on Elliptic (honest baseline, no diffusion): small, promising, healthier training than GAT
- Config: configs/elliptic_full_graphsage_diff.yaml — model.name=graphsage_diff (models/gnn/
  graphsage.py's new GraphSAGEDiff: concat [x, neighbor_mean(x), x-neighbor_mean(x)] before the
  first SAGEConv layer), otherwise identical to the honest GraphSAGE baseline config (alpha=0.25).
- **Result: Test F1=0.7496, AUC=0.8579** vs the honest GraphSAGE baseline's F1=0.7453/AUC=0.8422
  (Run 38) — a small improvement (+0.004 F1, +0.016 AUC), with BOTH precision_fraud (0.890->0.935)
  AND recall_fraud (0.319->0.355) up, not just a threshold-shift artifact. best_epoch=398 — normal,
  healthy convergence, none of GAT's early-plateau pathology (Runs 46/48).
- Given this session's repeated lesson about single-seed noise (Run 45's multi-seed check showed
  ~0.0075-0.028 F1 spread from seed alone in various comparisons), a +0.004 F1 improvement is NOT
  yet confirmed as real — needs the same multi-seed treatment before trusting it. But directionally
  positive with healthy training dynamics is already a meaningfully better outcome than GAT's two
  clear-negative, early-converging attempts.
- Next: (a) multi-seed GraphSAGEDiff vs plain GraphSAGE to confirm whether this small gain is real;
  (b) combine GraphSAGEDiff with the best-known diffusion recipe (untested combination) — the
  natural next test now that the architecture alone shows a non-negative signal, unlike GAT; (c) if
  neither GraphSAGEDiff nor GAT meaningfully closes the RF gap, the next real lever is a proper
  neighbor-selection mechanism (CARE-GNN/PC-GNN-style) rather than more tuning of what's been tried
  so far — per 2026-07-21 discussion, H100/more compute would NOT address GAT's actual problem
  (an optimization/convergence issue on a tiny graph that already trains fine, not a scale issue).

## [2026-07-21] Run 50 — Multi-seed GraphSAGE vs GraphSAGEDiff: the strongest signal of the session, not yet conventionally significant
- Config: infra/multi_seed.py compare(), configs/elliptic_full.yaml vs
  configs/elliptic_full_graphsage_diff.yaml, seeds 0-4 (paired), Test F1-macro.
- **Results**:
  | seed | GraphSAGE F1 | GraphSAGEDiff F1 | diff wins? |
  |---|---|---|---|
  | 0 | 0.7204 | 0.7450 | yes (+0.0246) |
  | 1 | 0.7482 | 0.7395 | no (-0.0087) |
  | 2 | 0.7432 | 0.7523 | yes (+0.0091) |
  | 3 | 0.7316 | 0.7450 | yes (+0.0134) |
  | 4 | 0.7241 | 0.7597 | yes (+0.0356) |
  | **mean +- std** | **0.7335 +- 0.0120** | **0.7483 +- 0.0078** | |
  Wilcoxon signed-rank: statistic=1.0, **p=0.125** — does not cross the conventional 0.05
  threshold, but n=5 is a genuinely underpowered test; 4 of 5 seeds favor GraphSAGEDiff, mean
  improvement is +0.0148 F1, and variance is ALSO lower (0.0078 vs 0.0120) — a combined signal
  (better mean, more stable, consistent direction) stronger than any other paired comparison this
  session (contrast with Run 45's all-vs-template comparison, p=0.4375, no consistent direction).
- **Provisionally the most promising architectural lead found this session** for closing the gap
  to Random Forest (mean 0.789), though still short of it, and not yet confirmed at a conventional
  significance level — treat as a strong lead, not a settled result, pending either more seeds or
  the eventual full 10-seed consolidation pass.
- Next: combine GraphSAGEDiff with the best-known diffusion recipe (alpha=0.75, n_synthetic=1731,
  ddim_steps=100) — untested combination, the natural next step now that the architecture alone
  shows a real (if not fully confirmed) positive signal.

## [2026-07-21] Run 51 — GraphSAGEDiff + diffusion combined: another "doesn't stack" result
- Config: configs/elliptic_diffusion_graphsage_diff.yaml — GraphSAGEDiff architecture (Run 49/50)
  combined with the best diffusion recipe (alpha=0.75, n_synthetic=1731, ddim_steps=100, Run 40).
- **Result (single seed=42): Test F1=0.7524, AUC=0.8404.**
  | Config | Test F1 | Test AUC |
  |---|---|---|
  | GraphSAGEDiff alone (Run 49) | 0.7496 | 0.8579 |
  | Plain diffusion alone (Run 40) | 0.7749 | 0.8465 |
  | **GraphSAGEDiff + diffusion (this run)** | **0.7524** | **0.8404** |
  Lands BETWEEN the two individual techniques, closer to GraphSAGEDiff-alone than to
  plain-diffusion-alone — not additive, consistent with Run 42's adversarial+ddim_steps combo
  (which also landed short of either individual best).
- **This is now the THIRD instance this session of two independently-positive techniques failing
  to compound when combined** (Run 42: adversarial loss + ddim_steps; Run 51 here: GraphSAGEDiff +
  diffusion). Worth calling out as a recurring pattern rather than three isolated coincidences:
  each of our interventions may be addressing an overlapping piece of the same underlying gap
  (class-imbalance handling, distributional realism, etc.) rather than genuinely orthogonal
  improvements — stacking them re-corrects for something already partially fixed instead of
  compounding. Still well short of RF (mean 0.789).
- Next: given this recurring non-additivity, further combination attempts are lower-priority than
  understanding WHY structure isn't helping in the first place (the feature-enrichment and
  feature-ablation experiments already queued) — that's more likely to produce a genuinely new
  lever than continuing to combine variations on what's already been tried.

## [2026-07-21] Run 52 — Elliptic feature ablation (RF, local-only vs aggregated-only vs both): refines the RF-vs-GNN hypothesis
- Added feature_slice support to evaluation/rf_baseline.py. Elliptic's 165 columns split as
  f0-f92 (93 "local"/per-transaction columns) + f93-f164 (72 "aggregated" one-hop-neighbor columns,
  computed by the dataset's original authors). Ran RF (5 seeds each) on all three subsets.
- **Results**:
  | Feature set | Mean F1 | Mean AUC |
  |---|---|---|
  | Local only (93 cols) | 0.7618 | 0.8050 |
  | Aggregated only (72 cols) | 0.7434 | 0.8569 |
  | **All 165 (local + aggregated)** | **0.7890** | **0.8669** |
- **Refines, rather than confirms, the original "aggregated columns already contain the graph
  signal, our GNN is redundant" hypothesis**: RF on local features ALONE already beats or matches
  most GNN attempts this session (0.7618 vs GraphSAGEDiff's multi-seed mean 0.7483, plain
  GraphSAGE's 0.7335) — a humbling number on its own. Critically, neither subset alone comes close
  to the combined 0.789 — local and aggregated features are COMPLEMENTARY, not redundant, so it
  isn't simply that the aggregated columns make our GNN's message-passing pointless. Also notable:
  aggregated-only carries strong AUC (0.857, close to the full 0.867) but weak F1 (0.743) — good
  for ranking, weaker for the precision/recall balance at a fixed threshold; local-only shows the
  opposite pattern (weaker AUC, comparatively stronger F1).
- **New interpretation**: RF's advantage may come less from "the graph doesn't matter" and more
  from RF being much better at modeling NONLINEAR INTERACTIONS between local and aggregated (i.e.
  graph-derived) features than our GNN's classifier head is — a single linear layer on top of
  GraphSAGE embeddings is a much weaker function class for this than an ensemble of decision trees.
- Next: the natural synthesis, not yet tried — train GraphSAGE, extract its learned node
  embeddings, concatenate with raw features, and feed that into RF/XGBoost instead of a linear
  layer. This combines the GNN's structural encoding with RF's proven strength at feature
  interactions, rather than treating them as competing approaches.

## [2026-07-21] Run 53 — GNN-embeddings + RF hybrid: promising, second-best F1 yet, still short of RF, and a real AUC/F1 divergence
- Implemented evaluation/gnn_rf_hybrid.py (new embed() methods on GraphSAGE/GraphSAGEDiff,
  handler.py dispatch via config.hybrid.enabled) — trains the GNN normally, concatenates its
  learned node embeddings with raw features, feeds that into RandomForestClassifier instead of
  the GNN's own linear classifier head. Caught and fixed a real device-mismatch bug on first
  attempt (data.edge_index needed a just-in-time .to(device) before embed(), matching
  train_gnn.py's own memory-conscious pattern — missed replicating it here).
- Config: configs/elliptic_gnn_rf_hybrid.yaml — GraphSAGEDiff (no diffusion augmentation) + RF.
- **Result: Test F1=0.7745, AUC=0.8030.**
  | Config | Test F1 | Test AUC |
  |---|---|---|
  | GraphSAGEDiff alone, multi-seed mean (Run 50) | 0.7483 | — |
  | Best plain diffusion, single-seed (Run 40, likely inflated per Run 45) | 0.7749 | 0.8465 |
  | **GNN embeddings + RF hybrid (this run, single-seed)** | **0.7745** | **0.8030** |
  | Random Forest alone, multi-seed mean (Run 47) | 0.7890 | 0.8669 |
- **Second-best F1 of the entire Elliptic investigation, achieved WITHOUT diffusion augmentation
  at all** — a strong standalone lead. Still short of RF's more reliably-established mean (0.789),
  but notably: this used NO diffusion, so combining the hybrid with the best diffusion recipe is
  still untested and could plausibly close more of the gap.
- **Real, noteworthy divergence**: F1 improved substantially (vs GraphSAGEDiff alone) while AUC
  dropped (0.803 vs plain GraphSAGE's 0.842, vs RF's 0.867) — RF's probability calibration/ranking
  behavior over the concatenated [raw+embedding] feature space differs from a linear classifier's;
  improving the threshold-0.5 decision doesn't necessarily mean improving the ranking across all
  thresholds. Precision_fraud is notably high (0.908) at the cost of recall_fraud (0.409) — a
  different point on the precision/recall tradeoff than RF-alone's own operating point.
- Given this session's repeated single-seed-noise lesson, this result (single seed=42) needs
  multi-seed confirmation before being trusted as a real improvement over GraphSAGEDiff alone —
  but it's the most theoretically-motivated experiment run this session (directly targets Run 52's
  "RF's advantage is about feature-interaction modeling, not just structure" hypothesis) and the
  closest any GNN-based approach has come to RF.
- Next: (a) multi-seed this hybrid to confirm; (b) combine with the best diffusion recipe
  (untested); (c) investigate the AUC/F1 divergence — check RF's predicted-probability
  distribution/calibration on the hybrid features vs raw-only.

## [2026-07-21] Run 54 — Hybrid + diffusion: fourth instance of non-additive combination this session
- Config: configs/elliptic_gnn_rf_hybrid_diffusion.yaml — GNN+RF hybrid (Run 53) combined with the
  best diffusion recipe (alpha=0.75, n_synthetic=1731, ddim_steps=100).
- **Result: Test F1=0.7617, AUC=0.8174** — vs the hybrid alone (Run 53: F1=0.7745, AUC=0.8030),
  F1 got WORSE while AUC got BETTER — the opposite direction of divergence from Run 53's own
  GraphSAGEDiff-alone-to-hybrid transition (where F1 improved and AUC dropped). Neither metric
  moves consistently with these interventions; each seems to shift the model's calibration/
  threshold behavior in its own direction rather than there being a clean "better in both" combo.
- **This is the FOURTH instance this session of two independently-reasonable techniques failing
  to compound when combined**: alpha+diffusion (original Run 32/33 arc), adversarial-loss+
  ddim_steps (Run 42), GraphSAGEDiff+diffusion (Run 51), and now hybrid+diffusion. At this point
  it's a well-established, repeatable pattern for this project rather than coincidence — diffusion
  augmentation specifically seems to interact negatively or neutrally with every other intervention
  tried on top of it, on Elliptic. Worth treating as a standing prior: test new architecture ideas
  WITHOUT diffusion first, and only try adding diffusion on top once an architecture idea is
  independently confirmed, rather than assuming it'll compound.
- Current standing best (no diffusion needed): GNN+RF hybrid, single-seed F1=0.7745 (Run 53),
  multi-seed confirmation in progress. Still short of RF alone's confirmed mean (0.789).
- Next: multi-seed hybrid-alone result (in progress) is the one to trust; deprioritize further
  hybrid+diffusion combination attempts given this now-4x-repeated non-additivity pattern.

## [2026-07-21] Run 55 — Multi-seed GNN+RF hybrid confirmation: BEST result of the entire Elliptic investigation
- Config: infra/multi_seed.py run(), configs/elliptic_gnn_rf_hybrid.yaml (GraphSAGEDiff embeddings
  + raw features -> RandomForestClassifier, no diffusion), seeds 0-4.
- **Results**: F1 = [0.7708, 0.7673, 0.7600, 0.7587, 0.7783], **mean=0.7670, std=0.0081**.
- **Multi-seed mean comparison across everything tried on Elliptic this session**:
  | Approach | Mean F1 | Gap to RF |
  |---|---|---|
  | Plain GraphSAGE (Run 50) | 0.7335 | 0.0555 |
  | GraphSAGEDiff (Run 50) | 0.7483 | 0.0407 |
  | **GNN+RF hybrid (this run)** | **0.7670** | **0.0220** |
  | Random Forest alone (Run 47) | 0.7890 | — |
- **This closes most of the RF gap** (from 0.0555 down to 0.0220 — roughly 60% of the gap closed)
  and is confirmed across 5 seeds with variance (std=0.0081) comparable to GraphSAGEDiff's own
  (0.0078) — not a fluke. This directly validates Run 52's hypothesis: RF's advantage over plain
  GNN approaches is substantially about modeling nonlinear feature interactions (which RF does via
  tree ensembles and a linear GNN classifier head doesn't), not simply "graph structure is useless
  on this dataset" — once RF gets to use BOTH the GNN's structural embeddings AND do its own
  interaction modeling on top of them, most (though not quite all) of the gap disappears.
- **This is the best-supported, most significant finding of the entire Elliptic investigation** —
  more so than any single diffusion/architecture tweak, and unlike most of this session's other
  leads, already confirmed at the multi-seed level rather than resting on a single favorable seed.
- Next: (a) does diffusion augmentation help THIS hybrid when properly isolated (Run 54's single-
  seed combo test showed it doesn't, but that's not yet multi-seeded either — worth one proper
  multi-seed check before fully closing the door, given how much single-seed noise has misled
  conclusions this session); (b) try XGBoost/LightGBM instead of RF as the hybrid's final
  classifier — gradient boosting sometimes captures feature interactions even better than RF's
  bagged trees; (c) update CLAUDE.md with this as the headline Elliptic result.

## [2026-07-21] Run 56 — Pure-GNN test (stronger classifier head + deeper message passing): flat, does not close the RF gap
- Motivation: user pushback — "I believe in GNN, why can't we beat this" — asking whether a
  stronger classifier (MLP head, matching the nonlinear-interaction power RF gets in Run 53/55)
  plus more message-passing hops (deeper receptive field) can close the gap WITHOUT relying on RF,
  staying entirely inside the GNN family.
- Config: configs/elliptic_graphsage_diff_mlp3layer.yaml — GraphSAGEDiff, num_layers=3 (vs 2),
  classifier_hidden_dim=64 (MLP head: Linear->ReLU->Dropout->Linear, vs plain nn.Linear). Compared
  via the new infra/multi_seed.py `compare` + `format_report()` wrapper (5 seeds, paired Wilcoxon).
- **Result**:
  | Config | Mean F1 | Std | n |
  |---|---|---|---|
  | GraphSAGEDiff, 2-layer, linear head (Run 50) | 0.7483 | 0.0050 | 5 |
  | GraphSAGEDiff, 3-layer, MLP head (this run) | 0.7504 | 0.0108 | 5 |

  Wilcoxon signed-rank: statistic=7.00, **p=1.0000**, n_pairs=5.
- **No real effect** — the +0.0021 mean bump is fully inside noise (p=1.0 is as flat as this test
  gets), and variance actually roughly doubled (std 0.0050 -> 0.0108), i.e. the change made results
  less stable, not better. Note this run confounds two changes at once (deeper layers AND MLP head
  together) — since neither shows any effect, no follow-up ablation to separate them is warranted.
- **This is real evidence, not just a hunch, on why "more layers" doesn't trivially help here**:
  each additional SAGEConv layer expands a node's receptive field, but Elliptic's fraud-touching
  edges are only 41% homophilic (Run 46) — a minority signal inside a much larger legit-dominated
  neighborhood. Going deeper averages over an even bigger, even-more-legit-dominated neighborhood,
  which plausibly cancels out any benefit from reaching more distant fraud-cluster paths (classic
  GNN over-smoothing, Li et al. 2018). The MLP head alone doesn't help either — reinforcing Run 52's
  finding that RF's edge isn't just "nonlinear classifier," it's tree-based feature-interaction
  modeling specifically applied over BOTH raw and structural features together (which the hybrid,
  not a bigger head on the GNN's own single embedding, is what actually captures).
- **Conclusion**: architectural depth/capacity increases alone, within the plain-GraphSAGEDiff
  family, are a dead end for closing the RF gap. The GNN+RF hybrid (Run 55, mean F1=0.7670) remains
  the best-performing approach and the right lever to keep pulling, not deeper/wider pure-GNN
  variants.
- Next: node/edge feature-encoder MLPs (separate from the classifier head, applied to raw features
  BEFORE message passing) — different mechanism than this run's classifier-side MLP, worth testing
  independently before writing off "more MLP" entirely.

## [2026-07-21] Run 57 — Input-side feature-encoder MLP: consistently WORSE, not just noise
- Motivation: Run 56 tested a classifier-side (output) MLP and found no effect. This tests a
  different mechanism — an MLP encoder applied to the raw 165-dim features BEFORE the self-vs-
  neighbor deviation feature is computed (model.feature_encoder_hidden_dim=64), so the network
  gets to learn nonlinear raw-feature combinations upstream of structural aggregation, closer to
  what RF does directly on raw features.
- Config: configs/elliptic_graphsage_diff_encoder.yaml vs the confirmed
  elliptic_full_graphsage_diff.yaml baseline (2-layer, linear head, no encoder). 5-seed compare via
  infra/multi_seed.py.
- **Result**:
  | Config | Mean F1 | Std | n |
  |---|---|---|---|
  | GraphSAGEDiff baseline (no encoder) | 0.7471 | 0.0060 | 5 |
  | GraphSAGEDiff + feature-encoder MLP (this run) | 0.7231 | 0.0163 | 5 |

  Wilcoxon signed-rank: statistic=0.00, **p=0.0625**, n_pairs=5.
- **statistic=0.00 means the encoder was worse in every single one of the 5 paired seeds** — a
  perfectly consistent direction, not noise like Run 56's p=1.0 result. p=0.0625 is the minimum
  achievable two-sided p-value at n=5 pairs — with even one more seed confirming the same
  direction this would already cross conventional significance. Variance also nearly tripled
  (std 0.0060 -> 0.0163), i.e. the encoder makes results both worse AND less stable.
  Baseline's mean here (0.7471) is a fresh 5-seed draw and lands close to Run 50's 0.7483 —
  consistent with normal run-to-run noise, not a baseline drift.
- **Working hypothesis for why it hurts**: compressing 165 raw features down to a 64-dim learned
  representation BEFORE computing the self-vs-neighbor deviation feature likely destroys some of
  the fine-grained, per-original-feature camouflage signal that GraphSAGEDiff's deviation feature
  (Run 49) relies on — the deviation is more informative computed directly on the original 165
  interpretable features (e.g. specific balance/amount statistics) than on a lossy nonlinear
  compression of them. This is the opposite of Run 56's classifier-head MLP (applied downstream of
  aggregation, where it had zero effect either way) — where in the pipeline you add capacity
  matters, not just whether you add it.
- **Conclusion**: input-side feature encoding, at least applied before the deviation computation,
  is a real (if not-yet-fully-significant) negative — do not pursue further in this position.
  GNN+RF hybrid (Run 55, mean F1=0.7670) remains the best-performing lever.
- Next: GraphSAGEGated (CARE-GNN/PC-GNN-inspired camouflage-resistant neighbor gating) — a fraud-
  specific architecture that changes WHAT gets aggregated rather than adding capacity anywhere in
  the pipeline (encoder or classifier side), directly targeting the Run 46 homophily-dilution root
  cause; currently dispatched via the new infra/campaign.py auto-compare-and-promote runner.

## [2026-07-21] Run 58 — RF error analysis: missed fraud (FN) has HIGHER same-class neighbor density than caught fraud (TP)
- Motivation: instead of another architecture guess, directly ask which test samples RF (our
  best classifier, structure-blind) gets wrong, and whether the graph structure it never sees would
  have explained those specific errors — a data-driven test of whether structure is really the
  missing piece, rather than assuming it from the aggregate homophily number alone (Run 46).
- New: evaluation/error_analysis.py — trains the same RF as evaluation/rf_baseline.py locally
  (~1s, no GPU), then for every test node computes degree, count of KNOWN-labeled neighbors, and
  fraction of those known neighbors that are fraud, and breaks down TP/FN/TN/FP by these stats.
- **Result** (Elliptic test split, RF seed=42):
  | Group | n | mean frac_fraud_neighbors | mean known_neighbors | isolated (0 known nbrs) |
  |---|---|---|---|---|
  | TP (fraud, caught) | 194 | 0.193 | 0.567 | 59.8% |
  | **FN (fraud, MISSED)** | 214 | **0.283** | **1.023** | 24.8% |
  | TN (legit, cleared) | 8384 | 0.004 | 1.674 | 26.3% |
  | FP (legit, false alarm) | 49 | 0.000 | 1.490 | 4.1% |
- **The fraud RF catches (TP) and the fraud RF misses (FN) look structurally different, in the
  opposite direction from a naive guess**: FN nodes are LESS isolated (only 25% have zero known
  neighbors, vs 60% for TP) and have a HIGHER fraction of same-class (fraud) neighbors (0.283 vs
  0.193) than the fraud RF successfully catches. In other words: RF's hits are disproportionately
  the "obviously fraud from features alone, often isolated" cases; RF's misses are
  disproportionately better-connected fraud nodes sitting in elevated-fraud-density neighborhoods
  — exactly the population a structure-aware model (not a raw-feature-only one) should have a real,
  specific shot at recovering, since RF has literally zero access to this signal.
- **FP (false alarms) show the opposite pattern — zero structural signal at all** (frac_fraud_
  neighbors=0.0, matching TN's near-zero baseline): RF's false alarms are driven entirely by raw-
  feature ambiguity, not misleading neighborhoods. Structure-aware models are unlikely to fix these
  specific errors (nothing in the neighborhood points away from the false flag either).
- **This gives a concrete, falsifiable prediction for GraphSAGEGated and any other structure-aware
  candidate**: if the architecture is doing what it's supposed to, it should disproportionately
  recover FN-profile nodes (well-connected, elevated same-class neighbor fraction) rather than
  moving performance uniformly across all fraud nodes. Worth checking directly once a structure-
  aware model's per-node test predictions are available — not just the aggregate F1/AUC delta.
- Caveat: Elliptic is very sparse overall (mean known_neighbors even for the "well-connected" FN
  group is ~1.0), so this is a modest, not overwhelming, structural signal — consistent with why
  plain mean-aggregation (which dilutes this modest signal across a 59%-non-fraud neighborhood
  on average, Run 46) hasn't been enough on its own, and why a SELECTIVE (not just deeper/bigger)
  mechanism is the more promising direction.
- Next: once GraphSAGEGated's multi-seed result lands, cross-reference its per-node test errors
  against this same FN/TP breakdown to see if it specifically recovers the FN profile.

## [2026-07-21] Run 59 — GraphSAGEGated (CARE-GNN/PC-GNN-inspired camouflage gate): flat/no effect
- Config: configs/elliptic_full_graphsage_gated.yaml vs the confirmed elliptic_full_graphsage_diff.yaml
  baseline. 5-seed compare, dispatched via the new infra/campaign.py auto-compare-and-promote runner
  (first real use of it) — required an infra fix mid-run: RunPod was routing jobs to long-lived
  worker processes that had the OLD build_model() (missing the 'graphsage_gated' branch) already
  imported in memory; a per-job `git pull` (already in handler.py) updates files on disk but can't
  reload already-imported Python modules — a documented limitation, not a new bug. Fixed by
  repointing the endpoint at a freshly created template (forces RunPod to recycle workers), verified
  via a direct single-seed test reporting the correct worker_git_commit before re-dispatching the
  real comparison.
- **Result**:
  | Config | Mean F1 | Std | n |
  |---|---|---|---|
  | GraphSAGEDiff baseline | 0.7488 | 0.0068 | 5 |
  | GraphSAGEDiff + learned neighbor gate (this run) | 0.7453 | 0.0096 | 5 |

  Wilcoxon signed-rank: statistic=6.00, **p=0.8125**, n_pairs=5. Campaign auto-decision: baseline
  kept, candidate not promoted.
- **No real effect, if anything slightly worse** — essentially the same flat-noise signature as
  Run 56 (MLP head) rather than Run 57's consistent-direction regression. A single-seed=42 sanity
  test run during the infra-fix verification (F1=0.7617) looked promising in isolation — another
  instance of this session's now well-established single-seed-noise trap (Run 45/50/56/57 all
  warned about this; glad this wasn't reported as a result on its own).
- **Why the "lightweight CARE-GNN" idea likely didn't work**: the gate MLP (Run 59's
  GraphSAGEGated.gate_mlp) is trained purely end-to-end via the downstream focal-loss gradient — it
  has no direct supervision telling it "this neighbor is actually same-class, trust it more." It can
  only use FEATURE similarity between src/dst as a proxy for LABEL similarity. But Elliptic's fraud
  is camouflaged specifically to look similar in feature space too (that's the whole detection
  difficulty) — so a feature-similarity proxy may simply not track true same-class similarity well
  enough to produce a useful gate. This is plausibly exactly why CARE-GNN/PC-GNN use RL-tuned or
  explicitly label-supervised similarity scoring rather than a plain differentiable MLP — the
  "extra" complexity in those papers may not be incidental, it may be doing necessary work that this
  simplified version skipped.
- **A more faithful next attempt** (not yet built): add an auxiliary loss that explicitly supervises
  the gate — e.g. binary cross-entropy between the gate's output and y_src==y_dst on TRAIN-labeled
  edges only (no leakage), so the gate is trained to predict actual same-class-ness directly, not
  hoping the classification loss discovers it indirectly through a much noisier gradient path.
- **Conclusion**: current standing leaderboard on Elliptic is unchanged — GNN+RF hybrid (Run 55,
  mean F1=0.7670) remains the best-performing approach; every pure-GNN architecture variant tried
  since (MLP head/deeper layers, feature encoder, similarity gate) has been flat or negative.
- Next: (a) label-supervised gate auxiliary loss, if worth the added complexity; (b) cross-reference
  against Run 58's FN/TP structural profile once/if a promising structure-aware model exists to
  check.

## [2026-07-21] Run 60 — Cross-referencing GraphSAGEGated against Run 58's FN/TP profile: prediction REFUTED
- Directly tests Run 58's falsifiable prediction: if GraphSAGEGated's neighbor gating works as
  intended, it should disproportionately recover the specific fraud nodes RF misses (better-
  connected, elevated same-class-neighbor density), not just move F1 uniformly.
- New: training/train_gnn.py's run_from_config() gained an opt-in `debug.return_test_predictions`
  config flag that includes raw per-node test probs/labels in the job's returned output (off by
  default, no effect on normal runs) — needed to get node-level predictions out of a RunPod job,
  not just aggregate metrics. Dispatched configs/elliptic_graphsage_gated_debug_predictions.yaml
  (seed=0), then cross-referenced against a locally-reproduced RF fit (same seed=42 as Run 58) on
  the exact same test_mask ordering (verified identical y_true arrays before comparing).
- **Result — the prediction is REFUTED, decisively**:
  | | count | outcome |
  |---|---|---|
  | RF's FN (missed fraud), recovered by GraphSAGEGated | 4 / 214 (1.9%) | essentially none |
  | RF's FN, still missed by GraphSAGEGated | 210 / 214 (98.1%) | |
  | RF's TP (caught fraud), also caught by GraphSAGEGated | 156 / 194 (80.4%) | |
  | **RF's TP, LOST by GraphSAGEGated (new miss)** | **38 / 194 (19.6%)** | net harmful |
  | RF's FP (false alarms), also made by GraphSAGEGated | 3 / 49 | roughly consistent, no change |

  Of the tiny handful (4) it did recover, mean frac_fraud_neighbors=0.5 — actually consistent with
  the hypothesis direction — but this is far too small a count to be anything but noise, and it's
  swamped by losing 38 nodes RF already got right using features alone.
- **This is the real reason Run 59's aggregate F1 was flat, not an accident**: the gate mechanism
  isn't recovering a different-but-overlapping set of frauds (which would show up as a wash even
  with real structural benefit) — it's net LOSING ground on the feature-obvious cases (that a plain
  linear/RF classifier already nails) while providing no real structural payoff. This is a stronger,
  more specific negative result than the aggregate metric alone: it directly falsifies the working
  hypothesis behind this architecture, rather than leaving it ambiguous.
- **Reinforces Run 59's diagnosis**: an end-to-end-learned gate with no explicit label-similarity
  supervision isn't picking up the real (if modest, per Run 58's own caveat) same-class-density
  signal — it's most likely converging to something close to noise/near-uniform weighting, which
  would explain both the near-zero FN recovery AND the collateral TP losses (a noisy gate can easily
  down-weight a helpful neighbor for an otherwise easy case just as readily as it fails to
  up-weight one for a hard case).
- **Conclusion**: do not continue refining differentiable-similarity-proxy gating without adding
  explicit label supervision to the gate itself (Run 59's proposed auxiliary BCE-on-known-edges
  loss) — this implementation as-is is a clear negative, not an ambiguous one. GNN+RF hybrid
  (Run 55, mean F1=0.7670) remains the best, and now most differentially validated, approach on
  Elliptic.
- Next: decide whether the label-supervised gate variant is worth building (meaningfully more
  complex — needs a same-class auxiliary loss term, training-edge sampling for it, and its own
  ablation) versus redirecting effort toward the feature-enrichment work already queued, or toward
  IEEE-CIS (a different, real dataset) per the project's standing priority on real data over
  further Elliptic tuning.

## [2026-07-21] Run 61 — Label-supervised gate auxiliary loss: WORSE, not better
- Implements Run 59/60's proposed fix: models/gnn/graphsage.py's GraphSAGEGated gained
  `gate_logits()` (raw pre-sigmoid gate score, exposed separately from embed() to avoid changing
  its tensor-only contract used by the hybrid), and training/train_gnn.py's training loop gained
  an optional `loss.gate_auxiliary_weight` term: BCE between the gate and y_src==y_dst on known-
  labeled TRAIN edges only (no leakage). Config: elliptic_full_graphsage_gated_supervised.yaml,
  gate_auxiliary_weight=0.5, otherwise identical to elliptic_full_graphsage_gated.yaml. 5 seeds,
  reusing Run 59's already-collected baseline/plain-gate per-seed values (no need to re-pay for
  them) and computing the paired Wilcoxon locally.
- **Result**:
  | Config | Mean F1 | Std |
  |---|---|---|
  | GraphSAGEDiff baseline (Run 59) | 0.7488 | 0.0068 |
  | GraphSAGEGated, unsupervised gate (Run 59) | 0.7453 | 0.0096 |
  | **GraphSAGEGated, label-supervised gate (this run)** | **0.7375** | **0.0116** |

  vs unsupervised gate: Wilcoxon statistic=1.0, p=0.125 (4/5 seeds worse). vs baseline: statistic=2.0,
  p=0.1875. Neither fully significant, but consistently in the WRONG direction — supervision made
  it worse, not better.
- **Working hypothesis for why**: the auxiliary task (predict y_src==y_dst) is trivially easy on
  Elliptic's class-imbalanced graph — legit vastly outnumbers fraud, so the large majority of edges
  are legit-legit, and "always predict same-class" is correct most of the time by default without
  the auxiliary loss being class-balanced (no pos_weight/focal-style reweighting was applied, unlike
  the main classification loss's own alpha=0.25). The gate likely took this cheap shortcut —
  converging toward near-uniform "always similar" weighting — rather than learning the specific
  minority signal that actually matters (correctly identifying rare fraud-fraud edges and rare
  heterophilic fraud-legit edges). That would also explain why it landed close to, but below, the
  UNSUPERVISED gate's result rather than clearly diverging from it: pulling toward "same-class by
  default" partially undoes GraphSAGEGated's whole mechanism, moving it toward (but with extra
  optimization noise from the competing objective, slightly below) plain unweighted aggregation.
- **Conclusion**: this specific fix does not work as implemented; a properly class-balanced version
  of the auxiliary loss might still be worth trying in principle, but per the 2026-07-21 discussion
  plan (fix the gate first — quick — then try a genuinely different mechanism if that doesn't pan
  out), this closes out the "patch the existing gate" direction for now rather than continuing to
  iterate on it further. GNN+RF hybrid (Run 55, mean F1=0.7670) remains the best Elliptic result.
- Next: BWGNN/GHRN-style spectral (multi-scale graph filter) approach — a fundamentally different,
  established mechanism for exactly this camouflage/heterophily problem, rather than another
  attention-gate variant.

## [2026-07-21] Run 62 — GraphSAGESpectral (simplified BWGNN/GHRN multi-band filter): clear negative, likely a fixable implementation gap
- Config: configs/elliptic_full_graphsage_spectral.yaml (num_bands=3: h0-h1, h1-h2, h2-h3, plus
  final smoothed h3) vs the confirmed elliptic_full_graphsage_diff.yaml baseline. 5-seed compare
  via infra/campaign.py. Verified fresh worker code (worker_git_commit matched HEAD) before
  dispatching, avoiding the earlier stale-worker trap.
- **Result**:
  | Config | Mean F1 | Std |
  |---|---|---|
  | GraphSAGEDiff baseline | 0.7473 | 0.0069 |
  | **GraphSAGESpectral (this run)** | **0.7211** | **0.0069** |

  Wilcoxon statistic=0.00, **p=0.0625** (minimum possible at n=5) — worse in **5/5 seeds**, the same
  signature as Run 57's feature-encoder regression (real, consistent effect, not noise).
- **Likely root cause — a real implementation gap, not evidence against spectral filtering as an
  idea**: GraphSAGEDiff explicitly keeps raw `x` itself as one of its three concatenated input
  terms (`[x, neighbor_mean, x-neighbor_mean]`) — the model always has direct access to the
  untouched original 165 features. GraphSAGESpectral's bands (`x-h1`, `h1-h2`, `h2-h3`, `h3`) never
  include raw `x` itself — only differences and progressively smoothed versions of it. Since RF's
  advantage partly comes from exploiting raw feature VALUES directly (specific thresholds in the
  original columns, not just graph-relative quantities), stripping direct access to `x` before the
  first layer plausibly threw away exactly the kind of signal that made GraphSAGEDiff's "keep x
  as-is" choice valuable in the first place.
- **Conclusion**: inconclusive on whether multi-scale spectral filtering itself is a dead end for
  this problem — the negative result is confounded with this specific design gap. Not re-testing
  immediately (redirecting toward IEEE-CIS per project priority — see below), but if spectral
  filtering is revisited, the fix (add raw `x` as its own band alongside the differences) is
  cheap and should be tried before concluding the mechanism itself doesn't work.
- **Standing Elliptic leaderboard unchanged**: GNN+RF hybrid (Run 55, mean F1=0.7670) remains the
  best result; every pure-GNN architecture variant tried this session (GAT, deeper layers/bigger
  head, feature encoder, learned gate with/without supervision, spectral filter) has been flat or
  negative. Combined with external validation (arXiv 2604.19514) and this session's own repeated,
  differently-diagnosed negative results, treating RF's edge on Elliptic as a genuine, dataset-
  specific finding (sparse graph + already feature-engineered tabular columns) rather than a bug.
- Next: pivot to a third, real dataset (IEEE-CIS) per the project's standing priority on real data
  over further Elliptic tuning — see below.

## [2026-07-21] Run 63 — IEEE-CIS Fraud Detection: third dataset, pipeline built, honest first baseline
- New dataset: IEEE-CIS Fraud Detection (real e-commerce transactions, ~590k rows, ~3.5% fraud) --
  the third dataset per the project's standing priority on real data. Unlike Elliptic (native
  graph) or PaySim (simulated), this is real transaction data with NO native graph -- structure had
  to be constructed, same philosophy as PaySim's account-sharing edges.
- The real competition dataset requires accepting Kaggle competition rules via browser (401 via
  API without it) -- used a community-uploaded plain-dataset mirror (lnasiri007/ieeecis-fraud-
  detection) with an IDENTICAL file structure (train_transaction.csv/train_identity.csv) instead,
  confirmed via file listing before committing to it (data/download.py's
  ensure_downloaded_ieee_cis()).
- New: data/ieee_cis_preprocess.py -- transaction-as-node graph, edges from shared card1/addr1
  entity keys (proxies for "same underlying account," since there's no direct user ID), capped at
  max_node_degree per key (same construction as PaySim's build_edges). Temporal split by
  TransactionDT. Feature scope deliberately limited for this first slice: TransactionAmt, ProductCD,
  card1-6, addr1/2, dist1/2, P/R_emaildomain, C1-14, D1-15, M1-9, DeviceType/DeviceInfo --
  SKIPPING the 339 anonymized V-columns (heavy missingness, would ~triple feature count), matching
  this project's "MVP slice first" convention. High-cardinality categorical/ID columns are
  frequency-encoded (count in TRAIN only) rather than one-hot, avoiding dimensionality blowup.
- **Caught and fixed a real bug during smoke-testing**: the stratified-subsample logic (adapted
  from data/paysim_preprocess.py's stratified_subsample) silently produced a 100%-fraud graph when
  subsample_size (5000 in the smoke test) was smaller than IEEE-CIS's real fraud count (20663) --
  `n_legit_keep = max(size - n_fraud, 0)` clamped to zero, dropping every legit row with no error.
  Added an explicit ValueError guard. PaySim's own subsample_size configs never hit this (always
  far larger than PaySim's ~8213 fraud rows), so this specific bug never manifested there, but the
  same latent assumption exists in that code too if it's ever used with a smaller subsample_size.
- **Local RF calibration baseline** (100k-row stratified subsample, matching the Elliptic/PaySim
  "cheap RF sanity check first" pattern): Test F1=0.7848, AUC=0.8921 -- sane, non-trivial numbers,
  confirms the feature/graph pipeline isn't broken before committing to an expensive full-scale GPU
  run.
- **First full-dataset GNN run** (configs/ieee_cis_full.yaml, plain GraphSAGE, alpha=0.25, full
  ~590k-node graph, RunPod): **Test F1=0.5624, AUC=0.8517**, recall_fraud=**0.077** (only 7.7% of
  real fraud caught), precision_fraud=0.915. best_epoch=1000 (hit the epoch cap, never converged/
  early-stopped).
- **Same failure signature as PaySim's own original alpha=0.25 baseline**: decent ranking ability
  (AUC=0.85) but the model is far too conservative at the default 0.5 threshold under this class
  imbalance (~3.5% fraud) -- a loss-weighting problem, not a broken pipeline or a bad graph
  construction. PaySim's alpha sweep (Runs 24-28) found monotonic improvement all the way to
  alpha=0.95 (F1 0.859->0.946) from this exact starting signature. Dispatched the same sweep here
  (loss.alpha in [0.5, 0.75, 0.9, 0.95]) before drawing any conclusion about IEEE-CIS being
  "harder" or the entity-key graph construction being weak -- results pending.
- Next: alpha sweep results; then RF baseline on the FULL dataset (not just the 100k calibration
  subsample) for a true apples-to-apples comparison point, matching Elliptic's own RF-vs-GNN
  investigation.

## [2026-07-21] Run 64 — IEEE-CIS alpha sweep: peaks at 0.75, NOT monotonic like PaySim
- configs/ieee_cis_full.yaml, loss.alpha in [0.5, 0.75, 0.9, 0.95], via infra/sweep.py, full
  ~590k-node dataset.
  | alpha | Test F1 | Test AUC | Test G-mean |
  |---|---|---|---|
  | 0.25 (Run 63, honest baseline) | 0.5624 | 0.8517 | 0.2766 |
  | 0.50 | 0.6413 | 0.8575 | 0.4321 |
  | **0.75** | **0.6832** | 0.8567 | 0.5426 |
  | 0.90 | 0.6654 | 0.8568 | 0.6914 |
  | 0.95 | 0.5920 | 0.8545 | 0.7575 |
- **Unlike PaySim's alpha sweep (Runs 24-28, monotonic all the way to 0.95), IEEE-CIS peaks at
  0.75 and gets WORSE past that** — alpha isn't a free lunch here; overweighting the fraud class
  too far past the optimum measurably hurts F1 even though AUC stays roughly flat throughout
  (0.85-0.86 across the whole sweep) and G-mean keeps climbing with alpha (0.28->0.76). This is a
  real, dataset-specific difference, not a bug: AUC-flat-while-F1-varies means alpha is mostly
  moving the DECISION THRESHOLD's precision/recall tradeoff at a fixed ranking quality, and IEEE-
  CIS's optimal tradeoff point for F1-macro specifically sits at a lower alpha than PaySim's did —
  plausibly because IEEE-CIS's fraud rate (~3.5%) and its noisier, more real-world feature/graph
  signal (vs PaySim's simulator artifacts) genuinely shift where over-correcting for imbalance
  starts costing more false positives than it buys in recall.
- alpha=0.75 (F1=0.6832) is now the standing IEEE-CIS baseline, closing much of the gap from
  alpha=0.25's F1=0.5624 but still clearly below Elliptic's GNN numbers (~0.75) and PaySim's
  tuned numbers (~0.95) — expected, given IEEE-CIS's harder, noisier real-world signal and a
  first-pass, deliberately reduced feature set (no V-columns yet).
- Next: RF baseline on the FULL dataset (not the 100k calibration subsample) for the real
  apples-to-apples comparison point; consider whether the skipped 339 V-columns are worth adding
  given F1 is still well below Elliptic/PaySim's tuned numbers.

## [2026-07-21] Run 65 — Elliptic: Node2Vec (unsupervised) + RF hybrid — another confirmed non-effect
- Different mechanism than the supervised GNN+RF hybrid (Run 55): LABEL-FREE structural embeddings
  (random-walk skip-gram, capturing community/degree/role in the graph, no classification signal
  involved at all) concatenated with raw features, fed to RF. Motivation: complementary to the
  supervised hybrid's embeddings, and much cheaper (no GPU).
- New: evaluation/node2vec_rf_hybrid.py — hand-rolled random walk (vectorized numpy, CSR adjacency)
  + skip-gram w/ negative sampling, NOT torch_geometric.nn.Node2Vec (that class requires pyg-lib,
  confirmed via `uv pip install pyg-lib` failing with "not found in the package registry" — no
  macOS wheel exists for it). First version (walks_per_node=5, walk_length=10, window=2 ->
  34.6M skip-gram pairs) was too slow locally (5.5 min for ONE epoch, exceeded the 300s
  auto-background threshold) -- killed and cut scope down (walks_per_node=2, walk_length=6,
  window=1 -> 4.1M pairs) to ~45s/run, still a meaningful structural signal at this graph size.
  Trained on train_edge_index only (same leakage-safe convention as GNN training).
- **Single-seed=42 result looked exciting**: Test F1=0.7920, AUC=0.8662 — ABOVE RF alone's own
  mean (0.7890) and above the supervised hybrid's mean (0.7670). Per this session's now
  extensively-confirmed single-seed-noise lesson, ran 5 seeds locally (cheap, ~4 min total, no
  RunPod) before trusting it.
- **5-seed result: the effect evaporates**:
  | Config | Mean F1 | Std |
  |---|---|---|
  | RF alone (fresh 5-seed local run, same seeds) | 0.7890 | 0.0025 |
  | Node2Vec + RF (this run) | 0.7893 | 0.0043 |

  Wilcoxon: statistic=6.00, p=0.8125 — no signal, essentially identical to RF alone. The single-
  seed=42 result was another (6th, after Runs 45/50/56/57/59) instance of the lucky-seed trap this
  session has now repeatedly documented.
- **This actually reinforces rather than contradicts the overall Elliptic picture**: unsupervised
  structural embeddings add nothing beyond RF's raw-feature baseline, same conclusion reached via
  the supervised hybrid, the learned gate (Run 59-61), and the spectral filter (Run 62) -- it's not
  that structure is hard to ACCESS via any particular mechanism, it's that Elliptic's graph
  genuinely doesn't carry much signal beyond what RF already extracts from raw features (small
  average degree, weak/diluted homophily — Run 46/58).
- Next: FAGCN (Frequency-Adaptive GCN, Bo et al. 2021) as a properly-established heterophily-
  specific architecture, rather than another homemade mechanism — GraphSAGESpectral's failure was
  confounded with a real implementation gap (Run 62), so this is worth trying on validated ground
  before concluding heterophily-aware architectures categorically don't help here.

## [2026-07-21] Run 66 — campaign: elliptic_fagcn vs elliptic_graphsage_diff
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_full_graphsage_diff.yaml,
  candidate=configs/elliptic_full_fagcn.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_full_graphsage_diff.yaml | 0.7484 | 0.0046 | 5 |
| configs/elliptic_full_fagcn.yaml | 0.5994 | 0.1227 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.5993854986745212 vs 0.7484380098749598, p=0.0625)
- Observations: FAGCN (Bo et al. 2021) is an established, validated heterophily-specific
  architecture (signed edge gate in [-1,1], can actively subtract a dissimilar neighbor's
  contribution rather than only ever down-weighting toward zero like GraphSAGEGated's sigmoid
  gate) -- yet this is not just "no effect," it's genuine training instability: std=0.1227 (vs
  baseline's 0.0046), individual per-seed F1 ranging 0.486-0.729, and a pre-campaign single-seed
  check showed unusually low AUC=0.7756 (every other Elliptic architecture this session has landed
  in the 0.83-0.87 AUC range). The tanh-based signed gate plausibly makes optimization harder here
  (a gate that can flip sign is a much less constrained/well-behaved function to learn than a
  sigmoid in [0,1], especially with FAGCN's default eps=0.2/2-layer setup untuned for this specific
  dataset) rather than the frequency-adaptive MECHANISM itself being wrong.
- Next: if pursuing FAGCN further, try lower learning rate / more conservative eps first to check
  whether the instability is fixable with better hyperparameters before concluding the mechanism
  doesn't work; otherwise, this closes out the third and final architecture-family attempt at
  beating RF on Elliptic (learned gate, spectral filter, FAGCN) -- GNN+RF hybrid (Run 55, mean
  F1=0.7670) stands as the best-supported result of the whole Elliptic investigation.

## [2026-07-21] Run 67 — The "hard core": 52% of test fraud is missed by BOTH RF and the best GNN, and it's a real feature-space outlier group
- Extends Run 58's RF-only error analysis by cross-referencing RF's per-node test predictions
  against GraphSAGEDiff's (via training/train_gnn.py's debug.return_test_predictions, same
  mechanism as Run 60) -- not the failed gate this time, the actual best/standing GNN result, to
  ask a sharper question: is there a genuinely IRREDUCIBLE set of hard fraud cases neither approach
  gets, and what characterizes it?
- **Result** (408 real fraud nodes in the Elliptic test split):
  | | n | % of all fraud |
  |---|---|---|
  | RF misses | 214 | 52.5% |
  | GraphSAGEDiff misses | 265 | 65.0% |
  | **Missed by BOTH ("hard core")** | **213** | **52.2%** |
  | GraphSAGEDiff recovers what RF misses | 1 | 0.2% |
  | RF recovers what GraphSAGEDiff misses | 52 | 12.7% |
- **GraphSAGEDiff (the actual best GNN, not the failed gate) still only recovers 1 of RF's 214
  misses** -- confirms, with the best real architecture rather than a failed one (Run 60's same
  check on GraphSAGEGated found 4/214), that nothing built this session genuinely reaches past
  RF's blind spot on Elliptic. The "hard core" IS effectively RF's own FN set (213 of 214).
- **New finding #1 -- the hard core is a real feature-space outlier group, not just a structural
  one**: mean L2 distance from the centroid of fraud BOTH methods correctly catch is **9.81** for
  the hard core vs **3.16** for correctly-caught fraud (~3x farther). This isn't explained by
  Run 46/58's structural/homophily story alone -- these look like a qualitatively different kind of
  fraud in raw feature space, plausibly an underrepresented pattern in training (consistent with
  temporal drift: fraud tactics evolving between the train and test periods, a well-documented real
  challenge, not specific to our pipeline).
- **New finding #2 -- GraphSAGEDiff's own EXTRA mistakes (beyond RF) concentrate on isolated
  nodes**: the 52 cases RF catches but GraphSAGEDiff doesn't are 46.2% isolated (zero known
  neighbors) vs only 24.9% for the hard core. This is a clean, specific mechanism for PART of the
  RF-GNN gap: for isolated fraud nodes, structural aggregation has nothing extra to add and may
  actively dilute an otherwise-clean feature signal that RF (feature-only) handles fine.
- **Practical implication**: any future architecture change is very unlikely to close much of the
  remaining gap by trying harder at STRUCTURE (that lever is close to exhausted -- gate, spectral
  filter, FAGCN, node2vec all failed or added nothing) -- the bigger, still-open opportunities are
  (a) whatever is qualitatively different about the hard-core feature-space outliers (worth
  visualizing/characterizing further, e.g. which of the 165 features differ most from the caught-
  fraud centroid), and (b) a structural ON/OFF switch so the GNN can fall back to feature-only
  behavior specifically for isolated nodes instead of always aggregating regardless of whether
  there's anything useful to aggregate.
- Next: (a) identify which specific raw features most separate the hard-core outliers from
  caught fraud; (b) consider an isolated-node bypass (skip/soften aggregation when a node has zero
  or near-zero known neighbors) as a targeted fix for finding #2, rather than another blanket
  architecture change.

## [2026-07-21] Run 68 — campaign: elliptic_graphsage_diff_dropedge_0.1 vs elliptic_graphsage_diff
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_full_graphsage_diff.yaml,
  candidate=configs/elliptic_graphsage_diff_dropedge_0.1.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_full_graphsage_diff.yaml | 0.7486 | 0.0060 | 5 |
| configs/elliptic_graphsage_diff_dropedge_0.1.yaml | 0.7494 | 0.0086 | 5 |

Wilcoxon signed-rank: statistic=5.00, p=0.6250, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.7493848739142079 vs 0.7486178447434756, p=0.625)
- Observations: DropEdge(0.1) is flat -- essentially identical to baseline (+0.0008 mean, well
  within noise), unlike Run 57's feature-encoder or Run 62's spectral filter (both consistently
  worse across all 5 seeds). A mild positive direction with no real signal yet -- worth seeing
  0.2/0.3 before concluding anything; DropEdge's typical useful range in the literature is often
  higher than 0.1 for smaller/sparser graphs.
- Next: wait for rate=0.2 and 0.3 results (same campaign, in progress) before drawing a conclusion
  on this augmentation angle.

## [2026-07-21] Run 69 — campaign: elliptic_graphsage_diff_dropedge_0.2 vs elliptic_graphsage_diff
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_full_graphsage_diff.yaml,
  candidate=configs/elliptic_graphsage_diff_dropedge_0.2.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_full_graphsage_diff.yaml | 0.7478 | 0.0047 | 5 |
| configs/elliptic_graphsage_diff_dropedge_0.2.yaml | 0.7505 | 0.0095 | 5 |

Wilcoxon signed-rank: statistic=4.00, p=0.4375, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.7505315844692877 vs 0.7478169340546867, p=0.4375)
- Observations: Same flat pattern as rate=0.1 (Run 68) -- a small positive nudge (+0.0027 mean),
  not significant, no clear trend yet across 0.1/0.2. DropEdge doesn't appear to be a strong lever
  on this graph either way so far.
- Next: rate=0.3 in progress (same campaign); if it's also flat, DropEdge closes out as another
  non-effect on Elliptic, joining the other structure-perturbation attempts this session.

## [2026-07-21] Run 70 — The hard core is out-of-distribution relative to TRAIN fraud itself, not just hard to classify -- a real ceiling on diffusion augmentation
- Motivated by a direct question: is graph-structure-aware diffusion (GRAD/MiDi-style, generating
  whole realistic synthetic fraud subgraphs instead of TabDDPM's feature-only + heuristic
  attachment) worth building? This tests the load-bearing assumption directly instead of guessing:
  diffusion models sample FROM their training distribution -- if Run 67's "hard core" (52% of test
  fraud missed by both RF and GraphSAGEDiff) is itself outside where TRAIN fraud lives in feature
  space, no amount of better synthetic-sample REALISM (structural or otherwise) can help, since the
  generator would still only ever sample things resembling what it was trained on.
- **Result**: distance from the TRAIN fraud feature centroid --
  | Group | mean L2 distance | n |
  |---|---|---|
  | Train fraud's own self-distance (typical spread) | 5.53 (std 3.54) | -- |
  | Correctly-caught test fraud (both RF+GNN) | 6.98 | 142 |
  | **Hard core (missed by both)** | **11.30 (~2x typical train-fraud spread)** | 213 |
- **This is a real, load-bearing finding, not just a restatement of Run 67**: the hard core isn't
  merely "hard to classify" or "structurally different" (Run 67's earlier framing) -- it sits
  roughly TWICE as far from the train-fraud centroid as train fraud typically spreads from itself.
  This looks like genuine concept drift (fraud tactics in the test period genuinely different from
  anything in the training period) rather than a modeling or structural-realism gap.
- **Direct implication for the diffusion-augmentation research direction**: TabDDPM (or any
  fancier graph-aware diffusion, GRAD/MiDi-style) is trained on real TRAIN fraud features and, by
  construction, generates synthetic samples resembling that same distribution. It cannot generate
  examples resembling out-of-distribution hard-core patterns almost by definition -- diffusion
  models interpolate/densify their training distribution, they don't extrapolate beyond it. This
  means a large engineering investment in graph-structure diffusion (discrete diffusion over
  adjacency + continuous over features) is UNLIKELY to move the metric it would be justified by,
  given the biggest single error category (52% of missed fraud) is specifically the category this
  entire approach-family structurally cannot address.
- **This doesn't mean diffusion augmentation was pointless overall** -- it's a real, separate
  finding about THIS SPECIFIC remaining gap, not a retraction of the original validated diffusion
  win (Run 29 first showed a genuine +0.04 F1 improvement over the honest baseline). It specifically
  caps how much further pushing on synthetic-sample REALISM (vs. e.g. the alpha/architecture levers
  already explored) can close the RF gap.
- **Practical redirect**: if the goal is closing more of the RF gap, the evidence this session
  points toward directions that don't require synthesizing out-of-distribution examples --
  (a) better exploiting genuinely in-distribution signal the GNN still misses (the GNN+RF hybrid,
  Run 55, remains the best lever); (b) accepting that ~52% of fraud may be structurally
  undetectable from THIS train period's data by ANY model/method until concept drift is addressed
  with fresher labeled data, which is a data/labeling problem, not a modeling one.
- Next: if pursuing this further, check whether the hard core clusters into a FEW distinct novel
  patterns (vs. being uniformly diffuse) -- a small number of emerging fraud archetypes would be a
  very different, more actionable finding than uniformly-scattered novelty.

## [2026-07-21] Run 71 — campaign: elliptic_graphsage_diff_dropedge_0.3 vs elliptic_graphsage_diff
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_full_graphsage_diff.yaml,
  candidate=configs/elliptic_graphsage_diff_dropedge_0.3.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_full_graphsage_diff.yaml | 0.7486 | 0.0055 | 5 |
| configs/elliptic_graphsage_diff_dropedge_0.3.yaml | 0.7594 | 0.0086 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.7594478416201158 vs 0.7486097775555, p=0.0625)
- Observations: statistic=0.00 means DropEdge(0.3) won on ALL 5 seeds -- the maximum possible
  significance signature at n=5 (same as several other confirmed-real effects this session), and
  there's a clear ascending trend across the whole DropEdge sweep: 0.1->0.7494 (Run 68), 0.2->0.7505
  (Run 69), 0.3->0.7594 (this run). This is the most promising lead of the whole DropEdge campaign
  and arguably the most promising pure-architecture result since GraphSAGEDiff itself. p=0.0625 is
  the statistical FLOOR at n=5 pairs, not weak evidence -- more seeds are needed to actually resolve
  significance, not more seeds at the SAME n.
- Next: extend to 10 seeds (5 new + the known 0-4 values) to get real statistical power on this
  specific candidate; also worth testing 0.4/0.5 to see if the ascending trend continues or peaks.

## [2026-07-21] Run 72 — campaign: elliptic_graphsage_diff_degree_aware vs elliptic_graphsage_diff
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_full_graphsage_diff.yaml,
  candidate=configs/elliptic_full_graphsage_diff_degree_aware.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_full_graphsage_diff.yaml | 0.7470 | 0.0032 | 5 |
| configs/elliptic_full_graphsage_diff_degree_aware.yaml | 0.7511 | 0.0049 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.1250, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.7511019670119181 vs 0.7470061689823659, p=0.125)
- Observations: (fill in manually -- see Run 73's correction after extending both this and Run 71 to
  10 seeds)
- Next: (fill in manually)

## [2026-07-21] Run 73 — CORRECTION: both DropEdge(0.3) (Run 71) and degree-aware (Run 72) evaporate at n=10 -- another confirmed lucky-seed trap
- Run 71 (DropEdge 0.3) hit the maximum-possible-significance signature at n=5 (statistic=0.00,
  p=0.0625, all 5 seeds agreed) and Run 72 (degree-aware) showed a consistent, if less extreme,
  same-direction result (p=0.125). Both looked like the most promising leads of the session's later
  half. Per this session's own repeatedly-learned lesson (never trust n=5, especially a "maximum
  significance" result, without extending it), dispatched 5 MORE seeds (5-9) for both candidates
  and their baseline before believing either.
- **DropEdge(0.3), full 10 seeds**: baseline mean=0.7513 (std=0.0053), DropEdge(0.3) mean=0.7533
  (std=0.0105). Wins for DropEdge: **4/10** (not 5/5). Wilcoxon p=**0.734** (fully non-significant).
  The new seeds (5-9) alone showed DropEdge LOSING in 4 of 5 -- the exact opposite direction of the
  first 5 seeds.
- **Degree-aware, full 10 seeds**: baseline mean=0.7505 (std=0.0049), degree-aware mean=0.7495
  (std=0.0064) -- now marginally WORSE, not better. Wins: 4/10. Wilcoxon p=**0.820**.
- **Both "promising" leads were lucky 5-seed draws, not real effects.** This is a clean,
  clarifying demonstration of why "statistic=0.00, p=0.0625 at n=5" should never be read as strong
  evidence on its own -- it is the FLOOR of what's achievable at that sample size, achieved by
  chance alone whenever a coin-flip-level true effect happens to land the same way 5 times running
  (which happens ~6% of the time under pure noise, by construction of the test). The correct
  response to hitting that floor is exactly what was done here: get more seeds before believing it,
  not treat the floor itself as proof.
- **Conclusion**: neither DropEdge nor the degree-aware fix meaningfully changes anything on
  Elliptic. Every pure-GNN architecture lever tried this entire session (deeper layers, bigger
  head, feature encoder, learned gate w/ and w/o supervision, spectral filter, FAGCN, node2vec,
  DropEdge, degree-awareness) has now been tested and found flat or negative. The GNN+RF hybrid
  (Run 55, mean F1=0.7670) remains the only result that meaningfully closes the RF gap, and Run 70's
  out-of-distribution finding gives a principled reason why: much of what's left (the hard core) may
  not be closeable by ANY model/method trained on this data, regardless of architecture.
- Next: treat the Elliptic architecture search as closed for now. Redirect effort to IEEE-CIS
  (in progress: mini-batch fix for the full-scale OOM issue) and/or the feature-enrichment work
  queued earlier, rather than continuing to generate new single-mechanism GNN variants for
  Elliptic -- the evidence base is now large and consistent enough to trust this conclusion.

## [2026-07-21] Run 74 — HEADLINE FINDING: the "hard core" is a temporal cliff, not an intrinsic-difficulty or architecture problem
- Deep-dive into Run 67/70's hard core: recovered per-node `step` (not stored in the processed
  Data object, reloaded fresh from elliptic_txs_features.csv in the same row order as
  preprocessing) and cross-referenced against RF's test predictions across the 8 test steps
  (42-49; train<=34, val=35-41).
- **Result -- RF's recall_fraud by test step**:
  | Test step | n_fraud | RF recall_fraud |
  |---|---|---|
  | **42** (first test step) | 239 | **79.9%** (191/239) |
  | 43 | 24 | 0.0% |
  | 44 | 24 | 4.2% |
  | 45 | 5 | 0.0% |
  | 46 | 2 | 50.0% (n=2, noise) |
  | 47 | 22 | 0.0% |
  | 48 | 36 | 0.0% |
  | 49 | 56 | 1.8% |
- **This is a CLIFF, not a gradual decline**: excellent detection immediately after the train/val
  boundary, then near-total collapse for the remaining 7 of 8 test steps, never recovering. Hard
  core's mean step is 45.76 vs correctly-caught-fraud's mean step of 42.06 -- almost the entire
  "hard core" from Run 67/70 is just fraud from steps 43-49, i.e. this is close to a single
  underlying phenomenon, not a diffuse population of independently-hard cases.
- **Per-feature analysis** (hard core vs correctly-caught fraud, standardized effect size): a mix
  of local (11/20 of the top differentiators) and aggregated (9/20) features differ meaningfully --
  no single feature block dominates. Some very high Cohen's d values (f4/f13 ~2.4) come from
  near-zero-variance features where a tiny absolute difference produces an inflated standardized
  effect size -- treat those specific ones with caution; f52/f54/f100/f102/f141/f113/f143 show
  large AND practically meaningful raw differences.
- **This reframes the entire session's central question.** Every architecture tried (GraphSAGE,
  GraphSAGEDiff, GAT, gated, spectral, FAGCN, DropEdge, degree-aware) and RF alike are being asked
  to generalize from a fixed training window (steps 1-41) to fraud that is, on average, ~4 steps
  further into the future than what "easy" fraud looks like. Run 70's out-of-distribution finding
  is a SYMPTOM of this, not an independent cause: features genuinely drift over Elliptic's ~49-step
  timeline (documented in the literature around this dataset's real-world time span), so fraud
  further from the training cutoff naturally looks more unlike anything trained on. No architecture
  change was ever going to fix a problem that both RF and every GNN variant suffer identically --
  the real bottleneck is DATA RECENCY, not modeling technique. This also reframes why RF's
  advantage over GNNs looked so stable across this whole investigation: both are hitting the same
  temporal wall, so any gap between them is being measured entirely WITHIN the narrow window of
  fraud both can still generalize to, not across the full test set's actual difficulty.
- **Practical validation still needed**: confirm this is really a "one step past most recent
  training data generalizes, more than that collapses" RULE (not something specific to step 42) by
  re-running with a SHIFTED split (e.g. train<=44, val=45-47, test=48-49) and checking whether the
  same cliff-at-first-test-step pattern reappears at the new boundary.
- **This changes what's worth building next far more than any further architecture search would**:
  continual/online learning, a rolling retraining window, or treating "how many steps out does
  generalization survive" as the actual research question, rather than more GNN variants for a
  fixed, aging train/test split.
- Next: (a) shifted-split validation test above; (b) check whether GraphSAGEDiff's per-node
  predictions show the identical cliff shape (very likely, given the hard-core intersection is
  already 213/214 of RF's own misses); (c) consider whether this changes how IEEE-CIS's temporal
  split should be interpreted/re-checked too.

## [2026-07-21] Run 75 — Run 74's hypothesis refined TWICE: it's not "N steps out decays" and not "a single learnable regime change" either
- Directly tested Run 74's "one step past training generalizes, more collapses" hypothesis with
  three quick, local (RF-only, no RunPod) shifted-split checks:
- **Test 1 -- shifted boundary** (train<=44, val=45-47, test=48-49): if generalization horizon were
  the mechanism, step 48 (the new first test step) should show ~80% recall like original step 42
  did. **It shows 0.0% (0/36).** REFUTES the generic "one step past training" hypothesis.
- **Test 2 -- entirely pre-42 window** (train<=25, val=26-30, test=31-49 all used as reference):
  recall stays strong (52-97%) all the way through step 42, THEN collapses at step 43 onward --
  **regardless of how far from the train cutoff (25) each step is** (step 42 is 17 steps past
  training and still gets 74.5% recall; step 43 is only 18 steps past and gets 0%). This confirmed
  the break is tied to a SPECIFIC absolute point in Elliptic's real timeline (~step 42/43), not a
  generic decay-with-distance curve.
- **Test 3 -- train INCLUDING post-break data** (train<=46, which includes steps 43-46 as direct
  labeled training examples, test=47-49): if the mechanism were "a single new regime that just
  needs its own training examples," this should now generalize reasonably to 47-49. **It doesn't --
  still 0.0%/0.0%/1.8% recall**, despite the model having seen real labeled fraud from the
  immediately preceding steps.
- **Conclusion, now well-evidenced through three separate falsification attempts**: this isn't a
  generalization-HORIZON problem (fixable by training closer to test time) and it isn't a single
  discrete regime-change problem either (fixable by including some post-break examples). The most
  consistent explanation left: steps 43-49 either keep genuinely drifting step-to-step (no stable
  target to learn even within "post-break"), or simply have too few fraud examples each (24, 24, 5,
  2, 22, 36, 56 -- much sparser than step 42's 239 or step 32's 342) to form any learnable pattern,
  closer to scattered individual anomalies than a coherent class.
- **Direct answer to "should we add a temporal architecture (TGN/EvolveGCN/diffusion-forcing-style
  variable-noise sequence modeling)"**: the evidence argues against it. Temporal GNN architectures
  need the target regime to be representable from SOME training signal, even an evolving one --
  Test 3 shows that even DIRECT labeled examples from the immediately-preceding period don't help
  predict the next period, meaning the bottleneck isn't "the model can't track how things evolve
  over time," it's that late-period fraud may not form a stable, learnable pattern from what's
  available in this dataset at all. Diffusion forcing specifically is a generative-sequence
  technique (variable per-step noise for flexible video/sequence generation) with no direct
  bearing on a classification-generalization problem. No architecture -- temporal or otherwise --
  manufactures labels or stable patterns that aren't there in the data.
- **This is a DATA problem, not a modeling problem** -- consistent with, and a much more specific/
  falsifiable version of, Run 74's framing. The practical implication for any Elliptic benchmark
  comparison (ours or the literature's): a real production system facing this exact pattern would
  need continual retraining as fresh labels arrive, not a fancier static architecture -- and no
  static train/test split on this specific dataset can resolve that, because the needed labels for
  late-period fraud may not meaningfully exist within it.
- Next: (a) treat this as effectively closed for Elliptic -- no further architecture search
  (temporal or otherwise) is likely to move this number; (b) redirect fully to IEEE-CIS, which
  doesn't have this dataset's known historical-timeline artifact; (c) if ever revisited, checking
  the RAW fraud counts/dates around step 42-43 against Elliptic's documented real-world time span
  would confirm (or refute) the "real external event" reading of this break.

## [2026-07-21] Run 76 — Would resampling/reweighting rare-but-present fraud types help? No -- hard core has essentially no near neighbor in train fraud at all
- Direct test of a specific, reasonable idea: if the hard core represents fraud "types" that ARE
  present in training but just rare/underrepresented, targeted resampling (oversampling, focal-
  loss-style reweighting) could plausibly help by amplifying their training signal. Tested this
  literally: nearest-neighbor distance (in raw 165-dim feature space) from every test fraud node to
  its closest TRAIN-fraud example.
- **Result**:
  | Group | mean NN dist to nearest train fraud | median |
  |---|---|---|
  | Train fraud's own typical spacing (self-baseline, leave-one-out) | 1.27 | 0.33 |
  | Caught-both fraud (RF+GNN both catch) | 2.79 | 2.49 |
  | **Hard core (both miss)** | **8.33** | **8.28** |
- **Essentially zero distributional overlap**: hard core's 10th percentile (5.74) is still farther
  than caught-both's 90th percentile (3.74). At a generous 3x-typical-spacing threshold, 90.8% of
  caught-both cases have a nearby train-fraud neighbor; only 1.9% of hard-core cases do. Even at an
  extremely generous 8x threshold, 20.7% of hard-core cases STILL have no train-fraud neighbor
  within range.
- **This directly answers the resampling question: no.** Resampling/reweighting only changes how
  much EXISTING training examples count toward the loss -- it cannot manufacture density in a
  region of feature space where there's essentially nothing nearby to begin with. This is a
  stronger, more concrete version of Run 70's out-of-distribution finding: not just "farther from
  the train-fraud centroid on average," but "no meaningfully close neighbor exists in training,"
  for the large majority of hard-core cases specifically. If these are distinct fraud archetypes,
  they're not underrepresented in training data -- they're absent from it.
- **This also rules out oversampling-based fixes as a category**, alongside the already-ruled-out
  architecture-mechanism fixes (Run 73) and generative/diffusion-based fixes (Run 70) -- three
  independent families of "make the existing data work harder" all fail for the same underlying
  reason (Run 74/75's regime break). The only remaining lever that could plausibly work is
  literally new labeled data from the post-break period, which isn't available within Elliptic.
- Next: treat this line of investigation as fully closed. Redirect entirely to IEEE-CIS.

## [2026-07-21] Run 77 — Does legit ALSO drift? Yes, ~half as much as fraud, correlated direction. CORAL domain adaptation: decisively negative
- Two more direct tests of the regime break's character, motivated by "do we need domain
  adaptation / a distribution-alignment technique" (rotation/alignment framing) and "is this a
  volume/scale artifact" (scale-free framing).
- **Volume-normalization check**: correlation of the top 9 drifting features (Run 74) with raw
  per-step transaction volume is weak for most (f52 +0.04, f100 -0.06, f141 +0.02), moderate for a
  few (f113 +0.34, f149 +0.25, f155 +0.23). Not primarily a normalization/count artifact.
- **Does legit ALSO drift across the step-42/43 break?** Across all 165 features, pre-42 vs
  post-42: legit mean|Cohen's d|=0.221 (6 features with |d|>1.0); fraud mean|Cohen's d|=0.392
  (17 features with |d|>1.0) -- **fraud drifts about 2x as much as legit, roughly 3x as many
  features cross the "large effect" threshold**. But direction is highly correlated between the
  two (r=0.796 across all 165 features) -- confirming a real SHARED covariate shift component
  (something changed dataset-wide, not just for fraud), with a substantial fraud-specific EXCESS
  drift on top of it.
- **CORAL domain adaptation test** (Sun & Saenko 2016): aligned TRAIN feature covariance to the
  UNLABELED post-break period's covariance (all nodes, any label, steps 43-49 -- genuine domain
  adaptation, no target labels used), trained RF on CORAL-aligned train features + original
  labels, evaluated on real (unaligned) test features.
  | | Overall test F1 |
  |---|---|
  | Baseline RF (no alignment) | 0.7888 |
  | **CORAL-aligned RF** | **0.4882** |
  Step-by-step: CORAL didn't recover ANY of steps 43-49 (still 0% each) and DESTROYED step 42's
  previously-strong 79.9% recall down to 0%. A decisive, not ambiguous, negative result.
- **Why CORAL actively hurts, not just fails to help**: it's an unsupervised, label-blind global
  linear transform -- it aligns the ENTIRE feature covariance (dominated by the majority legit
  class) to the target domain. Since fraud's drift is ~2x legit's and not a simple scaled/uniform
  version of it (a real class-conditional divergence, not just a shared linear shift), a single
  global correction has no mechanism to apply the right amount of correction per class -- it ends
  up distorting perfectly good existing structure (breaking step 42, which didn't need fixing) in
  service of a correction that's fundamentally mis-specified for a problem with a real class-
  asymmetric component.
- **Implication for metric learning too** (predicted, not yet tested): metric learning uses labels
  (unlike CORAL), so it wouldn't blindly scramble class structure the same way -- but it's still
  fit ONLY on pre-break labeled examples. For genuinely novel test points (Run 76: ~8x typical
  train-fraud spacing away from anything in training), a learned distance function has no training
  signal informing where such a point should sit relative to the classes. Expected to perform
  similarly to raw-feature RF on steps 43-49, not better -- same fundamental "no signal to learn
  from" ceiling as every other approach tried.
- **Conclusion**: this is now the FOURTH distinct family of fix shown to fail on this specific
  problem (architecture/Run 73, generative-diffusion/Run 70, resampling/Run 76, and now domain
  adaptation/Run 77) -- each for a related but distinct reason, converging on the same root cause:
  steps 43-49 require information (either new labels, or a correction that accounts for a real
  class-conditional divergence) that plain feature-space or covariance-level access to unlabeled
  target data cannot supply.
- Next: this closes the regime-break investigation very thoroughly. Full redirect to IEEE-CIS.

## [2026-07-21] Run 78 — Metric-learning GNN (triplet loss + nearest-centroid): first genuine, no-cost recovery on the hard core
- Direct empirical test (not theoretical dismissal) of a specific proposal: train a GNN encoder
  (GraphSAGEDiff) via triplet loss instead of a classification head, classify by nearest centroid
  (fraud vs legit/normal) in the learned embedding space. New: evaluation/metric_learning.py,
  dispatched via serverless/handler.py's config.metric_learning.enabled flag.
- **Overall result**: Test AUC=0.873, F1=0.727 (best_epoch=445) -- AUC competitive with supervised
  RF (0.867), though F1 below RF's 0.789 and GraphSAGEDiff's classification-head 0.748.
- **Per-step AUC (the real test) is meaningfully above random for EVERY post-break step**,
  unlike CORAL (Run 77, near/below random) or plain RF (near-zero recall/poor ranking): step 43
  AUC=0.730, 44=0.662, 45=0.918, 46=0.776, 47=0.703, 48=0.868, 49=0.674. The embedding retains real
  ranking signal for post-break fraud that the default 0.5 threshold doesn't translate into recall
  -- suggesting a calibration gap, not a total absence of signal.
- **Hard-core cross-reference (the decisive test)**: of Run 67's 213 hard-core cases (missed by
  BOTH RF and GraphSAGEDiff), plain triplet-loss metric learning RECOVERS 16 (7.5%) while losing
  ZERO of the 142 cases both RF and GraphSAGEDiff already catch. **This is the first approach all
  session with a clean, no-tradeoff improvement on the hard core** -- every prior attempt either
  failed outright (CORAL, anomaly detection, DropEdge, degree-aware, spectral, FAGCN, gate) or
  recovered a few hard cases at a real cost elsewhere (GraphSAGEGated: 4/214 recovered, but 38/194
  lost, Run 60).
- **Distance-distribution finding (visualized, artifact published)**: distance-to-legit-centroid
  for PRE-break fraud (mean 1.581) is far from legit (mean 0.349) -- clean separation. POST-break
  fraud (mean 0.527) sits MUCH closer to legit's own value (0.362) -- nearly overlapping. This is a
  sharper, more specific characterization than generic "concept drift": post-break fraud's
  embedding profile literally collapses toward looking like normal transactions, consistent with
  fraud camouflage/evasion evolving over time rather than fraud becoming an unrelated, unknown
  pattern.
- **Compression (Center) loss variant** (explicit pull toward each class's own centroid, weight=
  0.5, motivated by legit's spread being ~2.7x fraud's in the plain-triplet embedding): tightened
  both spreads substantially (fraud 0.126->0.068, legit 0.344->0.110) and improved OVERALL F1
  (0.716->0.746), but recovered FEWER hard-core cases (4/213, 1.9%) and didn't improve individual
  post-break step recall (step 48 dropped from 13.9% to 0%). Reads as: compression sharpens the
  boundary for the already-learnable majority, at the cost of the modest hard-core reach the plain
  version had -- a real trade-off, not a strict improvement.
- **Conclusion**: metric learning is a genuinely different, partially-successful lever -- not a
  full fix (92.5% of the hard core is still missed even by the best variant), but the first result
  all session to make ANY dent in it without a corresponding loss elsewhere. Worth a proper
  multi-seed confirmation before trusting the exact 7.5% number, and worth trying without
  compression loss (or with a lower weight) given the plain version did better on the specific
  thing we care about.
- Next: (a) multi-seed the plain triplet-loss result to confirm 16/213 isn't a lucky single-seed
  draw (this session's own repeated lesson); (b) if confirmed, this becomes the new best lead for
  Elliptic's regime-break problem specifically, distinct from Run 55's GNN+RF hybrid (best overall
  result, doesn't specifically target the hard core); (c) redirect primary effort to IEEE-CIS
  regardless, given Elliptic's core investigation is otherwise well-closed.

## [2026-07-21] Run 79 — Batch-hard triplet mining: backfires badly; a real bug caught (silent None output on oversized payload)
- Tested batch-hard mining (Hermans et al. 2017): every epoch, each fraud anchor trained against
  its CURRENTLY farthest fraud peer and CURRENTLY closest legit point (computed over the full
  train population, not just a mini-batch), directly targeting post-break fraud as exactly the
  "hard positive" random sampling under-weights.
- **Result: F1=0.325** (vs plain random-sampling triplet's 0.727) -- recall_fraud=0.855 but
  precision_fraud=0.061, because predicted_positive_rate=64.3% (the model flags nearly two-thirds
  of ALL test nodes as fraud). best_epoch=67, stopped early. A clear, decisive negative --
  consistent with a documented failure mode of pure batch-hard mining: the "hardest" pair keeps
  shifting between epochs and dominates the gradient, destabilizing the whole embedding space
  rather than sharpening the boundary (vs. semi-hard mining or curriculum approaches that ease into
  hard examples, which weren't tried here).
- **Separately, caught a real infra bug while testing return_embeddings=true for the planned
  dimension/probe analysis**: a job returning full test+train embeddings (~8841 test + ~3462 train
  fraud + ~5500 train legit, 128-dim, as JSON) came back with status=COMPLETED but output=None --
  no error raised anywhere. Almost certainly a RunPod output-payload size limit (~30-40MB
  estimated) being hit and failing SILENTLY rather than with a clear error. Fixed in
  evaluation/metric_learning.py by keeping ALL fraud (rare, small in absolute count, most
  important) but subsampling legit to 1500 and rounding to 5 decimals -- cuts the payload to
  ~8MB. Worth remembering as a general pattern: a "successful" RunPod job with an empty/None
  output should be suspected of a silent size-limit failure, not treated as a real empty result.
- Next: re-dispatch return_embeddings=true with the fix for the dimension-separation/linear-probe
  analysis; do not pursue hard mining further without a gentler variant (semi-hard mining, or hard
  mining phased in only after a warmup period on random triplets) if metric learning is revisited.

## [2026-07-21] Run 80 — Dimension-separation and linear-probe analysis: the residual signal is real but diffuse, no smoking-gun dimension, no easy linear signal left on the table
- With the Run 79 payload-size bug fixed, obtained real embeddings (test fraud/legit sample +
  train fraud/legit) from the plain (no compression, no hard-mining) triplet-loss model. Test
  fraud rows' original test_mask position (hence real `step`) recovered locally without a further
  server round-trip, by exploiting that both the local recomputation of test-fraud indices and the
  server's own construction sort disjoint fraud/legit index sets, which preserves each subset's
  relative order through the sort -- letting fraud rows in the returned (subsampled) embeddings be
  zipped back to their true step by position alone.
- **Per-dimension separation (post-break fraud vs. all legit)**: top 15 of 128 embedding
  dimensions all show MODEST effect sizes (Cohen's d 0.53-0.76) -- no dominant dimension carries
  most of the signal. Mean |d| across all 128 dims: PRE-break fraud vs legit = 2.145 (huge,
  explains why early steps classify so well) vs POST-break fraud vs legit = 0.253 (real, but
  small and spread thinly across nearly the whole embedding, not concentrated).
- **Linear probe (logistic regression fit on train fraud+legit embeddings, evaluated on test)**:
  AUC=0.705 for post-break fraud vs legit -- genuinely better than random, in the same range
  Run 78 already found via the simpler centroid-distance rule (0.66-0.92 per step). Mean
  predicted P(fraud): pre-break fraud=0.742 (confidently flagged), **post-break fraud=0.051**
  (indistinguishable from legit's own 0.025) despite the non-trivial AUC.
- **This decisively answers "where does the shift come from" and "is there easy signal left to
  extract"**: the shift is a genuine, diffuse, dataset-wide phenomenon (consistent with Run 77's
  "shared covariate shift + fraud-specific excess" framing) rather than a few identifiable
  directions we could isolate and amplify -- and a properly-optimized linear classifier does NOT
  meaningfully outperform the simple geometric centroid-distance rule already in use. We are not
  leaving free linear signal on the table by using a simpler decision rule. The residual AUC~0.70
  is real (not noise) but reflects a genuine, modest information ceiling for a LINEAR read of this
  embedding, not an extraction inefficiency.
- **Practical implication**: further gains would require either (a) a fundamentally different
  training objective that CONCENTRATES the currently-diffuse signal into a more separable
  direction (hard mining, Run 79, tried exactly this and made things categorically worse -- not
  a promising direction without a gentler variant), or (b) accepting this as close to the real
  ceiling given the labels available in this dataset. This closes out the metric-learning
  investigation with a precise, well-quantified answer rather than an open question.
- Next: full redirect to IEEE-CIS. The Elliptic investigation (Runs 46-80) is now closed on solid,
  multiply-corroborated ground: RF beats every GNN variant tried; the remaining gap is dominated
  by a real temporal regime break (Run 74-77) that is out-of-distribution relative to training in a
  diffuse, not concentrated, way (Run 80); metric learning recovers a small, real, no-cost slice of
  it (Run 78) but hits a genuine information ceiling, not a modeling gap.

## [2026-07-21] Run 81 — Reopened: alternative read-outs (Mahalanobis/MLP) confirm the ceiling, but a real camouflaged-fraud sub-cluster is found
- User pushback (correctly) on treating Run 80 as fully closed -- recall/AUC on post-break steps
  genuinely improved over RF/GraphSAGEDiff's near-total collapse, and the diffuse-signal finding
  doesn't mean nothing more can be done, only that simple linear readouts are near their ceiling.
- **Mahalanobis-distance nearest-centroid and MLP probe, tested on the SAME embeddings** (no
  retraining): both did WORSE than plain logistic regression (0.705) -- Mahalanobis=0.530,
  MLP=0.610. Shrinkage sweep (Ledoit-Wolf auto + manual alpha 0.3-0.99) shows AUC climbing
  monotonically from 0.53 toward 0.717 as shrinkage approaches pure identity (i.e., degenerating
  back to plain Euclidean distance) -- strong evidence the empirical covariance structure is
  mostly NOISE, not usable signal, given only ~3462 train-fraud points to estimate a 128x128
  covariance. Every readout method converges to the same 0.53-0.72 range, confirming the
  bottleneck is the embedding's training, not the classification rule reading it.
- **But: a real, cleanly-interpretable camouflaged-fraud sub-cluster exists.** Following up on the
  visible bimodality in the published distance-distribution chart -- re-identified correctly as a
  property of STEP-42 (the first/only "pre-break" test-period) fraud specifically, not train fraud
  (an initial mis-identification, corrected) -- a 2-component GMM on step-42 fraud's distance-to-
  legit-centroid cleanly splits it into an "obvious" archetype (mean dist 1.886, n=144) and a
  "camouflaged" archetype (mean dist 1.138, n=95, ~40% of step-42 fraud).
- **POST-break fraud aligns overwhelmingly (95.3%) with this camouflaged archetype specifically**
  (mean dist 0.950 to the camo centroid vs 1.835 to the obvious one) -- a much cleaner, more
  decisive signal than the aggregate fraud-vs-legit comparison. This means post-break fraud isn't
  an unprecedented pattern -- it's the SAME camouflaged archetype that already existed (as a
  ~40% minority) within the easy, well-classified period, just diluted into one aggregate fraud
  centroid/one aggregate training signal rather than being tracked as its own thing.
- **However, using the camo-centroid as the reference point does NOT itself raise AUC above the
  same ~0.70-0.72 ceiling** (0.702) -- consistent with Run 80: the underlying embedding wasn't
  trained to make this specific distinction sharp, so pointing at the right REFERENCE doesn't
  compensate for an embedding that was never optimized to separate camo-fraud from legit
  specifically. This is the clearest statement yet of what a better LOSS would need to do: don't
  just pull all fraud together (current triplet loss) -- explicitly preserve/sharpen the
  camouflaged-vs-legit boundary as a first-class training objective, not an emergent side effect
  of pulling all fraud toward one point.
- Next: (a) sweep compression_weight (0.1/0.25/1.0/2.0) alongside the existing 0/0.5 points to map
  the actual trade-off curve rather than one spot-check; (b) design and test a loss term that
  explicitly targets the camo/obvious sub-cluster distinction (e.g., a secondary triplet/contrastive
  term using GMM-assigned sub-cluster labels as an auxiliary signal, or triplets constructed
  specifically from the camo sub-cluster as hard positives) instead of only ever pulling all fraud
  toward one centroid.

## [2026-07-21] Run 82 — Camo-weighted triplet loss: NEW BEST on hard-core recovery (12.7%, zero cost); compression sweep; per-node anomaly-signature check; ArcFace correctly ruled out
- **Camo-weighted mining** (new evaluation/metric_learning.py `mining="camo_weighted"`): soft
  importance-weighting of RANDOM triplets by how close the fraud anchor currently is to the legit
  centroid (softmax weighting, not hard top-1 selection -- deliberately gentler than Run 79's
  hard mining, which destabilized training). Directly motivated by Run 81's camouflaged-sub-
  cluster finding.
- **Aggregate metrics look worse**: F1=0.6373 (vs plain triplet's 0.7273), precision_fraud
  collapsed to 0.238 (predicted_positive_rate jumped to 10.3%, a milder version of Run 79's
  hard-mining over-flagging problem). AUC=0.845, similar to plain.
- **But the metric that actually matters is BETTER, decisively**: cross-referenced against the
  established hard-core set (Run 67), camo-weighted recovers **27/213 (12.7%)** of hard-core cases
  -- nearly double plain triplet's 16/213 (7.5%, Run 78) -- while still losing **ZERO** of the 142
  already-caught-by-both cases. Aggregate F1 is a misleading proxy for this specific goal: the
  extra false positives camo-weighting introduces come from elsewhere in the legit population, not
  from cases anything else already correctly handles. **This is the new best result on hard-core
  recovery this entire investigation.**
- **Compression-weight sweep** (0.1/0.25/1.0/2.0, extending Run 78's single 0.5 point): F1 rises
  with weight (0.1->0.7428, 2.0->0.7592, both above plain's 0.7273) but AUC doesn't move
  monotonically (0.1 gives the BEST AUC=0.8919, even above plain's 0.8757). Per Run 78's own
  lesson, aggregate metrics alone don't establish whether any of these help hard-core recovery
  specifically -- not yet cross-referenced per-node; flagged as follow-up before drawing
  conclusions from this table alone.
- **Per-node anomaly-signature check** (different from Run 80's per-dimension AVERAGE separation):
  for each post-break fraud node, computed a per-dimension z-score vs. legit's own mean/std, then
  checked which dims most often appear in a node's own top-10 most-anomalous list. Result: a real,
  partial shared signature exists -- top dims (19, 23, 25, 51, 89...) appear in 18-31% of nodes'
  top-10 lists, 2-4x the uniform-random expectation (7.8%), tapering gradually rather than showing
  one/few dominant dims. Consistent with Run 80 (diffuse, not concentrated) while adding: it's not
  PERFECTLY uniform either -- a soft cluster of ~10-15 dims is disproportionately relevant.
- **ArcFace/angular-margin loss correctly ruled out before building it** (user catch): ArcFace's
  design assumes each class converges to ONE tight prototype direction with a uniform enforced
  margin -- appropriate for many-class problems with naturally single-mode classes (a face is one
  identity), but actively wrong here given Run 81's direct evidence that fraud is genuinely
  bimodal (camo ~40% / obvious ~60%). Forcing one fraud prototype would likely repeat/worsen the
  over-compression trade-off already seen with Center Loss. A multi-prototype approach (2+ fraud
  proxies, Proxy-Anchor-style) would be the principled version if this direction is pursued
  further, not vanilla ArcFace.
- **Conclusion**: camo-weighted triplet loss is now the standing best lead for closing part of the
  Elliptic hard core specifically (12.7% recovered, zero cost) -- a real, if partial, win, and a
  correction to Run 80's premature "ceiling" framing. Needs multi-seed confirmation before fully
  trusting the exact 12.7% number (this session's own repeated lesson).
- Next: (a) multi-seed camo-weighted to confirm; (b) cross-reference compression-weight sweep
  points against hard-core recovery specifically, not just aggregate F1/AUC; (c) if pursued
  further, a genuine multi-prototype (2 fraud proxies) loss rather than single-centroid framing.

## [2026-07-21] Run 83 — IEEE-CIS GraphSAGEDiff (mini-batch): succeeds mechanically, but same RF-beats-GNN pattern as Elliptic
- The mini-batch fix (configs/ieee_cis_graphsage_diff_minibatch.yaml, 90-min execution timeout)
  finally succeeded after repeated full-batch OOM-style failures (Run 63's saga) -- confirms
  mini-batch was the right fix for training at IEEE-CIS's full ~590k-node / 41M-edge scale.
- **Result: Test F1=0.6671, AUC=0.8572**, recall_fraud=0.260, precision_fraud=0.542.
  | Approach | Test F1 |
  |---|---|
  | RF alone (full dataset, Run 63/64-adjacent) | 0.7235 |
  | Plain GraphSAGE, best alpha=0.85 (Run 64) | 0.6939 |
  | **GraphSAGEDiff, mini-batch, alpha=0.75 (this run)** | **0.6671** |
- Not a fully clean comparison (mini-batch mode + alpha=0.75, not the later-discovered 0.85 peak,
  and GraphSAGE's own comparison point used full-batch) -- but the gap to RF is large enough to be
  a real, informative result regardless. **Same qualitative pattern as Elliptic: RF beats every
  GNN variant tried, including the architecture that was Elliptic's best lead.**
- Next: (a) IEEE-CIS GNN+RF hybrid with the same mini-batch fix (never yet dispatched at full
  scale -- the earlier hybrid attempts predate the mini-batch fix and failed on the same full-
  batch OOM); (b) re-run GraphSAGEDiff at alpha=0.85 for a cleaner comparison if this line is
  pursued further; (c) IEEE-CIS's own hard-core/regime-break characterization (analogous to
  Elliptic's Run 74-82) hasn't been done at all yet -- worth checking whether IEEE-CIS has a
  similar temporal fault line before assuming the two datasets fail for the same reason.

## [2026-07-21] Run 84 — Camo-weighted triplet loss CONFIRMED across 5 seeds: real, reproducible, zero-cost hard-core recovery
- Multi-seed check of Run 82's camo-weighted result (seeds 0, 1, 2, 3, in addition to the original
  seed=42), directly testing hard-core recovery per seed rather than trusting the single-seed
  12.7% number -- this session's own repeatedly-learned lesson (DropEdge/degree-aware, Run 73,
  both evaporated at n=10; this is the opposite outcome).
- **Result, hard-core recovery by seed**:
  | Seed | Recovers (of 213) | Loses (of 142 caught-both) |
  |---|---|---|
  | 0 | 50 (23.5%) | 0 (0.0%) |
  | 1 | 33 (15.5%) | 0 (0.0%) |
  | 2 | 26 (12.2%) | 0 (0.0%) |
  | 3 | 58 (27.2%) | 0 (0.0%) |
  | 42 (original) | 27 (12.7%) | 0 (0.0%) |
  Mean recovery ~18.2% across all 5 seeds (range 12.2-27.2%) -- **loses ZERO caught-both cases in
  EVERY single seed, no exceptions.**
- **This is a real, reproducible, multi-seed-confirmed effect, not a lucky draw.** It's also the
  first result in the ENTIRE Elliptic investigation (Runs 46-84) that both (a) meaningfully moves
  the hard-core needle and (b) survives proper multi-seed scrutiny without evaporating. Every other
  positive-looking lead this session (DropEdge, degree-aware, single-seed metric learning) either
  failed a multi-seed check or wasn't yet tested against one until now.
- **Practical read**: camo-weighted triplet-loss metric learning, layered on top of (not replacing)
  the existing best approaches, recovers a genuine ~18% slice of the previously-fully-unrecoverable
  hard core at zero cost to anything already working. Combined with the GNN+RF hybrid (Run 55,
  the best overall result) or used as a targeted secondary signal specifically for cases the
  primary classifier flags as legit, this is the most solid, defensible new lead to come out of
  the entire regime-break investigation.
- Next: (a) consider combining camo-weighted metric learning's predictions with the GNN+RF hybrid
  (e.g., flag as fraud if EITHER approach says fraud, given zero cost from camo-weighted alone) to
  see if the combination genuinely improves over either alone; (b) IEEE-CIS's own hard-core
  characterization, now that its pipeline is stable (Run 83); (c) subspace-restricted (top-N
  anomaly-signature dims only, per Run 82) distance scoring, testing whether that adds anything
  further on top of camo-weighted specifically.

## [2026-07-21] Run 85 — Subspace-restricted distance scoring (top anomaly-signature dims): real, stable improvement
- Direct follow-up to Run 82's per-node anomaly-signature finding (certain dims appear in 18-31%
  of post-break fraud nodes' top-10-most-anomalous list, vs 7.8% uniform-random expectation):
  tested whether restricting the fraud/legit centroid-distance SCORE to just those top-N dims
  (instead of all 128) improves discrimination, on the plain (non-camo-weighted) embedding.
  | Dims used | AUC (post-break fraud vs legit) |
  |---|---|
  | All 128 (baseline) | 0.713 |
  | Top 5 | 0.739 |
  | **Top 10** | **0.748** |
  | Top 15 | 0.742 |
  | Top 20 | 0.747 |
  | Top 30 | 0.732 |
  | Top 50 | 0.743 |
- **Real, stable improvement** -- not a fragile single lucky pick, holds across the whole 5-50 dim
  range tested, peaking around 10-20 dims (AUC 0.747-0.748 vs 0.713 full-128 baseline). Confirms
  the anomaly-signature dims aren't just descriptively interesting (Run 82) but functionally
  useful: the other ~110 dims were adding noise that diluted the full-128-dim Euclidean score,
  similar in spirit to (but more targeted/successful than) the Mahalanobis-shrinkage finding in
  Run 81 -- rather than trying to learn a full covariance-based reweighting (which overfit given
  the sample size), hand-selecting the informative subspace via the anomaly-signature method works.
- Next: combine with camo-weighted mining (Run 82/84) -- both are real, independent, zero/low-cost
  improvements (subspace selection is a pure readout-side change, doesn't need retraining); worth
  checking whether subspace-restricted scoring ALSO improves camo-weighted's already-strong
  hard-core recovery, or whether the two overlap/redundant.

## [2026-07-21] Run 86 — CORRECTION: Run 82/85's subspace analysis had real leakage; corrected result is smaller but still genuine
- User catch: Run 82's per-node anomaly-signature dims, and Run 85's subspace-restricted AUC
  improvement, were both computed by selecting dimensions using `fraud_emb_test[post_mask]` --
  literally the post-break TEST fraud, i.e. the exact population then scored on. This is circular:
  using the evaluation set's own labels to pick which dimensions to trust before evaluating on
  that same set inflates the reported improvement and doesn't reflect real generalization (in
  deployment you don't know in advance which unlabeled transactions are the fraud you're trying to
  catch).
- **Re-ran cleanly**: split TRAIN fraud in half, used ONLY one half (never touching any test
  label) to select the top anomaly-signature dims, then evaluated the resulting subspace-
  restricted score on the real held-out test set (both halves of train fraud, and the eventual
  scoring, never touch test labels for the SELECTION step).
  | | Full 128-dim AUC | Best subspace AUC |
  |---|---|---|
  | Leaky (Run 82/85, dims selected from test fraud) | 0.782 | 0.810 (n=20, INFLATED) |
  | **Clean (dims selected from train fraud only)** | 0.782 | **0.793 (n=20-15)** |
- **The corrected effect is smaller but still genuine** -- 0.782->0.793 is a real, non-circular
  improvement. Notably, only 5 of the top-20 dims overlap between the clean (train-derived) and
  leaky (test-derived) selections -- train fraud's own anomaly signature (dominated by the
  "obvious" archetype, since most train fraud IS obvious-type per Run 81) isn't the same set of
  dimensions as what's specifically anomalous about post-break fraud. Restricting to fewer dims
  still helps even with the "wrong" (train-derived, not test-optimal) subset, likely because ANY
  reduction in dimensionality cuts some noise dilution -- just not as much as cherry-picking the
  test-optimal subset would (dishonestly) show.
- **Scope of the correction**: this affects ONLY the subspace/dimension-selection readout analysis
  (Runs 82, 85). It does NOT affect the core camo-weighted TRAINING result (Runs 78/81/84 -- per-
  step recall, hard-core recovery, 5-seed confirmation) -- that only ever used TRAIN-period labels
  for both the training-time importance-weighting and the inference-time centroids, no test labels
  touched anywhere in that pipeline.
- Next: use the CLEAN (0.793) subspace number going forward, not the leaky 0.810 one, in any
  further comparison or combination with camo-weighted's own already-clean improvement.

## [2026-07-21] Run 87 — CORRECTION: "post-break fraud collapses onto legit" overstated the distance-distribution finding
- User catch, looking again at the published "Distance-to-normal" chart: the yellow (post-break
  fraud) mode has a noticeably bigger intercept/peak position than the "nearly collapses onto
  legit" framing (CLAUDE.md, Run 81/82 write-ups) implied.
- Re-checked the actual histogram data (`camo_weighted_full_analysis.json`'s `distance_dist`):
  | | mean dist-to-legit-centroid | peak (mode) location |
  |---|---|---|
  | Legit | 0.646 | 0.42 |
  | Post-break fraud | 0.910 | 0.67 |
  | Pre-break fraud | 1.540 | 1.68 |
- Post-break fraud moved dramatically closer to legit compared to pre-break fraud (mean
  1.54->0.91), and the two distributions overlap substantially in the 0.5-1.0 range -- but
  post-break fraud's mode stays clearly right-shifted from legit's own (0.67 vs. 0.42, and mean
  0.91 vs. 0.65 -- about 41% further out on average). That is real, measurable separation, not a
  near-collapse. "Moves sharply toward legit, retaining a distinct but much weaker mode" is the
  accurate characterization, not "nearly collapses onto legit's own."
- This is consistent with (and explains) why camo-weighted metric learning could recover part of
  the hard core at all -- if post-break fraud had truly collapsed onto legit's distribution with no
  remaining separation, no re-weighting scheme could have found signal to exploit. The residual gap
  is exactly the signal camo-weighted mining amplifies.
- Scope: text-only correction to how the distance-distribution finding is described (CLAUDE.md
  updated to match). Does not change any trained model, metric, or number in Runs 78-86 -- those
  numbers were always correct; only the qualitative "collapses" framing was too strong.

## [2026-07-22] Run 88 — IEEE-CIS hard-core investigation: bigger hard core, NO temporal cliff, ambiguous bimodality

- **Prerequisite fix**: `debug.return_test_predictions` only worked in train_gnn.py's non-mini_batch
  path; IEEE-CIS's scale requires mini_batch=True. Added mini-batch support (evaluate_batched now
  optionally returns raw per-node probs/y_true, seed-node order). While validating, found
  NeighborLoader can locally swap two ADJACENT seed nodes' order (27 swapped pairs / ~88.6K test
  nodes, 0.06%) -- almost certainly its internal node-dedup when one seed node is also sampled as a
  neighbor of another seed node in the same batch. Confirmed via clean adjacent-pair-swap pattern,
  corrected before use. Real, if narrow, caveat for any code assuming NeighborLoader's shuffle=False
  preserves exact input order.
- **RF vs GraphSAGEDiff on full IEEE-CIS test set (88,581 nodes, 3083 fraud)**:
  | | AUC | F1@0.5 |
  |---|---|---|
  | RF | 0.892 | 0.461 |
  | GraphSAGEDiff (mini-batch) | 0.855 | 0.345 |
  RF's edge is bigger here than on Elliptic (0.037 AUC gap vs. ~0.011).
- **Hard core (missed by both)**: 1895/3083 = **61.5%** of test fraud -- proportionally larger than
  Elliptic's 52%.
- **No temporal cliff.** Binned recall by day across the 31-day test window: noisy (RF recall
  bounces 0.16-0.56 day to day) but no sharp, sustained collapse like Elliptic's step-42/43 break.
  Structurally different difficulty profile: persistent noise, not a discrete regime break. Weakens
  the case that camo-weighted's specific "camouflage evolves at a break point" mechanism transfers
  as-is.
- **Raw-feature-space bimodality check (no learned embedding yet)**: distance-to-legit-centroid on
  ALL fraud (20,663, pooled across splits) gives bimodality coefficient 0.721, but the GMM
  breakdown is one component at 98.9% weight + a thin outlier tail (1.1%), and BIC never plateaus
  (1-comp 125189 -> 2-comp 95876 -> 3-comp 88805) -- the classic signature of fitting a skewed
  unimodal distribution with more Gaussians, not real 2-population structure. No meaningful
  bimodality in raw feature space.
- **Added mini-batch support to evaluation/metric_learning.py** (new code path, `train.mini_batch:
  true`) since the existing full-batch training loop would OOM/timeout at IEEE-CIS's scale the same
  way full-batch classification did. Per-epoch triplets embedded via a small NeighborLoader batch,
  aligned by `batch.n_id` (not assumed order, given the swap bug above); a fixed legit reference
  subsample (config: legit_ref_size) stands in for the ~400K-node full train-legit population for
  both camo_weighted's live centroid and eval-time scoring. hard mining explicitly unsupported in
  mini-batch mode (would need the full population every epoch).
- **Hit the same silent RunPod payload bug a THIRD time**, now properly root-caused: IEEE-CIS's
  test set (~88.6K) is 10x Elliptic's, so the *always-returned* scalar arrays (test_scores, test_y,
  dist_to_fraud, dist_to_legit), previously left unrounded on the assumption they'd stay small,
  pushed a return_embeddings=True payload to ~17MB and failed; the SAME scalars alone (unrounded, no
  embeddings) at ~7MB had succeeded, bracketing the real threshold well below the ~30-40MB estimated
  in Run 79. Fixed properly this time: round all returned float arrays to 5 decimals always (not
  just under return_embeddings), and added a `train_fraud_dist_to_legit` scalar array instead of
  needing raw 128-dim embeddings for this specific bimodality check at all -- scalars are far
  cheaper than vectors and this analysis never needed the embeddings themselves.
- **Bimodality in the LEARNED embedding space** (plain random-triplet metric learning,
  best_epoch=257, test AUC=0.823 but F1=0.349/precision=0.055 -- badly uncalibrated at the default
  0.5 threshold, a calibration-gap pattern matching Elliptic's Run 81): pooled fraud (train 14,538 +
  test 3,083 = 17,621, no time binding) distance-to-legit-centroid gives bimodality coefficient
  0.871 and 2-comp GMM: mean=0.788/std=0.002/weight=0.758 vs. mean=0.875/std=0.102/weight=0.242.
  More balanced than the raw-feature check (75.8/24.2% vs. 98.9/1.1%) and closer in spirit to
  Elliptic's 61.7/38.3% split -- but the majority component's std (0.002) is even tighter than the
  suspicious one flagged in Elliptic's own GMM (std=0.014), and BIC still improves at 3 components
  (-47667 -> -119829 -> -127216). Same overfitting-prone pattern, not yet independently confirmed.
- **Honest verdict**: weaker, more ambiguous evidence than Elliptic. Elliptic's camo/obvious split
  was ultimately validated by an INDEPENDENT behavioral test -- camo-weighted training actually
  recovering hard-core cases, confirmed across 5 seeds -- not just the GMM fit alone. IEEE-CIS has
  no such independent confirmation yet (camo-weighted hasn't been tried here). Combined with no
  temporal cliff and a bigger proportional hard core, I would NOT yet claim camo-weighted's specific
  mechanism transfers to IEEE-CIS. Next step, if pursued: just try camo_weighted mining on IEEE-CIS
  directly and see if it empirically recovers hard-core cases the way it did on Elliptic -- the
  fastest real answer, consistent with this session's "test, don't just theorize" rule.

## [2026-07-22] Run 89 — Four new metric-learning loss terms on Elliptic, run in parallel: semi-hard mining is a new single-run best, SupCon dissociates aggregate F1 from hard-core recovery

Filled the gap identified this session: semi-hard mining (the literature-standard middle ground
between random sampling and pure batch-hard mining, Run 79) had never been tried despite having
both extremes. Implemented and ran 4 new loss variants in parallel (all in `evaluation/
metric_learning.py`, smoke-tested locally on Elliptic full-batch before dispatch -- 25-epoch runs,
sensible loss/val-AUC trends, no NaN/crash, before committing to full training):

- **Semi-hard negative mining** (Schroff et al. 2015, FaceNet): for each anchor/positive fraud
  pair, picks the closest legit negative that's still farther than the positive -- informative but
  not destabilizing, unlike Run 79's pure batch-hard (F1 crashed to 0.325).
- **Multi-prototype**: refits a 2-component GMM on fraud's distance-to-legit LIVE every epoch,
  samples positives from the same sub-cluster as the anchor, classifies nearest-of-3
  (obvious-fraud/camo-fraud/legit) -- the structural response to Run 82's ArcFace catch, instead of
  camo-weighted's soft reweighting of a single centroid.
- **SupCon** (Khosla et al. 2020): pulls every anchor toward ALL same-label batch members and away
  from ALL different-label members at once, not one triplet at a time.
- **Alignment + uniformity** (Wang & Isola 2020): alignment pulls same-class pairs together;
  uniformity spreads all points across the hypersphere without actively compressing toward a
  centroid -- targets Run 82's Center Loss trade-off (compression helped aggregate F1 but hurt
  hard-core recovery specifically).

Cross-referenced against the same 213-node hard core (fraud missed by BOTH RF and GraphSAGEDiff,
Run 67) used throughout this investigation:

| mining | hard-core recovered | lost (of 142) | test F1 | test AUC |
|---|---|---|---|---|
| semi_hard | **31/213 (14.6%)** | 0/142 | 0.673 | 0.857 |
| align_uniform | 28/213 (13.1%) | 0/142 | 0.685 | 0.811 |
| camo_weighted (reference, single seed) | 28/213 (13.1%) | 0/142 | -- | -- |
| multi_prototype | 23/213 (10.8%) | 0/142 | 0.661 | 0.848 |
| supcon | 3/213 (1.4%) | 0/142 | **0.724** | 0.846 |

- **Semi-hard mining is a new single-run best** on hard-core recovery (14.6%, edging out
  camo-weighted's single-seed 13.1%) with zero losses -- but per this session's own repeated lesson
  (DropEdge/degree-aware evaporating at n=10 seeds after looking good at n=5), a single-seed win
  over camo-weighted's already-5-seed-confirmed 12-27% range (mean 18.2%) is NOT yet a confirmed
  result. Needs the same multi-seed treatment before trusting it beats camo-weighted for real.
- **SupCon is the standout finding of this run, for the opposite reason**: it has the BEST aggregate
  F1 of all 4 (0.724, beating even camo-weighted's classification-head baseline) but recovers almost
  NOTHING of the hard core (1.4%, essentially the pre-existing ceiling). A clean, concrete example of
  exactly the trap this whole investigation has been guarding against: aggregate metrics can improve
  while the specific hard segment everyone cares about barely moves. SupCon is winning by getting
  better at the ALREADY-EASY majority of fraud, not by touching the hard core at all.
- **Multi-prototype underperforms camo-weighted** despite being the more "structurally principled"
  response to the bimodality finding -- soft reweighting of one centroid apparently generalizes
  better than hard-splitting into two centroids here, possibly because the live GMM refit is noisy
  epoch-to-epoch (unlike camo-weighted's smooth, always-differentiable soft weight).
- **Zero losses across all 4** -- consistent with camo-weighted and unlike Run 79's hard mining or
  Run 82's plain Center Loss, none of these four traded away already-correct cases for hard-core
  gains.
- **Next**: multi-seed confirm semi_hard before treating it as a real improvement over camo_weighted
  (same discipline as Run 84). align_uniform ties camo_weighted exactly on this single seed -- also
  worth a multi-seed check given it's cheap and has no compression-style trade-off built in.

## [2026-07-22] Run 90 — Electrical-network (Kirchhoff/harmonic) node scoring: reveals a major structural fact (graph components = time steps, exactly), the naive version doesn't help

User's idea (from a prior thermal-conductivity-of-composites GNN project, which trained a network
to solve a Kirchhoff problem with a parallel "leakage" conductance through the insulating matrix):
could an analogous electrical-network potential -- inject current at fraud nodes, ground at legit
nodes, solve for the resulting potential field -- work as a graph-TOPOLOGY-based fraud score on
Elliptic? This is a real gap: every method tried on Elliptic so far this session (Mahalanobis, MLP
probe, LOF, kNN, KDE, full-dim density) operated purely in learned-embedding geometry; none has
used the graph's actual topology as the signal.

- **First test (naive)**: built the graph Laplacian (undirected, unweighted, conductance=1/edge),
  grounded ALL train-legit nodes at potential 0, injected unit current at ALL train-fraud nodes,
  solved the reduced Kirchhoff system once (single sparse solve, <1s at this scale) for every
  test-node's resulting potential. **AUC = 0.500 exactly** -- not noise, a hard mathematical
  degeneracy.
- **Root cause, verified directly**: Elliptic's graph has **49 connected components, and they
  correspond EXACTLY 1:1 to the 49 time steps** (every component spans exactly one step; every
  step is exactly one component; confirmed via groupby on both directions). There are literally
  ZERO edges connecting nodes from different time steps anywhere in this dataset. Consequence:
  **0 of 8841 test nodes share a connected component with ANY train-labeled node.** No amount of
  graph message-passing, at any number of GNN layers or any electrical-network relaxation, can EVER
  route information from a training example to a test node through this graph's actual edges --
  it's structurally impossible, not a training or capacity limitation.
- **This reframes the whole regime-break investigation**: the GNN's only possible structural
  advantage over RF (a pure feature-based method) is LOCAL, within-step neighbor aggregation --
  never "borrowing" proximity to a known-fraud example from a different time period, because that
  path doesn't exist in the graph. Everything camo-weighted/semi-hard/etc. have improved this
  session is necessarily a FEATURE-space generalization effect (the learned encoder function
  applied to a new node's own local-neighborhood-enriched features), not a topological-proximity
  effect. Worth keeping in mind for any future claim about "the GNN uses graph structure to relate
  test fraud to training fraud" -- on Elliptic specifically, it cannot, by construction.
- **Adapted, re-tested version**: since cross-step edges don't exist, the only legitimate use of
  this idea is WITHIN a test step's own real edges, using a base model's own OUT-OF-SAMPLE
  predictions as the seed (not leaked test labels) -- the "Correct and Smooth" pattern (Huang et al.
  2020). Tested: a few rounds of degree-weighted Jacobi smoothing of RF's own predicted
  probabilities among each test node's within-step neighbors, blended back with the original score
  at varying strengths.
  | blend (own vs. neighbor-mean) | AUC |
  |---|---|
  | baseline RF (no smoothing) | 0.8671 |
  | 90% own / 10% neighbor | 0.8668 |
  | 70% own / 30% neighbor | 0.8650 |
  | 50% own / 50% neighbor | 0.8617 |
  | 30% own / 70% neighbor | 0.8542 |
  Monotonically worse the more neighbor-smoothing is applied; hard-core recovery at the gentlest
  setting (90/10): **0/213, zero**.
- **Conclusion**: within-step neighbors' RF scores aren't informative enough beyond what a node's
  own features already carry -- degree-weighted averaging just dilutes real per-node signal with
  noisier neighbor signal here. The specific naive/first-pass implementation of this idea is a real
  negative result on Elliptic. Not necessarily fully closed off -- a version weighting edges by
  transaction similarity/amount instead of uniform conductance, or smoothing a better-calibrated
  score (GraphSAGEDiff's own probs, or the metric-learning centroid score) instead of RF's, hasn't
  been tried -- but the core finding (components == time steps, structurally severing any train-
  to-test graph path) is the more important, durable result of this experiment regardless of the
  smoothing outcome.

## [2026-07-22] Run 91 — Same electrical-network idea on IEEE-CIS: structurally viable here (unlike Elliptic), but the signal only reaches the easy fraud, not the hard core

Two follow-up questions from Run 90: (1) is Elliptic's component=step degeneracy really from the
raw dataset, not our own train/test split code? (2) does IEEE-CIS have the same problem, and can we
use real transaction amounts (Elliptic has none) to weight conductance?

- **(1) confirmed independently**: rebuilt the graph directly from `elliptic_txs_edgelist.csv` +
  `elliptic_txs_features.csv`, zero involvement of our preprocessing/masking code. Identical result:
  49 components, each spanning exactly one step, 0 edges crossing any step boundary. This is
  intrinsic to Elliptic's public release, not an artifact of our split.
- **Amount-weighting is not constructible on Elliptic**: `elliptic_txs_edgelist.csv` has only
  `txId1, txId2` (no weight/amount column, ever existed publicly); the 165 node features are
  Elliptic's own anonymized aggregates, not dollar values; zero duplicate edges (so no multiplicity
  proxy either).
- **(2) IEEE-CIS's graph is structurally the OPPOSITE of Elliptic's**: 1534 components, but the
  LARGEST one alone holds 99.0% of all 590,540 nodes, and **99.7% of test nodes (88,311/88,581)
  share a component with a training-labeled node** -- vs. Elliptic's 0%. Makes sense given the
  construction: IEEE-CIS connects transactions sharing a card1/addr1 to their temporal neighbors on
  that key (data/ieee_cis_preprocess.py's build_edges), and shared entities naturally get reused
  across the train/test boundary, unlike Elliptic's time-isolated snapshots.
- **Tested the harmonic-potential idea for real here** (ground train-legit at 0, inject unit
  current at train-fraud, solve the reduced Kirchhoff system via conjugate gradient -- converged in
  1.5s despite 590K nodes / 20.5M undirected edges): **AUC = 0.6835 using PURE GRAPH TOPOLOGY, zero
  node features** -- genuinely well above random, and a real, structurally-grounded result (unlike
  Elliptic's forced 0.5).
- **But the signal is concentrated entirely in the EASY fraud, not the hard core**: mean harmonic
  potential for hard-core fraud (missed by both RF and GNN, n=1895) is 0.049 -- barely above
  legit's own mean (0.030) -- while fraud already caught by both RF and GNN has mean 0.157, more
  than 3x higher. AUC restricted to hard-core-fraud-vs-legit is only 0.6005 (near the noise floor),
  vs. 0.6835 in aggregate. The topology signal is real but it's finding the SAME fraud RF/GNN
  already catch via features, not the population where a new signal source would actually help.
- **Stacking with RF makes things monotonically worse, not better**: blending rank-normalized RF
  score with rank-normalized harmonic potential at weight w:
  | w (harmonic weight) | AUC | hard-core flagged @ matched threshold |
  |---|---|---|
  | 0.0 (pure RF) | 0.8917 | 398/1895 |
  | 0.2 | 0.8820 | 341/1895 |
  | 0.5 | 0.8332 | 314/1895 |
  | 1.0 (pure harmonic) | 0.6835 | 158/1895 |
  No sweet spot -- pure RF dominates at every blend weight on both metrics.
- **Conclusion**: same pattern as Run 89's SupCon finding, in a different guise -- a real signal
  that improves on the easy majority while leaving the hard segment untouched, which looks good in
  aggregate and does nothing for the actual problem. IEEE-CIS's graph genuinely supports this family
  of ideas structurally (unlike Elliptic), but this specific realization (unweighted, train-fraud-
  as-source) doesn't crack the hard core. Amount-weighting (via `TransactionAmt`, which IEEE-CIS
  does have, unlike Elliptic) hasn't been tried yet -- open question whether reweighting by
  transaction size concentrates the topology signal differently, but given the signal already
  correlates with "how well-connected to OTHER already-easy fraud" rather than anything hard-core-
  specific, I wouldn't bet on amount-weighting changing this qualitatively.

## [2026-07-22] Run 92 — GraphSAGEDiff + GRU (per-entity temporal sequence) on IEEE-CIS: net negative, not just a wash

Follow-up to the 2026-07-22 discussion of diffusion forcing / self-forcing / framepack: none of
those transplant directly, but the underlying idea -- a per-entity temporal sequence encoder --
was real and testable on its own. IEEE-CIS only (Elliptic's nodes are one-off transactions, no
recurring entity to build a sequence from).

- **New infra**: `data/ieee_cis_preprocess.py`'s `build_entity_sequences` precomputes, for each
  transaction, its own card1's previous 8 transactions (causal by construction, oldest-first,
  right-aligned padding). 576,987/590,540 nodes (97.7%) have >=1 prior transaction. New
  `GraphSAGERNN` model (models/gnn/graphsage_rnn.py): GraphSAGEDiff branch (card1/addr1-sharing
  edges, unordered mean/diff aggregation) concatenated with a GRU branch (ordered per-entity
  history) before a linear classifier. Separate mini-batch training loop
  (evaluation/gnn_rnn_hybrid.py), same rationale as metric_learning.py/gnn_rf_hybrid.py being
  separate from train_gnn.py.
- **Caught a real bug locally before any RunPod dispatch**: the GNN branch's embedding covers the
  WHOLE NeighborLoader-sampled subgraph (seed nodes + neighbors), while the GRU branch only ever
  has seed-node sequences -- needed to slice the GNN embedding to the first seq_x.shape[0] rows
  before concatenating. Found via a local forward/backward-pass sanity check with fabricated
  tensors, not on RunPod.
- **Result, cross-referenced against the same 213-analogue hard core** (RF+GNN both miss,
  n=1895 for IEEE-CIS, corrected for the same NeighborLoader adjacent-seed-swap quirk as Runs 88/91,
  54/88581 positions, clean pairs):
  | | AUC (full test) | F1 (fraud) | hard-core recovered | lost (of 625 caught-by-both) |
  |---|---|---|---|---|
  | RF | 0.892 | 0.461 | -- | -- |
  | GraphSAGEDiff (plain) | 0.855 | 0.345 | -- (baseline) | -- |
  | **GraphSAGEDiff + GRU** | **0.842** | **0.330** | 75/1895 (4.0%) | **105/625 (16.8%)** |
  Worse than plain GraphSAGEDiff on EVERY aggregate metric, and net negative on hard-core trade
  (-30 correctly-classified fraud cases overall: +75 hard-core gained, -105 previously-easy lost).
  AUC restricted to hard-core-fraud-vs-legit specifically: 0.764, below plain GNN's 0.780.
- **Honest diagnosis**: the GRU branch is likely mostly REDUNDANT with the GNN branch, not
  complementary -- IEEE-CIS's edges are already built from card1/addr1-sharing temporal neighbors
  (data/ieee_cis_preprocess.py's build_edges), so the GNN is already seeing much of the same
  underlying information the GRU sees, just via a different aggregator (unordered mean/diff vs.
  ordered GRU). Combining two branches trained jointly from scratch with a single linear head
  converged early (best_epoch=21) -- plausibly a harder joint optimization landscape than either
  branch alone, landing on a worse local optimum rather than adding real new signal.
- **Conclusion**: a clean, honest negative result -- this specific architecture doesn't help and
  actively hurts net fraud recall. Doesn't rule out per-entity temporal modeling entirely (a
  version with a separately-pretrained sequence encoder, or one operating on a genuinely
  independent signal from the graph edges rather than largely the same underlying transactions,
  might fare differently), but this realization of the idea is a real negative, not worth pursuing
  further as-is.

## [2026-07-22] Run 93 — Camo-weighted metric learning on IEEE-CIS: an 89% hard-core "recovery" that's actually a threshold artifact; semi-hard mining wired into mini-batch mode; FAGCN fails again in metric-learning mode

Camo-weighted mining had never been tried on IEEE-CIS (only random mining was, for Run 88's
bimodality check) despite being the one lever that reliably worked on Elliptic. Dispatched it
(mini-batch, same infra as the plain metric-learning run).

- **Headline number, before checking**: recovered 1689/1895 (89.1%) of the hard core, **zero
  losses** among the 625 already caught by both RF and GNN. Looked like a huge win.
- **It isn't.** `predicted_positive_rate` for this run is **0.579** -- the model flags 58% of
  every single test node as fraud, against IEEE-CIS's true ~3.5% rate. That alone explains the
  "recovery": if you call more than half the graph fraud, you trivially catch almost everything,
  hard core included, with nothing lost from the already-easy population either (same mechanism as
  the Run 84 multi-seed-union trap on Elliptic, just via badly miscalibrated centroid distances
  instead of ensembling). Checked the threshold-free version to see if there's real signal
  underneath the bad calibration:
  | | AUC, hard-core-fraud vs. legit |
  |---|---|
  | RF | 0.827 |
  | Plain GraphSAGEDiff | 0.780 |
  | GraphSAGEDiff+GRU (Run 92) | 0.764 |
  | **Camo-weighted metric learning** | **0.742** |
  Camo-weighted's actual ranking ability on the hard core is WORSE than every other method tried
  on IEEE-CIS, not better. The 89.1%/zero-losses headline was entirely a threshold artifact, not
  real discrimination -- caught and corrected before being reported as a win, not after.
- **Semi-hard mining wired into mini-batch mode** for the same test (refactored
  `_semi_hard_triplets` into sampling + a shared `_semi_hard_negatives` selection function so
  mini-batch mode can reuse it against the fixed legit_ref pool, same pattern camo_weighted's live
  centroid already uses -- verified the refactor didn't change full-batch Elliptic behavior via a
  local smoke test first). 3-epoch RunPod smoke test passed cleanly; full run dispatched, result
  pending -- given camo-weighted's calibration problem here, will check the threshold-free
  hard-core AUC FIRST before reporting any recovery percentage from it.
- **FAGCN as a metric-learning encoder on Elliptic (camo_weighted mining, 3 seeds, 30 epochs)**:
  fails again, same signature as its earlier classification-head failure (Run ~60s-era, F1=0.599
  vs baseline 0.748, std=0.123 across seeds). Here: val AUC peaks at epoch 1-4 then DECLINES with
  more training in every seed (e.g. 0.874->0.856, 0.852->0.806, 0.870->0.811) while triplet loss
  keeps smoothly decreasing throughout -- textbook overfitting shape, but the fact that the
  identical instability signature (early peak, high variance, decline with more training) shows up
  in TWO completely different loss functions (cross-entropy classification AND triplet metric
  learning) points to something more structural than fixable-by-regularization: FAGCN's signed
  tanh gate (can actively flip from amplifying to subtracting a neighbor's contribution) is a much
  less constrained function to optimize than a sigmoid gate, plausibly destabilizing the embedding
  geometry itself during training. Test AUC (0.77-0.80 across 3 seeds) well below GraphSAGEDiff's
  established camo-weighted baseline (0.878 on Elliptic). Deprioritizing FAGCN as a metric-learning
  encoder -- this is its second failure with the same signature, not a hyperparameter gap.
- **Takeaway so far on IEEE-CIS's hard core**: nothing tried across Runs 88/91/92/93 (raw-feature
  bimodality, learned-embedding bimodality, harmonic-potential topology, GNN+RNN temporal
  sequence, camo-weighted metric learning) has produced a real, threshold-free improvement over
  plain RF/GraphSAGEDiff on the hard-core-fraud-vs-legit ranking task specifically. IEEE-CIS's
  hard core looks meaningfully more resistant than Elliptic's was to every lever that worked there.

- **Semi-hard mining's full IEEE-CIS run landed too**: best_epoch=75, aggregate AUC=0.822,
  same miscalibration signature as camo_weighted (predicted_positive_rate=0.589, ~59% of all test
  nodes flagged fraud). Threshold-free hard-core AUC: **0.7468** -- essentially identical to
  camo_weighted's 0.742, and still worse than RF (0.827) and plain GraphSAGEDiff (0.780).
- **All three metric-learning mining strategies tried on IEEE-CIS (random: Run 88, camo_weighted
  and semi_hard: this run) converge to the same ~0.74-0.75 hard-core AUC band, all below the
  simpler classification-head approaches.** The mining strategy that mattered enormously on
  Elliptic (camo_weighted +5-8 points of hard-core AUC over random, semi_hard the single-run best)
  makes no real difference here -- consistent with Run 88's finding that IEEE-CIS's bimodal
  camo/obvious split, unlike Elliptic's, was weak/ambiguous and never independently confirmed.
  Whatever mechanism made mining strategy matter on Elliptic (a real, exploitable, rare
  sub-archetype) doesn't appear to be present here for any mining strategy to exploit.

## [2026-07-22] Run 94 — Adding V-columns + UID (client) reconstruction features: the first real, clean win on IEEE-CIS's hard core

User shared the actual 1st/2nd-place Kaggle solution writeups for this competition (Yakovlev &
Deotte; Bryansky/CPMP/Giba). Neither winning team used a neural network or GNN at all -- plain
LightGBM/XGBoost/CatBoost won. Their edge was feature engineering, and UID (client) reconstruction
specifically mattered more than the raw V-columns (1st place: local val AUC 0.9245 -> 0.9377 from
UIDs alone, a bigger jump than most feature blocks). Implemented both:

- **V-columns** (`data/ieee_cis_preprocess.py`): added V1-V339 EXCEPT V322-V339, which the 1st-place
  writeup explicitly found failed their "time consistency" check (good on training period, didn't
  generalize forward) -- an evidence-based exclusion, not a guess, and directly the same
  generalization-failure mode this whole investigation has been chasing. Naive fillna(-999), no
  PCA/dedup of correlated blocks (the winners did that; not yet implemented here).
  Full-scale RF (V-cols only, no UID feats): F1=0.7300, AUC=0.8908 (vs. Run 83's F1=0.7235 without
  V-cols) -- small, real, modest gain.
- **UID reconstruction** (`build_uid_features`): card1+addr1+P_emaildomain+(day-D1) as the client
  key (day = TransactionDT // 86400, D1 tracks days-since-first-seen for a card, so day-D1
  approximates an "account creation day" anchor -- CPMP's exact recipe). 5 new features: prior
  transaction count, time-since-previous, mean/std of previous transaction amounts, and a
  frequency-smoothed TRAIN-only target encoding of the uid (CPMP's recipe: blend uid's own train
  fraud rate with the global train mean, weighted by count -- unseen/rare uids fall back toward
  the overall rate).
- **Deliberately made CAUSAL, unlike the source writeup**: CPMP's own `_next_dt`/`_next_amt`
  features use `gr.TransactionDT.shift(-1)` -- the NEXT transaction's timing -- which for a
  train-period row can reference a val/test-period transaction that doesn't exist yet at real
  scoring time. Implemented the past-only version instead (same convention as
  `build_entity_sequences`), consistent with this whole project's discipline about exactly this
  leakage class (see data/temporal_edges.py).
- **Result is a big, real jump, validated on both a 100K subsample and the full 590K-node dataset**:
  | | F1 | AUC | recall_fraud |
  |---|---|---|---|
  | Baseline (no V, no UID) | 0.7235 | ~0.892 | -- |
  | + V-columns only | 0.7300 | 0.8908 | 0.344 |
  | **+ V-columns + UID features** | **0.7550** | **0.8967** | **0.460** |
  The target-encoded uid feature is THE single most important feature in the model by a wide
  margin -- rank #1 of 380 features, importance 0.284 (next highest feature: ~0.02-0.03) -- at both
  subsample and full scale. Exactly matches both winning teams' own account of what mattered most.
- **Decisive test: does this new RF crack the hard core (missed by both old RF and GraphSAGEDiff,
  n=1895) for real, not via a miscalibrated threshold like Run 93's camo-weighted mirage?**
  | | AUC, hard-core-fraud vs. legit | hard-core recovered |
  |---|---|---|
  | Old RF (V-cols only) | 0.827 | -- |
  | Plain GraphSAGEDiff | 0.780 | -- |
  | Best metric-learning variant (Run 93) | 0.747 | worst of everything, threshold artifact |
  | **New RF (V-cols + UID feats)** | **0.839** | **429/1895 (22.6%), only 37/625 lost** |
  Calibration sanity check: predicted_positive_rate=0.026 vs. true rate 0.035 -- reasonable, NOT
  the ~58%-flag-everything pattern that invalidated Run 93's camo-weighted number. This is the
  first genuine, clean win on IEEE-CIS's hard core after Runs 88/91/92/93 all failed or turned out
  to be artifacts. It came from feature engineering, not from the graph or the loss function --
  consistent with both winning teams never having needed a GNN at all.
- **Next**: rebuild `build_edges`/`build_entity_sequences` using the precise `uid` (not raw
  card1/addr1 separately) as the entity-grouping key, so the GNN's own neighbor aggregation
  benefits from the same more-precise client identity, not just the RF's node features. Also worth
  testing GraphSAGEDiff (classification head) and the mini-batch metric-learning variants on this
  richer feature set now that RF has a new, higher bar to beat.

## [2026-07-22] Run 95 — uid wired into the graph itself; GraphSAGEDiff classification on the richer feature set is a clean NEGATIVE, opposite direction from RF

Wired `uid` (Run 94's client reconstruction) into graph construction: `build_uid` extracted as a
shared helper, `uid_graph` config flag adds "uid" as a THIRD entity-edge column alongside
card1/addr1 (not a replacement -- more, higher-precision edges, not fewer/cruder ones). Validated
on a 100K subsample first: edge count only grows modestly (+0.7%, 6.20M->6.25M) since same-uid
pairs are mostly already implied by same-card1/same-addr1 pairs; the small increase specifically
recovers cases where the per-column max_node_degree cap missed a same-uid pair due to interleaving
with other transactions on a busy shared card1/addr1.

Dispatched GraphSAGEDiff classification (mini-batch) on the full richer graph+features
(configs/ieee_cis_graphsage_diff_full_minibatch.yaml), smoke-tested clean first.

- **Aggregate result is WORSE than the plain-feature baseline, the opposite direction from RF**:
  | | AUC | F1 | recall_fraud |
  |---|---|---|---|
  | Plain GraphSAGEDiff (plain features) | 0.855 | 0.664 | 0.263 |
  | RF (V-cols + UID feats, Run 94) | 0.897 | 0.755 | 0.460 |
  | **GraphSAGEDiff (V-cols + UID feats)** | **0.762** | **0.662** | **0.224** |
  RF improved with the richer feature set; GraphSAGEDiff got WORSE, on the exact same features.
- **Hard-core check confirms it's worse, not just a wash**: recovered 274/1895 (14.5%) but LOST
  307/625 (49.1%!) of cases previously caught by both old RF and old GraphSAGEDiff -- a large net
  negative (274 gained vs. 307 lost). Threshold-free AUC on hard-core-fraud-vs-legit: **0.7253**,
  the WORST GNN-classification result seen on this population, below even the plain-feature
  baseline (0.780).
- **Likely explanation**: GraphSAGEDiff's embed() computes `[x, neighbor_mean, x-neighbor_mean]`
  BEFORE the first SAGEConv -- with 380 raw features (up from ~54), the first layer's effective
  input is ~1140-dim, mostly noisy/uninformative given the naive fillna(-999) V-column inclusion
  (no PCA/dedup of correlated blocks, which the actual competition winners did specifically for
  this reason). A fixed-capacity 2-layer, hidden_dim=128 GNN with a single linear projection per
  layer plausibly struggles to compress this much higher-dimensional, noisier input within the
  same training budget, where RF's per-split implicit feature selection handles high-dimensional/
  noisy tabular data far more gracefully. This reinforces why the winning solutions didn't skip the
  V-column dimensionality-reduction step -- it may matter MORE for a GNN's fixed linear/conv
  layers than for tree ensembles.
- **Metric-learning run on the same richer graph/features dispatched separately, result pending**
  (an earlier attempt's job record vanished from RunPod entirely -- "job not found" 404 despite
  having been dispatched and (per the wrapper poller) never reaching COMPLETED/FAILED; redispatched
  cleanly, not yet landed).
- **Takeaway so far**: richer features are a real, clean win for RF specifically (Run 94) but hurt
  GraphSAGEDiff's classification head just as clearly -- reinforcing this investigation's repeated
  finding that RF's advantage isn't really about "using the graph" or "not using the graph," it's
  about how gracefully each model handles a wide, correlated, partially-noisy feature space. Naive
  feature richness is not automatically good for a GNN the way it is for RF.
- **Metric learning on the same richer feature set landed too (original dispatch's job record
  vanished from RunPod entirely -- genuine 404 "job not found" despite having been dispatched and
  never reaching a terminal state per the wrapper poller; redispatched cleanly)**: same story.
  Aggregate AUC=0.790 (down from plain-feature random mining's 0.823). Headline hard-core
  "recovery" of 69.2% (1312/1895) is, AGAIN, a threshold artifact (predicted_positive_rate=0.373,
  vs. true rate 0.035) -- threshold-free hard-core AUC is **0.7191**, the WORST result of
  everything computed on IEEE-CIS so far, below even GraphSAGEDiff classification's own regression
  (0.725) on the identical richer feature set.
- **Complete picture now**: richer features (V-cols + UID) help RF (0.827->0.839 hard-core AUC)
  and hurt BOTH GNN training approaches tried on them (classification: 0.780->0.725; metric
  learning: 0.747->0.719). This isn't noise -- it's the same direction, same magnitude ballpark,
  across two very different GNN objectives on the same richer input. Naive high-dimensional
  feature inclusion is specifically bad for this GNN architecture family, not just an
  classification-head-specific quirk.

## [2026-07-22] Run 96 — Two leakage catches: harmonic-potential feature is pure label-leakage; Run 94's target encoding had a real, milder self-reference bug, now fixed and CONFIRMED BETTER

Testing "add Run 91's harmonic-potential score as an RF feature" (to combine topology signal with
the winning feature-engineering approach) surfaced a real bug: for every TRAIN row, harmonic
potential exactly EQUALS the train label (0 or 1) -- Run 91's construction used train fraud/legit
as the Kirchhoff system's fixed boundary conditions, so a train row's own "solved" value is just
its own label restated. Confirmed directly (`np.array_equal(train_harmonic, train_y)` == True).
Feeding this as a feature crashed recall_fraud to 0.006 (RF learned "value very close to exactly
1.0 -> fraud" from training, which almost never occurs in test's genuinely-continuous values) while
becoming the #1 most "important" feature by a wide margin (0.371) -- exactly the kind of
suspiciously-dominant-importance signature that should trigger a leakage check, and did. Retracted;
this specific test doesn't tell us anything real about whether topology-plus-RF helps.

That check prompted a look at Run 94's ALREADY-REPORTED target-encoding feature for the same class
of bug, milder but real:
- **Found it**: for singleton uid groups (25.7% of train rows), the unfolded smoothed target
  encoding is `(1*own_label + 10*global_mean)/11` -- roughly 9% of a row's own encoded value comes
  from its own label (fraud singleton ~0.123 vs. legit singleton ~0.032, not identical like the
  harmonic-potential bug but a real, non-negligible self-reference).
- **Important scoping**: this did NOT invalidate Run 94's reported TEST metrics -- the encoding
  was fit using only train labels, and test rows never contributed to it at all, so F1=0.755/
  hard-core AUC=0.839 were honestly measured on genuinely held-out data. What was at risk: RF
  TRAINED on slightly self-referential values for ~26% of training rows, which could bias what
  relationship it learned for low-count UIDs, in an unpredictable direction.
- **Fixed with standard practice**: K-fold (5-fold) out-of-fold target encoding for TRAIN rows
  specifically (each row's encoding comes only from OTHER folds' uid statistics); val/test rows
  keep the single full-train-based encoding, unaffected either way since they never entered the
  aggregation basis. `build_uid_features` now takes `seed`/`n_folds`.
- **Re-verified full-scale RF with the fix -- result is BETTER, not worse**:
  | | F1 | AUC | hard-core AUC (vs. legit) | hard-core recovered |
  |---|---|---|---|---|
  | Leaky (Run 94) | 0.7550 | 0.8967 | 0.839 | 429/1895 (22.6%), 37 lost |
  | **Fixed (this run)** | **0.7687** | **0.9086** | **0.8561** | 386/1895 (20.4%), 45 lost |
  Every aggregate and threshold-free metric improved; raw hard-core recovery count is very slightly
  lower with slightly more losses, but net effect on genuine generalization is clearly positive.
  Consistent with what you'd hope: removing an artificially-inflated training signal let RF learn
  a more honestly-calibrated relationship instead of a slightly miscalibrated one.
- **Lesson for the rest of this investigation**: any feature engineered from the training labels
  themselves (target/mean encoding, or anything derived from a boundary-condition-style construction
  like Run 91's harmonic potential) needs an explicit check for row-level self-reference before
  trusting it, regardless of how well-motivated the underlying idea is -- a suspiciously dominant
  feature importance (0.28-0.37 vs. everything else at 0.01-0.03) is the first, cheap warning sign.
  Run 94's headline result stands, and is now slightly BETTER with the leak properly closed.

## [2026-07-22] Run 97 — GNN+RF hybrid on IEEE-CIS's richer feature set: underperforms plain RF, the OPPOSITE of its role on Elliptic

Flagged as a "next" step back in Run 83 (IEEE-CIS GNN+RF hybrid at full scale, never dispatched --
earlier attempts predate the mini-batch fix and failed on the same full-batch OOM every other
full-batch IEEE-CIS attempt hit). Added mini-batch support to evaluation/gnn_rf_hybrid.py
(`_train_gnn_for_embeddings_minibatch` + `_run_minibatch`, reusing metric_learning.py's `_embed_nodes`
helper rather than reimplementing it), smoke-tested clean, dispatched on the richer (V-cols +
leak-fixed UID) feature/graph set from Runs 94-96.

- **Result: F1=0.7319, AUC=0.8785 (gnn_best_epoch=24)** -- barely moved from a 2-epoch smoke test
  (F1=0.7294, AUC=0.8783), and clearly BELOW plain RF-alone on the same features (Run 96:
  F1=0.7687, AUC=0.9086).
- **Hard-core check confirms it, not just the aggregate**:
  | | AUC, hard-core-fraud vs. legit | hard-core recovered |
  |---|---|---|
  | RF alone (Run 96, leak-fixed) | **0.8561** | 386/1895 (20.4%), 45 lost |
  | **GNN+RF hybrid (this run)** | **0.8142** | 335/1895 (17.7%), 132 lost |
  Worse ranking AND a worse recovery/loss trade -- more losses for fewer gains than RF alone.
- **This is the opposite of the hybrid's role on Elliptic**, where it's the single best-performing
  lever in the project's history (Run 55, closes most of RF's gap over a plain GNN). Consistent
  with Run 95's finding: GraphSAGEDiff's classification-head performance itself REGRESSED on this
  richer feature set (AUC 0.855->0.762), meaning the embeddings this hybrid concatenates with raw
  features are of genuinely lower quality here than on Elliptic -- adding ~128 noisy/low-value
  embedding dimensions to an already-strong 380-dim raw feature set dilutes RF rather than helping
  it, even though RF is far more robust to noisy additions than the GNN itself was.
- **Conclusion**: on IEEE-CIS specifically, plain RF on the well-engineered raw feature set (Run 96,
  F1=0.7687, AUC=0.9086, hard-core AUC=0.8561) remains the best result of the whole investigation.
  Every GNN-involving approach tried on the richer feature set (classification: Run 95; metric
  learning: Run 95; GNN+RF hybrid: this run) has underperformed RF alone. The graph/GNN family
  hasn't yet found anything on IEEE-CIS that beats good tabular feature engineering by itself.

## [2026-07-22] Run 98 — Two Elliptic architecture fixes, finished and re-tested: both work in the sense of "no longer broken," neither beats camo-weighted on the hard core

Two diagnosed-but-never-finished fixes from earlier this session, completed and dispatched in
parallel, each smoke-tested locally first.

**Fix 1 — GraphSAGESpectral raw-x band.** Run 62 found the original version (bands = differences
and smoothed copies of x, but never untouched x itself) failed in 5/5 seeds, root-caused to
exactly that gap (GraphSAGEDiff always keeps raw x as one of its three concatenated terms; RF's
edge partly comes from exploiting raw feature values directly). Added raw x as its own band
(`models/gnn/graphsage.py`).
- **Aggregate result: fixed.** F1=0.7506-0.7558 (two dispatches), AUC=0.854-0.855 -- back to
  parity with the GraphSAGEDiff baseline (~0.7473-0.7484), a large jump from the broken version's
  F1=0.7211. The diagnosis was right.
- **Hard-core result: not actually better.** Recovered only 2/213 (0.9%) and LOST 8/142 -- a net
  negative on the specific population this investigation cares about, despite the healthy
  aggregate numbers. Fixing the bug made it competitive again, not genuinely better than the
  baseline it was compared against.

**Fix 2 — Directly-supervised GraphSAGEGated gate + camo-weighted metric learning (not BCE).**
`gate_aux_weight` (already coded in train_gnn.py, never run as a logged experiment) supervises the
per-edge gate directly against y_src==y_dst on train edges, instead of only the indirect signal
through the main loss that showed no effect in Runs 59-61. Combined with camo_weighted mining as
the main objective (not BCE/focal classification) on the theory that BCE isn't a good enough
objective for this task's structure, so the gate deserved a fairer test under a better main loss
too. New `evaluation/metric_learning.py` support for `gate_aux_weight`.
- **Aggregate result**: F1=0.6908, AUC=0.8525 (best_epoch=297) -- plausible, not obviously broken.
- **Hard-core result: worse than plain camo-weighted.** Recovered only 10/213 (4.7%), zero losses
  -- clean, but clearly BELOW plain camo-weighted's already-established 12-27% (5-seed confirmed)
  range. Direct gate supervision, even combined with the better main objective, didn't help the
  specific problem it was meant to help with.

**Conclusion**: both fixes did what they were diagnosed to do (removed a specific implementation
gap / added a specific missing supervision signal) and both are now legitimately working,
non-broken architectures -- but neither adds anything beyond the existing best lever (camo-weighted
metric learning alone, 12-27% hard-core recovery, 5-seed confirmed) on the metric that actually
matters. "Fixed the diagnosed bug" and "now beats the state of the art" turned out to be different
claims. Single-seed for both -- given this session's own repeated lesson about small-n results
(DropEdge/degree-aware evaporating at n=10 after looking good at n=5), neither result is strong
enough to warrant a multi-seed follow-up given they're already below the existing bar on a single
seed each.

## [2026-07-22] Run 99 — gate_aux_weight sweep (0.0-5.0): 0.25 beats plain camo-weighted on a single seed, multi-seed confirmation dispatched

Followed up Run 98's single-point gate_aux_weight=1.0 test (worse than plain camo-weighted) with a
grid sweep over [0.0, 0.1, 0.25, 0.5, 2.0, 5.0] via infra/sweep.py, checking hard-core recovery
(not just aggregate F1/AUC) at each point:

| gate_aux_weight | test F1 | hard-core recovered | lost |
|---|---|---|---|
| 0.0 | 0.660 | 22/213 (10.3%) | 5/142 |
| 0.1 | 0.701 | 8/213 (3.8%) | 0/142 |
| **0.25** | 0.633 | **43/213 (20.2%)** | **0/142** |
| 0.5 | 0.684 | 20/213 (9.4%) | 0/142 |
| 2.0 | 0.660 | 27/213 (12.7%) | 0/142 |
| 5.0 | 0.653 | 10/213 (10.8%) | 0/142 |

- **gate_aux_weight=0.25 recovers 20.2% of the hard core with ZERO losses on a single seed --
  better than plain camo-weighted's single-seed 13.1%.** No monotonic trend across the sweep
  (0.1 is clearly worse, 0.25 clearly better, 0.5-5.0 back to middling) -- consistent with a
  genuinely noisy small-n landscape rather than a clean dose-response relationship, exactly the
  situation this session has repeatedly learned NOT to trust from a single seed (DropEdge/
  degree-aware both looked good at n=5 and evaporated at n=10).
- **Multi-seed (10-seed) confirmation of gate_aux_weight=0.25**: CONFIRMED a mirage, not a real
  improvement. Seeds 0-9: 14.6%, 15.5%, 16.4%, 13.6%, 12.2%, 13.1%, 11.7%, 16.9%, 10.3%, 19.2% --
  **mean 14.4%, std 2.6%, zero losses in every single seed.** Below plain camo-weighted's already-
  established 5-seed mean of 18.2% (range 12-27%). The single-seed sweep result that looked like a
  new best (43/213=20.2%) was exactly the small-n mirage this session has now caught three times
  (DropEdge, degree-aware, and now this) -- a genuinely noisy landscape at n=1 that regresses to
  the mean (or below) once properly confirmed.
- **Conclusion**: directly-supervised gate + camo-weighted metric learning, at its best-found
  weight, still doesn't beat plain camo-weighted alone. The gate mechanism (in either its indirect,
  BCE-direct, or now weight-swept forms) has not added anything across every variant tried this
  session (Runs 59-61, 98, 99). Plain camo-weighted (Run 84, 12-27%/mean 18.2%, 5-seed) remains the
  standing best Elliptic hard-core lever.

## [2026-07-22] Run 100 — GraphSAGECamoAgg: extends camo-weighting from the loss to the aggregation step; promising but not yet confirmed

New model (`models/gnn/graphsage.py`): reuses camo-weighted mining's own distance-to-legit-centroid
signal to weight neighbor AGGREGATION, not just the loss -- neighbors that look confidently legit
(fixed raw-feature-space distance to the train-legit centroid, set once via set_legit_centroid()
before training) get more weight in neighbor_mean, so GraphSAGEDiff's deviation feature
(x - neighbor_mean) is computed against a cleaner baseline instead of one contaminated by
already-ambiguous neighbors. Different mechanism from GraphSAGEGated's learned similarity gate
(Runs 59-61, 98-99, no real effect across every variant) -- reuses an already-validated signal at
a different pipeline point instead of learning a new one from scratch. Smoke-tested locally before
dispatch (10 epochs, no crash).

10-seed multi-seed run (camo_weighted mining, same as plain camo-weighted's own validation):
| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| recovery | 19.2% | 19.7% | 17.8% | 33.3% | 12.2% | 24.4% | 9.4% | 14.1% | 13.1% | 45.5% |

- **Mean 20.9% (vs. plain camo-weighted's 18.2%), zero losses in every single seed** -- same clean
  safety profile as camo-weighted, no regressions on already-correctly-classified cases anywhere.
- **But std is much higher (10.5%) and the mean is substantially pulled up by two outlier seeds**
  (33.3%, 45.5%). **Median across all 10 seeds (~18.5%) is nearly identical to camo-weighted's
  established mean** -- the honest read is "raises the ceiling on a good seed, without clearly
  raising the typical case," not a uniform improvement.
- **Paired check on the 4 seeds cached for both methods** (apples-to-apples, same exact seeds):
  camo-agg wins 3/4 (seed 0: 23.5%->19.2% worse; seed 1: 15.5%->19.7% better; seed 2: 12.2%->17.8%
  better; seed 3: 27.2%->33.3% better), mean 19.6%->22.5%. Wilcoxon p=0.375 at n=4 -- directionally
  positive but nowhere near enough statistical power to call this confirmed.
- **Conclusion**: the most promising result of this "extend camo-weighting" sub-investigation so
  far (Run 98-99's gate variants were clean negatives; this is a real, if not yet statistically
  confirmed, positive direction). Not treating this as a win yet given this session's own repeated
  lesson about small-n/high-variance results (DropEdge, degree-aware, gate_aux_weight=0.25 all
  looked good on a subset of seeds and regressed to baseline or worse on full confirmation). Next:
  a properly matched larger-n comparison (same seed set, more seeds) before drawing a real
  conclusion either way.

## [2026-07-22] Run 101 — Properly matched 10-seed comparison: GraphSAGECamoAgg is a WASH vs. plain camo-weighted, not an improvement

Dispatched plain camo_weighted on the exact same seeds (0-9) as Run 100's camo-agg run, for a real
paired comparison instead of mixing old cached seeds with new ones.

| seed | camo_weighted | camo_agg |
|---|---|---|
| 0 | 25.4% | 19.2% |
| 1 | 19.2% | 19.7% |
| 2 | 11.3% | 17.8% |
| 3 | 27.2% | 33.3% |
| 4 | 14.1% | 12.2% |
| 5 | 20.7% | 24.4% |
| 6 | 23.0% | 9.4% |
| 7 | 27.7% | 14.1% |
| 8 | 15.0% | 13.1% |
| 9 | 7.5% | 45.5% |

- **camo_weighted: mean=19.1%, std=6.6%, median=20.0%. camo_agg: mean=20.9%, std=10.5%,
  median=18.5%.** Paired Wilcoxon signed-rank: **statistic=27.5, p=1.0** -- as flat a result as
  this test produces, essentially zero evidence of a real difference. Wins split exactly 5/10.
  Zero losses for either method in every single seed (both fully safe).
- **Conclusion: GraphSAGECamoAgg is a wash, not an improvement.** Run 100's promising-looking mean
  (20.9% > 18.2%) was, once properly matched against the SAME seeds rather than a smaller cached
  set, revealed to be within noise -- camo_agg trades a couple of high outlier seeds (33.3%, 45.5%)
  for a couple of low ones (9.4%, 14.1% where camo_weighted got 23.0%/27.7% on the same seeds) --
  higher variance around the same center, not a real shift in the center itself.
- **This closes out the "extend camo-weighting beyond the loss" sub-investigation**: three variants
  tried (indirect gate, directly-supervised gate at a swept range of weights, camo-aware
  aggregation reusing the same distance signal) -- none beats plain camo-weighted's original,
  simplest formulation. Plain camo-weighted (Run 84, 12-27%/mean 18.2%, 5-seed; now further
  corroborated at mean 19.1%/10-seed in this run's own control arm) remains the standing best
  Elliptic hard-core lever, unchanged since Run 84.

## [2026-07-22] Run 102 — campaign: elliptic_metric_learning_camo_weighted_margin05 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin05.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6473 | 0.0410 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin05.yaml | 0.5888 | 0.0268 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.5888029099173464 vs 0.6472665330931208, p=0.0625)
- Observations: margin05 loses cleanly (0.589 vs 0.647, p=0.0625) -- consistent with Run 109's repeat and the later 10-seed matched confirmation (Run 119): margin05 is a genuine regression, not noise.
- Next: None -- confirmed bad. Do not revisit margin05.

## [2026-07-22] Run 103 — campaign: elliptic_metric_learning_camo_weighted_margin15 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin15.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6494 | 0.0358 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin15.yaml | 0.6576 | 0.0175 | 5 |

Wilcoxon signed-rank: statistic=7.00, p=1.0000, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.657640549391465 vs 0.6494052417626273, p=1.0)
- Observations: margin15 wins on mean (0.658 vs 0.649) but NOT a clean sweep (p=1.0, statistic=7) -- mixed within-seed results. Contrast with Run 110's repeat of the SAME candidate, which DOES show a clean 5/5 sweep (p=0.0625). Two runs of the same config giving different significance readings is itself informative: single-campaign non-significance doesn't rule out a real effect, see margin20's identical pattern below.
- Next: Compare against Run 110's repeat before concluding anything about margin15 specifically.

## [2026-07-22] Run 104 — campaign: elliptic_metric_learning_camo_weighted_margin20 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin20.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6374 | 0.0327 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin20.yaml | 0.6779 | 0.0299 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6779058093464244 vs 0.6374194485628847, p=0.0625)
- Observations: margin20 wins cleanly (0.637->0.678, p=0.0625, 5/5 sweep) -- the strongest single result in this whole 7-candidate sweep. NOT yet flagged as noteworthy at the time this ran (see Run 129 for why that was a miss: this deserved the matched 10-seed follow-up that Run 119 gave to margin05 instead).
- Next: This candidate specifically should get a matched multi-seed re-confirmation before any other candidate in this batch -- it has the largest, cleanest effect size of the seven.

## [2026-07-22] Run 105 — campaign: elliptic_metric_learning_camo_weighted_triplets1000 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_triplets1000.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6278 | 0.0252 | 5 |
| configs/elliptic_metric_learning_camo_weighted_triplets1000.yaml | 0.6183 | 0.0448 | 5 |

Wilcoxon signed-rank: statistic=7.00, p=1.0000, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6183038970580872 vs 0.6278191275449606, p=1.0)
- Observations: triplets1000 loses on mean (0.618 vs 0.628) with no significance (p=1.0) -- a clean null, no signal either direction.
- Next: None. Contrast with Run 113's repeat of the same candidate (also null, different specific numbers) -- consistent null across two independent dispatches.

## [2026-07-22] Run 106 — campaign: elliptic_metric_learning_camo_weighted_triplets4000 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_triplets4000.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6303 | 0.0256 | 5 |
| configs/elliptic_metric_learning_camo_weighted_triplets4000.yaml | 0.6603 | 0.0281 | 5 |

Wilcoxon signed-rank: statistic=2.00, p=0.1875, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6602677275077534 vs 0.6302663021375375, p=0.1875)
- Observations: triplets4000 wins on mean (0.660 vs 0.630) but not significant (p=0.1875) -- a positive-looking but unconfirmed direction.
- Next: Compare against Run 115's repeat of the same candidate before drawing conclusions -- consistently positive-but-not-significant across both runs is a different (weaker) pattern than margin20's consistently-significant one.

## [2026-07-22] Run 107 — campaign: elliptic_metric_learning_camo_weighted_comp005 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_comp005.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6276 | 0.0227 | 5 |
| configs/elliptic_metric_learning_camo_weighted_comp005.yaml | 0.6283 | 0.0321 | 5 |

Wilcoxon signed-rank: statistic=7.00, p=1.0000, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6282931247769361 vs 0.6275526658438715, p=1.0)
- Observations: comp005 essentially ties baseline (0.628 vs 0.628, p=1.0) -- a clean null.
- Next: None. Repeats in Run 117 with a similar null result.

## [2026-07-22] Run 108 — campaign: elliptic_metric_learning_camo_weighted_comp01 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_comp01.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6197 | 0.0343 | 5 |
| configs/elliptic_metric_learning_camo_weighted_comp01.yaml | 0.6638 | 0.0298 | 5 |

Wilcoxon signed-rank: statistic=1.00, p=0.1250, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6637682636049852 vs 0.6197436000852983, p=0.125)
- Observations: comp01 wins on mean (0.664 vs 0.620) but not significant (p=0.125) -- positive direction, unconfirmed, same qualitative pattern as triplets4000.
- Next: Compare against Run 118's repeat before drawing conclusions.

## [2026-07-22] Run 109 — campaign: elliptic_metric_learning_camo_weighted_margin05 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin05.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6299 | 0.0322 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin05.yaml | 0.5974 | 0.0172 | 5 |

Wilcoxon signed-rank: statistic=1.00, p=0.1250, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.5974456416009243 vs 0.6299237998974251, p=0.125)
- Observations: margin05 loses again (0.597 vs 0.630, p=0.125) -- SAME direction as Run 102's independent dispatch of this candidate. Two independent losses is a real, replicated signal (later confirmed definitively at n=10 in Run 119).
- Next: None needed -- margin05 is confirmed bad across two independent runs plus the later 10-seed check.

## [2026-07-22] Run 110 — campaign: elliptic_metric_learning_camo_weighted_margin15 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin15.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6358 | 0.0337 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin15.yaml | 0.6627 | 0.0212 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6627021342282325 vs 0.6357736239699513, p=0.0625)
- Observations: margin15 wins cleanly this time (0.636->0.663, p=0.0625, 5/5 sweep) -- unlike Run 103's mixed result for the SAME candidate. One win, one mixed, across two independent dispatches -- genuinely ambiguous, unlike margin20's two clean wins in a row (Runs 104/111) or margin05's two clean losses in a row (Runs 102/109).
- Next: margin15 sits at a genuinely unresolved point between margin05 (confirmed bad) and margin20 (confirmed good, Run 130) -- if a single 'recommended' margin value is ever needed, margin15 would be the natural candidate to test properly at n=10, since it may be a real middle-ground operating point. Not yet done as of Run 148.

## [2026-07-22] Run 111 — campaign: elliptic_metric_learning_camo_weighted_margin20 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin20.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6301 | 0.0368 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin20.yaml | 0.6802 | 0.0360 | 5 |

Wilcoxon signed-rank: statistic=0.00, p=0.0625, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6802284275167423 vs 0.6301203187248512, p=0.0625)
- Observations: margin20 wins cleanly AGAIN (0.630->0.680, p=0.0625, 5/5 sweep) -- second consecutive independent clean win for this exact candidate (see Run 104). Two independent p=0.0625 sweeps in the same direction is roughly 0.4% under a pure null -- this is the strongest replicated signal in the whole 7-candidate sweep and should have been flagged as the priority candidate for matched multi-seed follow-up.
- Next: This is the run that should have triggered a matched 10-seed re-confirmation immediately -- it didn't happen until Run 129/130, prompted by external review re-reading Runs 102-118 together rather than by real-time flagging.

## [2026-07-22] Run 112 — campaign: elliptic_metric_learning_camo_weighted_temp_sharp vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_temp_sharp.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6380 | 0.0365 | 5 |
| configs/elliptic_metric_learning_camo_weighted_temp_sharp.yaml | 0.6250 | 0.0000 | 1 |

- Decision: kept existing best (candidate did not win significantly: mean 0.6249870905594807 vs 0.6380379571046805, p=N/A)
- Observations: Crashed after 1/5 seeds (RunPod network issue) -- candidate did not get a fair comparison (n=1 vs baseline's n=5). Not informative on its own.
- Next: Re-run with a full 5 (or more) seeds before drawing any conclusion. This eventually happened at n=10 in Run 119, which found temp_sharp loses significantly on F1 (p=0.037) despite looking promising at this crashed n=1 snapshot.

## [2026-07-22] Run 113 — campaign: elliptic_metric_learning_camo_weighted_triplets1000 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_triplets1000.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6287 | 0.0302 | 5 |
| configs/elliptic_metric_learning_camo_weighted_triplets1000.yaml | 0.6056 | 0.0463 | 5 |

Wilcoxon signed-rank: statistic=6.00, p=0.8125, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6056051089811482 vs 0.6286749873703295, p=0.8125)
- Observations: triplets1000 loses again (0.606 vs 0.629, p=0.8125) -- consistent null/negative with Run 105's independent dispatch of the same candidate.
- Next: None. Confirmed null across two runs.

## [2026-07-22] Run 114 — campaign: elliptic_metric_learning_camo_weighted_dual vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_dual.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6288 | 0.0313 | 5 |
| configs/elliptic_metric_learning_camo_weighted_dual.yaml | 0.6802 | 0.0000 | 1 |

- Decision: kept existing best (candidate did not win significantly: mean 0.6801767577059613 vs 0.6287693279705119, p=N/A)
- Observations: Crashed after 1/5 seeds -- not a fair comparison (n=1 vs baseline's n=5), same issue as Run 112.
- Next: This candidate (camo_weighted_dual mining) was later properly tested in Run 119 (matched 10-seed) and found to wash against baseline -- this single crashed seed should not be read as evidence either way.

## [2026-07-22] Run 115 — campaign: elliptic_metric_learning_camo_weighted_triplets4000 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_triplets4000.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6286 | 0.0253 | 5 |
| configs/elliptic_metric_learning_camo_weighted_triplets4000.yaml | 0.6412 | 0.0101 | 5 |

Wilcoxon signed-rank: statistic=3.00, p=0.3125, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6411503058725667 vs 0.6286219300723709, p=0.3125)
- Observations: triplets4000 wins on mean again (0.641 vs 0.629) but still not significant (p=0.3125) -- consistent weak-positive-unconfirmed pattern with Run 106's independent dispatch.
- Next: Two independent runs both showing a positive-but-non-significant direction is a genuinely different (weaker) pattern than margin20's two clean significant wins -- not worth a matched re-confirmation given the consistently weak effect size, unlike margin20.

## [2026-07-22] Run 116 — campaign: elliptic_metric_learning_camo_weighted_margin_scale vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_margin_scale.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6336 | 0.0273 | 5 |
| configs/elliptic_metric_learning_camo_weighted_margin_scale.yaml | 0.6375 | 0.0317 | 5 |

Wilcoxon signed-rank: statistic=7.00, p=1.0000, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6374600833475013 vs 0.6336275746954996, p=1.0)
- Observations: margin_scale ties baseline almost exactly (0.637 vs 0.634, p=1.0) -- a clean null. Later confirmed via Run 119's matched 10-seed test (also a clean wash).
- Next: None. Confirmed null.

## [2026-07-22] Run 117 — campaign: elliptic_metric_learning_camo_weighted_comp005 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_comp005.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6367 | 0.0339 | 5 |
| configs/elliptic_metric_learning_camo_weighted_comp005.yaml | 0.6239 | 0.0346 | 5 |

Wilcoxon signed-rank: statistic=6.00, p=0.8125, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6239072028404913 vs 0.6366700076720571, p=0.8125)
- Observations: comp005 loses slightly (0.624 vs 0.637, p=0.8125) -- consistent with Run 107's independent dispatch showing an essential tie/null for the same candidate.
- Next: None. Confirmed null across two runs.

## [2026-07-22] Run 118 — campaign: elliptic_metric_learning_camo_weighted_comp01 vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_camo_weighted_comp01.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6380 | 0.0380 | 5 |
| configs/elliptic_metric_learning_camo_weighted_comp01.yaml | 0.6547 | 0.0242 | 5 |

Wilcoxon signed-rank: statistic=2.00, p=0.1875, n_pairs=5
- Decision: kept existing best (candidate did not win significantly: mean 0.6546624561143143 vs 0.6380036202579268, p=0.1875)
- Observations: comp01 wins on mean (0.655 vs 0.638) but not significant (p=0.1875) -- consistent weak-positive-unconfirmed pattern with Run 108's independent dispatch of the same candidate.
- Next: Same read as triplets4000 (Runs 106/115): consistently positive but never significant across two independent runs -- a weaker, less compelling pattern than margin20's, not prioritized for follow-up.

## [2026-07-22] Run 119 — Elliptic: 3 new camo-weighted loss-formula variants (temperature, dual-sided, adaptive-margin) all fail matched confirmation
- Implemented 3 generalizations of the confirmed camo-weighted triplet loss (`evaluation/metric_learning.py`), each modifying the loss MECHANISM rather than just sweeping an existing hyperparameter:
  - `camo_weighted_temp`: adds an explicit softmax temperature to the existing anchor-camouflage weight (`w_i = N*softmax(-dist_i/T)`, T=1.0 reproduces the original exactly).
  - `camo_weighted_dual`: extends the anchor-only camouflage signal to also up-weight triplets whose randomly-sampled legit NEGATIVE looks fraud-like (`combined_score = -dist(anchor,legit_centroid) - dist(neg,fraud_centroid)`).
  - `camo_weighted_margin_scale`: scales the MARGIN itself by camouflage severity instead of reweighting the loss value (`adaptive_margin = margin*(1 + margin_scale*(camo_weight-1))`).
- First (unmatched) n=5 dispatch of `camo_weighted_temp` (T=0.3) and the hyperparameter-sweep's `margin05` candidate looked very promising on hard-core recovery: 26.3% and 28.4% mean recovery respectively vs baseline's 18.2%, all/nearly-all seeds beating baseline, zero losses.
- **Properly matched 10-seed paired comparison (same seeds, same dispatch) told a different story**: both `temp_sharp` and `margin05` are significantly WORSE on aggregate F1 (Wilcoxon p=0.0371 and p=0.0195). Root cause confirmed directly: `margin05`'s hard-core recovery gain (paired +9.4pp, p=0.0098 -- genuinely significant on this narrow metric) costs ~360 extra false positives on legit per seed for only ~20 extra hard-core catches recovered (18:1 ratio) -- false-positive rate on legit jumps 9.3%->13.6%. This is the same threshold-sensitivity artifact caught before (Run 93): a globally more aggressive/lower-effective-threshold classifier, not a qualitatively better model.
- `camo_weighted_dual` and `camo_weighted_margin_scale` (at the originally-dispatched settings) also washed or regressed once matched.
- Infra note: two of the new mining modes initially crashed on RunPod (`ValueError: Unknown mining: 'camo_weighted_temp'`) because the worker process's in-memory Python modules don't get refreshed by the existing per-job `git pull` (file-level only, per `_git_pull_latest`'s own documented limitation) -- fixed by forcing a full worker recycle via the RunPod GraphQL `saveEndpoint` mutation (workersMax 3->0->3), which guarantees fresh containers/fresh imports on the next dispatch.
- Decision: none of the 3 new loss-formula variants beat baseline camo-weighted once properly confirmed. Closed for now.
- Next: subspace-restricted scoring combined with camo-weighted embeddings (see Run 120).

## [2026-07-22] Run 120 — Elliptic: subspace-restricted scoring on camo-weighted embeddings, single-seed inconclusive
- Direct follow-up to Run 85/86 (subspace-restricted distance scoring gave a real, leakage-free AUC gain, 0.782->0.793, on PLAIN GraphSAGEDiff embeddings) -- Run 86 explicitly flagged combining it with camo-weighted's own embeddings as untested.
- Dispatched a fresh `camo_weighted` run with `metric_learning.return_embeddings: true` to get properly-sized embeddings matching the current 8841-node test set (an old cached embeddings file from earlier in the session turned out to be from a stale/smaller 1908-node test config and had to be discarded).
- Discovered `return_embeddings`'s payload-size-limit workaround subsamples test LEGIT to 1500 nodes (keeps ALL test fraud, sorted by original position) -- recovered exact hard-core alignment by filtering both the embeddings dump and the reference masks to fraud-only, matching by preserved relative order.
- Clean (Run 86-style) dim selection: split TRAIN fraud in half, select top-N anomaly-signature dims (highest `|x-legit_centroid|/legit_std` in the top-10-per-node tally) from one half only, never touching test labels.
- Single-seed (seed=42) result: full 128-dim recovery=12.2%, best subspace (top-10) recovery=14.1%, AUC 0.867->0.865-0.875 across the dim grid -- modest, inconclusive, well within this project's established single-seed noise band.
- Dispatched a 5-seed version (seeds 0-4) for a proper multi-seed read before drawing any conclusion -- see next run for the result.
- Next: multi-seed subspace-vs-full-128 paired comparison (recovery, not just AUC, per this project's established decisive metric).

## [2026-07-22] Run 121 — IEEE-CIS: novel-client (uid unseen-in-train) subset test disconfirms the "topology helps where identity-lookup fails" hypothesis
- Motivation: RF's strength on IEEE-CIS leans heavily on UID (client-identity) reconstruction, essentially a lookup table -- hypothesized that GNN topology (message-passing over card1/addr1/uid-sharing edges) should have a relative edge specifically on transactions from a genuinely novel client, where lookup-based features carry no signal.
- First attempted definition (`prior_count==0`, i.e. zero prior transactions anywhere in the full time-sorted dataset) found **zero** such test rows exist -- every reconstructed client in the test period already has at least one prior occurrence somewhere in train/val. Corrected definition: uid never appears in the TRAIN split specifically (the split that fits the target-encoding/aggregate features) -- 43,149 test rows (48.7%), 1,111 fraud, a large and meaningful subset.
- RF (retrained locally on `ieee_cis_graph_full_v3_fixed.pt`) degrades substantially on this subset: AUC 0.909->0.834, recall_fraud 0.44->0.22.
- Dispatched a fresh plain-classification GraphSAGEDiff run on the SAME leak-fixed graph (previous cached GNN predictions were from the pre-fix `ieee_cis_graph_full_v2.pt`, not a valid comparison) to get paired test predictions; corrected the known NeighborLoader adjacent-seed-swap bug (54/88581 positions, confirmed clean adjacent pairs) before comparing.
- **Result: GNN degrades MORE on the novel subset, not less.** AUC 0.786->0.682 (-13%, vs RF's -8%), recall_fraud 0.23->0.047. Directly: of 1,111 novel-subset fraud cases, GNN catches 5 that RF misses; RF catches 194 that GNN misses.
- **Hypothesis disconfirmed.** Likely explanation: a client unseen in train also tends to have sparser/newer entity-sharing edges, so the graph structure gives the GNN less to work with exactly where it would need to compensate most for the missing lookup signal -- topology and identity-lookup appear to degrade together on cold-start clients, not substitute for each other.
- Next: no further pursuit of this specific angle planned; RF+UID+V-columns remains the best IEEE-CIS result (F1=0.769, AUC=0.909, Run 96).

## [2026-07-22] Run 122 — Elliptic: subspace-restricted scoring does NOT help camo-weighted embeddings (5-seed confirmed null, closes Run 120's open question)
- Follow-up to Run 120's single-seed inconclusive read. 5-seed (seeds 0-4) properly matched comparison: subspace-restricted score (top-5/10/15/20/30/50 anomaly-signature dims, clean train-fraud-half selection per Run 86's method) vs full 128-dim camo-weighted score, same embeddings, same seeds.
  | Dims | mean recovery | paired diff vs full-128 | wilcoxon p | wins |
  |---|---|---|---|---|
  | Full 128 (baseline) | 20.6% | -- | -- | -- |
  | Top-5 | 19.1% | -1.5pp | 0.3125 | 1/5 |
  | Top-10 | 19.2% | -1.3pp | 0.3750 | 1/5 |
  | Top-15 | 19.2% | -1.3pp | 0.3125 | 1/5 |
  | Top-20 | 20.0% | -0.6pp | 0.6250 | 2/5 |
  | Top-30 | 19.3% | -1.2pp | 0.1875 | 1/5 |
  | Top-50 | 19.7% | -0.8pp | 0.3125 | 1/5 |
- **All 6 dim options paired-lose to the full-128 baseline, none remotely significant.** Unlike Run 85/86 (subspace restriction gave a real 0.782->0.793 AUC gain on PLAIN, non-camo-weighted embeddings), it adds nothing here.
- Interpretation: camo-weighted's triplet loss already reshapes the embedding space around the camouflage signal specifically (that's the whole point of the up-weighting), so there's no leftover "noise-dilution" for dimension selection to cut the way there was in the unsupervised/classification embedding. The two levers don't stack -- confirms Run 86's open question with a clean no.
- Decision: subspace restriction is not a productive direction on top of camo-weighted. Closed.
- Next: GraphSAGEDualView (HOT-GNN-inspired decoupled homophily/heterophily aggregation) + camo-weighted -- see next run.

## [2026-07-22] Run 123 — campaign: elliptic_metric_learning_dual_view vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_dual_view.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6282 | 0.0381 | 7 |
| configs/elliptic_metric_learning_dual_view.yaml | FAILED (0/? seeds succeeded) |

- Decision: kept existing best (candidate did not win significantly: mean N/A vs 0.6282191783584749, p=N/A)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-22] Run 124 — campaign: elliptic_metric_learning_dual_view_encoded vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_dual_view_encoded.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | FAILED (0/? seeds succeeded) |
| configs/elliptic_metric_learning_dual_view_encoded.yaml | FAILED (0/? seeds succeeded) |

- Decision: kept existing best (candidate did not win significantly: mean N/A vs N/A, p=N/A)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 125 — campaign: elliptic_metric_learning_dual_view vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_dual_view.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6306 | 0.0287 | 10 |
| configs/elliptic_metric_learning_dual_view.yaml | 0.6273 | 0.0175 | 10 |

Wilcoxon signed-rank: statistic=26.00, p=0.9219, n_pairs=10
- Decision: kept existing best (candidate did not win significantly: mean 0.6273335378645979 vs 0.6306160744076652, p=0.921875)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 126 — campaign: elliptic_metric_learning_dual_view_encoded vs elliptic_metric_learning_camo_weighted
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_camo_weighted.yaml,
  candidate=configs/elliptic_metric_learning_dual_view_encoded.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_camo_weighted.yaml | 0.6317 | 0.0317 | 10 |
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6653 | 0.0352 | 10 |

Wilcoxon signed-rank: statistic=3.00, p=0.0098, n_pairs=10
- Decision: PROMOTED to new best (mean 0.6653 > 0.6317, p=0.0098 < 0.05)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 127 — Elliptic: GraphSAGEDualView + nonlinear encoder BEATS camo-weighted on aggregate F1 (matched, significant) -- first architecture to do so, with an honest hard-core caveat
- Built `GraphSAGEDualView` (`models/gnn/graphsage.py`): HOT-GNN-inspired decoupled homophily/heterophily aggregation -- computes pairwise src-dst cosine similarity (no external reference centroid needed, unlike GraphSAGECamoAgg), maintains TWO separate softmax-weighted aggregation channels (concentrating on similar vs dissimilar neighbors respectively), concatenated as `[x_enc, homophilic_agg, heterophilic_agg]` before the SAGEConv stack. Combined with camo-weighted mining (this project's own confirmed lever) instead of HOT-GNN's own classification objective.
- Two variants dispatched as a 10-seed matched campaign against the camo-weighted baseline:
  - `dual_view` (no feature encoder, similarity computed on raw x): clean wash, mean F1 0.6273 vs baseline 0.6306, p=0.92.
  - `dual_view_encoded` (feature_encoder_hidden_dim=64, similarity computed in a learned nonlinear space): **PROMOTED** -- mean F1 0.6653 vs baseline 0.6317, paired diff +0.034, wilcoxon p=0.0098, wins 8/10.
- **Confirmed NOT a threshold-sensitivity artifact** (the exact failure mode that invalidated Run 119's margin05/temp_sharp): predicted_positive_rate is LOWER for dual_view_encoded (0.090 vs baseline's 0.118) -- the model is more conservative, not more aggressive, and still wins on F1. This is a real precision/calibration improvement, not a shifted decision threshold.
- **Honest tradeoff**: hard-core recovery is slightly WORSE (15.6% vs baseline's 19.5%, paired diff -3.9pp, not significant p=0.084, 1/10 wins) and there's exactly ONE loss in one seed (a previously-caught-by-both case flipped to missed) -- breaking camo-weighted's clean zero-losses-in-every-seed record for the first time. Small (1/10 seeds, 1 case), but real and should not be glossed over.
- Interpretation: the nonlinear encoder appears to help the model separate homophilic/heterophilic neighborhoods more cleanly for the GENERAL population (driving the aggregate F1 gain via better precision), but very slightly at the expense of the specific hard-core/camouflaged cases camo-weighted's mining was designed to rescue -- plausibly two different populations respond differently to this architectural change.
- Also dispatched (compare(), not run_campaign) on IEEE-CIS's leak-fixed richer graph, same architecture/objective combination -- result pending.
- Next: (a) decide whether to adopt dual_view_encoded as the new best-overall Elliptic config (better aggregate F1, worse hard-core) or keep camo-weighted as best-for-hard-core-recovery and treat this as a separate, parallel result depending on which objective matters more for the eventual writeup; (b) IEEE-CIS result once it lands; (c) if adopted, re-run the earlier "extend camo-weighting" ideas (temp/dual/margin_scale, all failed on the OLD architecture) against this new base, since the interaction might differ.

## [2026-07-23] Run 128 — IEEE-CIS: GraphSAGEDualView vs camo-weighted, clean wash (Run 127's Elliptic win doesn't transfer)
- Same architecture/objective combination as Run 127 (GraphSAGEDualView, no encoder, + camo-weighted mining), dispatched on IEEE-CIS's leak-fixed richer graph (`ieee_cis_graph_full_v3_fixed.pt`), 3-seed matched compare against `ieee_cis_metric_learning_camo_weighted_v3fixed.yaml` (a new baseline config -- no prior camo-weighted run existed on this specific graph).
- First dispatch attempt (overnight) lost almost all seeds to a local network drop during laptop sleep (`Read timed out` / DNS resolution failures) -- redispatched cleanly once back online.
  | Config | Mean F1 | Std | n |
  |---|---|---|---|
  | camo_weighted (baseline) | 0.4761 | 0.0185 | 3 |
  | dual_view (candidate) | 0.4797 | 0.0097 | 3 |
- Wilcoxon p=1.0 -- clean wash, no signal either direction (n=3 given IEEE-CIS's much slower per-seed cost at full scale, ~10-20 min/seed vs Elliptic's ~30-90s/seed).
- **Note both numbers (~0.48 F1) are far below RF's established 0.769** on the same feature set (Run 96) -- metric learning as a whole already underperforms badly on IEEE-CIS regardless of aggregation architecture, consistent with Runs 95/97's finding that GNN-family approaches consistently lose to RF on IEEE-CIS's richer tabular-style features. Run 127's dual_view_encoded win looks Elliptic-specific, not a general improvement that transfers across datasets.
- Decision: no further pursuit of dual_view on IEEE-CIS. RF+UID+V-columns remains the best IEEE-CIS result.
- Next: possible follow-up on Elliptic only -- test dual_view_encoded combined with a couple of the earlier-failed loss-formula variants (temp/dual/margin_scale, Run 119) now that the base architecture differs, per Run 127's open item (c). Lower priority given the established pattern of stacked ideas mostly not compounding this session.

## [2026-07-23] Run 129 — CORRECTION/addendum: Run 127's "PROMOTED" is a metric switch, not a strict win; margin20 (Runs 102-118) deserves the matched re-check Run 119 gave margin05 instead
- User catch, re-reading Runs 102-118 (the two independent 7-candidate campaign dispatches) side by side:
  - **margin20 replicated cleanly twice, independently**: Run 104 (baseline 0.6374 vs candidate 0.6779, p=0.0625, 5/5) and Run 111 (baseline 0.6301 vs candidate 0.6802, p=0.0625, 5/5) -- same direction, same clean 5/5 sweep, in two separately-dispatched campaigns. Under a pure null, two independent p=0.0625 sweeps in the same direction is roughly 0.4% (0.0625^2), qualitatively different from the isolated single-run "promising" results (DropEdge, gate_aux_weight, camo_agg) that evaporated under matched n=10 checks earlier this session. There's also a monotonic dose-response across margin: margin05 clean LOSS both times (Runs 102/109, later confirmed at n=10 in Run 119), margin15 mixed (loss in Run 103, clean win in Run 110), margin20 clean win both times (Runs 104/111). **Run 119 tested margin05 (already the confirmed-bad one) instead of margin20 (the one with the actual signal) -- a real miss, not a judgment call.**
  - Verified the pairing mechanism itself is correct (`infra/multi_seed.py`'s `compare()` passes the identical seed value to both arms, matched by index for the Wilcoxon test) -- the cross-run drift in baseline's own mean (e.g. 0.6374 in Run 104 vs 0.6301 in Run 111, same nominal seeds 0-4) is real residual noise from elsewhere, not a pairing bug: `set_seed()` covers `random`/`numpy`/`torch.manual_seed` but never sets `torch.use_deterministic_algorithms` -- `torch_geometric.utils.scatter`, used throughout this codebase's neighbor aggregation, is well-documented as non-deterministic on CUDA regardless of seed (atomic-add ordering depends on GPU thread scheduling). This means Run 104 and Run 111 are genuinely independent replications despite nominally "the same" seed list, which if anything strengthens the joint-probability argument rather than undermining it.
  - Dispatched a proper 10-seed matched compare (baseline vs margin20, mirroring Run 119's methodology exactly) -- result pending, see next run.
  - **Preliminary context already on hand** (from an earlier, pre-Run-119 hard-core cross-reference of margin20/comp01/triplets4000's raw predictions): margin20 showed mean hard-core recovery ~8.5% (well below baseline's 18.2%) with a LOWER mean predicted_positive_rate (~0.074) than baseline's (~0.113-0.12) -- i.e. IF the F1 win replicates, margin20 looks mechanistically like Run 127's dual_view_encoded (more conservative, better aggregate precision, worse hard-core recovery), not like margin05's flat regression or the earlier threshold-inflation artifacts. A plausible, non-artifactual story: a larger triplet-loss margin demands more separation before the loss is satisfied, which could tighten the effective decision boundary generally at the cost of the specific camouflaged cases camo-weighted's mining targets.
- **Run 127 correction**: "PROMOTED to new best" in the campaign's own auto-decision logic is by aggregate f1_macro ONLY -- that is the campaign framework's built-in promotion criterion, not this project's own established decisive metric (hard-core recovery, used consistently since Run 78). By hard-core recovery, dual_view_encoded is WORSE (15.6% vs baseline's 19.5%) and breaks the zero-losses-in-every-seed record for the first time. These are two separate results answering two different questions -- "best on the aggregate metric" and "best on hard-core recovery" -- not one result superseding the other. Any write-up must report them as parallel, not nested, findings. (The original Run 127 entry already stated the tradeoff numbers; this entry makes the metric-switch explicit as its own point, per the user's framing.)
- **Contrast worth keeping explicit**: a feature encoder (`feature_encoder_hidden_dim`) was confirmed HARMFUL when added before GraphSAGEDiff's classifier head (Run 57: 5/5 seeds worse) but is what makes GraphSAGEDualView work at all (Run 125, no encoder: clean wash; Run 126, with encoder: p=0.0098). Not "more MLP capacity is good" in general -- the encoder's specific role here is computing the homophily/heterophily pairwise similarity in a learned nonlinear space rather than raw feature space, a mechanistically different job than encoding features upstream of a classifier. The same architectural knob helping in one place and hurting in another is itself the useful fact, not a contradiction to resolve away.

## [2026-07-23] Run 130 — Elliptic: margin20 CONFIRMED, strongest matched result in the investigation -- but it's a sharp hard-core-for-precision tradeoff, not a free win
- Direct follow-up to Run 129 (user-caught gap: margin20, not margin05, had the real dose-response signal across Runs 102-118). Fresh 10-seed matched compare (`infra/multi_seed.py`'s `compare()`, same methodology as Run 119):
  | | Mean F1 | Std |
  |---|---|---|
  | camo_weighted (baseline) | 0.6264 | 0.0233 |
  | margin20 (candidate) | 0.6747 | 0.0211 |
  Wilcoxon: statistic=0.00 (perfect 10/10 sweep, every seed favors margin20), **p=0.0020**. This is the THIRD independent confirmation in the same direction (Run 104: p=0.0625, 5/5; Run 111: p=0.0625, 5/5; this run: p=0.0020, 10/10) -- by a wide margin the most robustly replicated single-hyperparameter effect found this session.
- **Hard-core cross-reference on this same matched data tells the full story**:
  | | recovery | losses | predicted_positive_rate |
  |---|---|---|---|
  | baseline | 19.2% | 0 | 0.120 |
  | margin20 | **7.4%** | 1 | 0.071 |
  Paired recovery diff -11.7pp, wilcoxon p=0.0020, **10/10 seeds WORSE** (perfect sweep in the opposite direction from the F1 result). ppr drops from 0.120->0.071 -- margin20 is substantially MORE conservative, not less, ruling out the threshold-inflation artifact that killed margin05/temp_sharp in Run 119.
- **Interpretation**: a larger triplet-loss margin (2.0 vs the baseline's 1.0) demands more separation before the loss is satisfied, pushing the model toward a stricter, more confident decision boundary. This measurably improves aggregate precision/F1 on the general population, but strips away MOST of exactly the camouflaged-case recovery camo-weighted's soft up-weighting was designed to rescue -- recovery collapses to less than half its baseline value. This is not a mirage and not an artifact (unlike margin05/temp_sharp/dual_view/margin_scale, all null or artifactual) -- it's a real, large, doubly-orthogonal effect: strongly positive on one axis, strongly negative on the other, both at p<=0.002.
- **This sharpens Run 127's "two separate results" point into an explicit, quantified Pareto tradeoff** along a single, interpretable hyperparameter (the triplet margin) rather than an architectural change: margin=1.0 (baseline) sits at the hard-core-recovery end, margin=2.0 sits at the aggregate-precision end, both confirmed at high significance. dual_view_encoded (Run 127) shows the same qualitative direction (better F1, worse hard-core, lower ppr) at a smaller magnitude -- open question whether it's tapping the same underlying mechanism (a stricter effective decision boundary) via a different lever (aggregation architecture vs loss margin), or two independent effects that happen to trade off the same way. Not yet tested together.
- Decision: margin20 is a real, confirmed result -- report it explicitly as the aggregate-F1-optimized point on this tradeoff curve, alongside camo-weighted (hard-core-optimized) and dual_view_encoded (a milder version of the same direction). Do not present any one of the three as simply "the new best" without specifying which axis.
- Next: (a) test whether margin20 + dual_view_encoded stack (same direction, might compound OR might be redundant/hit a ceiling -- worth one matched check); (b) consider whether an intermediate margin value (e.g. 1.5, which showed a weaker/more mixed signal in Runs 103/110) sits at a genuinely different, perhaps more balanced point on this same curve, worth one more matched confirmation if the eventual write-up wants a single "recommended" operating point.

## [2026-07-23] Run 131 — Elliptic: hard-core recovery diagnostic — connectivity ruled out, camouflage+temporal drift confirmed as the driver
- Direct answer to "what's missing to recover more of the hard core" -- structural diagnostic on the 213-node hard core, split by camo-weighted's own 10-seed recovery frequency (Run 130's baseline arm): 137 (64.3%) never recovered in any of 10 seeds, 76 (35.7%) recovered in at least one.
- **Graph connectivity/isolation: ruled out.** Mean degree is statistically identical between never-recovered (3.46) and sometimes-recovered (3.47) groups; 0% isolated (degree=0) in either group. Isolation is NOT why these cases are hard -- both groups have comparable structural connectivity.
- **Embedding camouflage distance: confirmed driver.** Never-recovered mean dist-to-legit-centroid = 0.751, barely above the LEGIT population's own mean (0.646) -- these cases are embedded almost indistinguishably from normal transactions. Sometimes-recovered mean = 1.192, meaningfully more separable. This is exactly the population camo-weighted's up-weighting was designed to reach, and by construction it reaches the moderately-camouflaged half but not the most extreme half.
- **New finding: recoverability collapses with temporal distance from the train/val boundary, not just camouflage severity.** Per-step recovery rate: step 42 (closest to train) 57.4%, steps 43-48 mixed 23-66%, **step 49 (furthest from train, final test step) collapses to 5.5%** (52/55 hard-core cases at step 49 never recovered). Step 49 also has the lowest mean dist-to-legit of any step (0.794 among ALL step-49 fraud, 0.778 among step-49 hard-core specifically) -- the camouflage genuinely gets worse the further into the test period, not just "the model's reference happens to fail more" at that point.
- **Interpretation**: every intervention tried against this hard core so far (camo-weighted itself, dual-view aggregation, adaptive margin, temperature, subspace restriction, spectral filtering, gated aggregation) shares one structural property -- a reference (centroid, weight, subspace) computed ONCE from train-period data and then held FIXED across the entire test period. None of them re-anchor as the drift continues past the point they were calibrated at. This is consistent with camouflage being an ongoing, evolving evasion process (matching CLAUDE.md's existing framing) rather than a single one-time regime shift -- and explains why every static intervention plateaus around the same ~18-20% ceiling regardless of mechanism.
- **Untried lever this points to directly**: sequential/online adaptation WITHIN the test period -- using the model's own confident predictions on earlier test steps (42, 43...) to progressively update the legit/fraud reference as the timeline advances, instead of one static pass over steps 42-49 with a single frozen reference. This is a genuinely different mechanism from everything tried so far (all of which were static-reference variants), directly targeting the newly-confirmed temporal-decay pattern rather than another loss/architecture tweak on the same static-reference paradigm.
- Visualization published (recovery rate + camouflage distance by step, ruled-out-vs-driver comparison): https://claude.ai/code/artifact/28a1fc81-b074-41d4-aecd-0bb7c8437187
- Next: design and test a sequential test-period adaptation scheme (e.g. re-estimate the legit/fraud reference centroid after each test step using that step's own high-confidence predictions, evaluate step-by-step rather than as one static pass) -- the first genuinely new mechanism proposed since camo-weighted itself, motivated directly by this diagnostic rather than architectural guesswork.

## [2026-07-23] Run 132 — Elliptic: local density/clustering ruled out; neighbor-camouflage refines the diagnostic; literature confirms sequential test-time adaptation is unexplored on this dataset
- Follow-up to Run 131, testing whether graph-density analogs (beyond plain degree, already ruled out) or neighbor-level camouflage separate never-recovered from sometimes-recovered hard-core nodes.
- **Local clustering coefficient and k-core number: both uninformative.** Mean clustering 0.0003 (never-recovered) vs 0.0132 (sometimes-recovered) -- both near-zero, >98% of nodes in EITHER group have exactly zero clustering. K-core ~1.25 for both groups, no separation. This is a real structural property of this dataset, not a failed analysis: Elliptic's per-step transaction graph is essentially tree-like/DAG-like (directed payment flows rarely close into triangles), so triangle-based density measures have almost nothing to find here regardless of fraud status. Directed in/out-degree also flat (1.73 vs 1.74 both directions) -- confirms Run 131's plain-degree null result isn't an artifact of using undirected degree.
- **Neighbor-level camouflage: real signal, but explains the floor rather than breaking it.** Mean dist-to-legit-centroid of a hard-core node's OWN test-period neighbors: never-recovered = 0.666 (essentially identical to the legit population's own mean, 0.646), sometimes-recovered = 0.833 (meaningfully more separable, closer to the camouflaged-fraud zone). Only 75.1% of hard-core nodes even have a scoreable (test-period) neighbor at all. **Interpretation**: never-recovered cases aren't just individually camouflaged -- their entire local neighborhood is also unremarkable, so there's no "guilt by association" signal at ANY radius checked so far (node-level, 1-hop neighbor-level). Sometimes-recovered cases benefit from sitting near neighbors that are themselves somewhat suspicious, which is exactly the kind of signal neighbor-aggregation can exploit -- and exactly why it's absent for the remaining 64%.
- **Literature search** (via web research agent, distinguishing confirmed vs. unverified claims):
  - The step-42-49 progressive-collapse pattern is independently corroborated by arXiv 2604.19514's own per-timestep GraphSAGE numbers (recall 0.360@step42 -> 0.056@step43, staying collapsed through step 49) -- but no paper frames this as measurable embedding-space "camouflage" convergence; that framing/measurement appears to be this project's own contribution.
  - **No paper found implementing sequential/progressive test-time adaptation across successive time steps on Elliptic specifically.** One related paper exists (TEMG-TTA, arXiv 2605.29526) but on five OTHER blockchain datasets, with a ONE-SHOT (not step-by-step) adaptation mechanism, and no per-timestep breakdown reported. This is a genuine, unfilled gap -- the sequential-adaptation direction proposed in Run 131 would be a real contribution, not a reimplementation.
  - EvolveGCN's published F1=0.77 confirmed transductive per 2604.19514's own audit. ChronoWave-GNN's claimed F1=0.9799 (already flagged suspicious in CLAUDE.md) now has independent circumstantial support for being transductive/non-standard -- still unverified directly, protocol not confirmed either way.
- Decision: local density is a dead end (real structural property of this sparse transaction graph, not worth pursuing further). Neighbor-camouflage is real but only helps the already-recoverable fraction. Sequential test-time adaptation remains the most promising, literature-unexplored next direction.
- Next: design and dispatch a sequential/progressive test-time adaptation scheme for Elliptic (re-estimate legit/fraud reference after each test step using that step's own confident predictions, evaluated step-by-step through 42-49) -- still the standing next step from Run 131, now with added confidence it's a genuinely novel angle rather than something already tried and quietly abandoned elsewhere.

## [2026-07-23] Run 133 — Elliptic: camouflage-evolution analysis kills the sequential-adaptation direction (oracle ceiling test, before any RunPod dispatch)
- Direct follow-up to Run 131/132's proposed next step (sequential test-time adaptation of the legit/fraud reference across test steps). Analyzed the evolution first, cheaply and locally, before building/dispatching anything.
- **Per-step legit-vs-fraud distance-to-centroid breakdown**:
  | step | n_legit | legit dist-to-centroid | n_fraud | fraud dist-to-centroid | gap |
  |---|---|---|---|---|---|
  | 42 | 1915 | 0.656 | 239 | 1.536 | 0.881 |
  | 43 | 1346 | 0.717 | 24 | 0.943 | 0.226 |
  | 44 | 1567 | 0.732 | 24 | 0.899 | 0.166 |
  | 45 | 1216 | 0.603 | 5 | 1.112 | 0.508 |
  | 46 | 710 | 0.723 | 2 | 1.205 | 0.482 |
  | 47 | 824 | 0.619 | 22 | 0.827 | 0.207 |
  | 48 | 435 | 0.569 | 36 | 0.970 | 0.401 |
  | 49 | 420 | 0.677 | 56 | 0.794 | 0.117 |
  **Legit's own mean distance to the (fixed, train-derived) centroid stays flat across the whole test period (0.57-0.73, no drift)** -- it's specifically fraud moving closer to legit, not the legit reference going stale. Caveat: steps 43-48 have very few fraud examples (2-36 each) -- only step 42 (n=239) and step 49 (n=56) have large-enough samples to read cleanly; the middle-step numbers are individually noisy and shouldn't be over-interpreted as a smooth monotonic curve.
- **Oracle-sequential test (local, no RunPod dispatch needed)**: using TRUE test labels (not even self-generated pseudo-labels -- the best case a sequential-adaptation scheme could ever achieve) to progressively expand the "known fraud" pool step-by-step (42's true fraud folded in before scoring 43, etc.) and recompute the fraud centroid at each step, vs. the static train-only fraud centroid:
  | | static | oracle-sequential |
  |---|---|---|
  | Overall hard-core recovery | 12.2% | 12.7% |
  Per-step: identical at 6 of 8 steps, only step 44 moves (4.3%->8.7%, n=23). **Essentially no improvement, even with perfect future knowledge.**
- **Mechanistic explanation**: as newly-revealed test-period fraud is folded into the fraud centroid average, that fraud is itself progressively more camouflaged (closer to legit) -- so the fraud centroid drifts toward legit right along with it. There's no recalibration that restores separation, because the camouflage is a genuine overlap in embedding space, not a positional drift that a moving reference point can track down. This is not a stale-reference problem.
- **This changes Run 131/132's standing recommendation.** Sequential/online test-time adaptation of the reference centroid -- the most promising untried, literature-unexplored direction identified in Runs 131/132 -- is now ruled out at the oracle-ceiling level, before spending any RunPod compute building the realistic (pseudo-label-based, necessarily noisier-than-oracle) version. Saved a wasted dispatch by testing the ceiling locally first.
- **Updated interpretation**: the ~64% never-recovered floor increasingly looks like a genuine information-theoretic limit of the current feature/embedding representation for this specific camouflaged-fraud population, not a fixable artifact of a frozen reference, insufficient architecture, or insufficient loss engineering. Every static AND now every (oracle-level) sequential approach plateaus at essentially the same ceiling.
- Next: no further pursuit of centroid-recalibration-style ideas (static or sequential) against this specific hard core. Remaining un-explored directions would need a genuinely different signal source entirely (e.g. richer raw features beyond Elliptic's given 166, external chain-of-custody/exchange-side information not present in this dataset) rather than any reweighting of the existing embedding space -- likely outside what this dataset alone can support. Worth treating the current ~18-20% camo-weighted recovery as close to the practical ceiling for this specific investigation, and reporting the diagnostic (Runs 131-133) itself as a genuine contribution: a rigorous characterization of WHY this hard core resists every mechanism tried, not just another failed attempt.

## [2026-07-23] Run 134 — Elliptic: CORRECTION (user catch) on "collapse" framing + LDA/discriminant analysis refines, but doesn't reverse, Run 133's conclusion
- **Correction**: described the never-recovered group's mean dist-to-legit (0.751) as "almost exactly legit's own mean (0.646)" -- overstated. The gap is real (+16% relative), consistent with the established Run 87 framing (post-break fraud keeps a distinct, right-shifted mode, does not fully collapse onto legit). Should not have implied near-total indistinguishability from raw distance alone.
- Tested whether a properly-fit linear discriminant (not just isotropic Euclidean distance-to-centroid, which treats all 128 dims equally) finds separation the current scoring method misses, motivated by user's recollection of an earlier PCA analysis (`camo_weighted_full_analysis.json`) showing some shift for post-break fraud on PC1 (train_fraud=0.786, train_legit=-0.929, test_fraud_post=-0.619, test_legit=-0.883 -- post-break fraud sits within ~0.75 std of legit on PC1, but not AT legit).
- **Cross-validated LDA (fit and scored within never-recovered-vs-test-legit, 5-fold)**: AUC=0.854 -- looked very promising, but this setup implicitly assumes access to labeled test-period camouflaged-fraud examples to fit on, which isn't available in real deployment.
- **Honest, leak-free version (LDA fit ONLY on train_fraud/train_legit, zero test-label access)**: AUC=0.664 on the exact same never-recovered-vs-legit comparison -- barely above plain Euclidean distance's 0.685 on the same comparison. Sanity check confirms the method itself works (easy/caught-by-both fraud vs legit: AUC=0.997 with the same train-only LDA). **A smarter linear method does not meaningfully beat what's already used, when fit honestly.**
- **Interpretation of the gap between 0.854 and 0.664**: the camouflaged fraud population has real, consistent internal structure (a model fit on SOME camouflaged fraud generalizes to OTHER camouflaged fraud reasonably well) -- but that structure is not the same direction train-period fraud sits along, so a train-only model can't find it in advance. This is a more precise version of "evolving camouflage" than a simple centroid drift.
- **Oracle-sequential LDA test** (progressively fold in TRUE fraud from completed test steps, evaluate on the next step -- the proper discriminant-based version of Run 133's centroid-only oracle test): mixed, not a clean win.
  | step | n | static (train-only) AUC | oracle-sequential AUC |
  |---|---|---|---|
  | 43 | 24 | 0.710 | 0.757 |
  | 44 | 23 | 0.653 | 0.736 |
  | 45 | 5 | 0.890 | 0.826 (n too small to trust) |
  | 47 | 22 | 0.576 | 0.763 |
  | 48 | 36 | 0.746 | 0.798 |
  | **49** | **55** | **0.810** | **0.370 (catastrophic, below chance)** |
  Middle steps (all small samples, 22-36) show a modest, fairly consistent gain from oracle sequential updating. **Step 49 -- the largest, most reliable sample -- gets dramatically WORSE, not better**, when steps 42-48's true labels are folded into the fit. This is the opposite of what smooth drift-tracking would predict, and argues the evasion "generation" changes character between batches rather than sliding continuously in one trackable direction -- though several middle-step samples are thin enough (22-36) that part of the modest positive pattern there could be noise.
- **Net effect on Run 133's conclusion**: not reversed, but explained more precisely. It's not that there's no structure in the camouflage (there is, per the 0.854 oracle-CV number) -- it's that the structure isn't stable/extrapolable across the test period even with true labels and a proper discriminant (not just a centroid), and specifically fails hardest on the step where it would matter most. This is a stronger, more specific version of "no reliable sequential-adaptation win" than Run 133 alone established.
- Decision: no further pursuit of sequential/oracle-informed adaptation, static or discriminant-based, for this hard core. The ~18-20% camo-weighted ceiling and Runs 131-134's diagnostic (isolation ruled out, neighbor-camouflage explains but doesn't break the floor, temporal drift is real but not trackable even with oracle labels or a smarter discriminant) together form the honest, defensible characterization of this hard core for the write-up.

## [2026-07-23] Run 135 — Elliptic: whole-cloud density tracking + centroid-trajectory analysis confirms a genuine regime BREAK, not a trackable trend
- Tested two ideas from discussion: (a) track density evolution over the FULL population (labeled+unlabeled, 100% of the graph, using only raw features -- sidesteps the "no cross-step edges" limitation entirely since it needs no graph traversal), and (b) treat the illicit centroid's position over time as a trajectory and check whether its DERIVATIVE (step-to-step displacement) has a consistent, trackable direction.
- **Whole-cloud region-density tracking** (PCA-2D fit on train-labeled raw features, projecting ALL 203,769 nodes regardless of label): the fraction of ALL nodes (labeled+unlabeled) landing within the train-illicit region rises from ~19% (step 1) to a peak of 50-67% around steps 20-27, then collapses to near-zero (2-9%) by steps 39-49. This raw occupancy measure conflates population-wide feature drift (Bitcoin's overall on-chain characteristics evolving over the dataset's real ~150-day span) with fraud-specific evasion -- not a clean camouflage signal on its own, but confirms the whole population's feature distribution moves substantially over this timeline, not just labeled fraud.
- **Illicit centroid trajectory + step-to-step derivative**: computed the per-step illicit centroid position (PCA-2D, all 49 steps) and the displacement vector between consecutive steps. Cosine similarity between CONSECUTIVE displacement vectors oscillates unpredictably between -1.0 and +1.0 with no discernible pattern (e.g. 16->17 vs 17->18: cos=-1.000; 15->16 vs 16->17: cos=+0.992) -- classic signature of a noise-dominated random walk, consistent with many per-step illicit counts being tiny (5-30 labeled points), making step-to-step centroid estimates inherently noisy. **This directly explains Run 134's oracle-sequential-LDA catastrophic failure at step 49** -- extrapolating "the immediately preceding step's direction continues" is chasing noise, not signal.
- **But a genuine SMOOTH long-run trend does exist within the train period**: fit a linear trend (centroid position vs. step number, PCA-10) on TRAIN steps 1-34 only. Several PCA dims show strong linear fit (R²=0.87, 0.77, 0.63 on dims 1/3/4) -- a real, non-noisy directional drift exists when you look at the right timescale (whole-train-period regression, not step-to-step).
- **Critical test: does extrapolating this clean train-period trend into the test period help?** No -- it does not reliably beat (and sometimes clearly loses to) the naive static train-only centroid:
  | step | static AUC | trend-extrapolated AUC |
  |---|---|---|
  | 39 (pre-break) | 0.911 | 0.943 (better) |
  | 42 (the break itself) | 0.885 | 0.809 (worse) |
  | 43 | 0.671 | 0.510 (much worse, near chance) |
  | 47 | 0.813 | 0.786 (worse) |
  | 49 | 0.668 | 0.664 (wash) |
  A trend with R²=0.87 fit cleanly within train, if the underlying process were a smooth continuous evolution, should extrapolate BETTER into the test period than a static reference -- it does not, and specifically fails hardest right at and after the break (steps 42-43). This is the cleanest evidence yet that step 42/43 is a genuine DISCONTINUITY in the underlying process, not a continuation of a pre-existing slope.
- **Triangulated conclusion**: three independent tests (step-to-step derivative noise in Run 135, oracle-sequential LDA failure at step 49 in Run 134, and now clean-trend extrapolation failing specifically at the break in Run 135) all point to the same thing from different angles -- this is a regime CHANGE, not a trackable trend, consistent with CLAUDE.md's own "regime break" framing (not "gradual drift") from the original investigation. No extrapolation strategy tried (step-to-step, oracle-sequential, or smooth long-run trend) survives the break.
- Decision: closes the "can we track/extrapolate the camouflage evolution" line of inquiry. The break itself, not just the subsequent camouflage level, resists prediction from anything derivable purely from the train-period trajectory. Reinforces Runs 131-134's conclusion that the ~18-20% camo-weighted ceiling is close to the practical limit for this specific investigation using this dataset's features/graph.
- Next: no further pursuit of trend/trajectory-based extrapolation for this hard core. This diagnostic thread (Runs 131-135) is now comprehensive enough to write up as its own contribution.
- Visualization published (trajectory, derivative magnitude, direction-consistency): https://claude.ai/code/artifact/1d5b26eb-f165-4ba5-bf65-4d03a903e296

## [2026-07-23] Run 136 — CORRECTION (user catch, from the Run 135 visualization): trend IS real and visible from the start; the AUC failure isn't simple overshoot
- User, looking at the published trajectory plot, correctly noted a clear macro-trend is visible across the WHOLE 49-step path (not just noise) -- Run 135's "step-to-step derivative is noise, no consistent direction" framing conflated two different claims: LOCAL (consecutive-step) direction consistency is genuinely noisy (confirmed, cosine similarity oscillates -1..+1), but that does NOT mean no GLOBAL trend exists -- Run 135 itself already found a clean train-period trend (R²=0.87 on the best PCA dim). Should have been explicit that these are compatible facts (a real drift with noisy local fluctuation on top), not stated in a way that read as "no trend at all."
- **Checked whether the trend's positional error explains the AUC-extrapolation failure (it doesn't, cleanly)**: refit the train-only 2D trend, compared predicted vs actual centroid position at each test step.
  | step | overshoot distance | trend_extrap AUC vs static (Run 135) |
  |---|---|---|
  | 41 | 2.036 (LARGEST) | 0.812 vs 0.809 (no difference) |
  | 43 | 1.102 (moderate) | 0.510 vs 0.671 (WORST failure) |
  | 49 | 0.739 (one of smallest) | 0.664 vs 0.668 (no difference) |
  Positional overshoot does NOT correlate with which steps' AUC actually collapsed -- step 41 has the biggest position error with zero AUC impact, step 43 has a middling position error with the worst AUC impact. The y-coordinate trend tracks almost exactly (predicted 5.522 vs actual 5.519 at step 49); x has a persistent OFFSET (actual stays -3.1 to -4.2, predicted flat ~-2.75) rather than a growing divergence -- not what a runaway-extrapolation-error story would predict either.
- **Honest conclusion**: the long-run trend is real, visible, and reasonably accurate positionally (not what "regime break" alone would suggest) -- Run 135's correction stands (extrapolating it doesn't reliably improve scoring), but the MECHANISM isn't simple centroid overshoot. Something about how individual fraud points relate to legit's LOCAL density in the specific region near step 43 is driving that particular AUC collapse, not the aggregate centroid's positional accuracy. This is an open question, not a solved one -- flagging honestly rather than force-fitting an explanation.
- Decision: keep Run 133/134/135's core finding (no extrapolation strategy tried reliably improves hard-core recovery) but soften the "no trend" framing to "a real trend exists, extrapolating it doesn't reliably help for reasons not yet fully understood at the per-step level."

## [2026-07-23] Run 137 — Elliptic: added caught-vs-never-caught subgroup trajectories to the visualization
- Extended the Run 135/136 visualization with per-step centroids (same raw-feature PCA-2 space) for camo-weighted's three outcome groups: caught_both (caught by both RF+GNN), sometimes_recovered (camo-weighted's actual wins, ~18-20%), never_recovered (the persistent floor).
- **caught_both is not a trend at all**: 140 of 142 such cases land at step 42 alone (the post-break spillover of the pre-existing obvious archetype) -- shown as a single reference marker, not a line, since there's no meaningful multi-step trajectory to plot for it.
- **sometimes_recovered and never_recovered both spread across all test steps (42-49)** and were added as full trajectory lines for visual comparison.
- **Caveat, worth keeping explicit**: in this raw-feature 2D PCA space (~18% variance explained), sometimes_recovered's mean distance to the legit reference (5.49) and never_recovered's (5.67) are close to each other and to caught_both's (5.66) -- NOT the same clean separation found in Run 134's LDA/embedding analysis (never-recovered at 0.751 vs sometimes-recovered at 1.192 in the trained 128-dim camo-weighted embedding). This isn't a contradiction: the meaningful separation lives specifically in the SUPERVISED, triplet-loss-trained embedding space, not in this simple unsupervised raw-feature projection. The two analyses answer different questions and shouldn't be conflated.
- Visualization updated (same URL): https://claude.ai/code/artifact/1d5b26eb-f165-4ba5-bf65-4d03a903e296

## [2026-07-23] Run 138 — Elliptic: physics-analogy thread (continuity equation, eikonal/plane-wave) closed out — real characterizations, no usable signal
- Explored three physics-inspired framings for the camouflage evolution, prompted by discussion: (a) whether a crude "grab the whole nearby region" classifier works, (b) a fluid-continuity-equation analogy (∂ρ/∂t + ∇·(ρv) = 0) for local illicit-density change, (c) an eikonal/plane-wave analogy for the direction of illicit concentration relative to legit.
- **(a) Region-capture in raw-feature PCA-2 space**: computed directly -- flagging everything within radius R of the test-period illicit centroid. At 50% recall, 17.8% of ALL test legit gets falsely flagged (12.0% precision); at 90% recall, 60.6% of all legit is flagged. Confirms this space has essentially no separating power for a naive proximity rule -- consistent with margin20/dual_view_encoded's threshold-sensitivity pattern, just far more extreme since there's no trained discriminant involved at all.
- **(b) Continuity-equation / ∂ρ/∂t test**: measured local illicit-density growth (Gaussian-kernel-weighted enrichment, late-train minus early-train window, fully leak-free) at the position of each outcome group. Initially looked clean and monotonic: caught_both +0.073, sometimes_recovered +0.035, legit baseline +0.028, never_recovered +0.003 (never-recovered sits in unusually STATIC regions, below even legit's own ambient drift).
  **But confirmed via full-distribution/AUC check (not just means) that this is NOT a usable classifier**: 33.2% of a legit sample has growth >= caught_both's own mean; 58.7% has growth >= never_recovered's mean. AUC of growth alone: caught_both 0.656 (weak), sometimes_recovered 0.538 (near chance), **never_recovered 0.441 (BELOW chance)**. The whole population's raw-feature distribution drifts substantially over this dataset's real ~150-day span (confirmed earlier, Run 135's whole-cloud density finding) for reasons unrelated to fraud -- ∂ρ/∂t is confounded by this ambient population drift, not a clean fraud-specific signal.
- **(c) Eikonal/plane-wave angular-stability test**: computed the illicit population's angular position (relative to the fixed legit centroid) per step. Circular concentration (R) stays remarkably high (0.93-0.99) for most of the 49 steps, and the mean angle settles into a stable ~130-170 degree direction from step 17 onward, persisting unchanged through the step-42 break to step 49 -- a small number of steps (6, 24, 26, 29, 31, 32, 41) show transient low-R "wobble" episodes before snapping back, loosely analogous to caustic-like refocusing events. This IS a genuine, previously-unisolated structural fact: the illicit population's DIRECTION relative to legit is far more stable than its POSITION (which Run 135/136 already found to be noise-dominated step-to-step) -- separating angle from radius reveals a real stability the raw centroid-trajectory view didn't show.
  **But using this stable direction as a 1D projection classifier fails**: AUC=0.60 for never-recovered vs legit, and only 0.59 even for caught_both (easy, obviously-different fraud) -- barely above chance for either. PCA-2 (~18% of raw feature variance) is too coarse a cross-section to carry real separating power along any single fixed axis, regardless of how geometrically stable that axis is.
- **Overall conclusion**: both physics analogies produced genuine, verifiable structural facts about this dataset (ambient population-wide drift exists and confounds naive growth-based signals; the illicit population's direction, not just position, has real and previously-unmeasured stability) -- but neither produces a usable classifier, for two different reasons (growth is confounded with everything, direction alone lacks separating power in this reduced 2D space). This closes the physics-analogy thread. Combined with Runs 131-137, the hard-core diagnostic is now comprehensive across structural (isolation, neighbor), representational (LDA/embedding), and dynamical (trend, derivative, growth, direction) angles -- consistently converging on: real, confirmable structure exists, but nothing tried recovers meaningfully more of it than camo-weighted's original ~18-20%.
- Decision: no further pursuit of physics-inspired spatiotemporal signals for this hard core. Time to write up Runs 131-138 as a unified diagnostic contribution.

## [2026-07-23] Run 139 — Elliptic: legit's own trajectory added to the visualization, confirms ambient population drift directly
- Added legit's own per-step centroid trajectory (same PCA-2 space, same 49 steps) to the Run 135-137 visualization, to directly show the "ambient population drift" finding from Run 138 rather than leaving it as a number.
- **Legit's own centroid moves MORE per step than illicit's on average** (mean displacement 1.62 vs illicit's 0.94, median 1.36 vs 0.58) and is **just as directionally erratic** (consecutive-displacement cosine similarity: legit mean -0.325/std 0.716, illicit mean -0.237/std 0.759 -- both centered near zero, both noisy zigzags, no meaningful difference in erraticism between the two classes).
- This directly visualizes Run 138's conclusion: the whole population's raw-feature distribution drifts substantially over this dataset's real ~150-day span, and legit is swept up in it at least as much as illicit is -- reinforcing that the camouflage phenomenon is not simply "fraud moves, legit stays put," but rather "everything moves, and fraud's movement happens to end up overlapping more with wherever legit currently is."
- Visualization updated (same URL): https://claude.ai/code/artifact/1d5b26eb-f165-4ba5-bf65-4d03a903e296

## [2026-07-23] Run 140 — Elliptic: causal temporal-kernel smoothing gives a genuine, leak-free AUC improvement — refines (doesn't reverse) the "noisy derivative" finding
- Direct follow-up to the visual observation (Run 136-139 screenshots) that the raw per-step trajectory is noisy specifically because many steps have very few labeled illicit points (2-30). Tested denoising via a Gaussian time-kernel (bandwidth=3 steps) pooling nearby steps' raw points, weighted by temporal proximity -- fully causal (current + PAST steps only, no future information, deployable at inference time).
- **Direction consistency**: consecutive-displacement cosine similarity jumps from -0.237 (raw, effectively noise) to +0.988 (smoothed, std 0.029) -- confirms the earlier "erratic derivative" finding (Run 135) was substantially a small-sample estimation artifact, not a fact about the underlying process.
- **Contemporaneous gap** (smoothed illicit-centroid to smoothed legit-centroid distance) now shows the expected pattern cleanly: train mean 2.99 -> test mean 2.32, and step 49's gap (2.14) is smaller than step 42's (2.38) -- the raw/unsmoothed version (Run 139) had shown the OPPOSITE (step 49 gap larger than step 42's), which was itself a noise artifact.
- **AUC improvement, honestly leak-free**: scoring never-recovered fraud (vs test legit) using a causally-smoothed reference instead of a single static train-only centroid: 0.583 (static) -> 0.657 (causal-smoothed), a real ~13% relative gain, using ONLY information available at inference time (current step's data plus everything before it, nothing from the future).
- **Mechanistically distinct from Run 133's failed oracle-sequential test**: Run 133 used a cumulative, EQUALLY-WEIGHTED expanding average (all steps up to now diluted together, old data never decaying), which showed no improvement even with oracle labels. This uses a LOCAL Gaussian time-kernel that keeps the reference temporally relevant (nearby steps dominate, distant ones fade) rather than accumulating everything indiscriminately -- a genuinely different mechanism, not a re-test of what already failed.
- **Scope/caveat**: this test was run on the crude raw-feature 2D PCA space (~18% of feature variance), which has much weaker absolute separating power than the properly-trained 128-dim camo-weighted embedding (LDA there: 0.664 honest / 0.854 oracle, Run 134). The finding here is about the RELATIVE benefit of proper temporal regularization, not a claim that 0.657 beats anything already achieved.
- Visualization updated with a before/after trajectory comparison and the AUC table (same URL): https://claude.ai/code/artifact/1d5b26eb-f165-4ba5-bf65-4d03a903e296
- Decision: this reopens a concrete, promising, NOT-yet-tried direction -- apply this same causal Gaussian time-kernel smoothing to the properly-trained camo-weighted embedding's legit/fraud reference centroids (replacing the single frozen train-derived centroid), instead of assuming Run 133's cumulative-average result closed the door on ALL sequential-adaptation mechanisms. Distinct enough from what's been tried to warrant an actual RunPod dispatch + multi-seed confirmation + hard-core cross-reference before drawing conclusions.
- Next: implement causal time-kernel-smoothed reference centroids in `evaluation/metric_learning.py` (or as a post-hoc scoring change using existing per-seed embeddings), dispatch and multi-seed confirm on the real embedding, cross-reference against the hard-core mask.

## [2026-07-23] Run 141 — CORRECTION (external review catch): Run 140's "causal" smoothing was leaky; the real gain is marginal-to-negative
- External review (a second Claude instance reading Runs 123-140) caught that Run 140's "fully causal" kernel used `step_vals <= target_step`, which INCLUDES the current step itself -- e.g. scoring step 43's nodes used a centroid built partly from step 43's OWN true illicit labels. Temporal causality (using only steps <= t) is not the same as label availability at inference time when t itself is included; this is a direct leak, not "information available before scoring."
- **Re-tested with the current step properly excluded, at two lags**:
  | Version | AUC |
  |---|---|
  | max_lag=0 (original, leaky, includes current step) | 0.6505 |
  | **max_lag=1 (current step excluded, genuinely causal)** | **0.6032** |
  | max_lag=2 (current + immediately-prior step excluded) | 0.5771 |
  | static baseline (no smoothing) | 0.5830 |
- Almost the entire apparent improvement (0.583->0.650) was the leak. Properly causal (lag=1) smoothing gives only a marginal +0.02 over baseline, and lag=2 is actually WORSE than baseline. This is the same "too good for what the model is actually given" pattern as Run 86 (dimension selection on test labels) and Run 96 (target-encoding self-reference) -- both previously caught the same way.
- **Retracting Run 140's "genuine leak-free AUC improvement" claim and its "Next" step** (dispatching causal time-kernel smoothing on the real 128-dim embedding). The premise was false -- there is no clean, leak-free win here to transplant to the trained embedding. Not proceeding with that dispatch.
- What DOES still stand from Run 140: the mechanistic point that a LOCAL time-kernel differs from Run 133's cumulative equally-weighted average (real, and worth remembering), and the direction-consistency finding (raw derivative noise vs a coherent trend once smoothed) -- but the trend's smoothness was partly an artifact of the same leak (smoothing toward the answer), so even that visual claim needs re-verification with a properly lagged kernel before being trusted. Visualization not yet corrected -- flagging as still showing the leaky version, needs an update pass.
- Lesson reinforced: "fully causal, deployable at inference time" is a claim that must be verified by explicitly checking whether the current time-step's own labels enter the computation, not asserted from using `<=` in a step-index comparison alone.

## [2026-07-23] Run 142 — CORRECTION (external review): margin20's "Pareto tradeoff" was a threshold artifact; dual_view_encoded is a genuine, better-than-reported improvement
- External review (second Claude instance) questioned whether Run 130's margin20 "Pareto tradeoff" (better F1, worse hard-core recovery) and Run 127's dual_view_encoded (same qualitative pattern) were real tradeoffs between two different classifiers, or just the SAME ranking evaluated at different decision thresholds -- proposed two cheap tests on already-collected predictions: (a) threshold-free AUC (overall and hard-core-vs-legit) on the same paired seeds, (b) re-threshold the candidate to match baseline's predicted-positive-rate and check whether hard-core recovery reverts.
- **margin20 (Run 130): CONFIRMED to be a pure threshold artifact, not a real tradeoff.**
  | | overall AUC | hard-core-vs-legit AUC |
  |---|---|---|
  | baseline | 0.8477 | 0.7225 |
  | margin20 | 0.8493 | 0.7239 |
  | paired diff / p / wins | +0.0016 / p=0.8457 / 5-10 | +0.0014 / p=1.0000 / 5-10 |
  Both AUCs are statistically indistinguishable from baseline -- same ranking, not a different classifier. Re-thresholding margin20 to match baseline's mean ppr (0.120): hard-core recovery jumps from 7.4% (at margin20's own default 0.5 threshold) back to **17.7%** -- statistically indistinguishable from baseline's 19.2% (per-seed range 9.9-23.9%, overlapping baseline's own spread). **Run 130's "sharp hard-core-for-precision tradeoff, confirmed at p=0.002 both directions" is RETRACTED as a tradeoff claim** -- the F1 gain and recovery loss are both simple consequences of margin20 shifting the decision threshold along the IDENTICAL underlying ROC curve, not evidence of two different discriminative models. margin20 changes nothing about the model's ranking ability.
- **dual_view_encoded (Run 127): the SAME two tests give the OPPOSITE verdict -- it is a genuine, better-than-previously-reported improvement.**
  | | overall AUC | hard-core-vs-legit AUC |
  |---|---|---|
  | baseline | 0.8528 | 0.7320 |
  | dual_view_encoded | 0.8719 | 0.7681 |
  | paired diff / p / wins | +0.0191 / p=0.0488 / 8-10 | +0.0362 / p=0.0488 / 8-10 |
  Both AUCs are SIGNIFICANTLY higher for dual_view_encoded -- a real ranking improvement, not a threshold shift. Re-thresholding to match baseline's mean ppr (0.118): hard-core recovery = **22.7%**, actually HIGHER than baseline's own 19.5% (not lower, as Run 127's default-threshold comparison suggested; per-seed range 10.3-39.4%, noisy but clearly centered above baseline).
- **Corrected overall picture**: Run 127's "honest hard-core caveat" (worse recovery, breaks the zero-loss record) was itself an artifact of comparing at mismatched operating points (dual_view_encoded's own default-threshold ppr, 0.090, is more conservative than baseline's 0.120/0.118) -- not a real cost. At matched thresholds, dual_view_encoded is a strict improvement over baseline camo-weighted: better AUC (significant), better F1 (already confirmed, Run 127), and better-or-equal hard-core recovery. **dual_view_encoded should be promoted as a genuine upgrade over camo-weighted, not reported as a parallel Pareto-frontier point (retracting Run 129's "two separate results, not one superseding the other" framing for this specific pair).**
- Broader lesson (now the THIRD time this exact class of check has mattered this session, after margin05/temp_sharp in Run 119): whenever a candidate's aggregate metric improves alongside an apparent cost elsewhere (F1 up, recovery down, or vice versa), always check threshold-free AUC AND a matched-operating-point comparison before characterizing it as ANY kind of tradeoff -- it can hide either a non-finding (margin20) or an under-reported win (dual_view_encoded), and the only way to tell which is the AUC/matched-threshold check, not the raw default-threshold numbers.
- Decision: adopt dual_view_encoded as the new reference-best Elliptic config (genuine, multi-metric-confirmed improvement over camo-weighted). Do NOT pursue margin20 further (confirmed non-finding). Do NOT dispatch the Run 140-141 embedding-smoothing experiment (premise already retracted in Run 141).
- Next: re-run the earlier "extend camo-weighting" style loss variants (temp/dual/margin_scale, Run 119/129) against dual_view_encoded as the new base, now that it's confirmed as a real improvement worth building on -- per Run 127's original open item (c), now on firmer footing.

## [2026-07-23] Run 143 — campaign: elliptic_dual_view_encoded_temp vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_temp.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6681 | 0.0323 | 9 |
| configs/elliptic_dual_view_encoded_temp.yaml | FAILED (0/? seeds succeeded) |

- Decision: kept existing best (candidate did not win significantly: mean N/A vs 0.668125024411478, p=N/A)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 144 — campaign: elliptic_dual_view_encoded_dual vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_dual.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | FAILED (0/? seeds succeeded) |
| configs/elliptic_dual_view_encoded_dual.yaml | FAILED (0/? seeds succeeded) |

- Decision: kept existing best (candidate did not win significantly: mean N/A vs N/A, p=N/A)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 145 — campaign: elliptic_dual_view_encoded_margin_scale vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_margin_scale.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | FAILED (0/? seeds succeeded) |
| configs/elliptic_dual_view_encoded_margin_scale.yaml | FAILED (0/? seeds succeeded) |

- Decision: kept existing best (candidate did not win significantly: mean N/A vs N/A, p=N/A)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 146 — campaign: elliptic_dual_view_encoded_32 vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_32.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6714 | 0.0347 | 10 |
| configs/elliptic_dual_view_encoded_32.yaml | 0.6375 | 0.0306 | 10 |

Wilcoxon signed-rank: statistic=3.00, p=0.0098, n_pairs=10
- Decision: kept existing best (candidate did not win significantly: mean 0.6374788875671458 vs 0.6713660501509334, p=0.009765625)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 147 — campaign: elliptic_dual_view_encoded_128 vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_128.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6686 | 0.0387 | 10 |
| configs/elliptic_dual_view_encoded_128.yaml | 0.6779 | 0.0220 | 10 |

Wilcoxon signed-rank: statistic=20.00, p=0.4922, n_pairs=10
- Decision: kept existing best (candidate did not win significantly: mean 0.6778711637427058 vs 0.668629505599896, p=0.4921875)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 148 — campaign: elliptic_dual_view_encoded_temp vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_temp.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6671 | 0.0352 | 10 |
| configs/elliptic_dual_view_encoded_temp.yaml | 0.6546 | 0.0304 | 10 |

Wilcoxon signed-rank: statistic=9.00, p=0.0645, n_pairs=10
- Decision: kept existing best (candidate did not win significantly: mean 0.6546228541112252 vs 0.6671270072850518, p=0.064453125)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 151 — Elliptic: resolved the Run 135↔140 contradiction (external review catch) — extrapolation failure is NOT a noise artifact
- (Renumbered from a duplicate "Run 148" -- collided with infra/campaign.py's auto-appended entry of the same number, dispatched concurrently. See Run 148's own note on this class of numbering collision.)
- External review (second Claude instance) flagged an unresolved tension left across Runs 135/136/140: Run 135 concluded "regime break, not a trackable trend" (clean pre-break trend fails to extrapolate, crashing hardest at steps 42-43); Run 136 softened this (the trend IS real, R²=0.87 within train); Run 140 (before its own leak correction in Run 141) found near-perfect direction consistency (cosine=0.988) once the trajectory was denoised. None of these had actually tested whether extrapolating a SMOOTHED trend (rather than Run 135's original noisy-per-step-input trend) still fails at the same points -- leaving live the possibility that the original extrapolation failure was itself a noise artifact of fitting a trend to noisy inputs, not a real discontinuity.
- Also clarified precisely (per the same review): Run 141's leak finding is scoped to the INFERENCE-TIME AUC claim (0.583->0.657) and the word "deployable" -- it does NOT invalidate Run 140's descriptive finding (direction consistency, the contemporaneous-gap pattern) itself, since using true labels to CHARACTERIZE an already-observed phenomenon is legitimate descriptive analysis, not leakage. Only forward-predictive/inference-time claims needed retracting.
- **Direct test**: fit the same linear trend as Run 135/136 (PCA-10, steps 1-34), but on a train-internally-smoothed input (Gaussian time-kernel, bandwidth=3, using ONLY train data -- no test labels touched anywhere in the fit, so no leakage concern) instead of Run 135's raw noisy per-step centroids. Fit quality improved substantially (R²=0.545-0.992 across dims, vs Run 135's best-dim 0.87) -- a genuinely cleaner trend to extrapolate from.
  | step | static AUC | smoothed-trend-extrap AUC | raw-trend-extrap AUC (Run 135, reference) |
  |---|---|---|---|
  | 39 (pre-break) | 0.911 | 0.927 | 0.943 |
  | 42 (the break) | 0.885 | 0.786 | 0.809 |
  | **43** | 0.671 | **0.543** | 0.510 |
  | 44 | 0.459 | 0.507 | 0.507 |
  | 49 | 0.668 | **0.505** | 0.664 (smoothed input is WORSE here) |
- **Result: extrapolation still fails at exactly the same points, even with a much cleaner (R² up to 0.99) trend fit.** Step 43 remains the worst failure (0.543 vs static's 0.671) regardless of how well the pre-break trend itself is estimated; step 49 is actually WORSE with the smoothed-trend extrapolation than with the original raw one. **This definitively resolves the open question: the extrapolation failure was never about noisy trend-fitting -- it is a genuine discontinuity.** Run 136's stable-direction finding (cosine=0.988, hindsight/full-data description) and Run 135's extrapolation-failure finding (train-only fit, forward prediction) are both true and NOT in tension: a process can have a clean, well-fit direction in hindsight while still being fundamentally unpredictable in advance, if the thing that changes at the break is not the DIRECTION but something else (rate, or a genuine shift in the underlying generating process) that a directionally-faithful linear extrapolation cannot capture.
- Decision: this closes the Run 135/136/140 contradiction cleanly. No further pursuit of any trend/smoothing-based extrapolation scheme for this hard core, now on firmer footing than before (previously only tested on noisy inputs; now confirmed the same failure persists even with clean inputs).

## [2026-07-23] Run 149 — campaign: elliptic_dual_view_encoded_dual vs elliptic_metric_learning_dual_view_encoded
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_metric_learning_dual_view_encoded.yaml,
  candidate=configs/elliptic_dual_view_encoded_dual.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_metric_learning_dual_view_encoded.yaml | 0.6633 | 0.0278 | 10 |
| configs/elliptic_dual_view_encoded_dual.yaml | 0.6937 | 0.0396 | 10 |

Wilcoxon signed-rank: statistic=7.00, p=0.0371, n_pairs=10
- Decision: PROMOTED to new best (mean 0.6937 > 0.6633, p=0.0371 < 0.05)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 150 — campaign: elliptic_dual_view_encoded_margin_scale vs elliptic_dual_view_encoded_dual
- Dispatched via infra/campaign.py's auto-compare-and-promote runner (best=configs/elliptic_dual_view_encoded_dual.yaml,
  candidate=configs/elliptic_dual_view_encoded_margin_scale.yaml).
| Config | Mean f1_macro | Std | n |
|---|---|---|---|
| configs/elliptic_dual_view_encoded_dual.yaml | 0.6932 | 0.0389 | 10 |
| configs/elliptic_dual_view_encoded_margin_scale.yaml | 0.6797 | 0.0448 | 10 |

Wilcoxon signed-rank: statistic=13.00, p=0.1602, n_pairs=10
- Decision: kept existing best (candidate did not win significantly: mean 0.6797295089215365 vs 0.6931972358807298, p=0.16015625)
- Observations: (fill in manually)
- Next: (fill in manually)

## [2026-07-23] Run 152 — Elliptic: dual_view_encoded + camo_weighted_dual mining -- promising but not conclusively confirmed (AUC borderline at n=10)
- Campaign (Runs 148-150, auto-logged) testing 3 loss-formula variants combined with the confirmed-good dual_view_encoded architecture (replacing plain camo_weighted mining): temp (T=0.3) lost narrowly (p=0.064), camo_weighted_dual mining WON on F1 (0.663->0.694, p=0.037, promoted), margin_scale then lost against the new dual_view_encoded_dual best (p=0.16).
- Given Run 142's lesson (F1 wins can hide threshold artifacts in either direction), checked AUC and matched-ppr recovery before trusting the F1 result:
  | | dual_view_encoded (baseline) | dual_view_encoded_dual (candidate) |
  |---|---|---|
  | F1 | 0.6633 | 0.6937 |
  | hard-core recovery (default 0.5 threshold) | 15.5% | 12.3% |
  | predicted-positive-rate | 0.090 | 0.071 |
  | overall AUC | 0.8707 | 0.8791 |
  | hard-core-vs-legit AUC | 0.7655 | 0.7809 |
  AUC diffs (+0.0083 overall, +0.0154 hard-core) are consistently positive (9/10 seeds both) but do NOT clear conventional significance (p=0.084 both) -- unlike dual_view_encoded's own clean win over the original baseline (p=0.049, Run 142). This is a genuinely intermediate case: not a flat/tied AUC (margin20's pattern, a pure threshold artifact) and not a clearly-significant AUC gain (dual_view_encoded's own pattern) -- a small, fairly consistent positive signal that falls short of the bar this project has been holding results to.
  **Matched-ppr recovery** (re-thresholding the candidate to match baseline's own ppr): 17.4% vs baseline's 15.5% -- higher, not lower, consistent with the borderline-positive AUC direction rather than a threshold-artifact story.
- **Read**: likely a real but small incremental improvement (dual-sided camouflage weighting DOES add something on top of dual_view's own architecture, unlike on the old graphsage_diff architecture where it washed, Run 129) -- but n=10 isn't enough to call this confirmed given the AUC p-value sits just above 0.05. Needs a larger/independent seed count (e.g. n=20, or a fresh independent n=10 replication) before being promoted the way dual_view_encoded itself was.
- Decision: promising lead, not yet a confirmed result. Do not treat "PROMOTED" (the campaign framework's own F1-only auto-decision) as equivalent to "confirmed" -- exactly the distinction Run 142 already established needs to be checked every time, now demonstrated on a THIRD case that lands in the ambiguous middle rather than clearly at either extreme.
- Next: dispatch an independent n=10 (or larger) replication of dual_view_encoded vs dual_view_encoded_dual specifically, checking AUC significance directly rather than relying on the campaign's F1-only promotion criterion.

## [2026-07-23] Run 153 — Elliptic: dual_view_encoded_dual independent replication -- F1 robust, AUC inconsistent, matched recovery modestly positive
- Independent replication (fresh seeds 10-19, disjoint from Run 152's seeds 0-9) of dual_view_encoded vs dual_view_encoded_dual.
- **F1: robustly confirmed across BOTH independent batches** -- batch 1 (seeds 0-9): 0.663->0.694, p=0.037; batch 2 (seeds 10-19): 0.672->0.699, p=0.0059 (tighter still). This part is solid.
- **AUC: does NOT replicate cleanly** -- batch 1: overall +0.0083/p=0.084 (9/10 wins), hard-core +0.0154/p=0.084; batch 2: overall +0.0017/p=0.625 (5/10 wins, essentially flat), hard-core +0.0022/p=0.625. Pooled n=20: overall +0.0050/p=0.097 (14/20 wins), hard-core +0.0088/p=0.097 -- still not conventionally significant. This is exactly the "looked promising at n=10, weakens on fresh seeds" pattern this project has caught repeatedly (DropEdge, gate_aux_weight, GraphSAGECamoAgg) -- flagging the parallel explicitly rather than downplaying it.
- **Matched-ppr recovery: consistently positive in both batches** -- batch 1: 17.4% vs baseline's 15.5%; batch 2: 16.6% vs baseline's 14.8%. Pooled n=20 matched recovery: 17.0%, vs baseline's own pooled default recovery ~15.2% -- a modest (+1.8pp), directionally consistent improvement across all 20 seeds, not a reversal and not a dramatic win either.
- **Overall verdict**: F1 improvement is real and robust; the underlying separation improvement is modest and consistent (via matched-recovery) but NOT clean/dramatic the way dual_view_encoded's own win over the original camo_weighted baseline was (Run 142: AUC p=0.049, a much cleaner signal). This sits in a genuinely intermediate category -- more real than a pure threshold artifact (matched recovery moves the right way both times), but weaker and noisier than a fully confirmed architectural win. Treat as a small, adopted improvement, not a headline result.
- Decision: adopt dual_view_encoded_dual as the current best Elliptic config given the robust F1 signal and consistently-positive (if modest) matched-recovery, but do not oversell it in any write-up -- state the AUC inconsistency explicitly alongside the F1 result.

## [2026-07-23] Run 154 — Elliptic: streamline visualization (properly causal, lag≥1 smoothing) for both classes
- Built a "streamlines" visualization: both illicit and legit centroid trajectories, smoothed with the same Gaussian time-kernel as Run 140 but corrected per Run 141's fix (max_lag=1, current step strictly excluded -- no leak). Both classes shown together for the first time in this smoothed/causal form.
- Explicit in-artifact caveat carried over from Run 151: a clean-looking streamline is a descriptive fact (built from true labels, legitimate for characterization) and does NOT imply the movement is forecastable in advance -- Run 151 already showed the extrapolation fails at step 43 regardless of how cleanly the pre-break trend fits.
- Published: https://claude.ai/code/artifact/f3465139-1448-43bb-88d3-e3af71af3f33

## [2026-07-23] Run 155 — Elliptic: bootstrap-ensemble "many streamlines" visualization; TabDDPM archetype conditioning implemented
- Added a 40-line bootstrap ensemble (resample-with-replacement per class, same causal lag>=1 smoothing per Run 141's fix, applied independently to each resample) to the streamlines visualization -- a laminar-vs-turbulent read on trajectory confidence: a tight parallel bundle means the underlying smoothed trajectory is well-determined by the data at that point in the timeline; fanning/crossing lines flag genuine uncertainty (expected to concentrate where raw per-step samples are smallest). Published (same URL, updated in place): https://claude.ai/code/artifact/f3465139-1448-43bb-88d3-e3af71af3f33
- **Implemented targeted diffusion augmentation** (`training/train_diffusion.py`): added `_select_archetype()`, a one-time 2-component GMM split on TRAIN fraud's raw-feature distance-to-legit-centroid (same Run 81/82 methodology as `evaluation/metric_learning.py`'s `_fraud_prototypes`, but on raw features rather than live embeddings, since diffusion training happens before any GNN exists). New `diffusion.archetype` config key: `"all"` (default, preserves existing unconditional-generator behavior exactly), `"camouflaged"` (the GMM component with SMALLER mean distance-to-legit -- the rare, hard-to-classify sub-population), or `"obvious"` (the other component). Motivation: rather than another loss/architecture tweak on the same fixed feature representation, TRY TO STRENGTHEN the underlying signal by giving metric learning more synthetic examples of specifically the population it struggles with -- previous generic (unconditional) diffusion augmentation was tested BEFORE the camouflaged-archetype discovery and found not to stack additively with other interventions (CLAUDE.md), so this is a meaningfully different, more targeted test, not a rerun of a known negative.
- Next: build the camouflaged-archetype TabDDPM training config + augment config, generate synthetic camouflaged fraud nodes, wire into the current best (dual_view_encoded_dual) training pipeline, smoke-test before committing to a full multi-seed dispatch.

## [2026-07-23] Run 156 — Elliptic: QDA test confirms this is a mean-shift phenomenon, not a shape difference
- Direct test of whether a quadratic (2nd-order) decision surface -- QDA, which allows each class its own covariance matrix, unlike LDA's shared-covariance assumption -- captures separation LDA misses. Same honest (train-only fit) setup as Run 134's LDA test.
- **Result: QDA barely beats LDA.** Never-recovered-vs-legit AUC: LDA=0.664, QDA=0.675 (reg_param swept 0.3-0.999, all giving virtually identical 0.675 -- not a regularization-tuning issue, genuinely stable). Caught-both sanity check: 0.997 (LDA) vs 0.999 (QDA), also negligible difference.
- **Interpretation**: the separating signal in this embedding is almost purely a MEAN-SHIFT along one consistent direction (confirmed stable, Run 138/151), not a difference in class SHAPE/spread. QDA is specifically built to exploit shape differences; since there mostly aren't any, it has little extra to find beyond what LDA's linear projection already captures. This is a clean, coherent confirmation of the overall picture rather than a new lever.
- Decision: closes the "higher-order decision surface" question. No further pursuit of quadratic/covariance-based discriminant approaches for this hard core.
- Also: full seed=42 run of the camouflaged-diffusion-augmented dual_view_encoded_dual pipeline completed (F1=0.657, AUC=0.875) -- single seed only, not yet informative; matched multi-seed comparison against the non-augmented baseline still needed before drawing any conclusion.

## [2026-07-23] Run 157 — Elliptic: targeted camouflaged-archetype diffusion augmentation does NOT help (5-seed, consistent negative direction)
- 5-seed matched compare: dual_view_encoded_dual (current best, no augmentation) vs the same model trained on a graph augmented with 1500 synthetic camouflaged-archetype fraud nodes (TabDDPM trained on ONLY the GMM-identified camouflaged sub-population, `diffusion.archetype: camouflaged`, template-attached to preserve real local structure).
  | | F1 | hard-core recovery | ppr | AUC | AUC_hc |
  |---|---|---|---|---|---|
  | baseline (no augmentation) | 0.6836 | 15.2% | 0.080 | 0.8767 | 0.7771 |
  | + camouflaged diffusion augmentation | 0.6752 | 13.9% | 0.081 | 0.8662 | 0.7559 |
  | diff / p / wins | -0.0084 / p=0.81 / n=5 | -1.3pp / p=0.625 / 1-5 wins | flat | -0.0105 / p=0.44 / 1-5 wins | -0.0212 / p=0.31 / 1-5 wins |
- All four metrics point the same (non-positive) direction, though none individually reach significance at n=5. Critically, **ppr is essentially unchanged (0.080 vs 0.081)** -- ruling out a threshold-shift explanation; this is a genuine (if modest) lack of benefit, not an artifact.
- **This extends, rather than contradicts, this project's established pattern**: generic (unconditional) diffusion augmentation was already found not to stack additively with any other intervention tried, across 4+ independent instances (CLAUDE.md). This was a genuinely different, more targeted test (synthetic examples specifically resembling the rare camouflaged sub-population, not generic fraud) motivated by the hypothesis that the model might simply be data-starved for this specific population's structure -- and it still doesn't help. The hypothesis that "more data for the hard population would sharpen the model's read on it" is not supported.
- Decision: closing this direction. Given the consistent (if not individually significant) negative-to-flat direction across every metric, and that this matches an already well-established pattern in this project, a larger-n confirmation is not a good use of the heavier per-seed compute cost (full diffusion training + augmentation + GNN training per seed, vs a plain metric-learning run). Not pursuing further diffusion-augmentation variants (different n_synthetic, different attach_to, MIDI-style sequence diffusion) without a new, different hypothesis for why they'd behave differently from what's already been tried 5 times now (4 generic + 1 targeted).
- Broader read: this reinforces Run 156's finding (the signal is a mean-shift, not a shape difference) from a different angle -- if the issue were "not enough examples to characterize the camouflaged population's shape," more synthetic examples should help; it doesn't, consistent with there not being much class-specific SHAPE information to characterize in the first place. The bottleneck really does look like a fundamental, small mean-separation relative to within-class spread, not a data-scarcity problem fixable by augmentation.

## [2026-07-23] Run 158 — Elliptic: RBF-SVM gives a real gain in the TRAINED EMBEDDING (not the 2D PCA); critical clarification + scatter visualization resolves a representation mix-up
- Tested RBF-SVM (honest, train-only fit) on the SAME 128-dim camo-weighted embedding used for Run 134's LDA and Run 156's QDA tests:
  | Method | never-recovered-vs-legit AUC |
  |---|---|
  | LDA (linear) | 0.664 |
  | QDA (quadratic) | 0.675 |
  | **RBF-SVM (C=10, gamma=0.01)** | **0.710** |
  A genuine, if modest, gain -- most of the C/gamma grid (0.68-0.71) sits above both LDA and QDA, not just one cherry-picked point, though the specific "best" setting was selected by scanning against test AUC directly (a mild form of hyperparameter leakage -- the honestly-cross-validated number is probably somewhat lower, still likely above 0.664/0.675 given the whole grid clears that bar).
- **Important clarification/correction**: this RBF-SVM test was on the 128-dim TRAINED EMBEDDING, not the 2D raw-feature PCA space used throughout the streamline visualizations (Runs 154/155/157-adjacent). These are different representations and shouldn't be conflated.
- **Directly tested the actual question raised**: does a nonlinear (RBF) surface in the 2D PCA coordinates specifically get close to a clean separation, given how visually separated the centroid trajectories look? No -- honest RBF-SVM in the 2D PCA space gives AUC~0.53-0.55 for never-recovered-vs-legit (barely above chance) and only ~0.65-0.79 even for the EASY caught-both population (vs the embedding's 0.997) -- confirming the 2D projection (~18% of raw feature variance) has lost most of the useful separating signal; no amount of nonlinear flexibility recovers information that projection already discarded.
- **Root cause of the visual disagreement, resolved with a new panel**: the streamline plots show CENTROIDS -- averages over hundreds/thousands of individual points per step. Averaging cancels out individual-point overlap by construction; a cleanly-separated AVERAGE path says nothing about whether the actual points around that average overlap. Added a raw individual-point scatter (every labeled illicit point, a 4000-point legit sample, same PCA-2 coordinates, no smoothing/averaging at all) directly to the visualization so the actual overlap is visible, not just implied by the AUC numbers.
- Visualization updated (same URL) with the small-sample (n=10/step) ensemble and the raw scatter panel: https://claude.ai/code/artifact/f3465139-1448-43bb-88d3-e3af71af3f33
- Decision: RBF-SVM in the trained embedding is a genuinely promising, not-yet-fully-confirmed lever (needs proper CV-based hyperparameter selection + actual hard-core recovery testing with real thresholds, not just AUC, before being trusted) -- worth pursuing further. The 2D-PCA nonlinear-surface question is now closed with a definitive negative, and the centroid-vs-individual-point distinction is now directly visualizable, not just argued.

## [2026-07-23] Run 159 — CORRECTION: RBF-SVM's apparent edge over camo-weighted evaporates across seeds (does NOT replicate)
- Direct follow-up to Run 158's RBF-SVM finding, which looked strong and properly cross-validated on seed=42's embeddings (matched-FPR sweep showed a ~2x recovery advantage over plain centroid-distance at 20-30% FPR: e.g. 40.1% vs 22.6% at 20% FPR).
- Repeated the EXACT same matched-FPR comparison (same C=10/gamma=0.01, same methodology) on the 5 OTHER already-cached seeds' embeddings (seeds 0-4, no new dispatch needed).
  | seed | camo@20%FPR | SVM@20%FPR | camo@30%FPR | SVM@30%FPR |
  |---|---|---|---|---|
  | 0 | 21.2% | 21.9% | 35.8% | 35.8% |
  | 1 | 21.2% | 16.8% | 39.4% | 38.0% |
  | 2 | 8.8% | 6.6% | 23.4% | 25.5% |
  | 3 | 5.8% | 8.8% | 32.1% | 27.0% |
  | 4 | 14.6% | 21.2% | 51.1% | 43.8% |
  | mean | 14.3% | 15.0% (3/5 wins) | 36.4% | 34.0% (1/5 wins) |
- **The dramatic seed=42 advantage completely evaporates.** Mean difference at 20% FPR shrinks to a negligible +0.7pp (not the ~18pp seen on seed=42), and at 30% FPR the SVM is actually slightly WORSE on average (1/5 wins). Seed=42 was a favorable draw, not a real, general effect.
- **This is the same "looked strong on one instance, evaporated under multi-seed testing" pattern this project has hit repeatedly** (DropEdge, degree-aware, gate_aux_weight, GraphSAGECamoAgg) -- now confirmed for RBF-SVM scoring too, despite proper CV-based hyperparameter selection (which addressed a DIFFERENT concern, test-set peeking, not the separate concern of single-seed generalization). Proper CV does not substitute for multi-seed confirmation -- they guard against different failure modes.
- Decision: RBF-SVM does not offer a real, reliable improvement over plain centroid-distance scoring for this hard core. Retracting Run 158's implied promotion of this lever. Closing this thread -- the classifier-family comparison arc (LDA -> QDA -> RBF-SVM) ends with: none of them offer a robust improvement over the simplest (centroid-distance) scoring once properly multi-seed tested, reinforcing that the bottleneck is the underlying signal strength, not the choice of classifier on top of it.

## [2026-07-23] Run 160 — Elliptic: nearest-of-K-legit-sub-centroids also fails (6-seed test), closes the classifier-comparison arc
- Tested whether legit's known heterogeneity (small-sample streamlines showed much more spread/turbulence for legit than illicit, Run 155) means a SINGLE legit centroid is a poor summary -- scored by distance to the NEAREST of K=8 legit sub-centroids (k-means, train-only/honest) instead of the single legit mean, keeping fraud as a single centroid (matching the established 2-component archetype finding doesn't obviously extend to a benefit here either). Tested across all 6 available seeds' embeddings from the start this time (42 + 0-4), avoiding Run 158/159's mistake of checking one seed first.
  | seed | camo@20%FPR | nearestK@20%FPR | camo@30%FPR | nearestK@30%FPR |
  |---|---|---|---|---|
  | 0 | 21.2% | 19.7% | 35.8% | 38.7% |
  | 1 | 21.2% | 16.8% | 39.4% | 40.1% |
  | 2 | 8.8% | 8.8% | 23.4% | 23.4% |
  | 3 | 5.8% | 2.9% | 32.1% | 25.5% |
  | 4 | 14.6% | 13.1% | 51.1% | 33.6% |
  | 42 | 22.6% | 21.2% | 35.0% | 40.9% |
  | mean | 15.7% | 13.7% (0/6 wins) | 36.1% | 33.7% (3/6 wins) |
- **Clean negative at the primary (20% FPR) operating point**: 0/6 seeds favor nearest-of-K-centroids, mean recovery actually lower (13.7% vs 15.7%). At 30% FPR, a wash (3/6 wins, slightly lower mean). No benefit from modeling legit as multi-modal via k-means sub-centroids.
- **This closes the whole classifier-family comparison arc for this hard core**: LDA (Run 134, honest AUC 0.664) -> QDA (Run 156, 0.675, negligible gain -- confirms mean-shift not shape-difference) -> RBF-SVM (Run 158, looked like 0.710 + a 2x matched-FPR recovery gain, but Run 159 found this evaporates across seeds -- a favorable-draw mirage) -> nearest-of-K-legit-centroids (this run, a clean negative from the start). None of these decision-rule sophistication increases provide a robust improvement over the simplest single-centroid-distance scoring, once properly multi-seed tested. The bottleneck is the embedding's OWN separating power for this hard core, not the choice of classifier built on top of it -- consistent with, and now exhaustively confirmed alongside, Runs 131-157's structural/dynamical/augmentation-based findings.
- Decision: closing the "smarter classifier on top of the existing embedding" direction entirely. Combined with Runs 131-159, this hard core has now been probed from every angle this investigation has generated (structural isolation, neighbor camouflage, temporal drift/trend/derivative, physics analogies, targeted diffusion augmentation, and now linear/quadratic/kernel/multi-centroid classifiers) without finding anything that reliably moves the needle beyond camo-weighted's original ~18-20% recovery. This is the point to write up the diagnostic as its own contribution rather than continue searching for a fix.

## [2026-07-23] Run 161 — Elliptic: per-seed embedding trajectory visualization (six small panels)
- Built a small-multiples visualization directly answering "why do seeds matter if the data is fixed": one panel per seed (0-4, 42), each showing that seed's OWN trained 128-dim camo-weighted embedding's test-period (steps 42-49) illicit centroid trajectory, freshly PCA-projected to 2D per seed (axes not aligned across panels -- only shape/structure is comparable, not absolute position/rotation).
- Directly visualizes the point established in discussion: the DATA (graph, features, train/test split) is identical across all six panels -- what differs is the TRAINED MODEL (random init, triplet sampling order, dropout, early-stopping checkpoint selection, per `training/train_gnn.py`'s `set_seed()` and `evaluation/metric_learning.py`'s seeded triplet generator), which is why the same method's hard-core recovery ranged from 5.8% to 51.1% across these same six seeds (Run 160's table).
- Published: https://claude.ai/code/artifact/2a8ae765-a302-490f-b000-fd9ece7013c2

## [2026-07-23] Run 162 — MAJOR CORRECTION: at a realistic anti-fraud FPR budget (<=1%), hard-core recovery is 0% across all seeds; seed=4's 51.1% does not reproduce
- **Seed=4 non-reproducibility, tested directly**: retrained with the identical config (seed=4) as a fresh dispatch. Recovery at 30% FPR: 38.0%, NOT the original dispatch's 51.1%. Confirms (per Run 129's established GPU/scatter non-determinism finding) that even "the same seed" trains a genuinely different model on each dispatch -- the 51.1% was noise from one specific stochastic training run, not a discoverable or stabilizable property of "seed=4" as a config value. There is nothing to fix here; it was never a real, controllable difference.
- **Far more important: recomputed hard-core recovery at the FPR budget that actually matters for anti-fraud deployment (<=1%), across all 6 available seeds.**
  | seed | recovery@0.5%FPR | recovery@1%FPR | recovery@2%FPR | caught_both recovery@1%FPR |
  |---|---|---|---|---|
  | 42 | 0.0% | 0.0% | 0.0% | 99.3% |
  | 0 | 0.0% | 0.0% | 0.0% | 96.5% |
  | 1 | 0.0% | 0.0% | 0.0% | 98.6% |
  | 2 | 0.0% | 0.0% | 0.0% | 94.4% |
  | 3 | 0.0% | 0.0% | 0.0% | 91.5% |
  | 4 | 0.0% | 0.0% | 0.0% | 99.3% |
- **Hard-core recovery is exactly 0.0% at 1% FPR in every single seed, with zero exceptions.** Meanwhile the EASY population (caught_both -- fraud already caught by both RF and GNN) achieves 91.5-99.3% recovery at that same realistic threshold -- the existing classifiers already capture essentially everything achievable at 1% FPR.
- **This substantially reframes every "recovery" number reported in Runs 78-161.** Camo-weighted's headline "18-20% hard-core recovery, zero losses, confirmed across 5-10 seeds" (Run 84 onward) was computed at its DEFAULT 0.5 threshold, which corresponds to roughly 9-12% predicted-positive-rate -- itself already too loose for realistic anti-fraud deployment (flagging 1-in-10 legitimate transactions is not commercially viable). Every subsequent "improvement" tested (margin20, dual_view_encoded, RBF-SVM, etc.) was evaluated at even LOOSER operating points (15-30% FPR) to show any recovery at all. At the FPR an actual production anti-fraud system would need (<=1%), NONE of it survives -- this is not a modeling gap, it is a structural fact about how deeply this specific hard-core population overlaps with legit in feature space.
- **Directly answers "could an auxiliary MLP find a better weighting"**: no. This is the same argument that closed Run 159's RBF-SVM thread, compounded: (a) RBF-SVM is already at least as flexible as a small MLP in practice and its apparent advantage evaporated across seeds -- an MLP would face the identical generalization risk, worse given only 213 hard-core examples (137 never-recovered) to train/validate on; (b) more fundamentally, zero recoverable signal exists for ANY method at the realistic 1% FPR threshold -- this is a statement about where the two classes actually sit in feature space, not an unsolved function-approximation problem a learned reweighting could fix.
- **Revised overall conclusion for the write-up**: camo-weighted's hard-core recovery is a real, multi-seed-confirmed effect ON ITS OWN (technically true) but has NO practical value at a realistic anti-fraud operating point -- it should be reported as a research finding about embedding-space structure (there IS real, if modest, information about the camouflaged sub-population recoverable from the model, more than a null result would predict), not as an operational improvement to fraud detection. The honest headline for Elliptic is: RF's 0.789 F1 remains the practically deployable result; the entire hard-core investigation (Runs 78-162) is a rigorous characterization of a fundamental detection limit, not a fix for it.

## [2026-07-23] Run 163 — camo_weighted_mlp: learned replacement for camo_weighted's hand-designed weighting formula (implementation + dispatch)
- **Context**: Run 162 closed the "auxiliary MLP" question by analogy to RBF-SVM's evaporation (Run 159) and the 0%-recovery-at-1%-FPR finding, without actually building it. Explicit instruction this session to build and test it anyway rather than reason it away in advance ("давай softmax от выхода этого mlp, на вход побольше признаков - давай просто сделаем") -- implemented in `evaluation/metric_learning.py` as a new `camo_weighted_mlp` mining mode.
- **Mechanism**: `_CamoWeightMLP` (Linear -> ReLU -> Linear(1), hidden=32) takes a RICHER per-anchor input than the hand-designed formula's single scalar `dist_to_legit` -- the anchor's own 128-dim embedding, concatenated with `dist_to_legit` and `dist_to_fraud` (all DETACHED, so the main encoder can't game its own embedding position to win a favorable weight). Output softmax-normalized across the batch exactly like every other camo_weighted variant (mean weight ~= 1).
- **Known, central risk (identified before implementation, not resolved by it, only mitigated)**: `loss = mean(per_triplet_loss * weight)` gives any freely-trained weighting function a direct incentive to DOWN-weight high-loss (camouflaged) triplets and UP-weight low-loss (easy) ones -- the opposite of camo_weighted's domain-motivated prior, and the textbook degenerate-collapse failure mode of naive (non-bi-level) learned example reweighting (Ren et al. 2018). Three mitigations layered on top, all added during this session's design discussion rather than upfront:
  1. **40-epoch warmup**: encoder trains on plain unweighted triplet loss while the MLP trains quietly in the background (mirrors this codebase's own established convention for diffusion's adversarial/spectral-matching auxiliary losses -- gated to start partway through training, not epoch 1).
  2. **EMA-smoothed encoder weighting**: once warmup ends, `ema_weight_mlp` is SEEDED from the live MLP's just-warmed-up state (not carried from random init through the warmup period) and from then on is what actually reweights the ENCODER's gradient (decay=0.9/epoch) -- the live MLP itself still trains every epoch via a separate loss term, but the encoder isn't chasing a new, noisy scoring function every single epoch (the same non-stationarity concern already on record for why Run 79's hard-batch mining destabilized training).
  3. **Anchor regularizer**: the live MLP's own training term is `mean(per_triplet_loss.detach() * live_weight) + 0.1 * mean((live_weight - hand_designed_weight)^2)` -- pulling the learned weight toward camo_weighted's existing `softmax(-dist_to_legit)` formula makes the known degenerate (wrong-direction) solution costly instead of free, while still leaving room to use the richer features where informative.
- **Diagnostic wired in, not just the headline metric**: every 25 epochs, logs `weight`-vs-`dist_to_legit` correlation for both the live and (once active) EMA weight. Negative correlation = didn't collapse (still up-weights camouflaged anchors, like the hand-designed prior); positive = collapsed toward up-weighting easy/obvious fraud instead. This is checked BEFORE trusting any recovery-number improvement, not after.
- **Verified locally** (unit-level, not a full training run): the loss function's warmup/EMA-seeding/EMA-update/anchor-regularizer control flow was directly simulated end-to-end (forward + backward + optimizer.step() + EMA update across a short synthetic epoch loop) and runs without error, with gradient flowing into the MLP's parameters but NOT into the main embeddings through the weighting branch (only through the ordinary per_triplet_loss term, confirmed by inspecting which rows of a synthetic embeddings tensor accumulate gradient).
- **Config**: `configs/elliptic_metric_learning_camo_weighted_mlp.yaml`, plain `graphsage_diff` architecture (not `dual_view_encoded`) to isolate the loss mechanism's own effect first, same precedent as camo_weighted_temp/dual/margin_scale's original tests.
- Wired end-to-end: `run()`'s signature/optimizer construction, the full-batch mining elif chain, the ValueError message, and `serverless/handler.py`'s kwarg passing (`camo_mlp_hidden_dim`, `camo_mlp_ema_decay`, `camo_mlp_warmup_epochs`, `camo_mlp_anchor_weight`, all defaulted so every existing config/mining mode is unaffected). Mini-batch (IEEE-CIS) mode does NOT support this mining mode yet -- Elliptic-only for now, matching where the whole hard-core investigation lives.
- Significance/multi-seed check deferred until after a first single-seed sanity run confirms the mechanism trains at all and doesn't collapse in the wrong direction -- per established discipline (RBF-SVM's single-seed mirage, Run 158/159), a multi-seed comparison against camo_weighted's existing 18-20% baseline recovery is required before drawing any conclusion, not optional polish.
- Next: dispatch to RunPod (worker recycle needed first -- new mining-mode code, not just a config value, hit the same stale-in-memory-import issue for camo_weighted_temp per the infra note above), read back the collapse diagnostic and recovery numbers, THEN decide whether a multi-seed run is even worth running.

## [2026-07-23] Run 164 — camo_weighted_mlp collapse confirmed then fixed; centroid-separation loss; subcentroid feature; ablation campaigns + wandb grid sweep dispatched
- **Collapse CONFIRMED empirically, exactly as predicted**: Run 163's first real dispatch (anchor_weight_coeff=0.1, single seed, plain graphsage_diff) showed `live_corr`/`ema_corr` (learned-weight-vs-dist_to_legit correlation) jump to **+0.98 by epoch 25 and stay at +0.91 to +0.93** through epoch 250 -- POSITIVE correlation means the MLP was up-weighting easy/obvious fraud, the exact opposite of camo_weighted's domain-motivated direction. The anchor regularizer at weight 0.1 was roughly an order of magnitude too weak relative to the main per_triplet_loss-driven term to prevent it (test_f1=0.740/auc=0.866 that run, but on a collapsed mechanism -- not a result to trust).
- **Fix verified**: bumped `camo_mlp_anchor_weight` 0.1 -> 2.0. Re-dispatched (single seed, otherwise identical): correlation flipped to **-0.95 by epoch 50, settling around -0.99 to -0.998** through epoch 300 -- no longer collapsed, now strongly (arguably TOO strongly) tracking the hand-designed formula's own direction. test_f1=0.726/auc=0.865. This directly demonstrates the anchor_weight_coeff tradeoff flagged in the code's own docstring: too weak -> collapses to the wrong direction; too strong -> the MLP barely deviates from the fixed formula it was meant to generalize beyond. A middle value is needed to actually test whether the richer input features buy anything -- this is what the wandb grid sweep (below) is for.
- **Also added, requested mid-session**: (1) `_centroid_separation_loss` -- explicit, uncapped push for the fraud/legit centroids to separate further, motivated by the margin-based triplet loss saturating (F.relu(...)=0) once d_neg-d_pos exceeds margin=1.0, well short of unit-normalized embeddings' geometric max squared distance of 4; wired as a generic `separation_weight` term composable with any mining mode. (2) `_nearest_subcentroid_distance` -- KMeans over CURRENT legit embeddings (refit live, same spirit as `_fraud_prototypes`' GMM), giving the MLP a richer/more local "how camouflaged" signal than the single aggregate legit centroid; gated by `camo_mlp_subcentroid_k` (0=off, 15 in the active configs). Both unit-tested locally before dispatch.
- **Three-way ablation dispatched** (graphsage_diff, plain -- not dual_view_encoded, to isolate the loss mechanism) to decompose which piece (if any) actually helps, since bundling MLP + separation into one config would leave any effect uninterpretable:
  | Campaign | Mechanism | Seeds | Status |
  |---|---|---|---|
  | `elliptic_metric_learning_camo_weighted_separation_only` | plain camo_weighted + separation_weight=0.1 (NO MLP) | 5 | **DONE**: mean f1_macro=0.6600, std=0.0136 |
  | `elliptic_metric_learning_camo_weighted_mlp_nosep` | camo_weighted_mlp (anchor=2.0, subcentroid_k=15), separation_weight=0.0 | 5 | dispatched, in progress |
  | `elliptic_metric_learning_camo_weighted_mlp` | camo_weighted_mlp (anchor=2.0, subcentroid_k=15) + separation_weight=0.1 | 15 | dispatched, in progress |
- **wandb grid sweep dispatched** (sweep id `q75o7fly`, project `fraud-diffusion`, 3 parallel agent processes against the 3-worker endpoint): grid over `metric_learning.camo_mlp_anchor_weight` in {0.3, 0.5, 1.0, 2.0} x `metric_learning.separation_weight` in {0.0, 0.1, 0.3} (12 points, single seed=42 each) -- directly probes where the anchor-weight tradeoff stops collapsing but still leaves the MLP room to learn something the hand-designed formula can't express.
- **Important caveat, not yet resolved**: none of these campaigns' standard `test` metrics (f1_macro/auc_roc/etc.) are the hard-core recovery-at-realistic-FPR numbers Run 162 established as the actually meaningful metric for this investigation -- they're the ordinary classification metrics compute_metrics() reports. A real verdict on "does camo_weighted_mlp help the hard core" still requires the same matched-FPR-recovery analysis as Runs 158-162 (needs return_embeddings=True + the fixed hard_core_mask), not just comparing F1/AUC across configs. Flagging this now so it isn't skipped once the campaigns above finish and F1 numbers look tempting to just compare directly.
- Next: read back all four campaigns + the sweep once complete, THEN decide whether any variant is worth a proper hard-core-recovery check.

## [2026-07-23] Run 165 — Ablation campaigns + wandb grid sweep complete: separation loss looks like the active ingredient, MLP's own contribution unclear
- **Three-way ablation, all complete**:
  | Config | Mechanism | n | Mean f1_macro | Std |
  |---|---|---|---|---|
  | `elliptic_metric_learning_camo_weighted_separation_only` | separation loss alone (plain camo_weighted, no MLP), separation_weight=0.1 | 5 | 0.6600 | 0.0136 |
  | `elliptic_metric_learning_camo_weighted_mlp_nosep` | MLP alone (anchor_weight=2.0, subcentroid_k=15), separation_weight=0.0 | 5 | 0.6241 | 0.0254 |
  | `elliptic_metric_learning_camo_weighted_mlp` (full) | MLP (anchor_weight=2.0) + separation_weight=0.1 | 15 | **0.6810** | 0.0145 |
  Full mechanism > separation-alone > MLP-alone. The MLP-alone ablation is actually the WORST of the three (even below separation-alone), suggesting that at anchor_weight=2.0 (needed to avoid Run 164's collapse), the MLP's own learned weighting isn't adding value by itself -- whatever gain the full mechanism shows over separation-alone (+0.021, modest relative to the stds) is a small, not yet statistically confirmed edge, and could plausibly be more about the separation loss than the MLP.
- **wandb grid sweep (`q75o7fly`), all 12 points complete** -- `camo_mlp_anchor_weight` in {0.3, 0.5, 1.0, 2.0} x `separation_weight` in {0.0, 0.1, 0.3}, single seed=42 each:
  | anchor_weight | sep=0 | sep=0.1 | sep=0.3 |
  |---|---|---|---|
  | 0.3 | 0.671 | 0.753 | 0.752 |
  | 0.5 | 0.659 | 0.702 | 0.734 |
  | 1.0 | 0.698 | 0.713 | 0.729 |
  | 2.0 | 0.693 | 0.732 | 0.723 |
  Every nonzero-separation cell beats its own row's separation=0 baseline -- consistent across all 4 anchor-weight settings, a real pattern rather than one lucky cell. Best single point: anchor=0.3/sep=0.1 (0.753). **But this entire grid is seed=42 only** -- this session has repeatedly found seed=42 runs favorable in single-seed checks (the RBF-SVM mirage, the original seed=4 51.1%-that-didn't-reproduce, etc.), so this table identifies WHERE to point a proper multi-seed check, not a result to act on directly.
- **What this does and doesn't establish yet**:
  1. Reasonably solid (n=15 vs n=5, matches directionally across both ablations): separation_weight helps, at least at the anchor_weight=2.0 setting used throughout the ablation campaigns.
  2. NOT yet established: whether a looser anchor_weight (0.3-1.0, letting the MLP deviate further from the hand-designed formula) is genuinely better than 2.0, or just a seed=42 favorable draw -- the sweep's anchor=0.3/sep=0.1 single point (0.753) is nominally higher than the anchor=2.0/sep=0.1 config's already-established 15-seed mean (0.681), but a single point at a historically-favorable seed is exactly the pattern (Run 158/159) that evaporated on proper multi-seed testing before.
  3. Still entirely open, per Run 164's caveat: none of this is the hard-core recovery-at-realistic-FPR metric (Run 162) -- only ordinary classification F1/AUC. A real verdict requires that analysis regardless of which config wins this round.
- Next: multi-seed (5+) confirmation of the leading sweep candidates (anchor=0.3/sep=0.1, anchor=0.5/sep=0.3) before trusting the grid; if either survives, THEN run the hard-core-recovery-at-1%-FPR check against Run 162's fixed masks before calling this a real result.

## [2026-07-23] Run 166 — Multi-seed confirmation: looser anchor regularization (0.3-0.5) + higher separation_weight (0.3) is a genuine, stable improvement over the collapse-preventing anchor=2.0 config
- **Confirms Run 165's sweep lead was NOT a seed=42 mirage.** Dispatched both leading grid points at 6 seeds each (2 seeds lost per campaign to a transient network/DNS blip mid-run, not a training failure -- `infra.multi_seed`'s per-seed exception handling caught these cleanly, everything else completed normally):
  | Config | n | Mean f1_macro | Std |
  |---|---|---|---|
  | `elliptic_metric_learning_camo_weighted_mlp` (anchor=2.0, sep=0.1) -- Run 164/165's "fixed" config | 15 | 0.6810 | 0.0145 |
  | `elliptic_metric_learning_camo_weighted_mlp_anchor03` (anchor=0.3, sep=0.1) | 4 | 0.7245 | 0.0121 |
  | `elliptic_metric_learning_camo_weighted_mlp_anchor05sep03` (anchor=0.5, sep=0.3) | 5 | **0.7279** | **0.0013** |
- **Real effect, not noise**: both looser-anchor configs beat the anchor=2.0 baseline by a wide margin (+0.043 to +0.047 mean F1) with NON-OVERLAPPING seed distributions (anchor=2.0's own std alone is 0.0145; the gap is 3x that). The anchor=0.5/sep=0.3 point's std=0.0013 is the tightest variance seen anywhere in this entire investigation (contrast: every other multi-seed result this session, camo_weighted's own 19.1%+-6.6% recovery included, has shown substantial seed-to-seed spread) -- worth flagging as unusual and double-checking (are all seeds actually converging to a near-identical solution, or is there a subtle bug making the runs less independent than intended?) rather than just celebrating it.
- **Practical implication**: anchor_weight=2.0 (chosen purely to kill Run 164's collapse) was overly conservative -- it does prevent the WRONG-direction collapse, but at the cost of pinning the MLP so close to the hand-designed formula that it can't express anything useful beyond it. anchor=0.3-0.5 apparently still avoids collapse (not yet re-verified via the correlation diagnostic at these specific settings -- should confirm before fully trusting this) while leaving enough freedom for the richer input features (embedding, subcentroid distance) to matter.
- **Still pending before calling this a real result for the original investigation**: (1) re-check the collapse diagnostic (live/ema weight-vs-dist_to_legit correlation) at anchor=0.3/0.5 specifically -- these campaigns' logs weren't inspected for this, only the F1 numbers; (2) the hard-core recovery-at-1%-FPR check (Run 162) against the fixed masks -- still nothing here addresses that metric directly; (3) retry the 3 network-dropped seeds for full n=6 confirmation.
- Next: promote `anchor=0.5, sep=0.3` as the new default in `elliptic_metric_learning_camo_weighted_mlp.yaml` once the collapse diagnostic is confirmed clean at that setting.

## [2026-07-23] Run 167 — CORRECTION: anchor=0.5/sep=0.3's F1 gain is real but the mechanism is collapsed, not fixed
- **anchor=0.3 retry seeds landed**: full n=6 now for `elliptic_metric_learning_camo_weighted_mlp_anchor03` (0.3/0.1) -- values [0.7181, 0.7403, 0.7270, 0.7127, 0.7343, 0.7204], mean=0.7255. Consistent with Run 166's partial-n read.
- **Collapse diagnostic checked at anchor=0.5/sep=0.3 (the config just promoted in Run 166) -- it IS collapsed**:
  ```
  epoch 25:  live_corr=+0.979
  epoch 50:  live_corr=+0.986  ema_corr=+0.984
  epoch 100: live_corr=+0.816  ema_corr=+0.868
  epoch 150: live_corr=+0.807  ema_corr=+0.812
  epoch 200: live_corr=+0.800  ema_corr=+0.802
  ```
  POSITIVE correlation throughout, same wrong direction as the very first anchor=0.1 collapse (Run 163/164) -- the MLP is up-weighting EASY/obvious fraud, not camouflaged fraud. anchor_weight=0.5 was not actually strong enough to prevent the collapse; it just happened to also coincide with a real F1 improvement, for an unrelated reason.
- **What this means**: the +0.047 mean-F1 gain and the unusually tight variance (Run 166) are real and reproducible, but NOT evidence the learned-weighting mechanism works as intended. The far more likely explanation, consistent with the original 3-way ablation (Run 165: `mlp_nosep` alone was the WORST of the three variants, `separation_only` alone was solid, full mechanism best) -- the separation_weight term is doing essentially all the useful work, and the collapsed MLP is riding along, behaving like a fixed (if degenerate) easy-example emphasis that happens to be very STABLE across seeds (up-weighting "easy" is a far more consistent signal seed-to-seed than up-weighting "camouflaged" ever was, which plausibly explains the tight std too).
- **Practical risk, not yet checked**: a mechanism that up-weights easy fraud instead of camouflaged fraud could plausibly HURT hard-core recovery specifically, even while helping aggregate F1 (aggregate F1 is dominated by the easy majority of fraud, which the hard core is explicitly NOT part of). Promoting this config in Run 166 without first checking the diagnostic was premature -- corrected the config's own comment to flag this explicitly.
- **Revised recommendation**: do not treat `anchor=0.5/sep=0.3`'s F1 win as "camo_weighted_mlp fixed" -- it's a real, separately-useful finding (separation loss helps, confirmed twice now) bundled with an MLP mechanism that has NOT been shown to work as designed at any anchor_weight tried so far (0.1: collapsed; 0.5: collapsed; 2.0: didn't collapse but added no measurable value over separation-alone per Run 165's ablation). The honest state of the "learned camo-weighting" idea specifically is: still unconfirmed, arguably trending toward "doesn't work as hoped" every time it's actually been checked properly.
- Next: (1) run the hard-core recovery-at-1%-FPR check (Run 162's methodology) on BOTH the anchor=0.5/sep=0.3 config AND separation_only-no-MLP, to see whether the collapsed MLP is actively hurting hard-core recovery relative to separation-alone, which would be the clearest possible demonstration of why checking the diagnostic (not just the headline metric) matters; (2) hold off on the planned MLP-capacity/separation-weight follow-up sweep until this is resolved, since scaling up a collapsed mechanism isn't obviously useful.

## [2026-07-23] Run 168 — Adaptive anchor-weight controller successfully prevents collapse without pinning the MLP at a high constant
- **Directly addresses Run 167's finding**: neither anchor_weight=0.1 nor 0.5 (fixed constants) prevented collapse; only 2.0 did, at the cost of the MLP essentially just reproducing the hand-designed formula. Implemented `_adaptive_anchor_weight`: a one-epoch-lagged feedback controller (not a naively end-to-end learned gate, which would inherit the same degenerate incentive) -- grows the anchor coefficient automatically when live weight-vs-dist_to_legit correlation drifts positive (toward collapse), relaxes it back toward a low floor (0.3) once safely negative.
- **First dispatch (floor=0.3, single seed=42) -- worked as designed**:
  | epoch | anchor_weight | live_corr |
  |---|---|---|
  | 1 | 0.345 | +0.211 |
  | 25 | 4.529 | +0.970 (collapsing) |
  | 50 | 7.238 | -0.988 (corrected) |
  | 75 | 4.368 | -0.978 |
  | 100 | 2.636 | -0.981 |
  | 125 | 1.591 | -0.987 |
  | 150 | 0.960 | -0.987 |
  | 175 | 0.579 | -0.981 |
  The controller detected the drift early, spiked the anchor coefficient hard enough to flip the correlation negative, then relaxed it steadily back down toward the 0.3 floor -- and the correlation STAYED strongly negative (-0.98) throughout the relaxation, rather than snapping back to collapse the moment pressure eased. This is evidence the MLP settled into a genuinely stable, correctly-signed regime rather than just being held there by constant brute-force regularization -- the average anchor weight over this run is far below the fixed 2.0 that was previously required to avoid collapse.
- test_f1_macro=0.7179, auc_roc=0.8602 (single seed) -- consistent with the strong range seen in Run 166/167's runs, but on a mechanism that is ACTUALLY NOT collapsed this time (unlike anchor=0.5/sep=0.3's F1 win, which Run 167 showed WAS collapsed).
- **Still open**: (1) multi-seed confirmation that the controller reliably avoids collapse (not just this one seed/one lucky trajectory); (2) whether the low, non-collapsed anchor weight actually lets the MLP express something beyond the hand-designed formula, or whether corr=-0.98 (very close to -1.0) means it's still basically just reproducing the formula's ranking even at low regularization pressure; (3) the hard-core recovery-at-1%-FPR check, still the real bar, still not done for ANY variant.
- Next: multi-seed confirmation of the adaptive config, then finally the hard-core recovery check comparing (a) separation_only (no MLP), (b) anchor=2.0 fixed (not collapsed, no measured gain per Run 165's ablation), (c) adaptive (not collapsed, unclear if it adds anything beyond the formula) -- this three-way comparison is what actually answers whether the learned-MLP idea has ever added real value, independent of the separation loss.

## [2026-07-23] Run 169 — First-ever nonzero hard-core recovery at realistic FPR (small, separation_weight-driven, not clearly about MLP collapse)
- **The actual bar, finally checked**: multi-seed (n=5) recovery-at-fixed-FPR against Run 162's fixed hard_core_mask/caught_both_mask/y_test, for all four configs compared this session:
  | Config | Collapsed? | hc@0.5%FPR | hc@1%FPR | hc@2%FPR | caught_both@1%FPR | F1 |
  |---|---|---|---|---|---|---|
  | `separation_only` (no MLP) | n/a | 0.38% | 0.66% | 2.35% | 96.3% | 0.676 |
  | `anchor=2.0/sep=0.1` (full mechanism, not collapsed) | No | 0.19% | 0.56% | 1.69% | 96.1% | 0.670 |
  | `anchor=0.5/sep=0.3` | **Yes** | 0.19% | 1.22% | 2.82% | 98.9% | 0.723 |
  | `adaptive` floor=0.3/sep=0.3 | No | 0.38% | 1.13% | 3.19% | 98.6% | 0.710 |
- **First qualitative change from Run 162's finding**: the original camo_weighted mechanism gave EXACTLY 0.0% hard-core recovery at 1% FPR across all 6 seeds, with zero exceptions. Every config tested here shows small but consistently NONZERO recovery (0.56-1.22% at 1%FPR, 1.69-3.19% at 2%FPR). Still tiny in absolute terms -- nowhere near the original ~18-20% headline number (which was measured at a much looser ~9-12% operating point) -- but a real, qualitative change from "structurally zero" to "small but present."
- **Collapse status does NOT cleanly predict hard-core recovery**: contrary to Run 167's hypothesis (a collapsed, easy-fraud-up-weighting mechanism might actively hurt hard-core recovery), the collapsed `anchor=0.5` and non-collapsed `adaptive` configs land close together (1.22% vs 1.13% at 1%FPR; 2.82% vs 3.19% at 2%FPR) and both clearly beat the lower-separation_weight configs (`separation_only` at 0.1, `anchor=2.0` at 0.1). This points to **separation_weight magnitude, not MLP collapse status, as the main driver of both the F1 gain and this small hard-core gain** -- pushing the aggregate centroids further apart seems to pull the whole score distribution in a way that helps the hard core a little too, independent of whether the MLP's own weighting mechanism is doing anything sensible.
- **caught_both recovery stays high (96-99%) everywhere**, consistent with Run 162 -- the easy population remains easily caught regardless of which variant is used; none of today's changes trade easy-population recall for hard-core gains, at least at these sample sizes.
- **Revised overall assessment**: today's investigation produced a real, if modest, win -- separation_weight (whether or not paired with a working MLP) measurably moves hard-core recovery off of zero for the first time. This is worth keeping and building on (via the higher-separation_weight follow-up sweep already queued, and the new boundary-ranking loss specifically targeting the decision-boundary region rather than aggregate centroids). It is NOT yet a result that changes the practical bottom line (RF's 0.789 F1, or camo-weighted's own original honest characterization) -- 1-3% recovery of a 213-example hard core is a few extra correctly-flagged transactions, not a solved problem.
- Next: dispatch and multi-seed-test the new `_boundary_ranking_loss` (targets the K closest-to-fraud legit train examples specifically, unlike every centroid/random-triplet-based mechanism tried so far) against this same hard-core-recovery metric -- if boundary-targeting beats aggregate-centroid-targeting on hc@1%/2%FPR specifically (not just F1), that would be the first mechanism in this entire investigation actually designed around the metric that matters.

## [2026-07-23] Run 170 — CORRECTION: Run 169's "0% to 1%" comparison was against a stale baseline, not a same-batch one
- **Dispatched the ORIGINAL plain camo_weighted baseline (no separation_weight, no MLP) fresh, same 5 seeds (0-4), same dispatch conditions as Run 169's four variants** -- specifically to get an apples-to-apples comparison instead of relying on Run 162's older numbers.
- **Result: this fresh baseline is NOT 0% at 1%FPR.** hc@1%FPR: mean=0.94%, std=0.51% (values: 1.41%, 0.94%, 1.41%, 0%, 0.94%) -- only 1 of 5 seeds actually landed at exactly 0%. hc@2%FPR: mean=3.85%, std=1.09%. overall recall@2%FPR: mean=43.6%.
- **This directly contradicts Run 169's framing.** Comparing properly (same-batch baseline vs. same-batch variants):
  | Config | hc@1%FPR | hc@2%FPR |
  |---|---|---|
  | baseline (plain camo_weighted, fresh dispatch) | 0.94% | 3.85% |
  | separation_only | 0.66% | 2.35% |
  | anchor=2.0/sep=0.1 | 0.56% | 1.69% |
  | anchor=0.5/sep=0.3 (collapsed) | 1.22% | 2.82% |
  | adaptive | 1.13% | 3.19% |
  Every variant sits WITHIN NOISE of the freshly-measured baseline (baseline std alone is 0.51 points at 1%FPR and 1.09 at 2%FPR) -- two of the four "improved" variants (separation_only, anchor=2.0) are actually LOWER than baseline, not higher. There is no clear win from anything tested in Runs 163-169 on the metric that actually matters, once compared against a properly matched baseline instead of Run 162's older numbers.
- **Root cause of the error**: this session established (Run 129, and directly re-demonstrated with seed=4's 51.1%->38.0% non-reproducibility) that GPU training is non-deterministic even at the same nominal seed value -- a fresh dispatch of "seeds 0-4" produces genuinely different trained models each time, not a replay of a prior dispatch's models. Comparing a NEW dispatch's numbers against an OLD dispatch's numbers (even nominally "the same seeds") is exactly the kind of comparison this session has repeatedly warned against, and Run 169 did precisely that.
- **What still might be real**: `_boundary_ranking_loss`'s single-seed point (hc@2%FPR=3.76%, f1=0.616) is the one number not yet directly compared against this fresh baseline on the SAME dispatch batch, and its multi-seed confirmation (5 seeds) is currently in flight. This is the only candidate left that hasn't already been shown to be noise-level.
- **Revised overall conclusion**: none of Runs 163-169's mechanisms (learned MLP weighting, centroid separation, adaptive anchor control) have yet demonstrated a real improvement in hard-core recovery at a realistic FPR, once measured against a fairly-matched baseline. The F1/AUC gains ARE real (properly multi-seed confirmed, non-overlapping distributions) but do not translate to the hard-core-specific metric that was the entire point of this investigation. `_boundary_ranking_loss` (not yet properly compared) is the only remaining open question before concluding that this entire "improve camo-weighting" line of work (Runs 163-170) has not moved the practical needle.
- Next: (1) let boundary_only's multi-seed run finish and compare directly against THIS SAME baseline dispatch (already have raw numbers, no stale-comparison risk); (2) if boundary_ranking also lands within baseline noise, close this investigation arc with the honest conclusion that separation/MLP-weighting mechanisms improve aggregate F1 without improving the specific hard-core-recovery goal that motivated the entire session.

## [2026-07-24] Run 171 — _boundary_ranking_loss: the first mechanism outside baseline noise on hard-core recovery
- **Directly compared against Run 170's freshly-dispatched baseline (same 5 seeds, same conditions)** -- the one comparison that matters after Run 170 found every earlier variant (separation_only, anchor=2.0, anchor=0.5, adaptive) sat within baseline noise.
  | Config | hc@1%FPR | hc@2%FPR | f1 |
  |---|---|---|---|
  | baseline (plain camo_weighted, fresh) | mean=0.94%, std=0.51% | mean=3.85%, std=1.09% | mean=0.629 |
  | `boundary_only` (boundary_ranking_weight=0.1, k=20) | **mean=1.69%, std=0.48%** | **mean=4.69%, std=1.75%** | mean=0.676 |
  Full per-seed hc@1%FPR values: [1.41%, 0.94%, 2.35%, 1.88%, 1.88%] (one seed needed a network retry, unrelated to training).
- **This is the first candidate this session that's actually outside baseline noise, not just within it.** The gap (0.75 points at 1%FPR) is roughly 1.5x either distribution's own std, and the direction is consistent across all 5 seeds (every boundary_only seed beats or ties the baseline mean). Still a MODEST effect, not dramatic -- n=5 each side gives limited statistical power, and this is not the kind of "clean, obviously real" separation the strongest confirmed findings elsewhere in this investigation have shown (e.g., camo_weighted's own original 5-seed confirmation, or dual_view_encoded's AUC p=0.049). Should be read as "the most promising lead so far," not "confirmed."
- **Why this one might be different, mechanistically**: `_boundary_ranking_loss` is the only mechanism tried in Runs 163-171 that explicitly targets the K legit TRAIN examples closest to the current decision boundary (highest fraud-likeness score), rather than aggregate class centroids (separation_weight, camo_weighted's own formula) or random triplets. This is closer in spirit to a partial-AUC / recall-at-fixed-FPR-targeted objective than anything else tested, which may explain why it's the first one to show a real (if modest) effect on exactly that metric.
- Next: (1) a proper significance test (Wilcoxon or Mann-Whitney given the small n) on this comparison; (2) sweep boundary_ranking_weight and boundary_k (only one untuned guess -- 0.1/20 -- has been tried); (3) test whether combining boundary_ranking_weight WITH separation_weight (both target different aspects: boundary-specific ranking vs. aggregate push) stacks additively; (4) re-run at a larger n (8-10 seeds) before treating this as a confirmed result, given how many "promising at n=5" leads this session has seen evaporate (RBF-SVM, DropEdge/degree-aware at n=10) or turn out to be noise (Run 170's own correction).

## [2026-07-24] Run 172 — CORRECTION + closing finding: boundary_ranking_loss's apparent edge was an n=5 mirage; NOTHING tried in Runs 163-171 improves hard-core recovery at realistic FPR
- **Extended baseline + boundary_only to seeds 5-9 (n=10 total each), same fixed masks/methodology.** The apparent edge from Run 171 (1.69% vs 0.94% at n=5) completely evaporated:
  | Config | n=10 mean hc@1%FPR | std |
  |---|---|---|
  | baseline | 1.36% | 1.08% |
  | boundary_only | 1.32% | 0.55% |
  Paired Wilcoxon (by seed, n=10): **statistic=21.5, p=0.9414**, wins=6/ties=1/losses=3 -- as close to pure noise as this test produces. This is exactly the small-sample-mirage pattern this session has repeatedly warned about (RBF-SVM, Run 158/159; DropEdge/degree-aware at n=10 evaporating from n=5 promise) -- and it just claimed the one candidate (Run 171) that looked different from the rest.
- **The baseline's own noise level is the real story here**: individual seeds range from 0% to 3.29% recovery with ZERO intervention (std=1.08% on a mean of 1.36%, over a metric whose absolute scale is already tiny -- 213 hard-core examples total). Any real effect would need to be large relative to this spread to be distinguishable at n=10, let alone n=5. This is as much a statistical-power problem (very few hard-core examples means very high-variance recovery estimates) as it is a "the interventions don't work" finding -- worth remembering before dismissing future candidates outright, but doesn't change today's practical conclusion.
- **boundary_ranking_weight x boundary_k sweep (12 points, single seed=42)**: values ranged 0.94%-3.76% at hc@1%FPR with no clear monotonic pattern in either hyperparameter -- entirely consistent with being noise once the baseline's own ~0-3.3% single-seed range is accounted for, not evidence of a real hyperparameter-dependent effect.
- **CLOSING ASSESSMENT for the "improve camo-weighted's hard-core recovery via loss engineering" arc (Runs 163-172)**: five distinct mechanisms were built and properly multi-seed tested this session -- learned MLP weighting (three variants: fixed anchor=2.0, fixed anchor=0.5/collapsed, adaptive controller), centroid separation loss, MLP-triplet separation loss, and boundary-ranking loss. ALL FIVE produced real, robustly-confirmed improvements in aggregate F1/AUC (properly multi-seed verified, non-overlapping distributions in several cases). NONE of them produced a hard-core-recovery improvement at realistic FPR that survives proper multi-seed comparison against a freshly-matched baseline. This mirrors and reinforces Run 162's own finding at a different scale: F1/AUC improvements on Elliptic keep decoupling from the specific, narrow metric (hard-core recovery at a deployable FPR) that motivated this entire multi-day investigation.
- **Recommended path forward**: stop iterating on loss-engineering variants for THIS specific goal -- five independent, well-motivated mechanisms failing identically is a strong signal the bottleneck is NOT "the right auxiliary loss hasn't been found yet." Write up the full Runs 78-172 arc as the deliverable: a rigorous, honest characterization of a real fraud-detection hard limit (a specific, identifiable sub-population that resists every tested intervention at a realistic operating threshold), which is itself a legitimate and interesting research contribution, distinct from (and more honest than) claiming a fix that doesn't survive scrutiny.

## [2026-07-25] Run 173 — Confident-learning label-noise check: ~41% of the hard core looks more confidently-legit than the model's own correctly-labeled legit examples
- **Motivation**: five independently-designed loss mechanisms (Runs 163-172) all failed identically to improve hard-core recovery at realistic FPR. Rather than trying a sixth loss variant, tested a different hypothesis: is some of the hard core's "unrecoverability" because it isn't cleanly separable camouflaged fraud, but partially mislabeled or genuinely feature-indistinguishable-from-legit data? Elliptic's own labels come from third-party intelligence and are noted in the literature as potentially containing misclassifications.
- **Method**: standard confident-learning logic (Northcutt et al.) adapted to binary: ensembled test-set fraud-likeness scores across 6 INDEPENDENTLY-TRAINED baseline models (plain camo_weighted, seeds 0-5, fresh dispatch) to reduce any single model's own bias/noise from masquerading as evidence of mislabeling. Computed the standard class-0 (legit) self-confidence threshold t0 = mean predicted P(legit) among examples the given labels call legit (t0=0.7052). Flagged a given-fraud example as a candidate label error if its ensemble-predicted P(legit) exceeds t0 -- i.e. it looks, on average across 6 independent models, more like a typical confidently-correct legit example than like fraud.
- **Result**:
  | Population | n | flagged (legit_conf > t0) | mean legit_conf |
  |---|---|---|---|
  | hard_core | 213 | **87 (40.8%)** | 0.626 |
  | caught_both (sanity control) | 142 | 0 (0.0%) | 0.205 |
  | other illicit (neither) | 53 | 1 (1.9%) | 0.345 |
- **The sanity control passed cleanly**: caught_both (fraud already well-classified by both RF and GNN, i.e. definitely correctly labeled) shows EXACTLY 0% false-flagging, and "other illicit" (neither hard-core nor easy) shows only 1.9% -- both far from hard_core's 40.8%. This rules out the check simply flagging "whatever the model finds hard" as a generic artifact; the effect is specifically concentrated in the hard-core population.
- **Interpretation, with an honest caveat**: this does NOT distinguish between two possibilities that are indistinguishable from a pure modeling standpoint: (a) genuine label noise (some hard-core examples aren't actually fraud), or (b) fraud so successfully camouflaged it is truly feature-indistinguishable from legit given Elliptic's 166 anonymized features. Both interpretations lead to the SAME practical conclusion: no feature-based classifier, however sophisticated the loss function, can recover this ~41% subset -- which would cleanly explain why five independently-motivated mechanisms (learned reweighting x3, centroid separation, MLP-triplet separation, boundary-ranking) all failed identically. This reframes the entire Runs 78-172 arc: the bottleneck may be a DATA-level ceiling (mislabeled or truly-indistinguishable examples), not a modeling-level one.
- **Not yet done**: verifying against any external ground truth (Elliptic's own follow-up documentation, or independent blockchain-forensics cross-referencing) to determine how much of the 40.8% is actually mislabeled vs. genuinely-but-successfully-camouflaged. Raw ensemble scores saved (`ensemble_test_scores.npy`, `ensemble_test_y.npy`) and the top-15 hard-core nodes by legit-confidence logged for potential manual/external follow-up.
- **This is the first genuinely new, distinct-axis finding since the loss-engineering arc closed (Run 172)** -- a data-quality hypothesis rather than another architecture/loss variant, directly motivated by literature on Elliptic's known label-quality limitations.

## [2026-07-25] Run 174 — Correction to Run 173: removing flagged label-noise candidates doesn't unlock recoverable signal (denominator artifact)
- **Internal-consistency check, done correctly to avoid circularity**: evaluated hard-core recovery on 5 FRESH models (seeds 6-10, independent of the seeds 0-5 ensemble used to flag candidates in Run 173) against both the full 213-example hard_core_mask and a cleaned ~126-example version excluding the 87 flagged candidates.
  | Metric | full (n=213) | cleaned (n=126) | ratio |
  |---|---|---|---|
  | hc@1%FPR | mean=1.22% | mean=2.06% | 1.69x |
  | hc@2%FPR | mean=3.66% | mean=6.19% | 1.69x |
- **The ratio is EXACTLY 213/126=1.69 in every single seed, for both metrics.** This means the RAW COUNT of correctly-recovered hard-core examples is identical between full and cleaned (verified directly: e.g. seed=6 recovers ~2.00 examples either way, seed=8 recovers ~1.00 either way) -- the 87 flagged candidates were never being caught at these tight thresholds regardless of whether they're in the denominator, consistent with them having been flagged FOR having low fraud-likelihood scores in the first place. Removing them from the denominator mechanically inflates the percentage without recovering a single additional real example.
- **Corrected interpretation**: Run 173's confident-learning finding (41% of hard-core looks more confidently-legit than the model's own correctly-labeled legit examples, with a clean 0% sanity-control pass on caught_both) still stands as a real, methodologically sound observation -- it is NOT retracted. But the practical hope that removing these candidates would reveal a smaller, genuinely-tractable "real" hard core underneath is not supported: the remaining ~126 examples are recovered at the exact same absolute rate (1-3 examples per model across all seeds) as the full 213 always were. Whether or not ~41% of the hard core is mislabeled, the PRACTICALLY RECOVERABLE fraud count doesn't change either way.
- **Overall assessment, combining Run 173+174**: the label-noise/successful-camouflage hypothesis remains a plausible and methodologically well-supported explanation for WHY five independent loss mechanisms all failed identically (there may be little-to-no genuine feature-based signal in a meaningful chunk of this population) -- but it does not, by itself, suggest an actionable fix (no amount of "cleaning" the labels recovers real detectable fraud, since the fraud that IS detectable was already being detected regardless of which examples share the denominator). This is a genuine, rigorously-established structural finding about the dataset, not a lead toward improved recall.
