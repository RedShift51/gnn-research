# fraud-diffusion — MVP slice

Minimal end-to-end vertical slice: PaySim → transaction-node graph → GraphSAGE (Focal Loss) →
metrics → logged in `LAB_JOURNAL.md`. Everything beyond this (GAT/GCN, diffusion augmentation,
IEEE-CIS/Elliptic, ablations) is deferred — see `../CLAUDE.md` for the full project spec.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Kaggle credentials for download: place `kaggle.json` at `~/.kaggle/kaggle.json`
(Kaggle account → Settings → API → Create New Token). If missing, `data/download.py` prints
instructions and you can manually download `paysim1` from
https://www.kaggle.com/datasets/ealaxi/paysim1 into `data/raw/paysim.csv`.

## Run the MVP pipeline

```bash
python -m data.download                                  # → data/raw/paysim.csv
python -m data.paysim_preprocess --config configs/paysim.yaml   # → data/processed/paysim_graph.pt
python -m training.train_gnn --config configs/paysim.yaml       # trains, evaluates, appends to LAB_JOURNAL.md
```

Each training run appends a dated section to `LAB_JOURNAL.md` — the source of truth for
what was tried and what it scored.

## Graph construction (PaySim)

- Only `TRANSFER` and `CASH_OUT` rows are kept (all fraud lives there).
- Stratified subsample for local runs (`data.subsample_size` in config) — keeps all fraud,
  downsamples legit transactions.
- **Nodes** = transactions. **Edges** = pairs of transactions sharing `nameOrig` or `nameDest`
  (same account), degree-capped per node.
- **Split** is temporal (sorted by `step`), not random — avoids leaking future info into training.

## Compute

Local runs (this MVP) are sized to run on CPU/MPS in minutes. Full-dataset and future diffusion
training moves to a RunPod GPU Pod — see `../CLAUDE.md` § GPU Compute Infrastructure and
`infra/runpod_launcher.py` (added when the MVP is validated locally).
