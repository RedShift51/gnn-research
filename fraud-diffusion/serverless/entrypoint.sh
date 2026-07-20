#!/usr/bin/env bash
# Runs once per worker start (not per-job). Materializes Kaggle credentials from env vars
# (injected via RunPod Secrets — see console: Settings -> Secrets, referenced in the endpoint
# template) into ~/.kaggle/kaggle.json, since data/download.py expects that file.
set -e

if [ -n "$KAGGLE_USERNAME" ] && [ -n "$KAGGLE_KEY" ] && [ ! -f ~/.kaggle/kaggle.json ]; then
    mkdir -p ~/.kaggle
    printf '{"username":"%s","key":"%s"}' "$KAGGLE_USERNAME" "$KAGGLE_KEY" > ~/.kaggle/kaggle.json
    chmod 600 ~/.kaggle/kaggle.json
fi

# The image bakes in only the base image + Python deps (see Dockerfile) — application code
# (data/, models/, training/, serverless/, configs/) is pulled fresh from GitHub here instead of
# via `COPY . .` at build time. This means code/config changes take effect on the NEXT worker
# spin-up with zero image rebuild — only genuine requirements.txt changes still need one.
# Caveat: this only re-syncs at container start, not per-job — a long-lived warm worker serving
# many jobs back-to-back won't see a mid-session code change until RunPod recycles it
# (idle_timeout in the endpoint config).
if [ -d /root/repo/.git ]; then
    git -C /root/repo pull --ff-only
else
    git clone "https://${GITHUB_REPO_TOKEN}@github.com/RedShift51/gnn-research.git" /root/repo
fi

# Exposed so job output can reveal a stale worker even when it ISN'T missing a file (e.g. a few
# commits behind but everything it needs already exists) — see LAB_JOURNAL.md's stuck-worker
# incident, where a worker silently never advanced past an old commit despite idle_timeout.
export GIT_COMMIT_SHA
GIT_COMMIT_SHA=$(git -C /root/repo rev-parse HEAD)

cd /root/repo/fraud-diffusion
exec python3 -u -m serverless.handler
