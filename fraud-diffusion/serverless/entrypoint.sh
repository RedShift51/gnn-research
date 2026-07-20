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

exec python3 -u -m serverless.handler
