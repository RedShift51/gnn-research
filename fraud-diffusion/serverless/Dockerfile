# RunPod Serverless worker image. Base image already has CUDA + torch matching what we
# validated on the RunPod Pod (torch 2.4.1+cu124) — avoids re-downloading a huge torch wheel.
#
# Built by .github/workflows/build-serverless-image.yml on a native amd64 GitHub runner (same
# arch as RunPod's GPU hosts) — deliberately NOT built locally on an Apple Silicon Mac, where
# cross-arch QEMU emulation of this (large, CUDA-devel-based) image is unreliable and can hang
# rather than just run slowly. --platform is still pinned explicitly so the image is correct
# even if someone does build it from a different host arch.
#
# Deliberately does NOT `COPY . .` the application code — only the base image, system deps, and
# Python deps are baked in. entrypoint.sh clones/pulls the actual code from GitHub at container
# start instead, so code/config changes need zero rebuild, only genuine requirements.txt changes
# do. See entrypoint.sh for the tradeoff (only re-syncs on worker spin-up, not mid-session).
FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# NeighborLoader (mini-batch/neighbor-sampling training, used for GAT — full-batch attention
# doesn't fit this graph in GPU memory at any reasonable capacity, see LAB_JOURNAL.md) needs a
# compiled sampling backend. pyg-lib isn't on plain PyPI — it ships from PyG's own wheel index,
# pinned to the exact torch+CUDA combo of the base image (torch 2.4.1+cu124 here). Not
# installable this way on Apple Silicon Mac, which is why local dev only exercises full-batch.
RUN pip install --no-cache-dir pyg-lib -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

COPY serverless/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

CMD ["/root/entrypoint.sh"]
