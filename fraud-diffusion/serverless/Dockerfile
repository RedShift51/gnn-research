# RunPod Serverless worker image. Base image already has CUDA + torch matching what we
# validated on the RunPod Pod (torch 2.4.1+cu124) — avoids re-downloading a huge torch wheel.
#
# Built by .github/workflows/build-serverless-image.yml on a native amd64 GitHub runner (same
# arch as RunPod's GPU hosts) — deliberately NOT built locally on an Apple Silicon Mac, where
# cross-arch QEMU emulation of this (large, CUDA-devel-based) image is unreliable and can hang
# rather than just run slowly. --platform is still pinned explicitly so the image is correct
# even if someone does build it from a different host arch.
FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /root/fraud-diffusion

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x serverless/entrypoint.sh

CMD ["serverless/entrypoint.sh"]
