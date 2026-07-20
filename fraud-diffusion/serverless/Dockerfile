# RunPod Serverless worker image. Base image already has CUDA + torch matching what we
# validated on the RunPod Pod (torch 2.4.1+cu124) — avoids re-downloading a huge torch wheel.
# --platform is pinned to amd64 (what RunPod's GPU hosts run) since this is built on Apple
# Silicon (arm64) — without it, docker builds for the host's native arch and either fails
# under emulation or produces an image RunPod can't run.
FROM --platform=linux/amd64 runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /root/fraud-diffusion

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x serverless/entrypoint.sh

CMD ["serverless/entrypoint.sh"]
