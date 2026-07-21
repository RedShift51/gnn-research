"""Train a TabDDPM-style diffusion model on fraud node features (TRAIN split only), for
synthetic fraud augmentation of the training graph — see data/augment_graph.py for the next
stage. Reused by the CLI entrypoint (main, below) and by serverless/handler.py."""

import argparse
import logging
from pathlib import Path

import torch
import yaml

from models.diffusion.tabddpm import GaussianDiffusion, TabDDPMDenoiser

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_from_config(config: dict) -> Path:
    logger.info(f"train_diffusion config: {config}")
    dcfg = config["diffusion"]
    device = pick_device()
    logger.info(f"Using device: {device}")

    data = torch.load(ROOT / config["data"]["processed_path"], weights_only=False)

    # TRAIN split only, fraud only — sampling from val/test here would leak test-time
    # information into the generator that later augments the training set.
    fraud_mask = data.train_mask & (data.y == 1)
    fraud_features = data.x[fraud_mask].to(device)
    logger.info(f"Training diffusion model on {fraud_features.shape[0]} fraud node feature vectors "
                f"(dim={fraud_features.shape[1]}) from the TRAIN split")

    diffusion = GaussianDiffusion(num_timesteps=dcfg["num_timesteps"], device=device)
    denoiser = TabDDPMDenoiser(
        feature_dim=fraud_features.shape[1], hidden_dim=dcfg["hidden_dim"],
    ).to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=dcfg["lr"])

    batch_size = min(dcfg["batch_size"], fraud_features.shape[0])
    log_every = dcfg.get("log_every", 10)

    for epoch in range(1, dcfg["epochs"] + 1):
        denoiser.train()
        perm = torch.randperm(fraud_features.shape[0], device=device)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, fraud_features.shape[0], batch_size):
            batch = fraud_features[perm[i:i + batch_size]]
            optimizer.zero_grad()
            loss = diffusion.training_loss(denoiser, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % log_every == 0 or epoch == 1 or epoch == dcfg["epochs"]:
            logger.info(f"Diffusion epoch {epoch:4d}/{dcfg['epochs']} | loss={avg_loss:.4f}")

    out_path = ROOT / dcfg["model_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "denoiser_state": denoiser.state_dict(),
        "feature_dim": fraud_features.shape[1],
        "hidden_dim": dcfg["hidden_dim"],
        "num_timesteps": dcfg["num_timesteps"],
    }, out_path)
    logger.info(f"Saved diffusion model to {out_path}")
    return out_path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    run_from_config(config)


if __name__ == "__main__":
    main()
