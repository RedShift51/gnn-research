"""Train a TabDDPM-style diffusion model on fraud node features (TRAIN split only), for
synthetic fraud augmentation of the training graph — see data/augment_graph.py for the next
stage. Reused by the CLI entrypoint (main, below) and by serverless/handler.py."""

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn.functional as F
import wandb
import yaml

from models.diffusion.tabddpm import Discriminator, GaussianDiffusion, TabDDPMDenoiser

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def init_wandb(config: dict):
    """Same enable/disable conventions as training/train_gnn.py's init_wandb (auto-online when
    WANDB_API_KEY is present, config["wandb"]["enabled"]=False overrides) but NOT reused directly
    from there — this needs its own run (name suffixed "_diffusion") since handler.py runs this
    BEFORE the GNN training phase's own wandb run in the same process, and they'd otherwise share
    a confusingly-identical display name. Added specifically to answer "is the adversarial loss
    miscalibrated or is something actually wrong" with real per-epoch curves instead of only the
    tiny smoke test's printed logs — see LAB_JOURNAL.md's adversarial diffusion entry."""
    wcfg = config.get("wandb", {})
    if not wcfg.get("enabled", True):
        return wandb.init(mode="disabled")
    mode = wcfg.get("mode") or ("online" if os.environ.get("WANDB_API_KEY") else "disabled")
    base_name = config.get("journal", {}).get("run_name", "diffusion")
    return wandb.init(
        project=wcfg.get("project", "fraud-diffusion"),
        name=f"{base_name}_diffusion",
        group=config["data"].get("dataset", "paysim"),
        config=config.get("diffusion", {}),
        mode=mode,
        reinit=True,
    )


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

    wandb_run = init_wandb(config)

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

    # Optional adversarial fine-tuning phase (off by default — existing configs unaffected).
    # Per discussion: start partway through training (the denoiser needs to already be
    # reasonably good before an adversarial signal is useful — an adversarial term against a
    # near-random early denoiser just destabilizes both networks from the start), not from epoch 1.
    # L_total = L_denoise + lambda * L_adversarial, L_adversarial = BCE(D(x0_pred), "real").
    # x0_pred (the denoiser's one-step x0 estimate, not a full multi-step sample) is what's fed to
    # the discriminator — computing a real ddim/ddpm sample every training batch would be far too
    # expensive. Clamped to real fraud features' own mean+-3*std for the same reason the samplers
    # clamp it (numerically unstable at large t, see GaussianDiffusion.predict_x0's docstring) —
    # here specifically to keep the discriminator's gradient signal well-behaved.
    adv_cfg = dcfg.get("adversarial", {})
    adv_enabled = adv_cfg.get("enabled", False)
    discriminator, disc_optimizer, adv_start_epoch, adv_lambda, clamp_x0 = None, None, None, None, None
    if adv_enabled:
        adv_start_epoch = adv_cfg.get("start_epoch", dcfg["epochs"] // 2)
        adv_lambda = adv_cfg.get("lambda_adv", 0.1)
        discriminator = Discriminator(
            feature_dim=fraud_features.shape[1], hidden_dim=adv_cfg.get("discriminator_hidden_dim", 128),
        ).to(device)
        disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=adv_cfg.get("discriminator_lr", dcfg["lr"]))
        mu, sd = fraud_features.mean(dim=0), fraud_features.std(dim=0).clamp(min=1e-3)
        n_clamp_std = adv_cfg.get("clamp_std", 3.0)
        clamp_x0 = (mu - n_clamp_std * sd, mu + n_clamp_std * sd)
        logger.info(f"Adversarial phase enabled: starts at epoch {adv_start_epoch}/{dcfg['epochs']}, "
                    f"lambda={adv_lambda}")

    batch_size = min(dcfg["batch_size"], fraud_features.shape[0])
    log_every = dcfg.get("log_every", 10)

    for epoch in range(1, dcfg["epochs"] + 1):
        denoiser.train()
        adversarial_active = adv_enabled and epoch >= adv_start_epoch
        if adversarial_active:
            discriminator.train()
        perm = torch.randperm(fraud_features.shape[0], device=device)
        epoch_denoise_loss, epoch_adv_loss, epoch_disc_loss = 0.0, 0.0, 0.0
        n_batches = 0
        for i in range(0, fraud_features.shape[0], batch_size):
            batch = fraud_features[perm[i:i + batch_size]]
            b = batch.shape[0]
            t = torch.randint(0, diffusion.num_timesteps, (b,), device=device)
            noise = torch.randn_like(batch)
            x_t = diffusion.q_sample(batch, t, noise)
            pred_noise = denoiser(x_t, t)
            loss_denoise = F.mse_loss(pred_noise, noise)

            if adversarial_active:
                x0_pred = diffusion.predict_x0(x_t, t, pred_noise).clamp(*clamp_x0)

                disc_optimizer.zero_grad()
                d_real = discriminator(batch)
                d_fake = discriminator(x0_pred.detach())
                loss_disc = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                             + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))
                loss_disc.backward()
                disc_optimizer.step()

                d_fake_for_gen = discriminator(x0_pred)
                loss_adv = F.binary_cross_entropy_with_logits(d_fake_for_gen, torch.ones_like(d_fake_for_gen))
                loss_total = loss_denoise + adv_lambda * loss_adv
                epoch_adv_loss += loss_adv.item()
                epoch_disc_loss += loss_disc.item()
            else:
                loss_total = loss_denoise

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            epoch_denoise_loss += loss_denoise.item()
            n_batches += 1

        avg_loss = epoch_denoise_loss / max(n_batches, 1)
        avg_adv_loss = epoch_adv_loss / max(n_batches, 1) if adversarial_active else None
        avg_disc_loss = epoch_disc_loss / max(n_batches, 1) if adversarial_active else None
        wandb.log(
            {"epoch": epoch, "loss_denoise": avg_loss, "loss_adv": avg_adv_loss, "loss_disc": avg_disc_loss,
             "adversarial_active": adversarial_active},
            step=epoch,
        )
        if epoch % log_every == 0 or epoch == 1 or epoch == dcfg["epochs"] or epoch == adv_start_epoch:
            msg = f"Diffusion epoch {epoch:4d}/{dcfg['epochs']} | loss_denoise={avg_loss:.4f}"
            if adversarial_active:
                msg += f" | loss_adv={avg_adv_loss:.4f} | loss_disc={avg_disc_loss:.4f}"
            logger.info(msg)

    out_path = ROOT / dcfg["model_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "denoiser_state": denoiser.state_dict(),
        "feature_dim": fraud_features.shape[1],
        "hidden_dim": dcfg["hidden_dim"],
        "num_timesteps": dcfg["num_timesteps"],
    }, out_path)
    logger.info(f"Saved diffusion model to {out_path}")
    wandb.finish()
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
