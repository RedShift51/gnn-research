"""TabDDPM-style diffusion model for synthetic fraud node feature generation.

MLP denoiser + Gaussian diffusion (Kotelnikov et al., "TabDDPM", ICML 2023 — the general recipe
of running DDPM on tabular feature vectors with a small MLP denoiser instead of a U-Net). Trained
on fraud-only node features from the TRAIN split; sampling generates new synthetic fraud feature
vectors to add to the graph (see data/augment_graph.py) — the actual research contribution this
project is built around, not just another GNN baseline.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Standard transformer-style sinusoidal embedding of the diffusion timestep."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t.float()[:, None] * freqs[None]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TabDDPMDenoiser(nn.Module):
    """Predicts the noise added to a node feature vector at timestep t."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128, time_dim: int = 32):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(feature_dim + time_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, self.time_embed(t)], dim=-1))


class GaussianDiffusion:
    """Linear beta-schedule DDPM: training loss (predict the noise) plus two samplers —
    ancestral DDPM sampling (all `num_timesteps` steps) and deterministic DDIM sampling (a
    configurable, much smaller number of steps, same trained denoiser)."""

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: str = "cpu"):
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None]
        sqrt_1m_ac = self.sqrt_one_minus_alphas_cumprod[t][:, None]
        return sqrt_ac * x0 + sqrt_1m_ac * noise

    def training_loss(self, denoiser: nn.Module, x0: torch.Tensor) -> torch.Tensor:
        b = x0.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        pred_noise = denoiser(x_t, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def ddpm_sample(self, denoiser: nn.Module, n_samples: int, feature_dim: int,
                     device: torch.device, clamp_x0: tuple[float, float] | None = None) -> torch.Tensor:
        """clamp_x0, if given, clips the predicted x0 at every reverse step to (min, max) — a
        standard DDPM safeguard against reverse-process error compounding over many steps. Without
        it, a slightly-off prediction early in the chain drifts further out-of-distribution each
        subsequent step instead of being corrected (observed on Elliptic's 165-dim features: fully
        trained 1000-epoch model still produced samples with std~80-110 vs real fraud's std~0.56 —
        see LAB_JOURNAL.md Run 33/34)."""
        denoiser.eval()
        x = torch.randn(n_samples, feature_dim, device=device)
        for t_step in reversed(range(self.num_timesteps)):
            t = torch.full((n_samples,), t_step, device=device, dtype=torch.long)
            pred_noise = denoiser(x, t)
            alpha = self.alphas[t_step]
            alpha_cumprod = self.alphas_cumprod[t_step]
            beta = self.betas[t_step]

            if clamp_x0 is not None:
                x0_pred = (x - torch.sqrt(1 - alpha_cumprod) * pred_noise) / torch.sqrt(alpha_cumprod)
                x0_pred = x0_pred.clamp(*clamp_x0)
                mean = (torch.sqrt(alpha) * (1 - alpha_cumprod / alpha) * x
                        + torch.sqrt(alpha_cumprod / alpha) * beta * x0_pred) / (1 - alpha_cumprod)
            else:
                coef = beta / torch.sqrt(1 - alpha_cumprod)
                mean = (1 / torch.sqrt(alpha)) * (x - coef * pred_noise)

            if t_step > 0:
                x = mean + torch.sqrt(beta) * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(self, denoiser: nn.Module, n_samples: int, feature_dim: int,
                     device: torch.device, ddim_steps: int = 50,
                     clamp_x0: tuple[float, float] | None = None) -> torch.Tensor:
        """Deterministic (eta=0) DDIM sampling — far fewer forward passes than full DDPM.
        clamp_x0: see ddpm_sample's docstring."""
        denoiser.eval()
        step_indices = torch.linspace(0, self.num_timesteps - 1, ddim_steps, device=device).long()
        step_indices = torch.flip(step_indices, dims=[0])

        x = torch.randn(n_samples, feature_dim, device=device)
        for i, t_step in enumerate(step_indices):
            t = torch.full((n_samples,), t_step.item(), device=device, dtype=torch.long)
            pred_noise = denoiser(x, t)
            alpha_cumprod_t = self.alphas_cumprod[t_step]
            x0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * pred_noise) / torch.sqrt(alpha_cumprod_t)
            if clamp_x0 is not None:
                x0_pred = x0_pred.clamp(*clamp_x0)

            if i + 1 < len(step_indices):
                alpha_cumprod_prev = self.alphas_cumprod[step_indices[i + 1]]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            x = torch.sqrt(alpha_cumprod_prev) * x0_pred + torch.sqrt(1 - alpha_cumprod_prev) * pred_noise
        return x
