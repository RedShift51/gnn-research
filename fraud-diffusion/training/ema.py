import torch


class EMA:
    """Exponential moving average of a model's weights, started partway through training
    (see train_gnn.py's `ema_start_epoch`) and initialized from the model's weights at that
    point — not from zero or the random initial weights.

    Starting the average from step 0 either drags in noisy random-init weights for a long time
    (if the shadow starts at the initial weights) or needs Adam-style bias correction to
    compensate, which only works if the shadow instead starts at zero — easy to mix up. Simpler
    and just as effective: let the model train normally for a warm-up period, snapshot its
    weights as the EMA starting point, and average from there.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)

    def state_dict(self) -> dict:
        return self.shadow
