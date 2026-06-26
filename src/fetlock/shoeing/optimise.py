import math
from typing import Iterable

import torch

from fetlock.config import OptimSpec, ScheduleSpec


def build_optimizer(params: Iterable[torch.nn.Parameter], spec: OptimSpec) -> torch.optim.Optimizer:
    if spec.name != "adamw":
        raise ValueError(f"unsupported optimizer {spec.name!r}")
    return torch.optim.AdamW(
        params,
        lr=spec.lr,
        betas=(spec.beta1, spec.beta2),
        eps=spec.eps,
        weight_decay=spec.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, spec: ScheduleSpec
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(1, spec.warmup_epochs)
    total = max(spec.epochs, warmup + 1)

    def factor(epoch: int) -> float:
        if epoch < spec.warmup_epochs:
            return (epoch + 1) / warmup
        progress = (epoch - spec.warmup_epochs) / max(1, total - spec.warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float = -math.inf if mode == "max" else math.inf
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
            self.should_stop = self.bad_epochs >= self.patience
        return improved
