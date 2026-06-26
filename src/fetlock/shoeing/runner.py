from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from fetlock.config import OptimSpec, RunSpec

LossClosure = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
ScoreClosure = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


def move(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def fit(
    model: torch.nn.Module,
    loader: DataLoader,
    closure: LossClosure,
    optimizer: torch.optim.Optimizer,
    run: RunSpec,
    optim_spec: OptimSpec,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR] = None,
    epochs: int = 1,
    max_steps: Optional[int] = None,
) -> List[float]:
    model.train()
    history: List[float] = []
    scaler = GradScaler(enabled=run.amp)
    taken = 0
    for _ in range(epochs):
        optimizer.zero_grad()
        for position, raw in enumerate(loader):
            batch = move(raw, device)
            with autocast(enabled=run.amp):
                output = closure(batch)
                scaled = output["loss"] / run.grad_accum
            scaler.scale(scaled).backward()
            if (position + 1) % run.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), optim_spec.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            history.append(float(output["loss"].detach()))
            taken += 1
            if max_steps is not None and taken >= max_steps:
                return history
        if scheduler is not None:
            scheduler.step()
    return history


def collect_scores(
    score_fn: ScoreClosure,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    scores: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for raw in loader:
            batch = move(raw, device)
            logit = score_fn(batch)
            scores.extend(torch.sigmoid(logit).cpu().numpy().tolist())
            labels.extend(batch["label"].cpu().numpy().tolist())
    return np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=np.int64)
