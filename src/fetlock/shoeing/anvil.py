import inspect
import os
from pathlib import Path
from typing import Any, Dict, Union

import torch
from torch import nn

from fetlock.shoeing.seed import rng_state


def save_checkpoint(
    path: Union[str, Path],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    seed: int,
    extra: Dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "seed": seed,
        "rng": rng_state(),
        "extra": extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    payload: Dict[str, Any] = torch.load(path, map_location=map_location, **kwargs)
    return payload
