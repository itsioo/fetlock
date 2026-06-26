import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def global_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def is_main() -> bool:
    return global_rank() == 0


@contextmanager
def distributed_session(backend: str = "nccl") -> Iterator[None]:
    started = False
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not is_distributed():
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        started = True
    try:
        yield
    finally:
        if started:
            dist.destroy_process_group()


def wrap_model(model: nn.Module) -> nn.Module:
    if not is_distributed():
        return model
    device_ids = [int(os.environ.get("LOCAL_RANK", "0"))] if torch.cuda.is_available() else None
    return DistributedDataParallel(model, device_ids=device_ids)
