from typing import Tuple

import torch
from torch import nn

from fetlock.config import EncoderSpec


class _TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=2, padding=kernel // 2)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class TemporalSpine(nn.Module):
    def __init__(self, spec: EncoderSpec) -> None:
        super().__init__()
        blocks = []
        in_ch = spec.in_channels
        for out_ch in spec.stages:
            blocks.append(_TemporalBlock(in_ch, out_ch, spec.kernel, spec.dropout))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.project = nn.Linear(in_ch, spec.embed_dim)
        self.embed_dim = spec.embed_dim

    def feature_map(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.blocks(x)).squeeze(-1)
        return self.project(pooled)


def encoder_output_shape(spec: EncoderSpec) -> Tuple[int]:
    return (spec.embed_dim,)
