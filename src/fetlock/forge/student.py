import torch
from torch import nn

from fetlock.config import StudentSpec


class _SeparableBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_ch, in_ch, kernel, stride=2, padding=kernel // 2, groups=in_ch
        )
        self.pointwise = nn.Conv1d(in_ch, out_ch, 1)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class AnkleStudent(nn.Module):
    def __init__(self, spec: StudentSpec) -> None:
        super().__init__()
        blocks = []
        in_ch = spec.in_channels
        for out_ch in spec.stages:
            blocks.append(_SeparableBlock(in_ch, out_ch, spec.kernel))
            in_ch = out_ch
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(in_ch, 1)

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(self.blocks(signal)).squeeze(-1)
        return self.classifier(pooled).squeeze(-1)


def parameter_count(module: nn.Module) -> int:
    return sum(int(p.numel()) for p in module.parameters())
