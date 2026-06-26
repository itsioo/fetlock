from typing import Dict

import torch
from torch import nn

from fetlock.config import EncoderSpec, PretrainSpec
from fetlock.forge.spine import TemporalSpine


class ProjectionHead(nn.Module):
    def __init__(self, embed_dim: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PretextBundle(nn.Module):
    def __init__(self, encoder_spec: EncoderSpec, pretrain_spec: PretrainSpec) -> None:
        super().__init__()
        self.spine = TemporalSpine(encoder_spec)
        self.temporal_order = nn.Linear(encoder_spec.embed_dim, 2)
        self.transform = nn.Linear(encoder_spec.embed_dim, pretrain_spec.n_transforms)
        self.projection = ProjectionHead(encoder_spec.embed_dim, pretrain_spec.projection_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        return self.spine(x)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        z_a = self.spine(batch["view_a"])
        z_b = self.spine(batch["view_b"])
        z_order = self.spine(batch["order_input"])
        return {
            "order_logits": self.temporal_order(z_order),
            "transform_logits": self.transform(z_a),
            "proj_a": self.projection(z_a),
            "proj_b": self.projection(z_b),
        }
