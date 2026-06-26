from typing import Dict

import torch
from torch import nn

from fetlock.config import AblationSpec, EncoderSpec, HeadSpec
from fetlock.forge.spine import TemporalSpine


class BodyMLP(nn.Module):
    def __init__(self, in_dim: int, embed_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SexHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def route_logits(
    male_logit: torch.Tensor, female_logit: torch.Tensor, sex: torch.Tensor
) -> torch.Tensor:
    male_mask = (sex == 0).to(male_logit.dtype)
    female_mask = (sex == 1).to(male_logit.dtype)
    unknown_mask = (sex < 0).to(male_logit.dtype)
    blended = 0.5 * (male_logit + female_logit)
    return male_mask * male_logit + female_mask * female_logit + unknown_mask * blended


class AnkleNet(nn.Module):
    def __init__(
        self, encoder_spec: EncoderSpec, head_spec: HeadSpec, ablation: AblationSpec
    ) -> None:
        super().__init__()
        self.use_sex_heads = ablation.sex_heads
        self.sex_as_feature = ablation.sex_as_feature
        self.spine = TemporalSpine(encoder_spec)
        body_in = encoder_spec.embed_dim + (1 if self.sex_as_feature else 0)
        self.body = BodyMLP(body_in, head_spec.embed_dim, head_spec.dropout)
        if self.use_sex_heads:
            self.male_head = SexHead(head_spec.embed_dim, head_spec.hidden, head_spec.dropout)
            self.female_head = SexHead(head_spec.embed_dim, head_spec.hidden, head_spec.dropout)
        else:
            self.shared_head = SexHead(head_spec.embed_dim, head_spec.hidden, head_spec.dropout)

    def representation(self, signal: torch.Tensor, sex: torch.Tensor) -> torch.Tensor:
        embedding = self.spine(signal)
        if self.sex_as_feature:
            sex_feature = sex.clamp_min(0).to(embedding.dtype).unsqueeze(-1)
            embedding = torch.cat([embedding, sex_feature], dim=-1)
        return self.body(embedding)

    def forward(self, signal: torch.Tensor, sex: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.representation(signal, sex)
        if self.use_sex_heads:
            male_logit = self.male_head(z)
            female_logit = self.female_head(z)
            return {
                "logit": route_logits(male_logit, female_logit, sex),
                "male_logit": male_logit,
                "female_logit": female_logit,
            }
        return {"logit": self.shared_head(z)}


def build_teacher(
    encoder_spec: EncoderSpec, head_spec: HeadSpec, ablation: AblationSpec
) -> AnkleNet:
    return AnkleNet(encoder_spec, head_spec, ablation)
