from typing import Dict

import torch
import torch.nn.functional as functional

from fetlock.config import DistillSpec, HeadSpec, PretrainSpec
from fetlock.temper.contrast import nt_xent

_EPS = 1e-6


def _bernoulli_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp(_EPS, 1.0 - _EPS)
    q = q.clamp(_EPS, 1.0 - _EPS)
    return p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()


def pretrain_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    spec: PretrainSpec,
) -> Dict[str, torch.Tensor]:
    l_temporal = functional.cross_entropy(outputs["order_logits"], batch["order_label"])
    l_transform = functional.cross_entropy(outputs["transform_logits"], batch["transform_label"])
    l_contrastive = nt_xent(outputs["proj_a"], outputs["proj_b"], spec.nt_xent_temperature)
    total = (
        spec.lambda_temporal * l_temporal
        + spec.lambda_transform * l_transform
        + spec.lambda_contrastive * l_contrastive
    )
    return {
        "loss": total,
        "temporal": l_temporal.detach(),
        "transform": l_transform.detach(),
        "contrastive": l_contrastive.detach(),
    }


def sex_aware_loss(
    outputs: Dict[str, torch.Tensor],
    label: torch.Tensor,
    spec: HeadSpec,
) -> Dict[str, torch.Tensor]:
    target = label.to(outputs["logit"].dtype)
    ce = functional.binary_cross_entropy_with_logits(outputs["logit"], target)
    if "male_logit" in outputs and "female_logit" in outputs:
        p_male = torch.sigmoid(outputs["male_logit"])
        p_female = torch.sigmoid(outputs["female_logit"])
        kl = _bernoulli_kl(p_male, p_female).mean()
    else:
        kl = torch.zeros((), device=ce.device, dtype=ce.dtype)
    total = ce + spec.kl_beta * kl
    return {"loss": total, "ce": ce.detach(), "kl": kl.detach()}


def distillation_loss(
    student_logit: torch.Tensor,
    teacher_logit: torch.Tensor,
    label: torch.Tensor,
    spec: DistillSpec,
) -> Dict[str, torch.Tensor]:
    temperature = spec.temperature
    p_student = torch.sigmoid(student_logit / temperature)
    p_teacher = torch.sigmoid(teacher_logit.detach() / temperature)
    soft = (temperature * temperature) * _bernoulli_kl(p_student, p_teacher).mean()
    hard = functional.binary_cross_entropy_with_logits(student_logit, label.to(student_logit.dtype))
    total = spec.alpha * soft + (1.0 - spec.alpha) * hard
    return {"loss": total, "soft": soft.detach(), "hard": hard.detach()}
