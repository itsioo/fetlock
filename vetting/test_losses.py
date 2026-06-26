import torch

from fetlock.config import AblationSpec, DistillSpec, EncoderSpec, HeadSpec, PretrainSpec
from fetlock.forge import build_teacher
from fetlock.temper import distillation_loss, nt_xent, pretrain_loss, sex_aware_loss


def test_nt_xent_lower_for_aligned_views():
    rng = torch.manual_seed(0)
    a = torch.randn(8, 16, generator=rng)
    aligned = nt_xent(a, a.clone(), temperature=0.5)
    misaligned = nt_xent(a, torch.randn(8, 16, generator=rng), temperature=0.5)
    assert aligned < misaligned


def test_nt_xent_matches_finite_difference():
    torch.manual_seed(1)
    a = torch.randn(4, 8, requires_grad=True, dtype=torch.float64)
    b = torch.randn(4, 8, requires_grad=True, dtype=torch.float64)
    loss = nt_xent(a, b, temperature=0.7)
    loss.backward()
    analytic = a.grad[0, 0].item()
    eps = 1e-6
    with torch.no_grad():
        a[0, 0] += eps
        high = nt_xent(a, b, temperature=0.7).item()
        a[0, 0] -= 2 * eps
        low = nt_xent(a, b, temperature=0.7).item()
    assert abs(analytic - (high - low) / (2 * eps)) < 1e-4


def test_pretrain_weights_sum_components():
    spec = PretrainSpec()
    batch = {
        "order_label": torch.tensor([0, 1, 0, 1]),
        "transform_label": torch.tensor([0, 1, 2, 3]),
    }
    outputs = {
        "order_logits": torch.zeros(4, 2),
        "transform_logits": torch.zeros(4, 4),
        "proj_a": torch.randn(4, 8),
        "proj_b": torch.randn(4, 8),
    }
    result = pretrain_loss(outputs, batch, spec)
    expected = (
        spec.lambda_temporal * result["temporal"]
        + spec.lambda_transform * result["transform"]
        + spec.lambda_contrastive * result["contrastive"]
    )
    assert torch.allclose(result["loss"], expected, atol=1e-6)


def test_sex_aware_ce_flows_only_through_matched_head():
    net = build_teacher(
        EncoderSpec(embed_dim=16), HeadSpec(embed_dim=16, kl_beta=0.0), AblationSpec()
    )
    signal = torch.randn(6, 6, 256)
    sex = torch.zeros(6, dtype=torch.long)
    label = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    out = net(signal, sex)
    sex_aware_loss(out, label, HeadSpec(kl_beta=0.0))["loss"].backward()
    female_grad = net.female_head.net[0].weight.grad
    male_grad = net.male_head.net[0].weight.grad
    assert female_grad is None or float(female_grad.abs().sum()) == 0.0
    assert male_grad is not None and float(male_grad.abs().sum()) > 0.0


def test_distillation_matches_manual():
    spec = DistillSpec(temperature=4.0, alpha=0.7)
    student = torch.tensor([0.5, -0.5])
    teacher = torch.tensor([1.0, -1.0])
    label = torch.tensor([1.0, 0.0])
    out = distillation_loss(student, teacher, label, spec)
    p_s = torch.sigmoid(student / 4.0)
    p_t = torch.sigmoid(teacher / 4.0)
    kl = (p_s * (p_s / p_t).log() + (1 - p_s) * ((1 - p_s) / (1 - p_t)).log()).mean()
    soft = 16.0 * kl
    hard = torch.nn.functional.binary_cross_entropy_with_logits(student, label)
    assert torch.allclose(out["loss"], 0.7 * soft + 0.3 * hard, atol=1e-6)
