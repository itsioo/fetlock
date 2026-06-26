import torch

from fetlock.config import AblationSpec, EncoderSpec, HeadSpec, StudentSpec
from fetlock.forge import AnkleStudent, build_teacher, parameter_count, quantize_student
from fetlock.forge.routing import route_logits
from fetlock.forge.spine import TemporalSpine


def test_spine_pools_to_embed_dim():
    spec = EncoderSpec(stages=(8, 16, 32), embed_dim=24)
    spine = TemporalSpine(spec)
    out = spine(torch.randn(5, 6, 256))
    assert out.shape == (5, 24)


def test_spine_downsamples_each_block():
    spec = EncoderSpec(stages=(8, 16), embed_dim=16)
    spine = TemporalSpine(spec)
    feature = spine.feature_map(torch.randn(2, 6, 256))
    assert feature.shape[2] == 64


def test_student_is_lighter_than_teacher():
    student = AnkleStudent(StudentSpec())
    teacher = build_teacher(EncoderSpec(), HeadSpec(), AblationSpec())
    assert parameter_count(student) < parameter_count(teacher)


def test_route_logits_selects_matching_head():
    male = torch.tensor([1.0, 2.0, 3.0])
    female = torch.tensor([-1.0, -2.0, -3.0])
    sex = torch.tensor([0, 1, -1])
    routed = route_logits(male, female, sex)
    assert torch.allclose(routed, torch.tensor([1.0, -2.0, 0.0]))


def test_quantization_keeps_output_shape_and_shrinks_weights():
    student = AnkleStudent(StudentSpec())
    before = student.blocks[0].pointwise.weight.detach().clone()
    quantized = quantize_student(student, torch.randn(16, 6, 1000))
    out = quantized(torch.randn(4, 6, 1000))
    assert out.shape == (4,)
    after = student.blocks[0].pointwise.weight.detach()
    assert not torch.allclose(before, after)


def test_sex_as_feature_changes_input_dim():
    ablation = AblationSpec(sex_heads=False, sex_as_feature=True)
    net = build_teacher(EncoderSpec(embed_dim=16), HeadSpec(embed_dim=16), ablation)
    out = net(torch.randn(3, 6, 256), torch.tensor([0, 1, -1]))
    assert "male_logit" not in out
    assert out["logit"].shape == (3,)
