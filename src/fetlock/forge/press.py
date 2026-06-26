from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class AffineQ:
    scale: float
    zero_point: int


class MinMaxObserver:
    def __init__(self) -> None:
        self.low = float("inf")
        self.high = float("-inf")

    def update(self, tensor: torch.Tensor) -> None:
        self.low = min(self.low, float(tensor.min()))
        self.high = max(self.high, float(tensor.max()))

    def qparams(self) -> AffineQ:
        low = min(self.low, 0.0)
        high = max(self.high, 0.0)
        scale = (high - low) / 255.0 if high > low else 1.0
        zero_point = round(-low / scale) - 128
        return AffineQ(scale=scale, zero_point=max(-128, min(127, zero_point)))


def fake_quant_affine(tensor: torch.Tensor, q: AffineQ) -> torch.Tensor:
    quantized = torch.clamp(torch.round(tensor / q.scale) + q.zero_point, -128, 127)
    return (quantized - q.zero_point) * q.scale


def fake_quant_symmetric(weight: torch.Tensor) -> torch.Tensor:
    scale = float(weight.abs().max()) / 127.0
    if scale == 0.0:
        return weight.clone()
    quantized = torch.clamp(torch.round(weight / scale), -127, 127)
    return quantized * scale


class QuantizedStudent(nn.Module):
    def __init__(self, student: nn.Module, input_q: AffineQ) -> None:
        super().__init__()
        self.student = student
        self.input_q = input_q

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.student(fake_quant_affine(signal, self.input_q))


def quantize_student(student: nn.Module, calibration: torch.Tensor) -> QuantizedStudent:
    student.eval()
    observer = MinMaxObserver()
    with torch.no_grad():
        observer.update(calibration)
        for module in student.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                module.weight.data = fake_quant_symmetric(module.weight.data)
    return QuantizedStudent(student, observer.qparams())


def int8_size_kb(module: nn.Module) -> float:
    params = sum(int(p.numel()) for p in module.parameters())
    return params / 1024.0


def fp32_size_kb(module: nn.Module) -> float:
    params = sum(int(p.numel()) for p in module.parameters())
    return params * 4.0 / 1024.0
