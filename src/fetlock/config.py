import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import attrs
from cattrs import Converter

if sys.version_info >= (3, 11):
    import tomllib as _toml
else:
    import tomli as _toml


@attrs.frozen
class EncoderSpec:
    in_channels: int = 6
    stages: Tuple[int, ...] = (32, 64, 128, 256, 256)
    kernel: int = 7
    embed_dim: int = 256
    dropout: float = 0.1


@attrs.frozen
class StudentSpec:
    in_channels: int = 6
    stages: Tuple[int, ...] = (16, 32, 32)
    kernel: int = 5


@attrs.frozen
class HeadSpec:
    embed_dim: int = 256
    hidden: int = 128
    dropout: float = 0.3
    kl_beta: float = 0.1


@attrs.frozen
class WindowSpec:
    sample_rate: int = 100
    window_seconds: float = 10.0
    overlap: float = 0.5
    channels: int = 6

    @property
    def length(self) -> int:
        return round(self.sample_rate * self.window_seconds)

    @property
    def hop(self) -> int:
        return max(1, round(self.length * (1.0 - self.overlap)))


@attrs.frozen
class SplitSpec:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15
    strata: Tuple[str, ...] = ("sex", "age_band", "severity")


@attrs.frozen
class PretrainSpec:
    lambda_temporal: float = 1.0 / 3.0
    lambda_transform: float = 1.0 / 3.0
    lambda_contrastive: float = 1.0 / 3.0
    nt_xent_temperature: float = 0.5
    n_transforms: int = 4
    projection_dim: int = 128


@attrs.frozen
class DistillSpec:
    temperature: float = 4.0
    alpha: float = 0.7
    calibration_samples: int = 500


@attrs.frozen
class OptimSpec:
    name: str = "adamw"
    lr: float = 2e-7
    weight_decay: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    grad_clip: float = 1.0


@attrs.frozen
class ScheduleSpec:
    epochs: int = 4000
    warmup_epochs: int = 200
    patience: int = 20
    monitor: str = "val_auc"
    name: str = "cosine"


@attrs.frozen
class RunSpec:
    world_size: int = 8
    per_device_batch: int = 256
    grad_accum: int = 24
    precision: str = "fp32"
    amp: bool = False
    seeds: Tuple[int, ...] = (42, 123, 256, 512, 789, 1024, 1337, 2048, 3141, 4096)

    @property
    def effective_batch(self) -> int:
        return self.world_size * self.per_device_batch * self.grad_accum


@attrs.frozen
class AblationSpec:
    tl_pretrain: bool = True
    domain_adapt: bool = True
    sex_heads: bool = True
    sex_as_feature: bool = False
    channels: str = "both"

    @property
    def channel_index(self) -> Tuple[int, ...]:
        if self.channels == "accel":
            return (0, 1, 2)
        if self.channels == "gyro":
            return (3, 4, 5)
        return (0, 1, 2, 3, 4, 5)


@attrs.frozen
class Config:
    stage: str = "finetune"
    seed: int = 42
    dataset: str = "most"
    data_fraction: float = 1.0
    data_root: str = "data"
    out_dir: str = "artifacts"
    ablation: AblationSpec = AblationSpec()
    encoder: EncoderSpec = EncoderSpec()
    student: StudentSpec = StudentSpec()
    head: HeadSpec = HeadSpec()
    window: WindowSpec = WindowSpec()
    split: SplitSpec = SplitSpec()
    pretrain: PretrainSpec = PretrainSpec()
    distill: DistillSpec = DistillSpec()
    optim: OptimSpec = OptimSpec()
    schedule: ScheduleSpec = ScheduleSpec()
    run: RunSpec = RunSpec()


_CONVERTER = Converter(forbid_extra_keys=True)


def _deep_merge(base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _nest(flat: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for dotted, value in flat.items():
        cursor = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return out


def load_config(
    path: Union[str, Path],
    overrides: Optional[Dict[str, Any]] = None,
) -> Config:
    with open(path, "rb") as handle:
        raw: Dict[str, Any] = _toml.load(handle)
    if overrides:
        raw = _deep_merge(raw, _nest(overrides))
    return _CONVERTER.structure(raw, Config)


def dump_config(config: Config) -> Dict[str, Any]:
    result = _CONVERTER.unstructure(config)
    return dict(result)
