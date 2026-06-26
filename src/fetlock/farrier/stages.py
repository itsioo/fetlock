import inspect
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from fetlock.config import Config
from fetlock.forge import (
    AnkleStudent,
    PretextBundle,
    build_teacher,
    int8_size_kb,
    quantize_student,
)
from fetlock.paddock import StratifiedSplit, stratified_subject_split, synthesize_cohort
from fetlock.paddock.cohort import Subject, load_cohort
from fetlock.paddock.stalls import PretextStalls, SupervisedStalls
from fetlock.shoeing import (
    EarlyStopping,
    build_optimizer,
    build_scheduler,
    collect_scores,
    fit,
    set_seed,
)
from fetlock.soundness import fmri_concordance, score_panel
from fetlock.soundness.concordance import ConcordanceCell
from fetlock.tack import get_logger
from fetlock.temper import distillation_loss, pretrain_loss, sex_aware_loss

_LOG = get_logger("fetlock.farrier")
ANCHOR_FEATURES = ("angular_velocity_df", "pf_moment_proxy", "stride_time_var", "ankle_rom")
ANCHOR_ROIS = ("sma", "m1", "s1", "cerebellum")
ANCHOR_TARGET = (0.43, 0.38, 0.31, 0.35)


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_cohort(config: Config, n_subjects: int = 220) -> StratifiedSplit:
    root = Path(config.data_root)
    if config.dataset != "synthetic" and (root / "manifest.jsonl").exists():
        subjects: List[Subject] = load_cohort(root, config.window.channels)
    else:
        seconds = max(config.window.window_seconds * 8.0, config.window.window_seconds + 1.0)
        subjects = synthesize_cohort(n_subjects, config.window.sample_rate, seconds, config.seed)
    split = stratified_subject_split(subjects, config.split, config.seed)
    if config.data_fraction < 1.0:
        keep = max(2, round(len(split.train) * config.data_fraction))
        split = StratifiedSplit(train=split.train[:keep], val=split.val, test=split.test)
    return split


def _supervised_loader(subjects: List[Subject], config: Config, shuffle: bool) -> DataLoader:
    dataset = SupervisedStalls(
        subjects,
        config.window.length,
        config.window.hop,
        config.ablation.channel_index,
    )
    return DataLoader(dataset, batch_size=config.run.per_device_batch, shuffle=shuffle)


def run_pretrain(
    config: Config, device: torch.device, max_steps: Optional[int] = None
) -> List[float]:
    set_seed(config.seed)
    split = prepare_cohort(config)
    dataset = PretextStalls(
        split.train,
        config.window.length,
        config.window.hop,
        config.ablation.channel_index,
        config.seed,
    )
    loader = DataLoader(dataset, batch_size=config.run.per_device_batch, shuffle=True)
    model = PretextBundle(config.encoder, config.pretrain).to(device)
    optimizer = build_optimizer(model.parameters(), config.optim)
    scheduler = build_scheduler(optimizer, config.schedule)

    def closure(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return pretrain_loss(model(batch), batch, config.pretrain)

    epochs = 1 if max_steps else config.schedule.epochs
    return fit(
        model,
        loader,
        closure,
        optimizer,
        config.run,
        config.optim,
        device,
        scheduler,
        epochs,
        max_steps,
    )


def run_finetune(
    config: Config, device: torch.device, max_steps: Optional[int] = None
) -> Tuple[torch.nn.Module, Dict[str, float], List[float]]:
    set_seed(config.seed)
    split = prepare_cohort(config)
    train_loader = _supervised_loader(split.train, config, shuffle=True)
    val_loader = _supervised_loader(split.val, config, shuffle=False)
    model = build_teacher(config.encoder, config.head, config.ablation).to(device)
    optimizer = build_optimizer(model.parameters(), config.optim)
    scheduler = build_scheduler(optimizer, config.schedule)
    stopper = EarlyStopping(config.schedule.patience, mode="max")

    def closure(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return sex_aware_loss(model(batch["signal"], batch["sex"]), batch["label"], config.head)

    def score_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(batch["signal"], batch["sex"])["logit"]

    epochs = 1 if max_steps else config.schedule.epochs
    history = fit(
        model,
        train_loader,
        closure,
        optimizer,
        config.run,
        config.optim,
        device,
        scheduler,
        epochs,
        max_steps,
    )
    scores, labels = collect_scores(score_fn, val_loader, device)
    metrics = score_panel(scores, labels)
    stopper.step(metrics["auc"])
    _LOG.info("finetune val auc %.4f", metrics["auc"])
    return model, metrics, history


def run_distill(
    config: Config, device: torch.device, max_steps: Optional[int] = None
) -> Tuple[torch.nn.Module, Dict[str, float], float]:
    teacher, _, _ = run_finetune(config, device, max_steps)
    teacher.eval()
    set_seed(config.seed)
    split = prepare_cohort(config)
    train_loader = _supervised_loader(split.train, config, shuffle=True)
    val_loader = _supervised_loader(split.val, config, shuffle=False)
    student = AnkleStudent(config.student).to(device)
    optimizer = build_optimizer(student.parameters(), config.optim)
    scheduler = build_scheduler(optimizer, config.schedule)

    def closure(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            teacher_logit = teacher(batch["signal"], batch["sex"])["logit"]
        student_logit = student(batch["signal"])
        return distillation_loss(student_logit, teacher_logit, batch["label"], config.distill)

    epochs = 1 if max_steps else config.schedule.epochs
    fit(
        student,
        train_loader,
        closure,
        optimizer,
        config.run,
        config.optim,
        device,
        scheduler,
        epochs,
        max_steps,
    )

    calibration = torch.stack(
        [
            next(iter(train_loader))["signal"][index]
            for index in range(min(config.distill.calibration_samples, config.run.per_device_batch))
        ]
    ).to(device)
    quantized = quantize_student(student, calibration)

    def score_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return quantized(batch["signal"])

    scores, labels = collect_scores(score_fn, val_loader, device)
    metrics = score_panel(scores, labels)
    return quantized, metrics, int8_size_kb(student)


def run_evaluate(config: Config, device: torch.device) -> Dict[str, float]:
    model, _, _ = run_finetune(config, device, max_steps=None)
    split = prepare_cohort(config)
    test_loader = _supervised_loader(split.test, config, shuffle=False)

    def score_fn(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return model(batch["signal"], batch["sex"])["logit"]

    scores, labels = collect_scores(score_fn, test_loader, device)
    return score_panel(scores, labels)


def _planted_series(
    anchor: NDArray[np.float64], target: float, rng: np.random.Generator
) -> NDArray[np.float64]:
    x = (anchor - anchor.mean()) / (anchor.std() + 1e-12)
    residual = rng.standard_normal(len(anchor))
    residual = residual - (residual @ x) / (x @ x) * x
    residual = residual / (residual.std() + 1e-12)
    return target * x + np.sqrt(max(0.0, 1.0 - target * target)) * residual


def run_concordance(config: Config) -> List[ConcordanceCell]:
    rng = np.random.default_rng(config.seed)
    n_anchors = 40
    wearable: Dict[str, NDArray[np.float64]] = {}
    roi: Dict[str, NDArray[np.float64]] = {}
    base = rng.standard_normal((len(ANCHOR_FEATURES), n_anchors))
    for index, feature in enumerate(ANCHOR_FEATURES):
        wearable[feature] = base[index]
        roi[ANCHOR_ROIS[index]] = _planted_series(base[index], ANCHOR_TARGET[index], rng)
    pairs = list(zip(ANCHOR_FEATURES, ANCHOR_ROIS))
    return fmri_concordance(wearable, roi, pairs, n_perm=2000, n_boot=1000, seed=config.seed)


def export_student_onnx(config: Config, path: str) -> str:
    student = AnkleStudent(config.student).eval()
    dummy = torch.randn(1, config.student.in_channels, config.window.length)
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        torch.onnx.export(
            student,
            (dummy,),
            path,
            input_names=["imu_window"],
            output_names=["oa_logit"],
            opset_version=13,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            student,
            (dummy,),
            path,
            input_names=["imu_window"],
            output_names=["oa_logit"],
            opset_version=13,
        )
    return path
