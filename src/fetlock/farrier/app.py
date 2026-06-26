import json
from pathlib import Path
from typing import Any, Dict, List

import typer

from fetlock.config import Config, load_config
from fetlock.farrier.stages import (
    export_student_onnx,
    pick_device,
    run_concordance,
    run_distill,
    run_evaluate,
    run_finetune,
    run_pretrain,
)
from fetlock.tack import get_logger

app = typer.Typer(
    add_completion=False, help="Sex-aware transfer-learning anklet for ankle OA screening."
)
_LOG = get_logger("fetlock.cli")


def _coerce(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            continue
    return value


def _overrides(items: List[str]) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise typer.BadParameter(f"override {item!r} must be key=value")
        key, raw = item.split("=", 1)
        parsed[key.strip()] = _coerce(raw.strip())
    return parsed


def _read(job: Path, overrides: List[str]) -> Config:
    config = load_config(job, _overrides(overrides))
    _LOG.info(
        "loaded %s stage=%s effective_batch=%d", job.name, config.stage, config.run.effective_batch
    )
    return config


JobOption = typer.Option(..., "--job", "-j", exists=True, dir_okay=False, help="experiment TOML")
SetOption = typer.Option([], "--set", "-s", help="dotted override, e.g. optim.lr=1e-6")


@app.command()
def pretrain(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    history = run_pretrain(config, pick_device())
    typer.echo(json.dumps({"final_loss": history[-1] if history else None}))


@app.command()
def adapt(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    history = run_pretrain(config, pick_device())
    typer.echo(json.dumps({"final_loss": history[-1] if history else None}))


@app.command()
def finetune(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    _, metrics, _ = run_finetune(config, pick_device())
    typer.echo(json.dumps(metrics))


@app.command()
def distill(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    _, metrics, size_kb = run_distill(config, pick_device())
    typer.echo(json.dumps({**metrics, "int8_size_kb": size_kb}))


@app.command()
def evaluate(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    typer.echo(json.dumps(run_evaluate(config, pick_device())))


@app.command()
def concordance(job: Path = JobOption, overrides: List[str] = SetOption) -> None:
    config = _read(job, overrides)
    cells = run_concordance(config)
    payload = [
        {
            "feature": cell.feature,
            "roi": cell.roi,
            "r": round(cell.r, 4),
            "p": round(cell.p_value, 4),
            "ci": [round(cell.ci_low, 4), round(cell.ci_high, 4)],
            "fdr": cell.fdr_reject,
        }
        for cell in cells
    ]
    typer.echo(json.dumps(payload, indent=2))


@app.command("export-onnx")
def export_onnx(
    job: Path = JobOption,
    out: Path = typer.Option(Path("artifacts/student.onnx"), "--out", "-o"),
    overrides: List[str] = SetOption,
) -> None:
    config = _read(job, overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(export_student_onnx(config, str(out)))


def run() -> None:
    app()


if __name__ == "__main__":
    run()
