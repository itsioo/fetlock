import json

import attrs
import onnx
import torch
from typer.testing import CliRunner

from fetlock.farrier.app import app
from fetlock.farrier.stages import export_student_onnx, run_distill, run_finetune

runner = CliRunner()


def test_finetune_then_distill_runs(bench_config):
    device = torch.device("cpu")
    _, metrics, history = run_finetune(bench_config, device, max_steps=3)
    assert len(history) == 3
    assert set(metrics) == {"auc", "f1", "sensitivity", "specificity", "accuracy"}
    _, dmetrics, size_kb = run_distill(bench_config, device, max_steps=3)
    assert size_kb > 0
    assert 0.0 <= dmetrics["auc"] <= 1.0


def test_two_step_loss_is_finite(bench_config):
    config = attrs.evolve(bench_config, schedule=attrs.evolve(bench_config.schedule, epochs=1))
    _, _, history = run_finetune(config, torch.device("cpu"), max_steps=2)
    assert all(loss == loss for loss in history)
    assert history[0] < 1e3


def test_onnx_export_is_valid(bench_config, tmp_path):
    path = export_student_onnx(bench_config, str(tmp_path / "student.onnx"))
    model = onnx.load(path)
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "imu_window"


def test_cli_finetune_emits_json(bench_config, jobs_dir):
    result = runner.invoke(
        app,
        ["finetune", "--job", str(jobs_dir / "_bench.toml"), "--set", "schedule.epochs=1"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert "auc" in payload


def test_cli_concordance_reports_four_pairs(jobs_dir):
    result = runner.invoke(app, ["concordance", "--job", str(jobs_dir / "_bench.toml")])
    assert result.exit_code == 0
    payload = json.loads(result.stdout.strip())
    assert len(payload) == 4
