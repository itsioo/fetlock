#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-data/most}"

fetlock pretrain  --job dockets/jobs/pretrain.toml
fetlock adapt     --job dockets/jobs/adapt.toml    --set data_root="${ROOT}"
fetlock finetune  --job dockets/jobs/main.toml     --set data_root="${ROOT}"
fetlock distill   --job dockets/jobs/distill.toml  --set data_root="${ROOT}"
fetlock export-onnx --job dockets/jobs/distill.toml --out artifacts/anklet_student.onnx
