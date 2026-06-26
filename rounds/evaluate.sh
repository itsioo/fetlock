#!/usr/bin/env bash
set -euo pipefail

for JOB in dockets/jobs/main.toml \
           dockets/jobs/ablation_no_tl.toml \
           dockets/jobs/ablation_no_sex.toml \
           dockets/jobs/ablation_no_da.toml \
           dockets/jobs/ablation_accel_only.toml \
           dockets/jobs/ablation_gyro_only.toml; do
  echo "== ${JOB} =="
  fetlock evaluate --job "${JOB}"
done

fetlock concordance --job dockets/jobs/main.toml
