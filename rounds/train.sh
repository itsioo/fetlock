#!/usr/bin/env bash
set -euo pipefail

JOB="${1:-dockets/jobs/main.toml}"
NPROC="${NPROC:-8}"

torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="${NPROC}" \
  -m fetlock.farrier finetune --job "${JOB}"
