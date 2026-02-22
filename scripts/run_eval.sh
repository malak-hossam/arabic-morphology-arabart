#!/usr/bin/env bash
set -euo pipefail

python -m src.pipelines.eval_pipeline "$@"

