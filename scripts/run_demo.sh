#!/usr/bin/env bash
set -euo pipefail

# Runs the full workflow on the bundled breast cancer configuration.
# Everything is written under ./experiments and ./artifacts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/breast_cancer_rf.yaml"

autotuneml optimize --config "$CONFIG" --trials 25 "$@"
autotuneml train --config "$CONFIG" --from-experiment breast_cancer_rf
autotuneml evaluate breast_cancer_rf
autotuneml report breast_cancer_rf
autotuneml export breast_cancer_rf
autotuneml compare
