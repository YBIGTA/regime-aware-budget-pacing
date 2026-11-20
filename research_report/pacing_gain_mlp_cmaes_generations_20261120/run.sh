#!/bin/bash
set -euo pipefail

WORKSPACE="/Users/chanwoong/pacing_two_state_20261114"
PYTHON="$WORKSPACE/.venv/bin/python"
SCRIPT="$WORKSPACE/research_report/pacing_gain_mlp_cmaes_generations_20261120/code/run_experiment.py"
LOG="$WORKSPACE/research_report/pacing_gain_mlp_cmaes_generations_20261120/logs/screen.log"

cd "$WORKSPACE"
"$PYTHON" "$SCRIPT" all 2>&1 | tee -a "$LOG"
