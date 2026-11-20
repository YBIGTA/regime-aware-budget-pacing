#!/bin/bash
set -euo pipefail

WORKSPACE="/Users/chanwoong/pacing_two_state_20261114"
BUNDLE="$WORKSPACE/research_report/pacing_prediction_residual_correction_20261120"
cd "$WORKSPACE"
"$WORKSPACE/.venv/bin/python" "$BUNDLE/code/run_experiment.py" all 2>&1 | tee -a "$BUNDLE/logs/screen.log"
