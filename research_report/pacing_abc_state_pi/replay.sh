#!/usr/bin/env bash
set -euo pipefail
bundle="$(cd "$(dirname "$0")" && pwd)"
repo="$(git -C "$bundle" rev-parse --show-toplevel)"
cd "$repo"
mkdir -p outputs
run="$(mktemp -d "$repo/outputs/pacing_replay.XXXXXX")"
printf '재실행 폴더  %s\n' "$run"

"${PACING_PYTHON:-python3}" -m venv "$run/.venv"
py="$run/.venv/bin/python"
"$py" -m pip install --disable-pip-version-check \
  'numpy==2.3.5' 'scipy==1.17.0' 'scikit-learn==1.8.0' \
  'joblib==1.5.3' 'threadpoolctl==3.6.0'

cp -R "$bundle/code/abc/." "$run/"
mkdir -p "$run/experiment2.3IKJDN" "$run/experiment2_B"
cp "$bundle/code/C/experiment2.py" "$run/experiment2.3IKJDN/"
cp "$bundle/code/B/experiment2_b.py" "$run/experiment2_B/"

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$py" -u "$run/experiment2.3IKJDN/experiment2.py" \
  2>&1 | tee "$run/experiment2.3IKJDN/run.log"
"$py" -u "$run/experiment2_B/experiment2_b.py" \
  2>&1 | tee "$run/experiment2_B/run.log"
