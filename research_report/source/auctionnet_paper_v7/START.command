#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p outputs
if [ -f outputs/runner.pid ]; then
    previous_pid=$(cat outputs/runner.pid)
    if kill -0 "$previous_pid" 2>/dev/null; then
        echo "Already running: PID $previous_pid"
        exit 0
    fi
fi
# Use the existing project's Python only to create a separate environment.
if [ -x ../auctionnet_lab/.venv/bin/python ]; then
    base_python=../auctionnet_lab/.venv/bin/python
else
    base_python=python3
fi
if [ ! -x .venv/bin/python ]; then
    "$base_python" -m venv .venv
fi
echo "Installing dependencies; details: outputs/install.log"
.venv/bin/python -m pip install --disable-pip-version-check numpy==1.26.4 pandas==2.2.3 pyarrow==19.0.1 'torch>=2.2,<2.7' > outputs/install.log 2>&1
echo "Checking BC and IQL training..."
.venv/bin/python -u overnight.py --smoke > outputs/smoke.log 2>&1
if command -v caffeinate >/dev/null 2>&1; then
    nohup caffeinate -i .venv/bin/python -u overnight.py > outputs/overnight.log 2>&1 < /dev/null &
else
    nohup .venv/bin/python -u overnight.py > outputs/overnight.log 2>&1 < /dev/null &
fi
printf '%s\n' "$!" > outputs/runner.pid
echo "STARTED PID: $(cat outputs/runner.pid)"
echo "LOG: $(pwd)/outputs/overnight.log"
echo "Keep the Mac plugged in with the lid open."
