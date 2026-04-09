#!/usr/bin/env bash
set -euo pipefail

echo "=== Preparing VLIW Performance Task ==="

# Verify Python is available
python3 --version || { echo "ERROR: Python 3 required"; exit 1; }

# Install requirements if any
if [ -f requirements.txt ] && [ -s requirements.txt ]; then
  pip install -q -r requirements.txt
fi

# Verify critical files exist
for f in perf_takehome.py problem.py tests/submission_tests.py tests/frozen_problem.py; do
  if [ ! -f "$f" ]; then
    echo "ERROR: Missing required file: $f"
    exit 1
  fi
done

# Quick sanity check: verify KernelBuilder imports and builds
echo "Running sanity check..."
python3 -c "
from perf_takehome import KernelBuilder
kb = KernelBuilder()
kb.build_kernel(10, 2047, 256, 16)
print(f'Instructions: {len(kb.instrs)}')
print('KernelBuilder OK')
"

echo "=== Preparation complete ==="
