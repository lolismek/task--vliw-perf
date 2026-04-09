#!/usr/bin/env bash
set -euo pipefail

BASELINE=147734

# --- Anti-cheat: verify protected files are unmodified ---
EXPECTED_HASH="81ed3af09bd63468b14ec1aa531eac84f2c89499876c2c93dce22b817d960e38"
ACTUAL_HASH=$(cat tests/submission_tests.py tests/frozen_problem.py problem.py | shasum -a 256 | awk '{print $1}')
if [ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]; then
  echo "WARNING: Protected files have been modified. Restoring originals."
  git checkout origin/main -- tests/ problem.py 2>/dev/null || true
fi

# --- Run submission tests ---
OUTPUT=$(python3 tests/submission_tests.py 2>&1) || true

# --- Parse cycle count (portable sed, no grep -P) ---
CYCLES=$(echo "$OUTPUT" | sed -n 's/.*CYCLES:[[:space:]]*\([0-9]*\).*/\1/p' | tail -1)
if [ -z "$CYCLES" ]; then
  echo "ERROR: Could not parse cycle count from test output"
  echo "$OUTPUT" | tail -20
  CYCLES=$((BASELINE * 2))
fi

# --- Compute speedup ---
SPEEDUP=$(python3 -c "print(f'{$BASELINE / $CYCLES:.2f}')")

# --- Count passing tests from unittest output ---
RAN=$(echo "$OUTPUT" | sed -n 's/.*Ran \([0-9]*\).*/\1/p' | tail -1)
FAILS=$(echo "$OUTPUT" | sed -n 's/.*failures=\([0-9]*\).*/\1/p')
ERRS=$(echo "$OUTPUT" | sed -n 's/.*errors=\([0-9]*\).*/\1/p')
FAILS=${FAILS:-0}
ERRS=${ERRS:-0}
RAN=${RAN:-0}
PASSED=$((RAN - FAILS - ERRS))

# --- Output in Hive format ---
echo "---"
echo "cycles:           $CYCLES"
echo "speedup:          $SPEEDUP"
echo "tests_passed:     ${PASSED}/${RAN}"
