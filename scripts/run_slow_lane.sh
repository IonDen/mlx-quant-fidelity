#!/usr/bin/env bash
# Run the slow lane (real-model oracles, network-marked tests) and keep a JUnit file plus a log.
#
# These tests load real models: run on a Mac that is on AC power, awake, with the GPU idle and
# enough free memory. Never run two heavy jobs at once.
#
# Usage: scripts/run_slow_lane.sh [junit-xml-path] [log-path]
#   defaults: slow-lane.xml and slow-lane.log in the current directory.
set -euo pipefail

cd "$(dirname "$0")/.."

# Optional pre-flight check of your own (AC power, free GPU, awake): point
# MQF_SLOW_LANE_PREFLIGHT at an executable and it runs first; a non-zero exit stops the lane.
preflight="${MQF_SLOW_LANE_PREFLIGHT:-}"
if [ -n "$preflight" ]; then
  "$preflight"
else
  echo "note: no pre-flight set; check yourself that the Mac is on AC power, awake," >&2
  echo "      and that no other model-loading job is running." >&2
fi

uv run --no-sync pytest --run-slow --run-network -q --junitxml "${1:-slow-lane.xml}" 2>&1 | tee "${2:-slow-lane.log}"
