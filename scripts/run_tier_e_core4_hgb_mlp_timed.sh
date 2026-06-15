#!/usr/bin/env bash
set -euo pipefail

TIMING_DIR="results/timing/tier_e_core4_hgb_mlp_timed"
TIME_OUT="${TIMING_DIR}/orchestrator_time.txt"
mkdir -p "${TIMING_DIR}"

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
TIME_BIN="${RAISE_ICT_TIME_BIN:-/usr/bin/time}"

if [[ -x "${TIME_BIN}" ]]; then
  "${TIME_BIN}" -v \
    -o "${TIME_OUT}" \
    python scripts/run_tier_e_core4_hgb_mlp_timed.py "$@"
else
  echo "Warning: ${TIME_BIN} not found; using shell wall-clock fallback." >&2
  start_epoch="$(date +%s)"
  start_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  set +e
  python scripts/run_tier_e_core4_hgb_mlp_timed.py "$@"
  status="$?"
  set -e
  end_epoch="$(date +%s)"
  end_iso="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  elapsed_s="$((end_epoch - start_epoch))"
  {
    echo "Timing mode: shell_wall_clock_fallback"
    echo "Command: python scripts/run_tier_e_core4_hgb_mlp_timed.py $*"
    echo "Start UTC: ${start_iso}"
    echo "End UTC: ${end_iso}"
    echo "Elapsed wall-clock seconds: ${elapsed_s}"
    echo "Exit status: ${status}"
    echo "Note: GNU time was unavailable at ${TIME_BIN}, so max RSS and verbose resource counters were not recorded."
  } > "${TIME_OUT}"
  exit "${status}"
fi
