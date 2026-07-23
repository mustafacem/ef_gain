#!/bin/bash
# Sequential runner for the remaining experiment batch.
#
# Deliberately has NO `while pgrep ...` guards. An earlier version gated each
# stage on `pgrep -f "<script>.py"`, but the waiting shell's own command line
# contains that string, so pgrep matched the waiter itself and every wrapper
# blocked forever -- three of them sat deadlocked for ~3 hours with the GPU
# idle. Plain sequential execution gives the same ordering guarantee with
# nothing to race against.
#
# Lives in the repo rather than the scratchpad because the scratchpad is temp
# storage and was cleared between sessions.
#
# Usage:  setsid nohup scripts/run_queue.sh > /dev/null 2>&1 &
# Watch:  tail -f results/queue_progress.txt

set -u
cd /home/cm/Documents/selfquant
LOGDIR=results/logs
mkdir -p "$LOGDIR"
PROG=results/queue_progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True

run () {  # run <label> <outfile> <cmd...>
  local label=$1 out=$2; shift 2
  if [ -f "$out" ]; then
    echo "[$(date +%H:%M)] SKIP $label (already have $out)" >> "$PROG"
    return
  fi
  echo "[$(date +%H:%M)] START $label" >> "$PROG"
  if "$@" > "$LOGDIR/$label.log" 2>&1; then
    echo "[$(date +%H:%M)] OK    $label" >> "$PROG"
  else
    echo "[$(date +%H:%M)] FAIL  $label (see $LOGDIR/$label.log)" >> "$PROG"
  fi
}

echo "[$(date +%H:%M)] queue started" >> "$PROG"

run lowbit_pareto results/lowbit_pareto.json \
    python scripts/lowbit_pareto.py

run highband_pareto results/highband_pareto.json \
    env SQ_BASE_BITS=4 SQ_REF_BITS=5 SQ_PATCH_BITS=3,4 \
        SQ_OUT=results/highband_pareto.json python scripts/band_pareto.py

run bitmap_ab results/bitmap_ab.json \
    python scripts/bitmap_ab.py

run gsm8k_frontier results/gsm8k_frontier.json \
    python scripts/gsm8k_frontier.py

echo "[$(date +%H:%M)] ALL DONE" >> "$PROG"
