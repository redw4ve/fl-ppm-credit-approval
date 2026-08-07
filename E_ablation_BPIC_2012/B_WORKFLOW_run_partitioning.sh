#!/usr/bin/env bash
# CONTENT: Run the BPIC 2012 exploration, preprocessing and summary pipeline.
# Usage: ./B_WORKFLOW_run_partitioning.sh
# 1. Execute B_01_explore_bpic2012.ipynb
# 2. Execute B_02_preprocessing_and_partitioning_strat.py for the supported configs:
#    - 3-bank: iid, weak, medium  (no strong on BPIC 2012; no LoanGoal / offer attributes to anchor it)
# 3. Execute B_03_dataset_summary.py
# NOTE: HETEROGENEITY and N_CLIENTS passed via environment-variables. The constants block is used as fallback default.

# Strict mode fails on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve notebooks, scripts, logs and caches from the workflow directory.
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKFLOW_DIR"

NOTEBOOK="${NOTEBOOK:-B_01_explore_bpic2012.ipynb}"
SCRIPT="${SCRIPT:-B_02_preprocessing_and_partitioning_strat.py}"
PYTHON="${PYTHON:-../fl-ppm/bin/python}"

# Create directory for log files.
mkdir -p logs

# 1. Run BPIC 2012 exploration.
echo "[INFO] Running $NOTEBOOK"

# Execute a copy, so the notebook keeps the outputs it was handed in with. Fresh outputs go to build/ (git-ignored).
# Timeout is disabled because a full pass over the raw event log runs for several minutes.
mkdir -p build
"$PYTHON" -m nbconvert \
    --to notebook \
    --execute \
    --output-dir build \
    --output "B_01_explore_bpic2012.executed.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    "$NOTEBOOK" 2>&1 | tee "logs/B_01_explore_bpic2012.log"
echo "[INFO] Notebook complete"

# 2. Iterate over simulated banks
N=3
for CFG in iid weak medium; do
    echo "[INFO] Running $CFG with $N banks"

    # Pass config through environment variables, leave script constants unchanged
    HETEROGENEITY="$CFG" N_CLIENTS="$N" "$PYTHON" "$SCRIPT" 2>&1 \
        | tee "logs/B_02_${CFG}_${N}banks.log"
done

echo "[INFO] All configurations complete"

# 3. Run the BPIC 2012 summary script
echo "[INFO] Running analysis script"
"$PYTHON" B_03_dataset_summary.py
echo "[INFO] Analysis complete. Files saved to data/processed/ and plots/"