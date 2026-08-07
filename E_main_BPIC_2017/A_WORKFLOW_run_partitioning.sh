#!/usr/bin/env bash
# CONTENT: Run the BPIC 2017 exploration, preprocessing and summary pipeline. Usage: ./A_WORKFLOW_run_partitioning.sh
# 1. Execute A_01_explore_bpic2017.ipynb
# 2. Execute A_02_preprocessing_and_partitioning_strat.py for 6 configs:
#    - 3-bank: iid, weak, medium, strong
#    - 5-bank: medium, strong (+ specialist banks D and E)
# 3. Execute A_03_dataset_summary.py
# NOTE: HETEROGENEITY and N_CLIENTS passed via env-vars. Constants block used as fallback default.

# Strict mode fails on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve notebooks, scripts, logs and caches from the workflow directory.
WORKFLOW_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKFLOW_DIR"

NOTEBOOK="${NOTEBOOK:-A_01_explore_bpic2017.ipynb}"
SCRIPT="${SCRIPT:-A_02_preprocessing_and_partitioning_strat.py}"
PYTHON="${PYTHON:-../fl-ppm/bin/python}"

# Create directory for log files.
mkdir -p logs

# 1. Run BPIC 2017 exploration.
echo "[INFO] Running $NOTEBOOK"

# Execute a copy, so the notebook keeps the outputs it was handed in with. Fresh outputs go to build/ (git-ignored).
# Timeout is disabled because a full pass over the raw event log runs for several minutes.
mkdir -p build
"$PYTHON" -m nbconvert \
    --to notebook \
    --execute \
    --output-dir build \
    --output "A_01_explore_bpic2017.executed.ipynb" \
    --ExecutePreprocessor.timeout=-1 \
    "$NOTEBOOK" 2>&1 | tee "logs/A_01_explore_bpic2017.log"
echo "[INFO] Notebook complete"

# 2. Iterate over simulated banks.
for N in 3 5; do
    # Iterate over heterogeneity levels.
    for CFG in iid weak medium strong; do
        # Skip designs outside the experiment matrix.
        if [[ $N -eq 5 && ($CFG == "iid" || $CFG == "weak") ]]; then
            continue
        fi

        echo "[INFO] Running $CFG with $N banks"

        # Pass config through environment variables, leave script constants unchanged.
        HETEROGENEITY="$CFG" N_CLIENTS="$N" "$PYTHON" "$SCRIPT" 2>&1 \
            | tee "logs/A_02_${CFG}_${N}banks.log"
    done
done

echo "[INFO] All configurations complete"

# 3. Run the BPIC 2017 summary script.
echo "[INFO] Running analysis script"
"$PYTHON" A_03_dataset_summary.py
echo "[INFO] Analysis complete. Files saved to data/processed/ and plots/"