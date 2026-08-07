#!/usr/bin/env bash
# CONTENT: Run the E_05 centralized and local baseline matrix, including the joint cross-dataset runs.
#   1. Resolve the selected production config and cache locations.
#   2. Run one selected baseline when SINGLE_RUN_MODE=true.
#   3. Run centralized and local baselines for all BPIC 2017, BPIC 2012 and joint configs.
# NOTE: A bare run uses raw/zscore/beta 0.1 with reg_medium outcome regularization.

# Strict mode fails on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve outputs, cache and scripts from the E_training workflow directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION

PYTHON_BIN="${PYTHON_BIN:-../fl-ppm/bin/python}"                      # Path location of the environment
SEED="${SEED:-42}"                                                    # Seed for all experiments (standard 42)
DEVICE="${DEVICE:-auto}"                                              # Auto prefers CUDA, then MPS, then CPU.
PROGRESS_BARS="${PROGRESS_BARS:-true}"                                # Always turn on progress bars
REPORTING_PROFILE="${REPORTING_PROFILE:-compact}"                     # Compact is the production reporting layout
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_outputs}"   # Output locations
CACHE_ROOT="${CACHE_ROOT:-$SCRIPT_DIR/prefix_tensor_cache}"           # Prefix tensor caches (outside output folder)
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/../E_prefix_encoding/encoded_metadata}"  # E_04 metadata root
RESILIENT="${RESILIENT:-false}"                                       # When true, a failed run is logged and skipped.
FAILED_RUNS_LOG="${FAILED_RUNS_LOG:-}"                                # Optional file to append labels of skipped failed runs
DRY_RUN="${DRY_RUN:-false}"                                           # Preview planned runs without executing

# Workflow mode: SINGLE RUN -> Can be used for testing a single configuration, default false.
SINGLE_RUN_MODE="${SINGLE_RUN_MODE:-false}"                           # Default is false
DATASET="${DATASET:-bpic2017}"                                        # Default dataset is BPIC2017
HETEROGENEITY="${HETEROGENEITY:-medium}"                              # Default heterogeneity is medium
N_CLIENTS="${N_CLIENTS:-3}"                                           # Default number of clients is 3
REGIME="${REGIME:-centralized}"                                       # Default regime is centralized
BANK="${BANK:-}"                                                      # Optional local-bank identifier

# Optimizer defaults (trial-and-error baseline tuning).
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"                              # AdamW learning rate
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"                                  # AdamW weight decay
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"                       # Maximum gradient norm before optimizer step

# Capture explicit schedule overrides before defaulting so the per-family resolver can respect them.
LR_SCHEDULER_T_MAX_OVERRIDE="${LR_SCHEDULER_T_MAX:-}"                 # Empty unless the caller pinned t_max
LR_SCHEDULER_MIN_LR_OVERRIDE="${LR_SCHEDULER_MIN_LR:-}"               # Empty unless the caller pinned min_lr

# LR schedule decays over the cosine period, then holds the floor.
# The family resolver overrides t_max and min_lr per dataset family.
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"                                # Scheduler type
LR_SCHEDULER_MIN_LR="${LR_SCHEDULER_MIN_LR:-1e-6}"                    # BPIC 2017 floor; resolver sets 1e-5 elsewhere
LR_SCHEDULER_T_MAX="${LR_SCHEDULER_T_MAX:-15}"                        # BPIC 2017 period; resolver sets 35 elsewhere

# Training loop defaults, one uniform budget and patience so early stopping and not the budget terminates every run.
MAX_EPOCHS="${MAX_EPOCHS:-40}"                                        # Uniform baseline budget
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-7}"               # Uniform patience
BATCH_SIZE="${BATCH_SIZE:-512}"                                       # Prefix samples per batch
NUM_WORKERS="${NUM_WORKERS:-0}"                                       # DataLoader worker processes

# Model architecture defaults.
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"                                     # LSTM hidden size
NUM_LAYERS="${NUM_LAYERS:-2}"                                         # Number of LSTM layers
DROPOUT="${DROPOUT:-0.30}"                                            # Dropout probability
HEAD_HIDDEN_SIZE="${HEAD_HIDDEN_SIZE:-64}"                            # Hidden size of each prediction head

# Loss and task-head LR defaults.
OUTCOME_LABEL_SMOOTHING="${OUTCOME_LABEL_SMOOTHING:-0.10}"            # Outcome CE label smoothing
OUTCOME_CLASS_WEIGHT_POWER="${OUTCOME_CLASS_WEIGHT_POWER:-0.5}"       # Tempered outcome class weight strength
OUTCOME_LR_SCALE="${OUTCOME_LR_SCALE:-0.3}"                           # Outcome head LR multiplier
NEXT_ACTIVITY_LR_SCALE="${NEXT_ACTIVITY_LR_SCALE:-1.0}"               # Next activity head LR multiplier
REMAINING_TIME_LR_SCALE="${REMAINING_TIME_LR_SCALE:-1.0}"             # RT head LR multiplier
OUTCOME_LOSS_WEIGHT="${OUTCOME_LOSS_WEIGHT:-1.0}"                     # Outcome loss weight
NEXT_ACTIVITY_LOSS_WEIGHT="${NEXT_ACTIVITY_LOSS_WEIGHT:-0.5}"         # Next activity loss weight
REMAINING_TIME_LOSS_WEIGHT="${REMAINING_TIME_LOSS_WEIGHT:-0.5}"       # RT loss weight

# RT target defaults: raw plus zscore encoded by E_04 and Huber beta leaning towards MAE.
REMAINING_TIME_TRANSFORM="${REMAINING_TIME_TRANSFORM:-raw}"           # RT target transform
REMAINING_TIME_SCALING="${REMAINING_TIME_SCALING:-zscore}"            # RT target scaling
REMAINING_TIME_HUBER_BETA="${REMAINING_TIME_HUBER_BETA:-0.1}"         # Huber beta for RT loss

# Outcome head dropout default, with empty OUTCOME_HEAD_DROPOUT disabling the override.
OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT-0.45}"                  # Extra dropout for the outcome head

# ----------------------------------------------------------------------------------------------------------------------
# 1. WORKFLOW HELPERS

# HELPER: Resolve the per-family cosine schedule unless the caller pinned it explicitly.
resolve_family_schedule() {
  local dataset="$1"
  if [[ -z "$LR_SCHEDULER_T_MAX_OVERRIDE" ]]; then
    case "$dataset" in
      bpic2017) LR_SCHEDULER_T_MAX=15 ;;
      bpic2012|joint) LR_SCHEDULER_T_MAX=35 ;;
      *) echo "Unknown dataset family for schedule: ${dataset}" >&2; exit 1 ;;
    esac
  fi
  if [[ -z "$LR_SCHEDULER_MIN_LR_OVERRIDE" ]]; then
    case "$dataset" in
      bpic2017) LR_SCHEDULER_MIN_LR=1e-6 ;;
      bpic2012|joint) LR_SCHEDULER_MIN_LR=1e-5 ;;
    esac
  fi
}

# HELPER: Run the E_05 entry point with the resolved workflow environment.
run_python() {
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY: DATASET=${DATASET} HETEROGENEITY=${HETEROGENEITY} N_CLIENTS=${N_CLIENTS} REGIME=${REGIME} BANK=${BANK}"
    echo "    DRY: SCHEDULE dataset=${DATASET} MAX_EPOCHS=${MAX_EPOCHS} T_MAX=${LR_SCHEDULER_T_MAX}" \
      "MIN_LR=${LR_SCHEDULER_MIN_LR} LR=${LEARNING_RATE} BATCH=${BATCH_SIZE} PATIENCE=${EARLY_STOPPING_PATIENCE}"
    echo "    DRY: REMAINING_TIME_TRANSFORM=${REMAINING_TIME_TRANSFORM}" \
      "REMAINING_TIME_SCALING=${REMAINING_TIME_SCALING} REMAINING_TIME_HUBER_BETA=${REMAINING_TIME_HUBER_BETA}"
    echo "    DRY: REPORTING_PROFILE=${REPORTING_PROFILE}"
    return 0
  fi
  DATASET="${DATASET}" \
  HETEROGENEITY="${HETEROGENEITY}" \
  N_CLIENTS="${N_CLIENTS}" \
  REGIME="${REGIME}" \
  BANK="${BANK}" \
  SEED="${SEED}" \
  DEVICE="${DEVICE}" \
  PROGRESS_BARS="${PROGRESS_BARS}" \
  LEARNING_RATE="${LEARNING_RATE}" \
  WEIGHT_DECAY="${WEIGHT_DECAY}" \
  GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM}" \
  LR_SCHEDULER="${LR_SCHEDULER}" \
  LR_SCHEDULER_MIN_LR="${LR_SCHEDULER_MIN_LR}" \
  LR_SCHEDULER_T_MAX="${LR_SCHEDULER_T_MAX}" \
  MAX_EPOCHS="${MAX_EPOCHS}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  NUM_WORKERS="${NUM_WORKERS}" \
  HIDDEN_SIZE="${HIDDEN_SIZE}" \
  NUM_LAYERS="${NUM_LAYERS}" \
  DROPOUT="${DROPOUT}" \
  HEAD_HIDDEN_SIZE="${HEAD_HIDDEN_SIZE}" \
  OUTCOME_LABEL_SMOOTHING="${OUTCOME_LABEL_SMOOTHING}" \
  OUTCOME_CLASS_WEIGHT_POWER="${OUTCOME_CLASS_WEIGHT_POWER}" \
  OUTCOME_LR_SCALE="${OUTCOME_LR_SCALE}" \
  NEXT_ACTIVITY_LR_SCALE="${NEXT_ACTIVITY_LR_SCALE}" \
  REMAINING_TIME_LR_SCALE="${REMAINING_TIME_LR_SCALE}" \
  OUTCOME_LOSS_WEIGHT="${OUTCOME_LOSS_WEIGHT}" \
  NEXT_ACTIVITY_LOSS_WEIGHT="${NEXT_ACTIVITY_LOSS_WEIGHT}" \
  REMAINING_TIME_LOSS_WEIGHT="${REMAINING_TIME_LOSS_WEIGHT}" \
  REMAINING_TIME_TRANSFORM="${REMAINING_TIME_TRANSFORM}" \
  REMAINING_TIME_SCALING="${REMAINING_TIME_SCALING}" \
  REMAINING_TIME_HUBER_BETA="${REMAINING_TIME_HUBER_BETA}" \
  OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT}" \
  "$PYTHON_BIN" -B E_05_central_and_local_baselines_final.py \
    --output-root "$OUTPUT_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --reporting-profile "$REPORTING_PROFILE"
}

# HELPER: Run one centralized or local E_05 baseline.
run_one() {
  DATASET="$1"
  HETEROGENEITY="$2"
  N_CLIENTS="$3"
  REGIME="$4"
  BANK="${5:-}"
  resolve_family_schedule "$DATASET"

  local label="E_05 ${DATASET} ${HETEROGENEITY}_${N_CLIENTS}banks ${REGIME} ${BANK} lr=${LEARNING_RATE}"
  echo "${label} out=${OUTPUT_ROOT}"
  if [[ "$RESILIENT" == "true" ]]; then
    if ! run_python; then
      echo "WARNING: skipped failed run: ${label}" >&2
      if [[ -n "$FAILED_RUNS_LOG" ]]; then echo "${label}" >> "$FAILED_RUNS_LOG"; fi
      return 0
    fi
  else
    run_python
  fi
}

# HELPER: Run the centralized baseline and all local bank baselines for one split.
run_matrix_entry() {
  local dataset="$1"
  local heterogeneity="$2"
  local n_clients="$3"
  local banks="$4"

  run_one "$dataset" "$heterogeneity" "$n_clients" "centralized"
  for bank in $banks; do
    run_one "$dataset" "$heterogeneity" "$n_clients" "local" "$bank"
  done
}

# ----------------------------------------------------------------------------------------------------------------------
# 2. SINGLE RUN MODE: Run only the selected E_05 configuration when requested.

if [[ "$SINGLE_RUN_MODE" == "true" ]]; then
  run_one "$DATASET" "$HETEROGENEITY" "$N_CLIENTS" "$REGIME" "$BANK"
  exit 0
fi

# ----------------------------------------------------------------------------------------------------------------------
# 3. BPIC 2017 MAIN MATRIX: Run the BPIC 2017 main experiment matrix.

run_matrix_entry "bpic2017" "iid" "3" "A B C"
run_matrix_entry "bpic2017" "weak" "3" "A B C"
run_matrix_entry "bpic2017" "medium" "3" "A B C"
run_matrix_entry "bpic2017" "strong" "3" "A B C"
run_matrix_entry "bpic2017" "medium" "5" "A B C D E"
run_matrix_entry "bpic2017" "strong" "5" "A B C D E"

# ----------------------------------------------------------------------------------------------------------------------
# 4. BPIC 2012 ABLATION MATRIX: Run the BPIC 2012 ablation matrix.

run_matrix_entry "bpic2012" "iid" "3" "A B C"
run_matrix_entry "bpic2012" "weak" "3" "A B C"
run_matrix_entry "bpic2012" "medium" "3" "A B C"

# ----------------------------------------------------------------------------------------------------------------------
# 5. JOINT CROSS-DATASET MATRIX: Run the joint BPIC 2017 plus BPIC 2012 baselines with dataset-qualified banks.

run_matrix_entry "joint" "iid" "6" "bpic2017:A bpic2017:B bpic2017:C bpic2012:A bpic2012:B bpic2012:C"
run_matrix_entry "joint" "weak" "6" "bpic2017:A bpic2017:B bpic2017:C bpic2012:A bpic2012:B bpic2012:C"
run_matrix_entry "joint" "medium" "6" "bpic2017:A bpic2017:B bpic2017:C bpic2012:A bpic2012:B bpic2012:C"
run_matrix_entry "joint" "medium" "8" \
  "bpic2017:A bpic2017:B bpic2017:C bpic2017:D bpic2017:E bpic2012:A bpic2012:B bpic2012:C"