#!/usr/bin/env bash
# CONTENT: Run the E_06 federated DP-SGD epsilon experiment. Usage: ./WORKFLOW_run_federated_dp_final.sh.
#   1. Resolve the production config, the DP budget and cache locations.
#   2. Run one DP configuration when SINGLE_RUN_MODE=true.
#   3. Sweep 3 splits and four epsilon levels with FedProx, for 12 production runs against the no-DP infinity baselines.
# NOTE: This is the separate DP experiment, kept out of the no-DP nightly in WORKFLOW_run_federated_final.sh.
#       It inherits the same baseline hyperparameters, so only the privacy path differs.
#       E_06 writes DP outputs under OUTPUT_ROOT/differential_privacy/.
#       Default DEVICE is auto, which prefers CUDA.
#       The 2026 DP check confirmed MPS and CPU parity on achieved epsilon, noise and utility.
#       The production DP experiment is FedProx only, with 12 runs at the fixed FEDPROX_MU=1e-4.
#       This workflow is the single source of truth for the DP device, strategy and mu.

# Fail on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve outputs, cache and scripts from the E_training workflow directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION

PYTHON_BIN="${PYTHON_BIN:-../fl-ppm/bin/python}"                      # Path location of the environment.
SEED="${SEED:-42}"                                                    # Seed for all experiments (fixed 42).
DEVICE="${DEVICE:-auto}"                                              # Auto prefers CUDA; the runbook passes cuda.
PROGRESS_BARS="${PROGRESS_BARS:-true}"                                # Progress bar switch (true or false).
REPORTING_PROFILE="${REPORTING_PROFILE:-compact}"                     # Compact is the production reporting layout.
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_outputs}"   # Output locations.
CACHE_ROOT="${CACHE_ROOT:-$SCRIPT_DIR/prefix_tensor_cache}"           # Prefix tensor caches (outside output folder).
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/../E_prefix_encoding/encoded_metadata}"  # E_04 metadata root.
RESILIENT="${RESILIENT:-false}"                                       # true -> failed run is logged and skipped.
FAILED_RUNS_LOG="${FAILED_RUNS_LOG:-}"                                # Optional file to append labels of skipped runs.
DRY_RUN="${DRY_RUN:-false}"                                           # Preview planned runs without executing.

# Select the workflow mode and one split for single-run checks.
SINGLE_RUN_MODE="${SINGLE_RUN_MODE:-false}"                           # Default is false.
DATASET="${DATASET:-bpic2017}"                                        # Default dataset is BPIC 2017.
HETEROGENEITY="${HETEROGENEITY:-medium}"                              # Default heterogeneity is medium.
N_CLIENTS="${N_CLIENTS:-3}"                                           # Default number of clients is 3.
# The production DP experiment defaults to FedProx for 12 runs; an explicit both still runs 24.
STRATEGY="${STRATEGY:-fedprox}"
# The DP grid is single-dataset BPIC 2017 with no joint runs, so head aggregation is sample and equal would be a no-op.
NEXT_ACTIVITY_HEAD_AGG="${NEXT_ACTIVITY_HEAD_AGG:-sample}"

# Configure federated rounds, local epochs and the DP-SGD path that is always on here.
MAX_ROUNDS="${MAX_ROUNDS:-40}"                                        # Uniform federated budget.
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-7}"               # Noisy DP losses make short patience unreliable.
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"                                     # Local epochs per client fit.
FEDPROX_MU="${FEDPROX_MU:-1e-4}"                                      # Fixed production FedProx proximal strength.
USE_DP="${USE_DP:-true}"                                              # DP-SGD is forced on for this experiment.
DP_TARGET_EPSILON="${DP_TARGET_EPSILON:-10.0}"                        # Per-client target epsilon for single-run mode.
DP_DELTA="${DP_DELTA:-1e-6}"                                          # Per-client target delta (7.2e-6 at 139k pref.).
DP_MAX_GRAD_NORM="${DP_MAX_GRAD_NORM:-1.0}"                           # Clipping norm per sample.
DP_SMOKE_MAX_BATCHES="${DP_SMOKE_MAX_BATCHES:-}"                      # Optional DP batch cap, empty means full pass.

# Configure the limited DP epsilon sweep that keeps the result set manageable.
DP_EXPERIMENT_SPLITS="${DP_EXPERIMENT_SPLITS:-bpic2017:iid:3 bpic2017:medium:3 bpic2017:strong:3}"
DP_EXPERIMENT_EPSILONS="${DP_EXPERIMENT_EPSILONS:-1 5 10 50}"         # Thesis epsilon grid.

# Configure optimizer defaults from the final E_05 workflow.
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"                              # AdamW base learning rate.
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"                                  # AdamW weight decay.
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"                       # Ignored under DP, where Opacus owns clipping.

# Use the long schedule for every DP run because DP noise slows convergence.
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"                                # Scheduler type.
LR_SCHEDULER_MIN_LR="${LR_SCHEDULER_MIN_LR:-1e-5}"                    # Long-schedule floor for every DP run.
LR_SCHEDULER_T_MAX="${LR_SCHEDULER_T_MAX:-35}"                        # Long-schedule period for every DP run.

# Configure local data loading and model architecture.
BATCH_SIZE="${BATCH_SIZE:-512}"                                       # Prefix samples per batch.
NUM_WORKERS="${NUM_WORKERS:-0}"                                       # Worker processes, 0 is stable on macOS.
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"                                     # LSTM hidden size.
NUM_LAYERS="${NUM_LAYERS:-2}"                                         # Number of LSTM layers.
DROPOUT="${DROPOUT:-0.30}"                                            # Shared trunk dropout.
HEAD_HIDDEN_SIZE="${HEAD_HIDDEN_SIZE:-64}"                            # Hidden size of each prediction head.

# Configure loss weights and task head LR multipliers.
OUTCOME_LABEL_SMOOTHING="${OUTCOME_LABEL_SMOOTHING:-0.10}"            # Outcome CE label smoothing.
OUTCOME_CLASS_WEIGHT_POWER="${OUTCOME_CLASS_WEIGHT_POWER:-0.5}"       # Tempered outcome class weight strength.
OUTCOME_LR_SCALE="${OUTCOME_LR_SCALE:-0.3}"                           # Outcome head LR multiplier.
OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT-0.45}"                   # Extra dropout for the outcome head.
NEXT_ACTIVITY_LR_SCALE="${NEXT_ACTIVITY_LR_SCALE:-1.0}"               # NA head LR multiplier.
REMAINING_TIME_LR_SCALE="${REMAINING_TIME_LR_SCALE:-1.0}"             # RT head LR multiplier.
OUTCOME_LOSS_WEIGHT="${OUTCOME_LOSS_WEIGHT:-1.0}"                     # Outcome loss contribution.
NEXT_ACTIVITY_LOSS_WEIGHT="${NEXT_ACTIVITY_LOSS_WEIGHT:-0.5}"         # Next-activity loss contribution.
REMAINING_TIME_LOSS_WEIGHT="${REMAINING_TIME_LOSS_WEIGHT:-0.5}"       # RT loss contribution.

# Configure the final remaining-time representation and Huber switch point.
REMAINING_TIME_TRANSFORM="${REMAINING_TIME_TRANSFORM:-raw}"           # RT target transform.
REMAINING_TIME_SCALING="${REMAINING_TIME_SCALING:-zscore}"            # RT target scaling.
REMAINING_TIME_HUBER_BETA="${REMAINING_TIME_HUBER_BETA:-0.1}"         # RT Huber beta.

# ----------------------------------------------------------------------------------------------------------------------
# 1. WORKFLOW HELPERS

# HELPER: Build optional CLI switches that cannot be represented by a scalar value.
optional_args() {
  OPTIONAL_ARGS=()
  if [[ "$PROGRESS_BARS" != "true" ]]; then OPTIONAL_ARGS+=(--no-progress-bars); fi
  OPTIONAL_ARGS+=(--use-dp)
  if [[ -n "$DP_SMOKE_MAX_BATCHES" ]]; then OPTIONAL_ARGS+=(--dp-smoke-max-batches "$DP_SMOKE_MAX_BATCHES"); fi
}

# HELPER: Reject non-DP runs and invalid strategies before the DP experiment starts.
guard_dp_workflow() {
  if [[ "$USE_DP" != "true" ]]; then
    echo "Refusing WORKFLOW_run_federated_dp_final.sh with USE_DP=${USE_DP}." >&2
    echo "This workflow is DP-only and always passes --use-dp." >&2
    exit 1
  fi
  if [[ "$STRATEGY" != "fedavg" && "$STRATEGY" != "fedprox" && "$STRATEGY" != "both" ]]; then
    echo "Refusing DP workflow with STRATEGY=${STRATEGY}. Use fedavg, fedprox or both." >&2
    exit 1
  fi
}

# HELPER: Resolve the strategy list, expanding the both keyword to FedAvg first and then FedProx.
strategies_to_run() {
  if [[ "$STRATEGY" == "both" ]]; then echo "fedavg fedprox"; else echo "$STRATEGY"; fi
}

# HELPER: Run the E_06 entry point with the resolved workflow environment.
run_python() {
  optional_args
  # Preview the planned run without loading torch or starting training.
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY: DATASET=${DATASET} HETEROGENEITY=${HETEROGENEITY} N_CLIENTS=${N_CLIENTS}"
    echo "    DRY: STRATEGY=${STRATEGY} DP_TARGET_EPSILON=${DP_TARGET_EPSILON}"
    echo "    DRY: SCHEDULE MAX_ROUNDS=${MAX_ROUNDS} T_MAX=${LR_SCHEDULER_T_MAX} MIN_LR=${LR_SCHEDULER_MIN_LR}"
    echo "    DRY: PATIENCE=${EARLY_STOPPING_PATIENCE} REPORTING_PROFILE=${REPORTING_PROFILE}"
    return 0
  fi
  OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT}" \
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  "$PYTHON_BIN" -B E_06_federated_training.py \
    --dataset "$DATASET" \
    --heterogeneity "$HETEROGENEITY" \
    --n-clients "$N_CLIENTS" \
    --strategy "$STRATEGY" \
    --next-activity-head-agg "$NEXT_ACTIVITY_HEAD_AGG" \
    --max-rounds "$MAX_ROUNDS" \
    --local-epochs "$LOCAL_EPOCHS" \
    --fedprox-mu "$FEDPROX_MU" \
    --dp-target-epsilon "$DP_TARGET_EPSILON" \
    --dp-delta "$DP_DELTA" \
    --dp-max-grad-norm "$DP_MAX_GRAD_NORM" \
    --seed "$SEED" \
    --device "$DEVICE" \
    --learning-rate "$LEARNING_RATE" \
    --weight-decay "$WEIGHT_DECAY" \
    --gradient-clip-norm "$GRADIENT_CLIP_NORM" \
    --lr-scheduler "$LR_SCHEDULER" \
    --lr-scheduler-min-lr "$LR_SCHEDULER_MIN_LR" \
    --lr-scheduler-t-max "$LR_SCHEDULER_T_MAX" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --hidden-size "$HIDDEN_SIZE" \
    --num-layers "$NUM_LAYERS" \
    --dropout "$DROPOUT" \
    --head-hidden-size "$HEAD_HIDDEN_SIZE" \
    --outcome-label-smoothing "$OUTCOME_LABEL_SMOOTHING" \
    --outcome-class-weight-power "$OUTCOME_CLASS_WEIGHT_POWER" \
    --outcome-lr-scale "$OUTCOME_LR_SCALE" \
    --next-activity-lr-scale "$NEXT_ACTIVITY_LR_SCALE" \
    --remaining-time-lr-scale "$REMAINING_TIME_LR_SCALE" \
    --outcome-loss-weight "$OUTCOME_LOSS_WEIGHT" \
    --next-activity-loss-weight "$NEXT_ACTIVITY_LOSS_WEIGHT" \
    --remaining-time-loss-weight "$REMAINING_TIME_LOSS_WEIGHT" \
    --remaining-time-transform "$REMAINING_TIME_TRANSFORM" \
    --remaining-time-scaling "$REMAINING_TIME_SCALING" \
    --remaining-time-huber-beta "$REMAINING_TIME_HUBER_BETA" \
    --output-root "$OUTPUT_ROOT" \
    --cache-root "$CACHE_ROOT" \
    --artifact-root "$ARTIFACT_ROOT" \
    --reporting-profile "$REPORTING_PROFILE" \
    "${OPTIONAL_ARGS[@]}"
}

# HELPER: Run one E_06 DP configuration at the current epsilon.
run_one() {
  DATASET="$1"
  HETEROGENEITY="$2"
  N_CLIENTS="$3"

  local label="E_06 DP ${DATASET} ${HETEROGENEITY}_${N_CLIENTS}banks ${STRATEGY} eps=${DP_TARGET_EPSILON}"
  echo "${label}"
  echo "device=${DEVICE} out=${OUTPUT_ROOT}"
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

# ----------------------------------------------------------------------------------------------------------------------
# 2. SINGLE RUN MODE

# Run only the selected DP configuration at DP_TARGET_EPSILON when requested.
guard_dp_workflow
if [[ "$SINGLE_RUN_MODE" == "true" ]]; then
  for strategy in $(strategies_to_run); do
    STRATEGY="$strategy"
    run_one "$DATASET" "$HETEROGENEITY" "$N_CLIENTS"
  done
  exit 0
fi

# ----------------------------------------------------------------------------------------------------------------------
# 3. DP EPSILON SWEEP

# Sweep every strategy, split and epsilon level; the no-DP matrix is the infinity baseline.
for strategy in $(strategies_to_run); do
  STRATEGY="$strategy"
  for split in $DP_EXPERIMENT_SPLITS; do
    IFS=':' read -r dp_dataset dp_het dp_n <<< "$split"
    for eps in $DP_EXPERIMENT_EPSILONS; do
      DP_TARGET_EPSILON="$eps"
      run_one "$dp_dataset" "$dp_het" "$dp_n"
    done
  done
done