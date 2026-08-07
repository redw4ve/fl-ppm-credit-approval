#!/usr/bin/env bash
# CONTENT: Run the E_06 federated training matrix, including the joint cross-dataset runs.
#   1. Resolve the selected production config and cache locations.
#   2. Run one configuration when SINGLE_RUN_MODE=true, expanding STRATEGY=both to FedAvg and FedProx.
#   3. Run the full BPIC 2017, BPIC 2012 and joint matrix for each selected strategy.
#      STRATEGY=both delivers FedAvg and FedProx.
# NOTE: STRATEGY defaults to both, so a bare run produces FedAvg and FedProx at the fixed FEDPROX_MU=1e-4.
#       The DP epsilon experiment lives in WORKFLOW_run_federated_dp_final.sh, not here.

# Fail on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve outputs, cache and scripts from the E_training workflow directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION

PYTHON_BIN="${PYTHON_BIN:-../fl-ppm/bin/python}"                      # Path location of the environment.
SEED="${SEED:-42}"                                                    # Seed for all experiments (fixed 42).
DEVICE="${DEVICE:-auto}"                                              # Auto prefers CUDA, then MPS, then CPU.
PROGRESS_BARS="${PROGRESS_BARS:-true}"                                # Progress bar switch (true or false).
REPORTING_PROFILE="${REPORTING_PROFILE:-compact}"                     # Compact is the production reporting layout.
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/training_outputs}"   # Output locations.
CACHE_ROOT="${CACHE_ROOT:-$SCRIPT_DIR/prefix_tensor_cache}"           # Prefix tensor caches (outside output folder).
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/../E_prefix_encoding/encoded_metadata}"  # E_04 metadata root.
RESILIENT="${RESILIENT:-false}"                                       # When true, a failed run is logged and skipped.
FAILED_RUNS_LOG="${FAILED_RUNS_LOG:-}"                                # Optional file to append labels of skipped failed runs.
DRY_RUN="${DRY_RUN:-false}"                                           # Preview planned runs without executing.

# Select the workflow mode and one split for single-run smoke checks.
SINGLE_RUN_MODE="${SINGLE_RUN_MODE:-false}"                           # Default is false.
DATASET="${DATASET:-bpic2017}"                                        # Default dataset is BPIC 2017.
HETEROGENEITY="${HETEROGENEITY:-medium}"                              # Default heterogeneity is medium.
N_CLIENTS="${N_CLIENTS:-3}"                                           # Default number of clients is 3.
STRATEGY="${STRATEGY:-both}"                                          # fedavg, fedprox or both.
NEXT_ACTIVITY_HEAD_AGG_OVERRIDE="${NEXT_ACTIVITY_HEAD_AGG:-}"         # Empty unless the caller pinned the mode.
NEXT_ACTIVITY_HEAD_AGG="${NEXT_ACTIVITY_HEAD_AGG:-sample}"            # The resolver selects equal for joint runs.
SECURE_AGGREGATION_SIMULATION="${SECURE_AGGREGATION_SIMULATION:-false}"  # Additive-masking simulation, off by default.
SECURE_AGGREGATION_SEED="${SECURE_AGGREGATION_SEED:-42}"              # Deterministic secure-aggregation mask seed.

# Configure federated rounds, local epochs and optional DP-SGD.
MAX_ROUNDS="${MAX_ROUNDS:-40}"                                        # Uniform budget, early stopping terminates.
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"                                     # Local epochs per client fit.
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-7}"               # Uniform patience across the matrix.
FEDPROX_MU="${FEDPROX_MU:-1e-4}"                                      # Tuned on strong_3banks, confirmed on strong_5banks.
USE_DP="${USE_DP:-false}"                                             # DP-SGD switch.
DP_TARGET_EPSILON="${DP_TARGET_EPSILON:-10.0}"                        # Per-client target epsilon.
DP_DELTA="${DP_DELTA:-1e-6}"                                          # Below one over the smallest client train count.
DP_MAX_GRAD_NORM="${DP_MAX_GRAD_NORM:-1.0}"                           # Per-sample clipping norm.
DP_SMOKE_MAX_BATCHES="${DP_SMOKE_MAX_BATCHES:-}"                      # Optional DP smoke cap, empty means full pass.

# Configure optimizer defaults from the final E_05 workflow.
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"                              # AdamW base learning rate.
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"                                  # AdamW weight decay.
GRADIENT_CLIP_NORM="${GRADIENT_CLIP_NORM:-1.0}"                       # No-DP maximum gradient norm.

# Capture explicit schedule overrides before defaulting so the per-family resolver can respect them.
LR_SCHEDULER_T_MAX_OVERRIDE="${LR_SCHEDULER_T_MAX:-}"                 # Empty unless the caller pinned t_max.
LR_SCHEDULER_MIN_LR_OVERRIDE="${LR_SCHEDULER_MIN_LR:-}"               # Empty unless the caller pinned min_lr.

# Decay learning rates over the cosine period, then hold the floor.
# The family resolver overrides t_max and min_lr per dataset family.
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"                                # Scheduler type.
LR_SCHEDULER_MIN_LR="${LR_SCHEDULER_MIN_LR:-1e-6}"                    # BPIC 2017 floor; resolver sets 1e-5 elsewhere.
LR_SCHEDULER_T_MAX="${LR_SCHEDULER_T_MAX:-15}"                        # BPIC 2017 period; resolver sets 35 elsewhere.

# Configure local data loading and model architecture.
BATCH_SIZE="${BATCH_SIZE:-512}"                                       # Prefix samples per batch.
NUM_WORKERS="${NUM_WORKERS:-0}"                                       # Worker processes, 0 is stable on macOS.
HIDDEN_SIZE="${HIDDEN_SIZE:-128}"                                     # LSTM hidden size.
NUM_LAYERS="${NUM_LAYERS:-2}"                                         # Number of LSTM layers.
DROPOUT="${DROPOUT:-0.30}"                                            # Shared trunk dropout.
HEAD_HIDDEN_SIZE="${HEAD_HIDDEN_SIZE:-64}"                            # Hidden size of each prediction head.

# Configure loss weights and task-head learning-rate multipliers.
OUTCOME_LABEL_SMOOTHING="${OUTCOME_LABEL_SMOOTHING:-0.10}"            # Outcome CE label smoothing.
OUTCOME_CLASS_WEIGHT_POWER="${OUTCOME_CLASS_WEIGHT_POWER:-0.5}"       # Tempered outcome class weight strength.
OUTCOME_LR_SCALE="${OUTCOME_LR_SCALE:-0.3}"                           # Outcome head learning-rate multiplier.
NEXT_ACTIVITY_LR_SCALE="${NEXT_ACTIVITY_LR_SCALE:-1.0}"               # Next-activity head learning-rate multiplier.
REMAINING_TIME_LR_SCALE="${REMAINING_TIME_LR_SCALE:-1.0}"             # Remaining-time head learning-rate multiplier.
OUTCOME_LOSS_WEIGHT="${OUTCOME_LOSS_WEIGHT:-1.0}"                     # Outcome loss contribution.
NEXT_ACTIVITY_LOSS_WEIGHT="${NEXT_ACTIVITY_LOSS_WEIGHT:-0.5}"         # Next-activity loss contribution.
REMAINING_TIME_LOSS_WEIGHT="${REMAINING_TIME_LOSS_WEIGHT:-0.5}"       # Remaining-time loss contribution.

# Configure the final remaining-time representation and Huber switch point.
REMAINING_TIME_TRANSFORM="${REMAINING_TIME_TRANSFORM:-raw}"           # Remaining-time target transform.
REMAINING_TIME_SCALING="${REMAINING_TIME_SCALING:-zscore}"            # Remaining-time target scaling.
REMAINING_TIME_HUBER_BETA="${REMAINING_TIME_HUBER_BETA:-0.1}"         # Remaining-time Huber beta.

# Keep single-dash expansion so an explicit empty value disables the override.
OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT-0.45}"                  # Extra dropout for the outcome head.

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
  if [[ -z "$NEXT_ACTIVITY_HEAD_AGG_OVERRIDE" ]]; then
    case "$dataset" in
      joint) NEXT_ACTIVITY_HEAD_AGG=equal ;;
      *) NEXT_ACTIVITY_HEAD_AGG=sample ;;
    esac
  fi
}

# HELPER: Build optional CLI switches that cannot be represented by a scalar value.
optional_args() {
  OPTIONAL_ARGS=()
  if [[ "$PROGRESS_BARS" != "true" ]]; then OPTIONAL_ARGS+=(--no-progress-bars); fi
  if [[ "$USE_DP" == "true" ]]; then OPTIONAL_ARGS+=(--use-dp); else OPTIONAL_ARGS+=(--no-use-dp); fi
  if [[ "$SECURE_AGGREGATION_SIMULATION" == "true" ]]; then OPTIONAL_ARGS+=(--secure-aggregation-simulation); fi
  if [[ -n "$DP_SMOKE_MAX_BATCHES" ]]; then OPTIONAL_ARGS+=(--dp-smoke-max-batches "$DP_SMOKE_MAX_BATCHES"); fi
}

# HELPER: Run the E_06 entry point with the resolved workflow environment.
run_python() {
  optional_args
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY: DATASET=${DATASET} HETEROGENEITY=${HETEROGENEITY} N_CLIENTS=${N_CLIENTS} STRATEGY=${STRATEGY}"
    echo "    DRY: NEXT_ACTIVITY_HEAD_AGG=${NEXT_ACTIVITY_HEAD_AGG}"
    echo "    DRY: SCHEDULE dataset=${DATASET} MAX_ROUNDS=${MAX_ROUNDS} T_MAX=${LR_SCHEDULER_T_MAX}" \
      "MIN_LR=${LR_SCHEDULER_MIN_LR} LR=${LEARNING_RATE} BATCH=${BATCH_SIZE} PATIENCE=${EARLY_STOPPING_PATIENCE}"
    echo "    DRY: FEDPROX_MU=${FEDPROX_MU} USE_DP=${USE_DP} REMAINING_TIME_TRANSFORM=${REMAINING_TIME_TRANSFORM}" \
      "REMAINING_TIME_SCALING=${REMAINING_TIME_SCALING}"
    echo "    DRY: REPORTING_PROFILE=${REPORTING_PROFILE}"
    return 0
  fi
  EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}" \
  OUTCOME_HEAD_DROPOUT="${OUTCOME_HEAD_DROPOUT}" \
  "$PYTHON_BIN" -B E_06_federated_training.py \
    --dataset "$DATASET" \
    --heterogeneity "$HETEROGENEITY" \
    --n-clients "$N_CLIENTS" \
    --strategy "$STRATEGY" \
    --next-activity-head-agg "$NEXT_ACTIVITY_HEAD_AGG" \
    --secure-aggregation-seed "$SECURE_AGGREGATION_SEED" \
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

# HELPER: Run one E_06 federated split for the selected strategy.
run_one() {
  DATASET="$1"
  HETEROGENEITY="$2"
  N_CLIENTS="$3"
  resolve_family_schedule "$DATASET"

  local label="E_06 ${DATASET} ${HETEROGENEITY}_${N_CLIENTS}banks ${STRATEGY}"
  label="${label} agg=${NEXT_ACTIVITY_HEAD_AGG} rounds=${MAX_ROUNDS} le=${LOCAL_EPOCHS}"
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

# HELPER: Run the nine BPIC 2017 and BPIC 2012 configurations for one strategy.
run_matrix_for_strategy() {
  STRATEGY="$1"
  # Run the BPIC 2017 main experiment matrix for this strategy.
  run_one "bpic2017" "iid" "3"
  run_one "bpic2017" "weak" "3"
  run_one "bpic2017" "medium" "3"
  run_one "bpic2017" "strong" "3"
  run_one "bpic2017" "medium" "5"
  run_one "bpic2017" "strong" "5"

  # Run the BPIC 2012 ablation matrix for this strategy.
  run_one "bpic2012" "iid" "3"
  run_one "bpic2012" "weak" "3"
  run_one "bpic2012" "medium" "3"

  # Run the joint cross-dataset matrix for this strategy.
  run_one "joint" "iid" "6"
  run_one "joint" "weak" "6"
  run_one "joint" "medium" "6"
  run_one "joint" "medium" "8"
}

# HELPER: Resolve the strategy list, expanding the both keyword to FedAvg first and then FedProx.
strategies_to_run() {
  if [[ "$STRATEGY" == "both" ]]; then echo "fedavg fedprox"; else echo "$STRATEGY"; fi
}

# ----------------------------------------------------------------------------------------------------------------------
# 2. SINGLE RUN MODE

# Run only the selected E_06 configuration when requested, expanding the both keyword to two strategy runs.
if [[ "$SINGLE_RUN_MODE" == "true" ]]; then
  for strategy in $(strategies_to_run); do
    STRATEGY="$strategy"
    run_one "$DATASET" "$HETEROGENEITY" "$N_CLIENTS"
  done
  exit 0
fi

# ----------------------------------------------------------------------------------------------------------------------
# 3. MATRIX MODE

# Run the full matrix for each selected strategy, so STRATEGY=both delivers FedAvg and FedProx in one nightly.
for strategy in $(strategies_to_run); do
  run_matrix_for_strategy "$strategy"
done