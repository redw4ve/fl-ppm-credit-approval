#!/usr/bin/env bash
# CONTENT: Run the final thesis pipeline from existing processed splits through all final matrices.
#   1. Optionally download raw BPIC logs and re-run preprocessing.
#   2. Optionally regenerate E_04 metadata, including joint metadata and the decentralized POC.
#   3. Run the baseline (incl joint) and federated (incl joint) matrices into one unified output root.
#   4. Run the secure-aggregation POC into secure_aggregation/, then generate the E_07 analysis.
# NOTE: A bare invocation produces the locked thesis configuration into training_outputs.
# NOTE: Encoding is off by default, so the bare run reuses the existing E_04 metadata and the warm prefix cache.
# The bare run therefore also skips the LLM experiment and the decentralized POC.
# NOTE: DP is excluded from the bare run and launched separately with RUN_FEDERATED_DP=true.
# NOTE: Training hyperparameters live in the three training workflows, here only orchestration.
# NOTE: Preprocessing (A_01 to A_03 and B_01 to B_03) is excluded and can be toggled on.

# Strict mode fails on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# Resolve all paths from the repository root.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ----------------------------------------------------------------------------------------------------------------------
# COMMAND LINE

# Print the accepted switches and exit.
usage() {
  cat <<'USAGE'
Usage: bash WORKFLOW_run_FULL_PIPELINE.sh [options]

  --full               Reproduce everything from source. This is the switch a fresh clone needs.
                       It downloads the raw logs when they are missing, reruns preprocessing and encoding,
                       rebuilds the prefix tensor cache from scratch and runs both LLM side experiments.
                       The download skips a log that already exists, so the switch is safe on a warm checkout.
                       It does NOT enable the DP grid, which needs CUDA and days of runtime. Add --dp for that.
                       Warning: it deletes an existing prefix tensor cache, which is large but regenerable.
  --dp                 Run the DP-SGD epsilon grid into the same output root.
                       Requires a CUDA machine.
  --no-dp              Skip the DP-SGD grid. This is the default.
  --strict             Abort on the first failed training run and fail the analysis on any warning.
  --resilient          Log and skip a failed training run so the matrix continues. This is the default.
  --dry-run            Print the planned stages without executing them.
  -h, --help           Print this message.

A bare invocation runs the training matrix and the analysis against artifacts that already exist. It reuses the
processed splits, the E_04 metadata and the prefix tensor cache, so it never overwrites data and never spends
money at the OpenAI API. A fresh clone has none of those artifacts and must use --full.

Every switch also exists as an environment variable, for example RUN_FEDERATED_DP=true or RESILIENT=false.
A switch is read before the configuration block, so it wins over the matching environment variable.
USAGE
}

# Read the switches before the configuration block, so a flag overrides the matching environment variable.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)
      # Reproduce every stage from source, which is what a fresh clone needs and what the clean run uses.
      RUN_DATA_DOWNLOAD=true
      RUN_PREPROCESSING=true
      RUN_ENCODING=true
      CLEAN_CACHE=true
      RUN_LLM_EXPERIMENT=true
      RUN_LLM_SCHEMA_EXPERIMENT=true
      ;;
    --dp) RUN_FEDERATED_DP=true ;;
    --no-dp) RUN_FEDERATED_DP=false ;;
    --strict) RESILIENT=false ;;
    --resilient) RESILIENT=true ;;
    --dry-run) DRY_RUN=true ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ----------------------------------------------------------------------------------------------------------------------
# CONFIGURATION

# Environment and preview
PYTHON_BIN="${PYTHON_BIN:-./fl-ppm/bin/python}"                      # Path location of the environment.
DRY_RUN="${DRY_RUN:-false}"                                          # Preview planned stages without executing.

# Stage toggles
RUN_DATA_DOWNLOAD="${RUN_DATA_DOWNLOAD:-false}"                      # Download raw logs from 4TU before preprocessing.
RUN_PREPROCESSING="${RUN_PREPROCESSING:-false}"                      # Rerun A and B preprocessing workflows.
RUN_TEST_SUITE="${RUN_TEST_SUITE:-true}"                             # Run the full test suite before any pipeline stage.
RUN_FOCUSED_TESTS="${RUN_FOCUSED_TESTS:-false}"                      # Run the focused joint and reporting unit tests.
RUN_ENCODING="${RUN_ENCODING:-false}"                                # Run E_04 encoding and POC workflow.
RUN_BASELINES="${RUN_BASELINES:-true}"                               # Run E_05 baseline matrix, including joint.
RUN_FEDERATED="${RUN_FEDERATED:-true}"                               # Run E_06 federated matrix, including joint.
# The DP grid stays off by default, so cloning the repository onto a laptop never blocks the matrix for days.
# It needs CUDA and is launched separately with --dp or RUN_FEDERATED_DP=true.
RUN_FEDERATED_DP="${RUN_FEDERATED_DP:-false}"                        # Run the E_06 DP-SGD epsilon experiment.
RUN_SECURE_AGGREGATION="${RUN_SECURE_AGGREGATION:-true}"             # Run the secure-aggregation POC stage.
RUN_TRAINING_ANALYSIS="${RUN_TRAINING_ANALYSIS:-true}"               # Generate E_07 analysis after training.

# ATTENTION: Clean slate is off by default
# If this is activated: the encoding metadata and prefix cache are deleted!!!
CLEAN_ENCODING="${CLEAN_ENCODING:-false}"                            # Delete generated E_04 metadata and POC.
CLEAN_CACHE="${CLEAN_CACHE:-false}"                                  # Delete the prefix tensor cache -> fresh rebuild.

# LLM side experiment toggles
RUN_LLM_EXPERIMENT="${RUN_LLM_EXPERIMENT:-false}"                    # Run LLM dataset-mapping experiment during encoding.
RUN_LLM_SCHEMA_EXPERIMENT="${RUN_LLM_SCHEMA_EXPERIMENT:-false}"      # Run LLM schema experiment during encoding.

# Reporting
REPORTING_PROFILE="${REPORTING_PROFILE:-compact}"                    # Compact is the production reporting layout.
RESILIENT="${RESILIENT:-true}"                                       # Gracefully failing.

# Roots directories
OUTPUT_ROOT="${OUTPUT_ROOT:-$SCRIPT_DIR/E_training/training_outputs}"   # Unified final training output root.
CACHE_ROOT="${CACHE_ROOT:-$SCRIPT_DIR/E_training/prefix_tensor_cache}"           # Shared prefix tensor cache root.
ARTIFACT_ROOT="${ARTIFACT_ROOT:-$SCRIPT_DIR/E_prefix_encoding/encoded_metadata}" # E_04 metadata root.
FAILED_RUNS_LOG="${FAILED_RUNS_LOG:-$OUTPUT_ROOT/FAILED_RUNS.log}"               # Destination for run logs.

# ----------------------------------------------------------------------------------------------------------------------
# 1. WORKFLOW HELPERS

# HELPER: Execute one command under caffeinate when available.
run_step() {
  local label="$1"
  shift
  echo ">>> ${label}"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY: $*"
    return 0
  fi
  if command -v caffeinate >/dev/null 2>&1; then
    caffeinate -is "$@"
  else
    "$@"
  fi
}

# HELPER: Require one processed split folder with a plausible number of parquets before downstream stages.
# A single stray parquet used to satisfy this check, so a placeholder directory passed silently.
# One split writes three parquets per bank, so three banks give at least nine files.
require_processed_split() {
  local path="$1"
  local minimum="$2"
  local found=0
  if [[ -d "$path" ]]; then found="$(find "$path" -maxdepth 1 -name "*.parquet" | wc -l | tr -d " ")"; fi
  if (( found >= minimum )); then return 0; fi
  echo "Processed split folder $path holds $found parquets, expected at least $minimum." >&2
  echo "Set RUN_PREPROCESSING=true or regenerate the processed splits before running the full pipeline." >&2
  exit 1
}

# HELPER: Verify the existing processed split roots needed by E_04 and training when preprocessing is skipped.
verify_processed_splits() {
  if [[ "$RUN_PREPROCESSING" == "true" || "$DRY_RUN" == "true" ]]; then return 0; fi
  local bpic2017_splits=("iid_3banks" "weak_3banks" "medium_3banks" "strong_3banks" "medium_5banks" "strong_5banks")
  local bpic2012_splits=("iid_3banks" "weak_3banks" "medium_3banks")
  local split
  local banks
  for split in "${bpic2017_splits[@]}"; do
    banks=3
    if [[ "$split" == *_5banks ]]; then banks=5; fi
    require_processed_split "$SCRIPT_DIR/E_main_BPIC_2017/data/processed/$split" $(( banks * 3 ))
    require_processed_split "$SCRIPT_DIR/E_main_BPIC_2017/data/processed/centralized/$split" 3
  done
  for split in "${bpic2012_splits[@]}"; do
    require_processed_split "$SCRIPT_DIR/E_ablation_BPIC_2012/data/processed/$split" 9
    require_processed_split "$SCRIPT_DIR/E_ablation_BPIC_2012/data/processed/centralized/$split" 3
  done
}

# HELPER: Print a loud, decisive banner when the LLM experiment is skipped for a missing API key.
print_llm_skip_banner() {
  echo ""
  echo "######################################################################"
  echo "#  OPENAI_API_KEY is NOT set."
  echo "#  The LLM dataset-mapping and schema experiments will be SKIPPED."
  echo "#  All encoding, training and analysis will still run normally."
  echo "######################################################################"
  echo ""
}

# HELPER: Record a timestamped marker that the LLM experiment was skipped for a missing key.
write_llm_skip_marker() {
  local marker_dir="E_prefix_encoding/mappings/llm_mapping/llm_analysis"
  local marker_file="$marker_dir/LLM_SKIPPED_MISSING_OPENAI_API_KEY.txt"
  if [[ "$DRY_RUN" == "true" ]]; then
    echo "    DRY: would write LLM skip marker to $marker_file"
    return 0
  fi
  mkdir -p "$marker_dir"
  {
    echo "LLM experiment skipped at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Reason: OPENAI_API_KEY was not set while RUN_ENCODING=true."
    echo "Encoding, training and analysis ran without the LLM dataset-mapping and schema experiments."
  } > "$marker_file"
}

# HELPER: Resolve the optional LLM experiment against RUN_ENCODING and the API key without ever aborting.
verify_llm_configuration() {
  # The LLM experiments only run inside encoding, so there is nothing to check when encoding is off.
  if [[ "$RUN_ENCODING" != "true" ]]; then return 0; fi
  # A present key runs the LLM experiment normally, with no banner and no marker.
  if [[ -n "${OPENAI_API_KEY:-}" ]]; then return 0; fi
  # A missing key never aborts. It disables only the LLM experiment, warns loudly and records a marker file.
  print_llm_skip_banner
  RUN_LLM_EXPERIMENT=false
  RUN_LLM_SCHEMA_EXPERIMENT=false
  write_llm_skip_marker
}

# HELPER: Fail fast when the Python environment is missing a package a selected stage needs.
verify_python_packages() {
  if [[ "$DRY_RUN" == "true" ]]; then return 0; fi
  local packages=(numpy pandas torch sklearn flwr matplotlib pyarrow)
  if [[ "$RUN_FEDERATED_DP" == "true" ]]; then packages+=(opacus); fi

  # The preprocessing workflows execute the A_01 and B_01 notebooks through nbconvert.
  if [[ "$RUN_PREPROCESSING" == "true" ]]; then packages+=(pm4py nbconvert); fi
  local missing=()
  local package
  for package in "${packages[@]}"; do
    if ! "$PYTHON_BIN" -c "import ${package}" >/dev/null 2>&1; then missing+=("$package"); fi
  done
  if (( ${#missing[@]} )); then
    echo "Missing Python packages for the selected stages: ${missing[*]}" >&2
    echo "Build the environment from requirements.txt before running the pipeline." >&2
    exit 1
  fi

  # Check the notebook launcher the preprocessing workflows actually invoke, not only the importable package.
  if [[ "$RUN_PREPROCESSING" == "true" ]] && ! "$PYTHON_BIN" -m nbconvert --version >/dev/null 2>&1; then
    echo "The nbconvert module is not runnable with $PYTHON_BIN." >&2
    echo "Install nbconvert from requirements.txt so the A_01 and B_01 notebooks can be executed." >&2
    exit 1
  fi
}

# HELPER: Fail fast when the raw BPIC logs are missing while preprocessing is requested.
# The download stage runs later, so a requested download is what satisfies this check on a fresh clone.
verify_raw_logs() {
  if [[ "$RUN_PREPROCESSING" != "true" || "$RUN_DATA_DOWNLOAD" == "true" || "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  local roots=("$SCRIPT_DIR/E_main_BPIC_2017/BPI Challenge 2017" "$SCRIPT_DIR/E_ablation_BPIC_2012/BPI Challenge 2012")
  local root
  for root in "${roots[@]}"; do
    if [[ -z "$(find "$root" -maxdepth 1 -name '*.xes*' -print -quit 2>/dev/null)" ]]; then
      echo "Missing raw BPIC log below: $root" >&2
      echo "Set RUN_DATA_DOWNLOAD=true to fetch the logs from 4TU.ResearchData first." >&2
      exit 1
    fi
  done
}

# HELPER: Fail fast when the E_04 metadata is missing while encoding is skipped.
verify_encoded_metadata() {
  if [[ "$RUN_ENCODING" == "true" || "$DRY_RUN" == "true" ]]; then return 0; fi
  if [[ -z "$(find "$ARTIFACT_ROOT" -name '*_encoding_spec.json' -print -quit 2>/dev/null)" ]]; then
    echo "Missing E_04 encoding metadata below: $ARTIFACT_ROOT" >&2
    echo "Run once with RUN_ENCODING=true to regenerate the metadata, then rerun the pipeline." >&2
    exit 1
  fi
}

# HELPER: Refuse the DP grid without CUDA, because the 12 DP runs are only practical on the lab server.
verify_dp_device() {
  if [[ "$RUN_FEDERATED_DP" != "true" || "$DRY_RUN" == "true" ]]; then return 0; fi
  if "$PYTHON_BIN" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" >/dev/null 2>&1; then
    return 0
  fi
  echo "DP-SGD was requested but torch.cuda.is_available() is False." >&2
  echo "The 12-run DP grid needs a CUDA machine. Launch it there with --dp, or drop the switch." >&2
  exit 1
}

# HELPER: Refuse to delete a cache path that is empty or not under the repository root.
safe_remove_cache() {
  local path="$1"
  if [[ -z "$path" || "$path" != "$SCRIPT_DIR"/* ]]; then
    echo "Refusing to delete cache path outside the repository: '$path'" >&2
    exit 1
  fi
  rm -rf "$path"
}

# ----------------------------------------------------------------------------------------------------------------------
# 2. CONFIGURATION SUMMARY

echo "Full pipeline configuration:"
echo "RUN_DATA_DOWNLOAD=${RUN_DATA_DOWNLOAD}"
echo "RUN_PREPROCESSING=${RUN_PREPROCESSING}"
echo "RUN_TEST_SUITE=${RUN_TEST_SUITE}"
echo "RUN_FOCUSED_TESTS=${RUN_FOCUSED_TESTS}"
echo "CLEAN_ENCODING=${CLEAN_ENCODING}"
echo "CLEAN_CACHE=${CLEAN_CACHE}"
echo "RUN_ENCODING=${RUN_ENCODING}"
echo "RUN_BASELINES=${RUN_BASELINES}"
echo "RUN_FEDERATED=${RUN_FEDERATED}"
echo "RUN_FEDERATED_DP=${RUN_FEDERATED_DP}"
echo "RUN_SECURE_AGGREGATION=${RUN_SECURE_AGGREGATION}"
echo "RUN_TRAINING_ANALYSIS=${RUN_TRAINING_ANALYSIS}"
echo "RUN_LLM_EXPERIMENT=${RUN_LLM_EXPERIMENT}"
echo "RUN_LLM_SCHEMA_EXPERIMENT=${RUN_LLM_SCHEMA_EXPERIMENT}"
echo "REPORTING_PROFILE=${REPORTING_PROFILE}"
echo "RESILIENT=${RESILIENT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "CACHE_ROOT=${CACHE_ROOT}"
echo "ARTIFACT_ROOT=${ARTIFACT_ROOT}"
echo "FAILED_RUNS_LOG=${FAILED_RUNS_LOG}"

# Run every preflight before the first stage, so a missing input fails in seconds and not after hours of training.
# The preprocessing workflows call "python -m nbconvert" directly, so no console script or dispatcher is involved.
# Some tooling still resolves helpers from PATH rather than from the interpreter prefix, so put the interpreter's
# own bin directory first and a clean clone works without activating the environment.
if [[ "$PYTHON_BIN" == */* && -d "$(dirname "$PYTHON_BIN")" ]]; then
  PATH="$(cd "$(dirname "$PYTHON_BIN")" && pwd):$PATH"
  export PATH
fi

verify_llm_configuration
verify_python_packages
verify_raw_logs
verify_processed_splits
verify_encoded_metadata
verify_dp_device

# A strict run aborts on the first failed training run and fails the analysis on any recorded warning.
# The switch stays a plain string, because bash 3.2 rejects an empty array under set -u.
STRICT_ANALYSIS_ARG=""
if [[ "$RESILIENT" != "true" ]]; then STRICT_ANALYSIS_ARG="--strict"; fi

# The training children default to their own relative interpreter path, which breaks a custom PYTHON_BIN.
# Hand them an absolute path, so a caller-selected interpreter reaches every stage.
CHILD_PYTHON_BIN="$PYTHON_BIN"
if [[ "$PYTHON_BIN" == */* && -d "$(dirname "$PYTHON_BIN")" ]]; then
  CHILD_PYTHON_BIN="$(cd "$(dirname "$PYTHON_BIN")" && pwd)/$(basename "$PYTHON_BIN")"
fi

# The DP workflow defaults to MPS, but the DP preflight verified CUDA, so the child must receive CUDA.
DP_DEVICE="${DEVICE:-cuda}"

# Create the output root early so resilient training stages can append to the failure log.
# Record the interpreter, the platform and the library versions of this invocation next to the output root.
if [[ "$DRY_RUN" != "true" ]]; then
  mkdir -p "$OUTPUT_ROOT"
  DP_SNAPSHOT_ARG=""
  if [[ "$RUN_FEDERATED_DP" == "true" ]]; then DP_SNAPSHOT_ARG="--use-dp"; fi
  "$PYTHON_BIN" -B -m E_training.training_reporting --output-root "$OUTPUT_ROOT" ${DP_SNAPSHOT_ARG}
fi

# ----------------------------------------------------------------------------------------------------------------------
# 3. TESTS, RAW DATA AND PREPROCESSING

# The full suite is self-contained and needs no data, so it runs first and a red test stops the run immediately.
if [[ "$RUN_TEST_SUITE" == "true" ]]; then
  run_step "Run the full test suite" "$PYTHON_BIN" -m pytest tests/ -q
  if [[ "$DRY_RUN" != "true" ]]; then
    echo "Test suite green, every test passed. The pipeline stages start now."
  fi
fi

if [[ "$RUN_FOCUSED_TESTS" == "true" ]]; then
  run_step "Run focused unit tests" "$PYTHON_BIN" -m unittest \
    tests.test_joint_run_specs \
    tests.test_prefix_encoding_joint_runtime \
    tests.test_e05_joint_baselines \
    tests.test_e06_federated_core \
    tests.test_prefix_encoding_decentralized_poc \
    tests.test_prefix_encoding_runtime \
    tests.test_joint_workflows \
    tests.test_training_reporting \
    tests.test_training_analysis
fi

if [[ "$RUN_DATA_DOWNLOAD" == "true" ]]; then
  run_step "Download BPIC source logs" "$PYTHON_BIN" -B "_helpers/download_bpic_from_4tu.py"
fi

if [[ "$RUN_PREPROCESSING" == "true" ]]; then
  run_step "Run BPIC 2017 preprocessing" bash "E_main_BPIC_2017/A_WORKFLOW_run_partitioning.sh"
  run_step "Run BPIC 2012 preprocessing" bash "E_ablation_BPIC_2012/B_WORKFLOW_run_partitioning.sh"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 4. CLEAN SLATE

# Refuse to delete the E_04 metadata when the encoding stage is off, because the run would then have none.
if [[ "$CLEAN_ENCODING" == "true" && "$RUN_ENCODING" != "true" ]]; then
  echo "CLEAN_ENCODING=true deletes the E_04 metadata, but RUN_ENCODING=false would not regenerate it." >&2
  echo "Set RUN_ENCODING=true to rebuild the metadata, or drop CLEAN_ENCODING." >&2
  exit 1
fi

# Delete generated E_04 metadata and the decentralized POC so the encoding regenerates from the frozen inputs.
if [[ "$CLEAN_ENCODING" == "true" ]]; then
  run_step "Clean generated E_04 metadata" rm -rf "E_prefix_encoding/encoded_metadata" "E_prefix_encoding/decentralized_poc"
fi

# Delete the prefix tensor cache so training rebuilds every tensor for the final run.
if [[ "$CLEAN_CACHE" == "true" ]]; then
  if [[ "$DRY_RUN" == "true" ]]; then
    echo ">>> Clean prefix tensor cache"
    echo "    DRY: safe_remove_cache $CACHE_ROOT"
  else
    echo ">>> Clean prefix tensor cache"
    safe_remove_cache "$CACHE_ROOT"
  fi
fi

# ----------------------------------------------------------------------------------------------------------------------
# 5. ENCODING

if [[ "$RUN_ENCODING" == "true" ]]; then
  run_step "Run E_04 encoding metadata and POC" env \
    RUN_LLM_EXPERIMENT="$RUN_LLM_EXPERIMENT" \
    RUN_LLM_SCHEMA_EXPERIMENT="$RUN_LLM_SCHEMA_EXPERIMENT" \
    RUN_MANUAL_ENCODING=true \
    RUN_DECENTRALIZED_POC=true \
    bash "E_prefix_encoding/WORKFLOW_run_encoding.sh"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 6. TRAINING MATRICES

if [[ "$RUN_BASELINES" == "true" ]]; then
  run_step "Run E_05 baseline matrix (incl joint)" env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    REPORTING_PROFILE="$REPORTING_PROFILE" \
    RESILIENT="$RESILIENT" \
    FAILED_RUNS_LOG="$FAILED_RUNS_LOG" \
    PYTHON_BIN="$CHILD_PYTHON_BIN" \
    bash "E_training/WORKFLOW_run_baseline_final.sh"
fi

if [[ "$RUN_FEDERATED" == "true" ]]; then
  run_step "Run E_06 federated matrix (incl joint)" env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    REPORTING_PROFILE="$REPORTING_PROFILE" \
    RESILIENT="$RESILIENT" \
    FAILED_RUNS_LOG="$FAILED_RUNS_LOG" \
    PYTHON_BIN="$CHILD_PYTHON_BIN" \
    bash "E_training/WORKFLOW_run_federated_final.sh"
fi

# The DP experiment runs last so its no-DP infinity baseline already exists in the same OUTPUT_ROOT for E_07.
if [[ "$RUN_FEDERATED_DP" == "true" ]]; then
  run_step "Run E_06 federated DP-SGD epsilon experiment" env \
    DEVICE="$DP_DEVICE" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    REPORTING_PROFILE="$REPORTING_PROFILE" \
    RESILIENT="$RESILIENT" \
    FAILED_RUNS_LOG="$FAILED_RUNS_LOG" \
    PYTHON_BIN="$CHILD_PYTHON_BIN" \
    bash "E_training/WORKFLOW_run_federated_dp_final.sh"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 6b. SECURE-AGGREGATION POC

# HELPER: Run one federated configuration with the secure-aggregation simulation on, routed into secure_aggregation/.
run_secure_poc() {
  local dataset="$1"
  local heterogeneity="$2"
  local n_clients="$3"
  run_step "Secure-aggregation POC ${dataset} ${heterogeneity}_${n_clients}banks fedprox" env \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    ARTIFACT_ROOT="$ARTIFACT_ROOT" \
    REPORTING_PROFILE="$REPORTING_PROFILE" \
    RESILIENT="$RESILIENT" \
    FAILED_RUNS_LOG="$FAILED_RUNS_LOG" \
    PYTHON_BIN="$CHILD_PYTHON_BIN" \
    SINGLE_RUN_MODE=true DATASET="$dataset" HETEROGENEITY="$heterogeneity" N_CLIENTS="$n_clients" \
    STRATEGY=fedprox SECURE_AGGREGATION_SIMULATION=true \
    bash "E_training/WORKFLOW_run_federated_final.sh"
}

# Run three secure-aggregation POC configurations after the matrices, each with a plain FedProx counterpart.
# joint medium_8banks exercises the masked next-activity head across eight banks and two datasets.
# bpic2017 medium_3banks exercises the masked model-update sum on the main dataset, the standard single-dataset aggregation.
# bpic2012 medium_3banks confirms the masked sum on the ablation dataset too.
if [[ "$RUN_SECURE_AGGREGATION" == "true" ]]; then
  run_secure_poc "joint" "medium" "8"
  run_secure_poc "bpic2017" "medium" "3"
  run_secure_poc "bpic2012" "medium" "3"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 7. ANALYSIS

# HELPER: Refuse the analysis stage when no training run completed, so an empty matrix can never look finished.
refuse_empty_matrix() {
  local completed
  completed="$(find "$OUTPUT_ROOT/baselines" "$OUTPUT_ROOT/federated" "$OUTPUT_ROOT/differential_privacy" \
    -type f \( -name 'E_05_run_report.json' -o -name 'E_06_run_report.json' \) 2>/dev/null | wc -l | tr -d ' ' || true)"
  if [[ "${completed:-0}" -eq 0 ]]; then
    echo "ERROR: no completed training run exists under $OUTPUT_ROOT, refusing the analysis stage." >&2
    if [[ -s "$FAILED_RUNS_LOG" ]]; then nl -ba "$FAILED_RUNS_LOG" >&2; fi
    exit 1
  fi
}

if [[ "$RUN_TRAINING_ANALYSIS" == "true" ]]; then
  if [[ "$DRY_RUN" != "true" ]]; then refuse_empty_matrix; fi
  run_step "Run E_07 generated training analysis" "$PYTHON_BIN" -B "E_training/E_07_generate_training_analysis.py" \
    --output-root "$OUTPUT_ROOT" \
    --analysis-root "$OUTPUT_ROOT/analysis" \
    ${STRICT_ANALYSIS_ARG}

  # E_08 runs after E_07 and reads the same run reports, so the robustness table lands beside the analysis outputs.
  run_step "Run E_08 outcome robustness" "$PYTHON_BIN" -B "E_training/E_08_outcome_robustness.py" \
    --output-root "$OUTPUT_ROOT" \
    --analysis-root "$OUTPUT_ROOT/analysis" \
    ${STRICT_ANALYSIS_ARG}
fi

# Surface any resilient skips so a partial matrix is never mistaken for a complete one.
if [[ "$DRY_RUN" != "true" && -s "$FAILED_RUNS_LOG" ]]; then
  echo "WARNING: some runs were skipped after failures. Review before using the numbers:"
  echo "  $FAILED_RUNS_LOG"
  nl -ba "$FAILED_RUNS_LOG"
fi

echo "Full pipeline finished."