#!/usr/bin/env bash
# CONTENT: Run the E_04 prefix encoding metadata workflow and LLM side experiment.
# USAGE: ./WORKFLOW_run_encoding.sh
#   1. Validate the fixed MANUAL contract, canonical schema and dataset mapping.
#   2. Execute 04_0_extract_contract_context.py for the federation contract context.
#   3. Execute 04_2_create_canonical_schema.py for LLM schema drafts across strategies 1 to 3.
#   4. Execute 04_3_create_dataset_mapping.py for a semantic baseline and LLM drafts using strategies 1 to 3.
#   5. Execute 04_4_runner.py once for the full BPIC 2017 and BPIC 2012 metadata matrix.
#   6. Execute 04_6_analyze_llm_outputs.py to compare semantic and LLM drafts against the manual ground truth.
#   7. Execute 04_7_decentralized_metadata_poc.py for the decentralized POC.
# NOTE: Manual JSON files are repository inputs. The workflow never recreates or overwrites them.
# NOTE: OPENAI_API_KEY is required only when RUN_LLM_EXPERIMENT=true. (Key can be set in .zshrc)

# Strict mode fails on command errors, undefined variables and failed pipe segments.
set -euo pipefail

# CONFIGURATION
PYTHON_BIN="${PYTHON_BIN:-../fl-ppm/bin/python}"                              # Interpreter of the project environment

# Select run mode
RUN_MANUAL_ENCODING="${RUN_MANUAL_ENCODING:-true}"                            # Encode the approved manual mapping
RUN_LLM_EXPERIMENT="${RUN_LLM_EXPERIMENT:-false}"                             # Needs OPENAI_API_KEY when true
RUN_LLM_SCHEMA_EXPERIMENT="${RUN_LLM_SCHEMA_EXPERIMENT:-false}"               # Only runs when the line above is true
RUN_DECENTRALIZED_POC="${RUN_DECENTRALIZED_POC:-true}"                        # Rebuild metadata from local aggregates
REMAINING_TIME_TRANSFORM="${REMAINING_TIME_TRANSFORM:-raw}"                   # RT target transform: raw | log
REMAINING_TIME_SCALING="${REMAINING_TIME_SCALING:-zscore}"                    # RT target scaling: raw | median | zscore

# Open Ai specific parameters for LLM experiments
OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}"                                    # Model for the mapping experiments
OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-240}"                       # OpenAI timeout per request
OPENAI_RETRIES="${OPENAI_RETRIES:-2}"                                         # Retries after a timeout or API error
LLM_DATASET_MAPPING_RUNS="${LLM_DATASET_MAPPING_RUNS:-3}"                     # Repeated runs for run-to-run variance

# Paths
MANUAL_CONTRACT_PATH="mappings/MANUAL_contract.json"                          # Approved field contract input
MANUAL_SCHEMA_PATH="mappings/MANUAL_canonical_schemas.json"                   # Approved schema input
MANUAL_MAPPING_PATH="mappings/MANUAL_dataset_mapping.json"                    # Approved dataset mapping input
LLM_TARGETS_PATH="mappings/llm_mapping/LLM_canonical_schema_targets.json"     # Ground truth for the LLM drafts
ARTIFACT_ROOT="encoded_metadata"                                              # Root folder for metadata artifacts
LLM_SCHEMA_ROOT="mappings/llm_mapping/canonical_schemas"                      # Output folder for schema drafts
LLM_MAPPING_ROOT="mappings/llm_mapping/dataset_mappings"                      # Output folder for generated mappings
LLM_ANALYSIS_ROOT="mappings/llm_mapping/llm_analysis"                         # Output folder for report and plots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Run one LLM draft step. A failure is logged and skipped so a bad draft never aborts the experiment.
run_llm_step() {
  if ! "$@"; then
    echo "WARNING: LLM draft step failed and was skipped: $*"
  fi
}

# ----------------------------------------------------------------------------------------------------------------------
# 1. VALIDATION

# Stop when someone tries to pass arguments to the workflow.
if [ "$#" -ne 0 ]; then
  echo "Usage: ./WORKFLOW_run_encoding.sh"
  exit 2
fi

# Stop early when the configured Python runtime is not available.
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Missing Python runtime: $PYTHON_BIN"
  exit 1
fi

# Check that fixed repository inputs exist before any generated output is written.
for path in "$MANUAL_CONTRACT_PATH" "$MANUAL_SCHEMA_PATH" "$MANUAL_MAPPING_PATH" "$LLM_TARGETS_PATH"; do
  if [ ! -f "$path" ]; then
    echo "Missing required E_04 input: $path"
    exit 1
  fi
done

# Require human approval for the manual schema and mapping consumed by the runner.
check_approved() {
  local path="$1"
  local label="$2"
  "$PYTHON_BIN" -B - "$path" "$label" <<'PY'
import json
import sys

path, label = sys.argv[1], sys.argv[2]
with open(path, encoding="utf-8") as handle:
    payload = json.load(handle)
if not payload.get("approved", False):
    raise SystemExit(f"{label} is not approved: {path}")
PY
}
check_approved "$MANUAL_SCHEMA_PATH" "manual canonical schema"
check_approved "$MANUAL_MAPPING_PATH" "manual dataset mapping"

# LLM experiment calls OpenAI only when the branch is enabled.
if [ "$RUN_LLM_EXPERIMENT" = true ] && [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is required because RUN_LLM_EXPERIMENT=true"
  exit 1
fi

# Pass API timeout controls to the Python LLM helpers.
export OPENAI_TIMEOUT_SECONDS
export OPENAI_RETRIES

# ----------------------------------------------------------------------------------------------------------------------
# 2. CONTRACT CONTEXT

# Export contract context per bank from the processed parquets inside the data owner boundary.
if [ "$RUN_DECENTRALIZED_POC" = true ]; then
  "$PYTHON_BIN" -B 04_0_extract_contract_context.py
fi

# ----------------------------------------------------------------------------------------------------------------------
# 3. LLM CANONICAL SCHEMA DRAFTS

# Generate LLM canonical schema drafts for all prompt strategies without touching manual ground truth files.
if [ "$RUN_LLM_EXPERIMENT" = true ] && [ "$RUN_LLM_SCHEMA_EXPERIMENT" = true ]; then
  mkdir -p "$LLM_SCHEMA_ROOT"

  run_llm_step "$PYTHON_BIN" -B 04_2_create_canonical_schema.py \
    --schema-mode llm \
    --openai-model "$OPENAI_MODEL" \
    --prompt-strategy strategy_1 \
    --force \
    --canonical-schema-path "$LLM_SCHEMA_ROOT/04_02_strategy_1_baseline_canonical_schema.json" \
    --canonical-schema-targets-path "$LLM_TARGETS_PATH"

  run_llm_step "$PYTHON_BIN" -B 04_2_create_canonical_schema.py \
    --schema-mode llm \
    --openai-model "$OPENAI_MODEL" \
    --prompt-strategy strategy_2 \
    --force \
    --canonical-schema-path "$LLM_SCHEMA_ROOT/04_02_strategy_2_field_rules_canonical_schema.json" \
    --canonical-schema-targets-path "$LLM_TARGETS_PATH"

  run_llm_step "$PYTHON_BIN" -B 04_2_create_canonical_schema.py \
    --schema-mode llm \
    --openai-model "$OPENAI_MODEL" \
    --prompt-strategy strategy_3 \
    --force \
    --canonical-schema-path "$LLM_SCHEMA_ROOT/04_02_strategy_3_target_recipe_canonical_schema.json" \
    --canonical-schema-targets-path "$LLM_TARGETS_PATH"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 4. DATASET MAPPING DRAFTS

# Generate a deterministic semantic baseline and the repeated LLM dataset-mapping drafts.
if [ "$RUN_LLM_EXPERIMENT" = true ]; then
  mkdir -p "$LLM_MAPPING_ROOT"

  # Deterministic semantic mappings as non-LLM baselines scored against the manual ground truth.
  "$PYTHON_BIN" -B 04_3_create_dataset_mapping.py \
    --schema-profile joint \
    --mapping-mode semantic \
    --semantic-activity-similarity character \
    --force \
    --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
    --dataset-mapping-path "$LLM_MAPPING_ROOT/run_semantic_character/04_03_semantic_character_dataset_mapping.json"

  "$PYTHON_BIN" -B 04_3_create_dataset_mapping.py \
    --schema-profile joint \
    --mapping-mode semantic \
    --semantic-activity-similarity word \
    --force \
    --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
    --dataset-mapping-path "$LLM_MAPPING_ROOT/run_semantic_word/04_03_semantic_word_dataset_mapping.json"

  for run in $(seq 1 "$LLM_DATASET_MAPPING_RUNS"); do
    run_id="$(printf "run_%02d" "$run")"
    run_mapping_root="$LLM_MAPPING_ROOT/$run_id"
    mkdir -p "$run_mapping_root"

    for semantic_similarity in character word; do
      semantic_variant="semantic_${semantic_similarity}"
      seed_mapping_root="$LLM_MAPPING_ROOT/$run_id/$semantic_variant"
      mkdir -p "$seed_mapping_root"

      run_llm_step "$PYTHON_BIN" -B 04_3_create_dataset_mapping.py \
        --schema-profile joint \
        --mapping-mode llm \
        --semantic-activity-similarity "$semantic_similarity" \
        --openai-model "$OPENAI_MODEL" \
        --prompt-strategy strategy_1 \
        --force \
        --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
        --dataset-mapping-path "$seed_mapping_root/04_03_strategy_1_baseline_dataset_mapping.json"
    done

    run_llm_step "$PYTHON_BIN" -B 04_3_create_dataset_mapping.py \
      --schema-profile joint \
      --mapping-mode llm \
      --semantic-activity-similarity character \
      --openai-model "$OPENAI_MODEL" \
      --prompt-strategy strategy_2 \
      --force \
      --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
      --dataset-mapping-path "$run_mapping_root/04_03_strategy_2_split_prompt_dataset_mapping.json"

    run_llm_step "$PYTHON_BIN" -B 04_3_create_dataset_mapping.py \
      --schema-profile joint \
      --mapping-mode llm \
      --semantic-activity-similarity character \
      --openai-model "$OPENAI_MODEL" \
      --prompt-strategy strategy_3 \
      --force \
      --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
      --dataset-mapping-path "$run_mapping_root/04_03_strategy_3_target_recipe_dataset_mapping.json"
  done
fi

# ----------------------------------------------------------------------------------------------------------------------
# 5. ENCODING METADATA MATRIX

# Build the full BPIC metadata matrix from the approved manual schema and mapping.
if [ "$RUN_MANUAL_ENCODING" = true ]; then
  "$PYTHON_BIN" -B 04_4_runner.py \
    --schema-profile all \
    --artifact-root "$ARTIFACT_ROOT" \
    --remaining-time-transform "$REMAINING_TIME_TRANSFORM" \
    --remaining-time-scaling "$REMAINING_TIME_SCALING" \
    --canonical-schema-path "$MANUAL_SCHEMA_PATH" \
    --dataset-mapping-path "$MANUAL_MAPPING_PATH"
fi

# ----------------------------------------------------------------------------------------------------------------------
# 6. LLM ANALYSIS

# Compare every semantic and LLM draft against the manual ground truth.
if [ "$RUN_LLM_EXPERIMENT" = true ]; then
  mkdir -p "$LLM_ANALYSIS_ROOT"
  "$PYTHON_BIN" -B 04_6_analyze_llm_outputs.py \
    --manual-schema-path "$MANUAL_SCHEMA_PATH" \
    --manual-mapping-path "$MANUAL_MAPPING_PATH" \
    --schema-root "$LLM_SCHEMA_ROOT" \
    --mapping-root "$LLM_MAPPING_ROOT" \
    --analysis-root "$LLM_ANALYSIS_ROOT" \
    --write-plots
fi

# ----------------------------------------------------------------------------------------------------------------------
# 7. DECENTRALIZED POC

# Reconstruct the central E_04 metadata from local aggregates and compare it against the runner artifacts.
if [ "$RUN_DECENTRALIZED_POC" = true ]; then

  # The POC rebuilds the central metadata, so the encoding matrix from section 5 must exist first.
  if [ ! -d "$ARTIFACT_ROOT" ] || [ -z "$(ls -A "$ARTIFACT_ROOT" 2>/dev/null)" ]; then
    echo "Missing central metadata for the decentralized POC: run with RUN_MANUAL_ENCODING=true first"
    exit 1
  fi

  "$PYTHON_BIN" -B 04_7_decentralized_metadata_poc.py --schema-profile all
fi

echo "E_04 workflow finished."