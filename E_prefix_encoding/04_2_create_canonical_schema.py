"""
Step 4.2: Create a canonical schema template for a new federation contract.

Run this script after the federation contract in `mappings/MANUAL_contract.json` is ready.
Manual mode writes `mappings/llm_mapping/LLM_canonical_schema_template.json`:
 -> An empty review template that exposes the contract fields and activity labels for manual completion.
 -> The SECTION that has to be filled out is at the very bottom and clearly marked.
LLM mode loads the same template plus `mappings/llm_mapping/LLM_canonical_schema_targets.json`.
 -> Then it writes an LLM-filled schema draft.

The script does not contain schema profiles specific to BPIC.
The manually filled BPIC canonical schema lives in `mappings/MANUAL_canonical_schemas.json`.

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_contract.json: contract loaded through 04_1_contract.py
    E_prefix_encoding/mappings/llm_mapping/LLM_canonical_schema_targets.json: LLM target brief, only used in llm mode

CREATED FILES:
    E_prefix_encoding/mappings/llm_mapping/LLM_canonical_schema_template.json: empty manual fill-in template
    E_prefix_encoding/mappings/llm_mapping/canonical_schemas/04_02_llm_canonical_schema_candidate.json: default LLM schema draft
"""

# IMPORTS
from __future__ import annotations
import argparse
import importlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

# Allow direct script execution from the repository root.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the contract python module and the encoding module.
contract = importlib.import_module("E_prefix_encoding.04_1_contract")
encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent  # Folder that contains this script
MAPPING_ROOT: Path = SCRIPT_DIR / "mappings"  # Root folder for reviewed E_04 mapping files

SCHEMA_MODE: str = "manual"  # "manual" writes a fill-in template, "llm" writes an LLM draft
OPENAI_MODEL: str = "gpt-5-nano"  # OpenAI model used only in llm mode
PROMPT_STRATEGY: str = "strategy_1"  # "strategy_1", "strategy_2" or "strategy_3" for LLM prompting
FORCE: bool = False  # Overwrite an approved schema file when True
OPENAI_TIMEOUT_SECONDS: int = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "240"))
OPENAI_RETRIES: int = int(os.environ.get("OPENAI_RETRIES", "2"))

# Manual fill-in template
CANONICAL_SCHEMA_PATH: Path = MAPPING_ROOT / "llm_mapping/LLM_canonical_schema_template.json"
# Default LLM draft
LLM_CANONICAL_SCHEMA_PATH: Path = MAPPING_ROOT / "llm_mapping/canonical_schemas/04_02_llm_canonical_schema_candidate.json"
# LLM target brief
CANONICAL_SCHEMA_TARGETS_PATH: Path = MAPPING_ROOT / "llm_mapping/LLM_canonical_schema_targets.json"


# ----------------------------------------------------------------------------------------------------------------------
# 1. CLI OVERRIDES

# Parse optional automation arguments while keeping script defaults.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an E_04 canonical schema template.")
    parser.add_argument("--schema-mode", default=SCHEMA_MODE, choices=["manual", "llm"])
    parser.add_argument("--openai-model", default=OPENAI_MODEL)
    parser.add_argument("--prompt-strategy", default=PROMPT_STRATEGY,
                        choices=["strategy_1", "strategy_2", "strategy_3"])
    parser.add_argument("--force", action="store_true", default=FORCE)
    parser.add_argument("--canonical-schema-path", type=Path, default=CANONICAL_SCHEMA_PATH)
    parser.add_argument("--canonical-schema-targets-path", type=Path, default=CANONICAL_SCHEMA_TARGETS_PATH)
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------------------------------------------------
# 2. OPENAI HELPERS

# Extract text from one Responses API result.
def _extract_response_text(response_payload: dict[str, Any]) -> str:
    # Use the direct text field when the API returns the compact shape.
    if "output_text" in response_payload: return str(response_payload["output_text"])

    # Fall back to collecting nested text chunks from output content blocks.
    texts: list[str] = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            if "text" in content: texts.append(str(content["text"]))
    return "\n".join(texts)


# Request one JSON draft only when llm mode is selected.
def request_openai_json(prompt: dict[str, Any], model: str, api_key: Optional[str] = None) -> dict[str, Any]:
    # Read the API key from the argument or the local environment.
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY is required for LLM mode.")

    # Send the schema prompt as JSON to the OpenAI Responses API.
    body = json.dumps({"model": model, "input": json.dumps(prompt)}).encode("utf-8")
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )

    # Retry transient API read errors before stopping the LLM side experiment.
    response_payload: dict[str, Any] = {}
    for attempt in range(1, OPENAI_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=OPENAI_TIMEOUT_SECONDS) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
            break
        except TimeoutError as exc:
            if attempt == OPENAI_RETRIES:
                raise RuntimeError(
                    f"OpenAI schema draft timed out after {OPENAI_TIMEOUT_SECONDS} seconds "
                    f"and {OPENAI_RETRIES} attempt(s)."
                ) from exc
        except urllib.error.URLError as exc:
            if attempt == OPENAI_RETRIES: raise RuntimeError(f"OpenAI schema draft failed: {exc}") from exc

    # Parse the model text as the requested JSON schema draft.
    text = _extract_response_text(response_payload)
    try:
        return json.loads(text)

    # Keep malformed model output visible for debugging and manual inspection.
    except json.JSONDecodeError:
        return {"llm_raw_response": response_payload, "llm_text": text}


# ----------------------------------------------------------------------------------------------------------------------
# 3. TEMPLATE AND VALIDATION

# Build one visible schema profile placeholder for manual completion.
def build_schema_profile_placeholder() -> dict[str, Any]:
    return {
        "datasets": [],
        "sequence_categorical_columns": [],
        "sequence_numerical_columns": [],
        "offer_numerical_columns": [],
        "max_prefix_length_for_encoding": None,
        "field_help": {
            "datasets": "Dataset ids covered by this profile.",
            "sequence_categorical_columns": "Categorical model inputs from contract_field_catalog.",
            "sequence_numerical_columns": "Numerical model inputs from contract_field_catalog.",
            "offer_numerical_columns": "Offer-state numerical inputs from contract_field_catalog.",
            "max_prefix_length_for_encoding": "Integer prefix cap used for padding and sample generation.",
        },
        "review_note": "Replace <profile_name> and fill every required field before approval.",
    }


# Build the canonical schema template from the contract.
def build_schema_template(schema_mode: str, openai_model: str) -> dict[str, Any]:
    return {
        "approved": False,
        "mode": schema_mode,
        "model": openai_model if schema_mode == "llm" else None,
        "reserved_tokens": {
            "pad": contract.PAD_TOKEN,
            "unknown": contract.UNK_TOKEN,
            "missing": contract.MISSING_TOKEN,
            "end": contract.END_TOKEN,
            "other_activity": contract.OTHER_ACTIVITY_TOKEN,
        },
        "contract_field_groups": contract.CONTRACT_FIELD_GROUPS,
        "contract_field_catalog": contract.FIELD_CATALOG,
        "canonical_activity_labels": contract.CANONICAL_ACTIVITY_LABELS,
        "schema_profiles": {
            "<profile_name>": build_schema_profile_placeholder(),
        },
        "review_notes": [
            "THE SECTION BELOW IS FOR MANUALLY FILLING IN YOUR DATASETS",
            "Copy the placeholder for each schema profile and remove it before approval.",
        ],
        "unresolved_items": [],
    }


# Load the target brief used only by LLM schema mode.
def load_schema_targets(path: Path) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(f"missing canonical schema target brief: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


# Check that one list contains only contract fields.
def _validate_field_list(profile_name: str, key: str, fields: Any, valid_fields: set[str]) -> None:
    if not isinstance(fields, list): raise ValueError(f"{profile_name}.{key} must be a list")
    unknown = sorted(str(field) for field in fields if str(field) not in valid_fields)
    if unknown: raise ValueError(f"{profile_name}.{key} contains unknown contract fields: {unknown}")


# Identify the manual placeholder allowed only in template files.
def is_schema_profile_placeholder(profile_name: str) -> bool:
    return profile_name == "<profile_name>"


# Validate one filled canonical schema before it enters the workflow.
def validate_canonical_schema_payload(payload: dict[str, Any], require_profiles: bool) -> None:
    # Require the sections used by manual review, LLM prompting and encoding.
    required_top_level = ["approved", "mode", "contract_field_groups", "contract_field_catalog",
                          "canonical_activity_labels", "schema_profiles", "review_notes", "unresolved_items", ]
    missing = [key for key in required_top_level if key not in payload]
    if missing: raise ValueError(f"canonical schema is missing top-level keys: {missing}")

    # Keep approval explicit and profile definitions keyed by profile name.
    if not isinstance(payload["approved"], bool):
        raise ValueError("approved must be a boolean")
    if not isinstance(payload["schema_profiles"], dict):
        raise ValueError("schema_profiles must be a dictionary keyed by profile name")
    if require_profiles and not payload["schema_profiles"]:
        raise ValueError("schema_profiles must contain at least one profile")

    # Use the contract as the only allowed source for fields and activity labels.
    valid_fields = set(contract.FIELD_CATALOG)
    activity_labels = payload["canonical_activity_labels"]
    if not isinstance(activity_labels, dict): raise ValueError("canonical_activity_labels must be a dictionary")

    # Reject activity labels not part of the contract (verifies the allowed canonical activity universe).
    unknown_labels = sorted(set(activity_labels) - set(contract.CANONICAL_ACTIVITY_LABEL_NAMES))
    if unknown_labels: raise ValueError(f"canonical_activity_labels contains unknown labels: {unknown_labels}")

    # Require every contract activity label to stay visible for review and mapping.
    missing_labels = sorted(set(contract.CANONICAL_ACTIVITY_LABEL_NAMES) - set(activity_labels))
    if missing_labels: raise ValueError(f"canonical_activity_labels is missing contract labels: {missing_labels}")

    # Validate every real schema profile selected for the experiment.
    for profile_name, profile in payload["schema_profiles"].items():

        # Allow the placeholder only in an unapproved manual template.
        if not require_profiles and is_schema_profile_placeholder(str(profile_name)): continue

        # Require the profile fields needed by the runner and on-the-fly encoder.
        for key in ["datasets", "sequence_categorical_columns", "sequence_numerical_columns",
                    "offer_numerical_columns", "max_prefix_length_for_encoding", ]:
            if key not in profile: raise ValueError(f"{profile_name} is missing {key}")

        # Each profile must name at least one dataset it can encode.
        if not isinstance(profile["datasets"], list) or not profile["datasets"]:
            raise ValueError(f"{profile_name}.datasets must be not empty list")

        # Profile input fields must be selected from the contract catalog.
        for key in ["sequence_categorical_columns", "sequence_numerical_columns", "offer_numerical_columns"]:
            _validate_field_list(str(profile_name), key, profile[key], valid_fields)

        # Prefix caps must be positive because they define tensor length and sample count.
        if int(profile["max_prefix_length_for_encoding"]) <= 0:
            raise ValueError(f"{profile_name}.max_prefix_length_for_encoding must be positive")


# Return contract fields that may appear as model inputs.
def allowed_model_input_fields() -> list[str]:
    return [
        contract.CANONICAL_ACTIVITY_TOKEN, contract.RESOURCE, contract.TIME_DELTA, contract.REQUESTED_AMOUNT_VALUE,
        contract.LOAN_GOAL, contract.APPLICATION_TYPE, contract.CREDIT_SCORE_VALUE, contract.MONTHLY_COST_VALUE,
        contract.OFFERED_AMOUNT_VALUE, contract.NUMBER_OF_TERMS_VALUE,
    ]


# Return contract fields that must not be selected as model inputs
def forbidden_model_input_fields() -> list[str]:
    return [
        contract.CASE_ID, contract.EVENT_INDEX, contract.TIMESTAMP, contract.RAW_ACTIVITY, contract.LIFECYCLE,
        contract.RAW_ACTIVITY_TOKEN, contract.NEXT_ACTIVITY_RAW, contract.REMAINING_TIME, contract.OUTCOME,
        contract.DATASET_ID, contract.CLIENT_ID, contract.SPLIT, contract.CANONICAL_ACTIVITY_LABEL,
        contract.NEXT_ACTIVITY_TARGET, contract.NEXT_ACTIVITY_MASK, contract.REMAINING_TIME_MASK,
        contract.REQUESTED_AMOUNT_MASK, contract.LOAN_GOAL_MASK, contract.APPLICATION_TYPE_MASK,
        contract.OFFER_PRESENT, contract.OFFER_FEATURE_MASK,
    ]


# Build an LLM helper recipe from the target brief.
def profile_recipe_from_targets(targets: dict[str, Any]) -> dict[str, Any]:
    recipes: dict[str, Any] = {}

    # Convert each requested profile into the schema fields the LLM should fill (canonical activity token as input).
    for profile_name, profile in targets.get("profiles", {}).items():
        categorical = [contract.CANONICAL_ACTIVITY_TOKEN]

        # Add the resource feature only when the target brief requests it.
        if profile.get("include_resource"): categorical.append(contract.RESOURCE)

        # Add BPIC 2017 static categories when the profile actually supports them.
        if profile.get("include_static_case_features") and str(profile_name) != "bpic2012":
            categorical.extend([contract.LOAN_GOAL, contract.APPLICATION_TYPE])

        # Every profile uses elapsed case time as numerical event input.
        numerical = [contract.TIME_DELTA]

        # Add the requested amount when static case features are active.
        if profile.get("include_static_case_features"): numerical.append(contract.REQUESTED_AMOUNT_VALUE)

        # Add offer numbers only for profiles with offer features.
        offers = list(contract.OFFER_NUMERICAL_COLUMNS) if profile.get("include_offer_features") else []

        # Store the filled profile skeleton in the exact schema format.
        recipes[str(profile_name)] = {
            "datasets": profile.get("datasets", []),
            "sequence_categorical_columns": categorical,
            "sequence_numerical_columns": numerical,
            "offer_numerical_columns": offers,
            "max_prefix_length_for_encoding": profile.get("prefix_cap"),
        }
    return recipes


# Build the prompt that asks the LLM to fill the schema template.
def build_llm_prompt(template: dict[str, Any], targets: dict[str, Any], prompt_strategy: str = "strategy_1") -> dict[
    str, Any]:
    # All strategies require the same profile fields in the returned JSON.
    required_profile_fields = [
        "datasets", "sequence_categorical_columns", "sequence_numerical_columns", "offer_numerical_columns",
        "max_prefix_length_for_encoding",
    ]

    # STRATEGY 1: Give the template, target brief and strict JSON instructions.
    # The Least guidance: Most LLM inference.
    prompt = {
        "instruction": (
            "Fill the E_04 canonical schema template. Return the completed JSON object only. "
            "Do not return review summaries. Preserve all top-level keys from the template. "
            "Keep approved false. schema_profiles must be an object keyed by profile name. "
            "Create one full profile for every name in profiles_to_create. "
            "Every profile must contain datasets, sequence_categorical_columns, sequence_numerical_columns, "
            "offer_numerical_columns and max_prefix_length_for_encoding. "
            "Use only contract fields and canonical activity labels from the template. "
            "Put uncertainty into unresolved_items."
        ),
        "required_schema_profiles": targets.get("profiles_to_create", []),
        "required_profile_fields": required_profile_fields,
        "schema_profile_field_meaning": {
            "datasets": "Dataset ids covered by the profile.",
            "sequence_categorical_columns": "Categorical model inputs active in the profile.",
            "sequence_numerical_columns": "Numerical model inputs active in the profile.",
            "offer_numerical_columns": "Offer-state numerical inputs active in the profile.",
            "max_prefix_length_for_encoding": "Static prefix padding length and prefix sample cap.",
        },
        "template": template,
        "target_brief": targets,
    }

    # STRATEGY 2: Add allowed and forbidden input fields to reduce invalid choices.
    # Fair guidance: Gives allowed and forbidden fields but still asks the LLM to decide.
    if prompt_strategy in {"strategy_2", "strategy_3"}:
        prompt["strategy"] = prompt_strategy
        prompt["allowed_model_input_fields"] = allowed_model_input_fields()
        prompt["forbidden_model_input_fields"] = forbidden_model_input_fields()
        prompt["field_selection_rules"] = [
            "Use canonical_activity_token as the activity input.",
            "Use resource when the target brief says include_resource is true.",
            "Use time_delta as the event-time input.",
            "Use requested_amount_value when static case features are active.",
            "Use loan_goal and application_type only when they are available for the selected profile.",
            "Use offer numerical columns only when offer features are active.",
            "Never select targets, raw source labels, ids, split labels, masks or timestamps as model inputs.",
        ]

    # STRATEGY 3: Add a target profile recipe as the strongest guidance.
    # Assisted recipe: Mostly checks whether the LLM can copy and format a schema derived from rules.
    if prompt_strategy == "strategy_3":
        prompt["target_derived_profile_recipe"] = profile_recipe_from_targets(targets)
        prompt["instruction"] += (
            " Use target_derived_profile_recipe as the preferred interpretation of the target brief. "
            "Only deviate if a listed field is not present in the contract."
        )

    return prompt


# Build the JSON body for either the manual template or the LLM draft.
def build_schema_payload(schema_mode: str, openai_model: str,
                         prompt_strategy: str = "strategy_1", targets_path: Optional[Path] = None, ) -> dict[str, Any]:
    # Start from the same contract template in both modes.
    template = build_schema_template(schema_mode, openai_model)

    # Manual mode stops at the fill-in template and does not call the API.
    if schema_mode != "llm":
        validate_canonical_schema_payload(template, require_profiles=False)
        return template

    # LLM mode loads the target brief and asks the model to fill the template.
    targets = load_schema_targets(targets_path or CANONICAL_SCHEMA_TARGETS_PATH)
    proposal: dict[str, Any] = request_openai_json(build_llm_prompt(template, targets, prompt_strategy), openai_model)

    # Wrap a non-object response so an invalid draft is still saved for review.
    if not isinstance(proposal, dict): proposal = {"llm_invalid_response": proposal}

    # Keep every LLM draft unapproved until a human reviews it.
    proposal["approved"] = False
    proposal["mode"] = "llm"
    proposal["model"] = openai_model

    # Record an invalid LLM draft instead of aborting so the side experiment can score it.
    try:
        validate_canonical_schema_payload(proposal, require_profiles=True)
    except ValueError as exc:
        proposal["validation_error"] = str(exc)
    return proposal


# Write the schema template or LLM-filled draft for human approval.
def write_canonical_schemas(path: Path, schema_mode: str, openai_model: str,
                            force: bool, prompt_strategy: str = "strategy_1", targets_path: Optional[Path] = None, ) -> \
dict[str, Any]:
    # Protect approved review files from accidental overwriting.
    encoding.refuse_approved_overwrite(path, force)

    # Build the selected schema review artifact, save the JSON file that the human reviews next.
    payload = build_schema_payload(schema_mode, openai_model, prompt_strategy, targets_path)
    encoding.save_json_artifact(path, payload)
    return payload


# ----------------------------------------------------------------------------------------------------------------------
# 4. MAIN

# Run the schema creation step.
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    output_path = args.canonical_schema_path
    if args.schema_mode == "llm" and output_path == CANONICAL_SCHEMA_PATH: output_path = LLM_CANONICAL_SCHEMA_PATH
    write_canonical_schemas(
        output_path, args.schema_mode, args.openai_model, args.force, args.prompt_strategy,
        args.canonical_schema_targets_path,
    )


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────