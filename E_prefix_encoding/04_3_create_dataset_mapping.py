"""
Step 4.3: Create the E_04 dataset mapping.

Run this script after `04_2_create_canonical_schema.py`.
It reads the approved canonical schema, inspects the processed split parquets for the selected schema profile and
writes a dataset mapping review file. The mapping connects source parquet columns to canonical schema fields.
It also groups raw activity labels into approved canonical activity labels.

- MANUAL mode writes a fill-in template.
- SEMANTIC mode writes a deterministic similarity draft.
- LLM mode writes an OpenAI-generated draft.
    - strategy_1: one prompt for everything.
    - strategy_2: separate prompts for columns and activities.
    - strategy_3: activity mapping as classification.

All modes keep `"approved": false.
Review the file, edit it if needed and then change only the approval flag to `true`.
Only an approved dataset mapping is used by `04_4_runner.py`.

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_contract.json: contract loaded through 04_1_contract.py
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved canonical schema
    E_main_BPIC_2017/data/processed/*/*.parquet: BPIC 2017 split parquets when selected
    E_ablation_BPIC_2012/data/processed/*/*.parquet: BPIC 2012 split parquets when selected
    OPENAI_API_KEY: required only in llm mode

CREATED FILES:
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: default manual review file
    E_prefix_encoding/mappings/llm_mapping/dataset_mappings/*.json: optional LLM side experiment drafts
"""

# IMPORTS
from __future__ import annotations
import argparse
import importlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional
import pandas as pd

# Allow direct script execution from the repository root.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the frozen contract and reusable encoding helpers.
contract = importlib.import_module("E_prefix_encoding.04_1_contract")
encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent                                  # Folder that contains this script
PROJECT_ROOT: Path = SCRIPT_DIR.parent                                              # Repository root
MAPPING_ROOT: Path = SCRIPT_DIR / "mappings"                                        # Reviewed schema / mapping files
SCHEMA_PROFILE: str = "bpic2017"                                                    # bpic2017 | bpic2012 | joint
MAPPING_MODE: str = "llm"                                                           # manual | semantic | llm
OPENAI_MODEL: str = "gpt-5-nano"                                                    # OpenAI model used in llm mode
PROMPT_STRATEGY: str = "strategy_1"                                                 # strategy_1 | strategy_2 | strategy_3
SEMANTIC_ACTIVITY_SIMILARITY: str = "character"                                     # character | word
FORCE: bool = False                                                                 # True: Overwrite approved mapping
OPENAI_TIMEOUT_SECONDS: int = int(os.environ.get("OPENAI_TIMEOUT_SECONDS", "240"))  # Timeout for OpenAI API
OPENAI_RETRIES: int = int(os.environ.get("OPENAI_RETRIES", "2"))                    # Retries for OpenAI API
METADATA_SAMPLE_FILES: int = 3                                                      # Train files inspected
CANONICAL_SCHEMA_PATH: Path = MAPPING_ROOT / "MANUAL_canonical_schemas.json"        # Approved canonical schema input
DATASET_MAPPING_PATH: Path = MAPPING_ROOT / "MANUAL_dataset_mapping.json"           # Mapping review output destination

DATASET_INPUTS = {
    "bpic2017": {
        "input_root": PROJECT_ROOT / "E_main_BPIC_2017/data/processed",
        "split_prefix": "A_02",
    },
    "bpic2012": {
        "input_root": PROJECT_ROOT / "E_ablation_BPIC_2012/data/processed",
        "split_prefix": "B_02",
    },
}

# ----------------------------------------------------------------------------------------------------------------------
# 1. CLI OVERRIDES

# Parse optional automation arguments while keeping script defaults for WORKFLOW_run_encoding.sh.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the E_04 dataset mapping.")
    parser.add_argument("--schema-profile", default=SCHEMA_PROFILE)
    parser.add_argument("--mapping-mode", default=MAPPING_MODE, choices=["manual", "semantic", "llm"])
    parser.add_argument("--openai-model", default=OPENAI_MODEL)
    parser.add_argument("--prompt-strategy", default=PROMPT_STRATEGY, choices=["strategy_1", "strategy_2", "strategy_3"])
    parser.add_argument("--semantic-activity-similarity", default=SEMANTIC_ACTIVITY_SIMILARITY, choices=["character", "word"])
    parser.add_argument("--force", action="store_true", default=FORCE)
    parser.add_argument("--canonical-schema-path", type=Path, default=CANONICAL_SCHEMA_PATH)
    parser.add_argument("--dataset-mapping-path", type=Path, default=DATASET_MAPPING_PATH)
    return parser.parse_args(argv)

# ----------------------------------------------------------------------------------------------------------------------
# 2. OPENAI HELPERS

# Extract model text from one Responses API result.
def _extract_response_text(response_payload: dict[str, Any]) -> str:

    # Use the direct text field when the API returns the compact shape.
    if "output_text" in response_payload: return str(response_payload["output_text"])

    # Fall back to collecting nested text chunks from output content blocks.
    texts: list[str] = []
    for item in response_payload.get("output", []):
        for content in item.get("content", []):
            if "text" in content: texts.append(str(content["text"]))
    return "\n".join(texts)

# Request one JSON draft (only when llm mode is selected).
def request_openai_json(prompt: dict[str, Any], model: str, api_key: Optional[str] = None) -> dict[str, Any]:

    # Read the API key from the argument or the local environment.
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key: raise RuntimeError("OPENAI_API_KEY is required for LLM mode.")

    # Send the mapping prompt as JSON to the OpenAI Responses API.
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
                    f"OpenAI mapping draft timed out after {OPENAI_TIMEOUT_SECONDS} seconds "
                    f"and {OPENAI_RETRIES} attempt(s)."
                ) from exc
        except urllib.error.URLError as exc:
            if attempt == OPENAI_RETRIES: raise RuntimeError(f"OpenAI mapping draft failed: {exc}") from exc

    # Parse the model text as the requested JSON mapping draft.
    text = _extract_response_text(response_payload)
    try: return json.loads(text)

    # Keep malformed model output visible for debugging and manual inspection.
    except json.JSONDecodeError: return {"llm_raw_response": response_payload, "llm_text": text}

# ----------------------------------------------------------------------------------------------------------------------
# 3. SCHEMA AND METADATA

# HELPER: Normalize names before deterministic matching.
def _name_tokens(value: str) -> list[str]:
    normalized = re.sub(r"[^A-Za-z0-9]+", " ", value).lower()
    tokens = [token for token in normalized.split() if token not in {"case", "event", "value"}]
    return tokens

# HELPER: Return the review name for one semantic activity similarity variant.
def _semantic_variant_name(similarity_mode: str) -> str:
    if similarity_mode not in {"character", "word"}: raise ValueError(f"unknown semantic activity similarity: {similarity_mode}")
    return f"semantic_{similarity_mode}"

# HELPER: Return the activity origin prefix used by BPIC labels.
def _activity_origin_prefix(value: str) -> str:
    text = str(value)
    return text.split("_", 1)[0] if "_" in text else ""

# HELPER: Pick the most likely raw activity label column from available columns.
def _activity_label_column(columns: list[str]) -> Optional[str]:
    for column in columns:
        tokens = _name_tokens(column)
        if tokens in [["activity"], ["activity", "name"], ["concept", "name"]]: return column
    return None

# HELPER: Resolve one schema profile from the approved schema file.
def get_schema_profile(schema_payload: dict[str, Any], schema_profile: str) -> dict[str, Any]:
    profiles = schema_payload.get("schema_profiles", {})
    if not isinstance(profiles, dict) or schema_profile not in profiles:
        raise ValueError(f"unknown schema profile: {schema_profile}")
    return profiles[schema_profile]

# HELPER: Find train split parquets for one dataset input block.
def _discover_train_parquets(input_root: Path, split_prefix: str) -> list[Path]:
    return sorted(Path(input_root).glob(f"*_*banks/{split_prefix}_bank_*_train.parquet"))

# HELPER: Find every split parquet for raw-label discovery in the LLM mapping experiment.
def _discover_activity_label_parquets(input_root: Path, split_prefix: str) -> list[Path]:
    return sorted(Path(input_root).glob(f"*_*banks/{split_prefix}_bank_*.parquet"))

# HELPER: Inspect available columns and raw activity labels from processed parquets.
def discover_dataset_metadata(dataset_id: str, input_block: dict[str, Any], require_files: bool) -> dict[str, Any]:

    # input_block names the processed parquet folder and A_02 or B_02 filename prefix.
    input_root = Path(input_block["input_root"])
    split_prefix = str(input_block["split_prefix"])
    paths = _discover_train_parquets(input_root, split_prefix)
    activity_paths = _discover_activity_label_parquets(input_root, split_prefix)
    if require_files and not paths:
        raise FileNotFoundError(f"no train split parquets found for {dataset_id} under {input_root}")

    # Collect train only column metadata, because columns and dtypes are stable across splits.
    columns: dict[str, str] = {}
    raw_activity_labels: set[str] = set()
    for path in paths[:METADATA_SAMPLE_FILES]:
        frame = pd.read_parquet(path)
        for column, dtype in frame.dtypes.items(): columns[str(column)] = str(dtype)

    # Read the full raw-label universe for the semantic and LLM mapping drafts.
    activity_column = _activity_label_column(list(columns))
    if activity_column:
        for path in activity_paths:
            frame = pd.read_parquet(path, columns=[activity_column])
            raw_activity_labels.update(str(value) for value in frame[activity_column].dropna().unique().tolist())
    elif paths:
        frame = pd.read_parquet(paths[0])
        activity_column = _activity_label_column([str(column) for column in frame.columns])
        if activity_column:
            raw_activity_labels.update(str(value) for value in frame[activity_column].dropna().unique().tolist())

    return {
        "dataset_id": dataset_id,
        "input_root": str(input_root),
        "split_prefix": split_prefix,
        "sampled_files": [str(path) for path in paths[:METADATA_SAMPLE_FILES]],
        "available_columns": columns,
        "raw_activity_labels": sorted(raw_activity_labels),
    }

# Collect metadata for every dataset in the selected profile.
def discover_profile_metadata(profile: dict[str, Any], require_files: bool) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for dataset_id in profile.get("datasets", []):
        # Stop when the schema references a dataset without configured parquet inputs.
        if dataset_id not in DATASET_INPUTS: raise ValueError(f"missing DATASET_INPUTS entry for {dataset_id}")
        metadata[str(dataset_id)] = discover_dataset_metadata(str(dataset_id), DATASET_INPUTS[str(dataset_id)], require_files)
    return metadata

# ----------------------------------------------------------------------------------------------------------------------
# 4. SEMANTIC DRAFTING (this is federation-specific, so it only works for BPIC experiments)

FIELD_ALIASES = {
    contract.CASE_ID: ["case id", "case concept name", "concept name case"],
    contract.TIMESTAMP: ["timestamp", "time timestamp", "event time"],
    contract.RAW_ACTIVITY: ["activity", "concept name", "event name"],
    contract.LIFECYCLE: ["lifecycle", "lifecycle transition"],
    contract.RAW_ACTIVITY_TOKEN: ["activity token"],
    contract.NEXT_ACTIVITY_RAW: ["next activity"],
    contract.RESOURCE: ["resource", "org resource"],
    contract.TIME_DELTA: ["time delta", "elapsed time"],
    contract.REMAINING_TIME: ["remaining time"],
    contract.OUTCOME: ["outcome", "target", "label"],
    contract.REQUESTED_AMOUNT_VALUE: ["requested amount", "amount req", "application amount"],
    contract.LOAN_GOAL: ["loan goal", "loan purpose"],
    contract.APPLICATION_TYPE: ["application type"],
    contract.CREDIT_SCORE_VALUE: ["credit score"],
    contract.MONTHLY_COST_VALUE: ["monthly cost"],
    contract.OFFERED_AMOUNT_VALUE: ["offered amount"],
    contract.NUMBER_OF_TERMS_VALUE: ["number terms", "number of terms"],
}

# Score one source column against one canonical field.
def _column_score(field: str, column: str) -> float:
    column_text = " ".join(_name_tokens(column))
    aliases = FIELD_ALIASES.get(field, [field.replace("_", " ")])
    # SequenceMatcher(...).ratio() returns a string similarity score between 0.0 and 1.0:
    # 1.0 = identical or almost identical 0.0 = no similarity -> higher value = more likely match.
    return max(SequenceMatcher(None, column_text, alias).ratio() for alias in aliases)

# Map source columns to canonical fields when confidence is high enough.
def build_column_mapping(metadata: dict[str, Any], empty: bool) -> tuple[dict[str, str], list[str]]:
    # Active fields are canonical fields that must come from a parquet column or a default value.
    active_fields = [
        field
        for field in contract.REQUIRED_COLUMNS
        if field not in {contract.DATASET_ID, contract.CLIENT_ID, contract.SPLIT, contract.EVENT_INDEX}
    ]

    # Manual mode returns empty fields so the human can fill the mapping directly.
    if empty: return {field: "" for field in active_fields}, active_fields

    # Semantic mode picks the closest source column by normalized name similarity (see SequenceMatcher(...)).
    columns = list(metadata["available_columns"])
    mapping: dict[str, str] = {}
    unresolved: list[str] = []
    for field in active_fields:
        best_column = ""
        best_score = 0.0
        for column in columns:
            score = _column_score(str(field), str(column))
            if score > best_score:
                best_column = str(column)
                best_score = score
        # Require a conservative similarity score -> weak name matches stay unresolved.
        if best_score >= 0.72:
            mapping[str(field)] = best_column
        else:
            mapping[str(field)] = ""
            unresolved.append(str(field))
    return mapping, unresolved

# Create default values for fields absent from one profile.
def build_default_values(profile: dict[str, Any]) -> dict[str, Any]:

    # Profile columns decide which static and offer fields are active for this schema.
    categorical_columns = set(profile.get("sequence_categorical_columns", []))
    offer_columns = set(profile.get("offer_numerical_columns", []))
    defaults: dict[str, Any] = {contract.REQUESTED_AMOUNT_MASK: 1}

    # Missing static category features get the missing token and a zero availability mask.
    if contract.LOAN_GOAL not in categorical_columns:
        defaults[contract.LOAN_GOAL] = contract.MISSING_TOKEN
        defaults[contract.LOAN_GOAL_MASK] = 0
    else:
        defaults[contract.LOAN_GOAL_MASK] = 1
    if contract.APPLICATION_TYPE not in categorical_columns:
        defaults[contract.APPLICATION_TYPE] = contract.MISSING_TOKEN
        defaults[contract.APPLICATION_TYPE_MASK] = 0
    else:
        defaults[contract.APPLICATION_TYPE_MASK] = 1

    # Profiles without offer features receive zero masks and missing offer values.
    if not offer_columns:
        defaults[contract.OFFER_PRESENT] = 0
        defaults[contract.OFFER_FEATURE_MASK] = 0
        for column in contract.OFFER_NUMERICAL_COLUMNS:
            defaults[column] = None
    return defaults

# HELPER: Read the contract-approved canonical activity label names.
def allowed_canonical_activity_labels(schema_payload: dict[str, Any]) -> list[str]:

    # The schema carries the contract activity universe into the mapping review file
    labels = schema_payload.get("canonical_activity_labels", contract.CANONICAL_ACTIVITY_LABELS)
    if not isinstance(labels, dict): raise ValueError("canonical_activity_labels must be a dictionary")
    return sorted(str(label) for label in labels)

# HELPER: Score one raw label against one contract-approved canonical label.
def _activity_score(raw_label: str, canonical_label: str, similarity_mode: str = "character") -> float:
    raw_tokens = [token for token in _name_tokens(raw_label) if token not in {"a", "o", "w"}]
    label_tokens = [token for token in _name_tokens(canonical_label) if token not in {"a", "o", "w"}]
    description = contract.CANONICAL_ACTIVITY_LABELS.get(canonical_label, {}).get("description", "")
    description_tokens = _name_tokens(str(description))
    if similarity_mode == "character":
        raw_text = " ".join(raw_tokens)
        label_text = " ".join(label_tokens)
        description_text = " ".join(description_tokens)
        return max(
            SequenceMatcher(None, raw_text, label_text).ratio(),
            SequenceMatcher(None, raw_text, description_text).ratio(),
        )
    if similarity_mode == "word":
        return max(
            SequenceMatcher(None, raw_tokens, label_tokens).ratio(),
            SequenceMatcher(None, raw_tokens, description_tokens).ratio(),
        )
    raise ValueError(f"unknown semantic activity similarity: {similarity_mode}")

# Pick the closest contract-approved activity label for semantic mode.
def _semantic_activity_label(raw_label: str, allowed_labels: list[str], similarity_mode: str = "character") -> Optional[str]:
    best_label = ""
    best_score = 0.0
    raw_prefix = _activity_origin_prefix(raw_label)
    for canonical_label in allowed_labels:
        # Preserve A, O and W namespaces during deterministic drafting.
        if _activity_origin_prefix(canonical_label) != raw_prefix: continue
        score = _activity_score(raw_label, canonical_label, similarity_mode=similarity_mode)
        if score > best_score:
            best_label = canonical_label
            best_score = score
    return best_label if best_score >= 0.55 else None

# Build a deterministic raw-label grouping from discovered labels.
def build_activity_mapping(metadata_by_dataset: dict[str, dict[str, Any]],
    empty: bool, allowed_labels: list[str], semantic_activity_similarity: str = "character") -> dict[str, Any]:

    # base is the review shape shared by manual, semantic and LLM mapping modes.
    base = {
        "allowed_canonical_activity_labels": allowed_labels,
        "canonical_activities": {},
        "unresolved_labels": [],
    }

    # Manual mode leaves the activity groups empty for human completion.
    if empty: return base

    # Semantic mode groups each discovered raw label under the closest allowed label.
    groups: dict[str, dict[str, Any]] = {}
    unresolved: list[str] = []
    for dataset_id, metadata in metadata_by_dataset.items():
        for raw_label in metadata.get("raw_activity_labels", []):
            canonical_label = _semantic_activity_label(
                str(raw_label), allowed_labels, similarity_mode=semantic_activity_similarity,
            )
            if canonical_label is None:
                unresolved.append(str(raw_label))
                continue

            # Store labels by dataset so BPIC 2017 and BPIC 2012 remain reviewable.
            group = groups.setdefault(
                canonical_label,
                {
                    "labels_by_dataset": {},
                    "token_overrides": {},
                    "rationale": "Deterministic draft from normalized activity label.",
                },
            )
            group["labels_by_dataset"].setdefault(dataset_id, []).append(str(raw_label))
    base["canonical_activities"] = groups
    base["unresolved_labels"] = sorted(unresolved)
    return base

# ----------------------------------------------------------------------------------------------------------------------
# 5. PAYLOAD CREATION

# Build dataset mapping blocks for every dataset in the selected profile.
def build_dataset_blocks(profile: dict[str, Any], metadata_by_dataset: dict[str, dict[str, Any]],
    empty: bool,) -> tuple[dict[str, Any], list[str]]:

    datasets: dict[str, Any] = {}
    unresolved: list[str] = []

    # offer_columns controls which source offer fields are active in this schema profile.
    offer_columns = set(profile.get("offer_numerical_columns", []))
    for dataset_id, metadata in metadata_by_dataset.items():
        # Build one dataset block with parquet columns, the defaults and computed event_index.
        mapping, missing_fields = build_column_mapping(metadata, empty)
        unresolved.extend(f"{dataset_id}:{field}" for field in missing_fields)
        input_block = DATASET_INPUTS[dataset_id]
        datasets[dataset_id] = {
            "input_root": str(input_block["input_root"]),
            "split_prefix": str(input_block["split_prefix"]),
            "column_mapping": mapping,
            "default_values": build_default_values(profile),
            "offer_source_columns": [field for field in contract.OFFER_NUMERICAL_COLUMNS if field in offer_columns],
            "computed_fields": {contract.EVENT_INDEX: {"operation": "cumcount", "group_by": contract.CASE_ID}},
            "unresolved_fields": missing_fields,
        }
    return datasets, unresolved

# Build the split LLM prompt for parquet-column mapping.
def build_llm_column_prompt(profile: dict[str, Any], metadata_by_dataset: dict[str, dict[str, Any]],
                            datasets: dict[str, Any], prompt_strategy: str,) -> dict[str, Any]:

    # Strategy 2 and 3 split the LLM task so column mapping is judged separately.
    return {
        "task": "Fill only the datasets column_mapping blocks for E_04.",
        "strategy": prompt_strategy,
        "output_contract": (
            "Return JSON with one top-level key named datasets. Preserve input_root, split_prefix, "
            "default_values, offer_source_columns and computed_fields from the seed. "
            "Only improve column_mapping and unresolved_fields."
        ),
        "rules": [
            "Use source parquet column names exactly as listed in available_columns.",
            "Prefer case:concept:name over concept:name for case_id.",
            "Use concept:name for raw_activity.",
            "Use activity_token for raw_activity_token.",
            "Use NextActivity for next_activity_raw.",
            "Leave a field unresolved only when no listed column or default can represent it.",
        ],
        "schema_profile": profile,
        "metadata_by_dataset": metadata_by_dataset,
        "seed_datasets": datasets,
    }

# Build the split LLM prompt for activity mapping.
def build_llm_activity_prompt(profile: dict[str, Any], metadata_by_dataset: dict[str, dict[str, Any]],
    allowed_labels: list[str], prompt_strategy: str,) -> dict[str, Any]:

    # label_descriptions give the LLM the contract meaning without showing manual mappings.
    label_descriptions = {
        label: contract.CANONICAL_ACTIVITY_LABELS[label]
        for label in allowed_labels
        if label in contract.CANONICAL_ACTIVITY_LABELS
    }
    prompt = {
        "task": "Map raw activity labels to contract-approved canonical activity labels.",
        "strategy": prompt_strategy,
        "output_contract": (
            "Return JSON with one top-level key named activity_mapping. "
            "The activity_mapping must contain allowed_canonical_activity_labels, canonical_activities and unresolved_labels. "
            "canonical_activities must be a dictionary keyed by canonical label. "
            "Each canonical activity group must contain labels_by_dataset, token_overrides and rationale."
        ),
        "rules": [
            "Use only labels from allowed_canonical_activity_labels as canonical group names.",
            "preserve the A, O and W prefix. Raw A labels map only to A labels. Raw O labels map only to O labels. Raw W labels map only to W labels.",
            "Map obvious spelling, tense, capitalization and Dutch-English equivalents instead of leaving them unresolved.",
            "Use unresolved_labels only when no same-prefix canonical label is semantically defensible.",
            "Do not invent canonical labels.",
        ],
        "schema_profile": profile,
        "allowed_canonical_activity_labels": allowed_labels,
        "canonical_activity_descriptions": label_descriptions,
        "canonical_activity_group_template": {
            "canonical_label": {
                "labels_by_dataset": {"dataset_id": ["raw activity label"]},
                "token_overrides": {},
                "rationale": "Short reason for the mapping.",
            }
        },
        "metadata_by_dataset": {
            dataset_id: {"raw_activity_labels": metadata.get("raw_activity_labels", [])}
            for dataset_id, metadata in metadata_by_dataset.items()
        },
    }

    # Strategy 3 frames the activity task as label classification to reduce summaries.
    if prompt_strategy == "strategy_3":
        prompt["classification_format"] = (
            "First classify every raw label independently. Then assemble canonical_activities from those classifications. "
            "This is a classification task, not a summarization task."
        )
    return prompt

# Merge an LLM column response into the seed mapping payload.
def merge_column_response(payload: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:

    # Preserve malformed output for review instead of silently discarding it.
    datasets = proposal.get("datasets", proposal)
    if not isinstance(datasets, dict):
        payload["llm_column_proposal"] = proposal
        return payload

    # Copy only the fields the column prompt is allowed to change.
    for dataset_id, dataset_block in datasets.items():
        if dataset_id not in payload["datasets"] or not isinstance(dataset_block, dict):
            continue
        if "column_mapping" in dataset_block:
            payload["datasets"][dataset_id]["column_mapping"] = dataset_block["column_mapping"]
        if "unresolved_fields" in dataset_block:
            payload["datasets"][dataset_id]["unresolved_fields"] = dataset_block["unresolved_fields"]
    return payload

# Merge an LLM activity response into the seed mapping payload.
def merge_activity_response(payload: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:

    # Preserve malformed activity output for manual inspection.
    activity_mapping = proposal.get("activity_mapping", proposal)
    if not isinstance(activity_mapping, dict) or not isinstance(activity_mapping.get("canonical_activities"), dict):
        payload["llm_activity_proposal"] = proposal
        return payload
    payload["activity_mapping"] = activity_mapping
    return payload

# Build one complete mapping payload without writing it.
def build_dataset_mapping_payload(schema_payload: dict[str, Any], schema_path: Path, schema_profile: str,
    mapping_mode: str, openai_model: str, prompt_strategy: str = "strategy_1",
    semantic_activity_similarity: str = "character") -> dict[str, Any]:

    # Resolve the approved schema profile and activity universe for this mapping run.
    profile = get_schema_profile(schema_payload, schema_profile)
    allowed_labels = allowed_canonical_activity_labels(schema_payload)

    # Manual mode can run without parquet files, draft modes inspect source metadata.
    require_files = mapping_mode in {"semantic", "llm"}
    metadata_by_dataset = discover_profile_metadata(profile, require_files=require_files)
    empty = mapping_mode == "manual"
    datasets, unresolved_fields = build_dataset_blocks(profile, metadata_by_dataset, empty)

    # payload is the single review artifact edited and approved by the human.
    payload = {
        "approved": False,
        "mode": mapping_mode,
        "semantic_variant": _semantic_variant_name(semantic_activity_similarity),
        "semantic_activity_similarity": semantic_activity_similarity,
        "model": openai_model if mapping_mode == "llm" else None,
        "schema_profile": schema_profile,
        "schema_path": str(schema_path),
        "schema_sha256": encoding.json_sha256(schema_path),
        "datasets": datasets,
        "activity_mapping": build_activity_mapping(
            metadata_by_dataset, empty, allowed_labels, semantic_activity_similarity=semantic_activity_similarity,
        ),
        "metadata_summary": metadata_by_dataset,
        "review_notes": [
            "Review column mappings, defaults, masks and activity groups before approval.",
        ],
        "unresolved_items": unresolved_fields,
    }

    # Validate the seed draft before optional LLM changes are merged.
    encoding.validate_activity_mapping_payload(payload)
    if mapping_mode != "llm": return payload

    # Strategy 2 and 3 use split prompts for columns and activities.
    if prompt_strategy in {"strategy_2", "strategy_3"}:
        column_prompt = build_llm_column_prompt(profile, metadata_by_dataset, datasets, prompt_strategy)
        activity_prompt = build_llm_activity_prompt(profile, metadata_by_dataset, allowed_labels, prompt_strategy)
        payload = merge_column_response(payload, request_openai_json(column_prompt, openai_model))
        payload = merge_activity_response(payload, request_openai_json(activity_prompt, openai_model))
        payload["prompt_strategy"] = prompt_strategy
        payload["semantic_variant"] = _semantic_variant_name(semantic_activity_similarity)
        payload["semantic_activity_similarity"] = semantic_activity_similarity
        try: encoding.validate_activity_mapping_payload(payload)
        except ValueError as exc: payload["validation_error"] = str(exc)
        return payload

    # Strategy 1 asks the LLM to improve the full seed payload in one request.
    prompt = {
        "instruction": (
            "Review and improve this E_04 dataset mapping draft. Return JSON only with approved false, "
            "schema_profile, schema_sha256, datasets, activity_mapping, review_notes and unresolved_items. "
            "Use only allowed_canonical_activity_labels for canonical activity groups."
        ),
        "schema_profile": profile,
        "allowed_canonical_activity_labels": allowed_labels,
        "metadata_by_dataset": metadata_by_dataset,
        "seed_payload": payload,
    }
    proposal = request_openai_json(prompt, openai_model)

    # Accept a complete mapping only when it preserves the required review sections.
    if "datasets" in proposal and "activity_mapping" in proposal:
        proposal["approved"] = False
        proposal["mode"] = "llm"
        proposal["model"] = openai_model
        proposal["schema_profile"] = schema_profile
        proposal["schema_path"] = str(schema_path)
        proposal["schema_sha256"] = encoding.json_sha256(schema_path)
        proposal["prompt_strategy"] = prompt_strategy
        proposal["semantic_variant"] = _semantic_variant_name(semantic_activity_similarity)
        proposal["semantic_activity_similarity"] = semantic_activity_similarity
        try: encoding.validate_activity_mapping_payload(proposal)
        except ValueError as exc: proposal["validation_error"] = str(exc)
        return proposal

    # Keep a partial LLM output visible when it cannot replace the seed payload.
    payload["llm_proposal"] = proposal
    payload["prompt_strategy"] = prompt_strategy
    return payload

# Write the selected dataset mapping for human approval.
def write_dataset_mapping(schema_path: Path, mapping_path: Path, schema_profile: str, mapping_mode: str,
    openai_model: str, force: bool, prompt_strategy: str = "strategy_1",
    semantic_activity_similarity: str = "character") -> dict[str, Any]:

    # Protect approved review files from accidental overwriting.
    encoding.refuse_approved_overwrite(mapping_path, force)

    # Load only approved canonical schemas before creating dataset mappings.
    schema_payload = encoding.load_approved_json(schema_path, "canonical schema")
    payload = build_dataset_mapping_payload(
        schema_payload, schema_path, schema_profile, mapping_mode, openai_model, prompt_strategy,
        semantic_activity_similarity,
    )

    # Save the mapping review file used by the runner after human approval.
    encoding.save_json_artifact(mapping_path, payload)
    return payload

# ----------------------------------------------------------------------------------------------------------------------
# 6. MAIN

# Run the dataset mapping step.
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    write_dataset_mapping(
        args.canonical_schema_path, args.dataset_mapping_path, args.schema_profile, args.mapping_mode,
        args.openai_model, args.force, args.prompt_strategy, args.semantic_activity_similarity,
    )

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────