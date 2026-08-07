"""
Step 4.1: Load the E_04 model contract.

The editable federation contract lives in `mappings/MANUAL_contract.json`.
This module validates that JSON file and exposes stable Python constants for the schema, mapping, runner and runtime encoder.
The contract is the model-facing definition of the data (model-facing universe).

Only tensor records stay in Python because later training imports those types directly.
Feature fields, masks, reserved tokens and canonical activity labels come from the JSON contract.

IMPORTANT: The contract has to be reconfigured for a new federation. This contract works for BPIC 2012 and BPIC 2017.
For a new federation a new contract has to be created. This is the only step that is not LLM supported.

REQUIRED FILES: E_prefix_encoding/mappings/MANUAL_contract.json: editable federation contract JSON
CREATED FILES: none
"""

# IMPORTS
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Optional
import torch

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent
CONTRACT_PATH: Path = SCRIPT_DIR / "mappings" / "MANUAL_contract.json"

# ----------------------------------------------------------------------------------------------------------------------
# 1. CONTRACT LOADING

# Load the JSON contract from the disk.
def load_contract_payload(path: Optional[Path] = None) -> dict[str, Any]:
    contract_path = Path(path or CONTRACT_PATH)
    if not contract_path.exists(): raise FileNotFoundError(f"missing contract JSON: {contract_path}")
    payload = json.loads(contract_path.read_text(encoding="utf-8"))
    validate_contract_payload(payload)
    return payload

# Look up one dictionary section in the contract JSON, throw an error if it does not exist.
def _require_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value: raise ValueError(f"contract JSON requires non empty {key}")
    return value

# Look up one list section in the contract JSON, throw an error if it does not exist.
def _require_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value: raise ValueError(f"contract JSON requires non empty {key}")
    return value

# Reject schema fields that are not defined in the contract.
# Field_catalog contains allowed fields for the model; fields contain selected fields from one schema section.
def _validate_field_refs(field_catalog: dict[str, Any], key: str, fields: list[Any]) -> None:
    unknown = sorted(str(field) for field in fields if str(field) not in field_catalog)
    if unknown: raise ValueError(f"{key} contains unknown contract fields: {unknown}")

# Validate the contract JSON before constants are exposed to the encoder.
def validate_contract_payload(payload: dict[str, Any]) -> None:

    # Load the main contract sections and require the expected object shape.
    reserved_tokens = _require_dict(payload, "reserved_tokens")
    event_fields = _require_dict(payload, "event_fields")
    field_groups = _require_dict(payload, "field_groups")
    field_catalog = _require_dict(payload, "field_catalog")
    activity_labels = _require_dict(payload, "canonical_activity_labels")

    # RESERVED TOKENS must exist because vocabularies depend on fixed token names.
    for key in ["pad", "unknown", "missing", "end", "other_activity"]:
        if key not in reserved_tokens or not str(reserved_tokens[key]):
            raise ValueError(f"reserved_tokens is missing {key}")

    # EVENT FIELDS define the canonical column names used throughout E_04.
    for key in ["case_id", "timestamp", "outcome", "canonical_activity_token", "remaining_time_mask"]:
        if key not in event_fields: raise ValueError(f"event_fields is missing {key}")

    # FIELD GROUPS collect fields by semantic purpose.
    for key in ["base_event_columns", "static_case_columns", "offer_state_columns", "mapped_columns"]:
        _validate_field_refs(field_catalog, key, _require_list(field_groups, key))

    # Check field lists against the catalog of allowed contract fields.
    for key in ["base_event_columns", "static_case_columns", "offer_numerical_columns", "offer_state_columns", "mapped_columns", "required_columns"]:
        _validate_field_refs(field_catalog, key, _require_list(payload, key))

    # Check dtype lists because the encoder uses them to parse text and numeric values.
    _validate_field_refs(field_catalog, "string_columns", _require_list(payload, "string_columns"))
    _validate_field_refs(field_catalog, "numeric_columns", _require_list(payload, "numeric_columns"))

    # Every event field value must also be present in the full catalog.
    missing_catalog_fields = sorted(set(event_fields.values()) - set(field_catalog))
    if missing_catalog_fields: raise ValueError(f"field_catalog is missing event fields: {missing_catalog_fields}")

    # Each field catalog entry must describe one valid field: Dictionary with field_name, group, value_type and role.
    for field_name, field_spec in field_catalog.items():
        if not isinstance(field_spec, dict):
            raise ValueError(f"field_catalog.{field_name} must be a dictionary")
        if field_spec.get("field_name") != field_name:
            raise ValueError(f"field_catalog.{field_name}.field_name must equal {field_name}")
        for key in ["group", "value_type", "role"]:
            if key not in field_spec: raise ValueError(f"field_catalog.{field_name} is missing {key}")

    # Check each canonical activity label used by the federation.
    for label, label_spec in activity_labels.items():

        # Keep activity labels in the A, O and W activity namespaces.
        if not str(label).startswith(("A_", "O_", "W_")):
            raise ValueError(f"canonical activity label has invalid prefix: {label}")

        # Require a metadata object for each canonical activity.
        if not isinstance(label_spec, dict):
            raise ValueError(f"canonical activity {label} must be a dictionary")

        # Require a short meaning for human review and LLM mapping.
        if not str(label_spec.get("description", "")):
            raise ValueError(f"canonical activity {label} is missing description")

        # Require a true boolean so terminal activity checks stay unambiguous.
        if not isinstance(label_spec.get("terminal_candidate"), bool):
            raise ValueError(f"canonical activity {label} terminal_candidate must be boolean")

# Load the approved contract one time and expose its core sections as module constants.
_CONTRACT = load_contract_payload()
_FIELDS = _CONTRACT["event_fields"]
_TOKENS = _CONTRACT["reserved_tokens"]

# ----------------------------------------------------------------------------------------------------------------------
# 2. RESERVED TOKENS

PAD_TOKEN = _TOKENS["pad"]
UNK_TOKEN = _TOKENS["unknown"]
MISSING_TOKEN = _TOKENS["missing"]
END_TOKEN = _TOKENS["end"]
OTHER_ACTIVITY_TOKEN = _TOKENS["other_activity"]

# ----------------------------------------------------------------------------------------------------------------------
# 3. CANONICAL EVENT FIELDS

CASE_ID = _FIELDS["case_id"]
EVENT_INDEX = _FIELDS["event_index"]
TIMESTAMP = _FIELDS["timestamp"]
RAW_ACTIVITY = _FIELDS["raw_activity"]
LIFECYCLE = _FIELDS["lifecycle"]
RAW_ACTIVITY_TOKEN = _FIELDS["raw_activity_token"]
NEXT_ACTIVITY_RAW = _FIELDS["next_activity_raw"]
RESOURCE = _FIELDS["resource"]
TIME_DELTA = _FIELDS["time_delta"]
TIME_SINCE_PREVIOUS = _FIELDS["time_since_previous"]
WEEKDAY_SIN = _FIELDS["weekday_sin"]
WEEKDAY_COS = _FIELDS["weekday_cos"]
HOUR_SIN = _FIELDS["hour_sin"]
HOUR_COS = _FIELDS["hour_cos"]
REMAINING_TIME = _FIELDS["remaining_time"]
OUTCOME = _FIELDS["outcome"]
DATASET_ID = _FIELDS["dataset_id"]
CLIENT_ID = _FIELDS["client_id"]
SPLIT = _FIELDS["split"]

REQUESTED_AMOUNT_VALUE = _FIELDS["requested_amount_value"]
REQUESTED_AMOUNT_MASK = _FIELDS["requested_amount_mask"]
LOAN_GOAL = _FIELDS["loan_goal"]
LOAN_GOAL_MASK = _FIELDS["loan_goal_mask"]
APPLICATION_TYPE = _FIELDS["application_type"]
APPLICATION_TYPE_MASK = _FIELDS["application_type_mask"]

OFFER_PRESENT = _FIELDS["offer_present"]
OFFER_FEATURE_MASK = _FIELDS["offer_feature_mask"]
CREDIT_SCORE_VALUE = _FIELDS["credit_score_value"]
MONTHLY_COST_VALUE = _FIELDS["monthly_cost_value"]
OFFERED_AMOUNT_VALUE = _FIELDS["offered_amount_value"]
NUMBER_OF_TERMS_VALUE = _FIELDS["number_of_terms_value"]

CANONICAL_ACTIVITY_LABEL = _FIELDS["canonical_activity_label"]
CANONICAL_ACTIVITY_TOKEN = _FIELDS["canonical_activity_token"]
NEXT_ACTIVITY_TARGET = _FIELDS["next_activity_target"]
NEXT_ACTIVITY_MASK = _FIELDS["next_activity_mask"]
REMAINING_TIME_MASK = _FIELDS["remaining_time_mask"]

# ----------------------------------------------------------------------------------------------------------------------
# 4. FIELD GROUPS AND FIELD CATALOG

BASE_EVENT_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["base_event_columns"])
STATIC_CASE_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["static_case_columns"])
OFFER_NUMERICAL_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["offer_numerical_columns"])
OFFER_STATE_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["offer_state_columns"])
MAPPED_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["mapped_columns"])
REQUIRED_COLUMNS: tuple[str, ...] = tuple(_CONTRACT["required_columns"])
STRING_COLUMNS = set(_CONTRACT["string_columns"])
NUMERIC_COLUMNS = set(_CONTRACT["numeric_columns"])
CONTRACT_FIELD_GROUPS = _CONTRACT["field_groups"]
FIELD_CATALOG = _CONTRACT["field_catalog"]
SPLITS: tuple[str, ...] = tuple(_CONTRACT["splits"])

# ----------------------------------------------------------------------------------------------------------------------
# 5. CANONICAL ACTIVITY LABELS

CANONICAL_ACTIVITY_LABELS = _CONTRACT["canonical_activity_labels"]
CANONICAL_ACTIVITY_LABEL_NAMES: tuple[str, ...] = tuple(CANONICAL_ACTIVITY_LABELS)

# ----------------------------------------------------------------------------------------------------------------------
# 6. DATA RECORDS

# These records define the typed handoff between metadata creation, on-the-fly encoding and training.
# They keep prefix references, artifact paths and batch tensors explicit without storing the full encoded datasets.

# Reference one prefix without storing padded tensors.
@dataclass(frozen=True)
class PrefixIndexRow:
    dataset_id: str
    case_id: str
    client_id: str
    split: str
    prefix_length: int
    label_pos: int

# Store compact JSON artifact paths for one E_04 run.
@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    encoding_spec: Path
    vocabulary: Path
    scaler: Path
    mapping_report: Path

# Store the tensor fields consumed by the next training stage.
@dataclass(frozen=True)
class EncodedBatch:
    categorical_ids: torch.Tensor
    numerical: torch.Tensor
    offer_numerical: torch.Tensor
    offer_feature_mask: torch.Tensor
    padding_mask: torch.Tensor
    prefix_length: torch.Tensor
    outcome_label: torch.Tensor
    next_activity_label: torch.Tensor
    next_activity_mask: torch.Tensor
    remaining_time_label: torch.Tensor
    remaining_time_mask: torch.Tensor

PUBLIC_HELPERS = {
    "PrefixIndexRow", "ArtifactPaths", "EncodedBatch", "load_contract_payload", "validate_contract_payload",
}

# Export contract constants, data records and public helper functions, globals() returns all module constants.
EXPORTED_NAMES = [
    name
    for name in globals()
    if name.isupper() or name in PUBLIC_HELPERS
]
__all__ = EXPORTED_NAMES

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────