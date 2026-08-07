"""
Step 4.5: Reusable E_04 encoding library.

This file is NOT a user workflow script.
It contains the shared runtime code used by the dataset mapping creator, runner and later E_05 training.

The stable model facing contract lives in `04_1_contract.py`.
This library loads approved JSON files, maps the processed split parquets into canonical events, fits vocabularies
and scalers, builds compact prefix references and creates prefix tensors on demand through the PrefixDataset class.

The library never saves full prefix tensors.
The runner writes compact JSON metadata, while PrefixDataset creates padded tensors when training asks for a sample.
On-the-fly encoded samples are cached in memory for faster training.

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_contract.json: contract loaded through 04_1_contract.py
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved schema when called by runner or training
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved mapping when called by runner or training
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: vocabulary metadata when PrefixDataset is loaded
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: scaler metadata when PrefixDataset is loaded for training

CREATED FILES:
    E_prefix_encoding/encoded_metadata/*/*/*_encoding_spec.json: created by write_run_artifacts through the runner
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: created by write_run_artifacts through the runner
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: created by write_run_artifacts through the runner
    E_prefix_encoding/encoded_metadata/*/*/*_mapping_report.json: created by write_run_artifacts through the runner
"""

# IMPORTS
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
import torch

# Load the stable model facing contract.
contract: Any = importlib.import_module("E_prefix_encoding.04_1_contract")

# EXPORT contract names used throughout runtime encoding and later training.
# This keeps 04_5_encoding.py as the single import surface for PrefixDataset callers.

# SECTION: Reserved vocabulary tokens
PAD_TOKEN = contract.PAD_TOKEN
UNK_TOKEN = contract.UNK_TOKEN
MISSING_TOKEN = contract.MISSING_TOKEN
END_TOKEN = contract.END_TOKEN
OTHER_ACTIVITY_TOKEN = contract.OTHER_ACTIVITY_TOKEN

# SECTION: Base event columns
CASE_ID = contract.CASE_ID
EVENT_INDEX = contract.EVENT_INDEX
TIMESTAMP = contract.TIMESTAMP
RAW_ACTIVITY = contract.RAW_ACTIVITY
LIFECYCLE = contract.LIFECYCLE
RAW_ACTIVITY_TOKEN = contract.RAW_ACTIVITY_TOKEN
NEXT_ACTIVITY_RAW = contract.NEXT_ACTIVITY_RAW
RESOURCE = contract.RESOURCE
TIME_DELTA = contract.TIME_DELTA
TIME_SINCE_PREVIOUS = contract.TIME_SINCE_PREVIOUS
WEEKDAY_SIN = contract.WEEKDAY_SIN
WEEKDAY_COS = contract.WEEKDAY_COS
HOUR_SIN = contract.HOUR_SIN
HOUR_COS = contract.HOUR_COS
REMAINING_TIME = contract.REMAINING_TIME
OUTCOME = contract.OUTCOME
DATASET_ID = contract.DATASET_ID
CLIENT_ID = contract.CLIENT_ID
SPLIT = contract.SPLIT

# SECTION: Static case feature columns
REQUESTED_AMOUNT_VALUE = contract.REQUESTED_AMOUNT_VALUE
REQUESTED_AMOUNT_MASK = contract.REQUESTED_AMOUNT_MASK
LOAN_GOAL = contract.LOAN_GOAL
LOAN_GOAL_MASK = contract.LOAN_GOAL_MASK
APPLICATION_TYPE = contract.APPLICATION_TYPE
APPLICATION_TYPE_MASK = contract.APPLICATION_TYPE_MASK

# SECTION: Offer feature columns
OFFER_PRESENT = contract.OFFER_PRESENT
OFFER_FEATURE_MASK = contract.OFFER_FEATURE_MASK
CREDIT_SCORE_VALUE = contract.CREDIT_SCORE_VALUE
MONTHLY_COST_VALUE = contract.MONTHLY_COST_VALUE
OFFERED_AMOUNT_VALUE = contract.OFFERED_AMOUNT_VALUE
NUMBER_OF_TERMS_VALUE = contract.NUMBER_OF_TERMS_VALUE

# SECTION: Canonical activity and target columns
CANONICAL_ACTIVITY_LABEL = contract.CANONICAL_ACTIVITY_LABEL
CANONICAL_ACTIVITY_TOKEN = contract.CANONICAL_ACTIVITY_TOKEN
NEXT_ACTIVITY_TARGET = contract.NEXT_ACTIVITY_TARGET
NEXT_ACTIVITY_MASK = contract.NEXT_ACTIVITY_MASK
REMAINING_TIME_MASK = contract.REMAINING_TIME_MASK

# SECTION: Contract field groups and catalogs
BASE_EVENT_COLUMNS = contract.BASE_EVENT_COLUMNS
STATIC_CASE_COLUMNS = contract.STATIC_CASE_COLUMNS
OFFER_NUMERICAL_COLUMNS = contract.OFFER_NUMERICAL_COLUMNS
OFFER_STATE_COLUMNS = contract.OFFER_STATE_COLUMNS
MAPPED_COLUMNS = contract.MAPPED_COLUMNS
REQUIRED_COLUMNS = contract.REQUIRED_COLUMNS
STRING_COLUMNS = contract.STRING_COLUMNS
NUMERIC_COLUMNS = contract.NUMERIC_COLUMNS
CONTRACT_FIELD_GROUPS = contract.CONTRACT_FIELD_GROUPS
FIELD_CATALOG = contract.FIELD_CATALOG
CANONICAL_ACTIVITY_LABELS = contract.CANONICAL_ACTIVITY_LABELS
CANONICAL_ACTIVITY_LABEL_NAMES = contract.CANONICAL_ACTIVITY_LABEL_NAMES
SPLITS = contract.SPLITS

# SECTION: Cyclical calendar features keep an identity scaler so they stay in the sin and cos range of -1 to 1
CYCLICAL_NUMERICAL_COLUMNS: tuple[str, ...] = (WEEKDAY_SIN, WEEKDAY_COS, HOUR_SIN, HOUR_COS)

# SECTION: RT target representation lives in E_04, so training consumes encoded targets directly
REMAINING_TIME_TRANSFORMS: tuple[str, str] = ("raw", "log")
REMAINING_TIME_SCALINGS: tuple[str, str, str] = ("raw", "median", "zscore")
DEFAULT_REMAINING_TIME_TRANSFORM: str = "raw"
DEFAULT_REMAINING_TIME_SCALING: str = "zscore"

# SECTION: Shared data records
PrefixIndexRow = contract.PrefixIndexRow
ArtifactPaths = contract.ArtifactPaths
EncodedBatch = contract.EncodedBatch

# CLASS: Store the RT target representation.
@dataclass(frozen=True)
class RemainingTimeTargetRepr:
    transform: str
    scaling: str
    center: float
    scale: float
    use_softplus: bool
    median_model_units: float
    train_median_seconds: float
    train_value_count: int

# Configure a module logger for runner and POC dispatches.
log = logging.getLogger("E_04_encoding")

# CLASS: Describe one source dataset participating in a joint run.
@dataclass(frozen=True)
class JointSourceSpec:
    dataset_id: str
    heterogeneity: str
    n_clients: int
    banks: tuple[str, ...]

# CLASS: Describe one joint cross-dataset run by total participant count.
@dataclass(frozen=True)
class JointRunSpec:
    run_id: str
    heterogeneity: str
    total_clients: int
    sources: tuple[JointSourceSpec, ...]

    @property
    def qualified_client_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{source.dataset_id}:{bank}"
            for source in self.sources
            for bank in source.banks
        )

# The locked joint matrix uses total participant counts in the run id.
JOINT_RUNS: dict[str, JointRunSpec] = {
    "iid_6banks": JointRunSpec(
        run_id="iid_6banks",
        heterogeneity="iid",
        total_clients=6,
        sources=(
            JointSourceSpec("bpic2017", "iid", 3, ("A", "B", "C")),
            JointSourceSpec("bpic2012", "iid", 3, ("A", "B", "C")),
        ),
    ),
    "weak_6banks": JointRunSpec(
        run_id="weak_6banks",
        heterogeneity="weak",
        total_clients=6,
        sources=(
            JointSourceSpec("bpic2017", "weak", 3, ("A", "B", "C")),
            JointSourceSpec("bpic2012", "weak", 3, ("A", "B", "C")),
        ),
    ),
    "medium_6banks": JointRunSpec(
        run_id="medium_6banks",
        heterogeneity="medium",
        total_clients=6,
        sources=(
            JointSourceSpec("bpic2017", "medium", 3, ("A", "B", "C")),
            JointSourceSpec("bpic2012", "medium", 3, ("A", "B", "C")),
        ),
    ),
    "medium_8banks": JointRunSpec(
        run_id="medium_8banks",
        heterogeneity="medium",
        total_clients=8,
        sources=(
            JointSourceSpec("bpic2017", "medium", 5, ("A", "B", "C", "D", "E")),
            JointSourceSpec("bpic2012", "medium", 3, ("A", "B", "C")),
        ),
    ),
}

# HELPER: Resolve one locked joint run from its total-client run name.
def resolve_joint_run(run_name: str) -> JointRunSpec:
    try: return JOINT_RUNS[str(run_name)]
    except KeyError as exc: raise ValueError(f"unknown joint run: {run_name}") from exc

# HELPER: Convert a client id into a file-system safe token.
def safe_client_id(qualified: str) -> str:
    return str(qualified).replace(":", "_")

# ----------------------------------------------------------------------------------------------------------------------
# 1. JSON AND APPROVAL HELPERS

# HELPER: Convert local Python and numpy values into JSON safe values.
def _json_default(value: object) -> object:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if pd.isna(value): return None
    return value

# HELPER: Write a compact JSON artifact with stable formatting.
def save_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

# HELPER: Load a JSON file with a clear path error.
def load_json_artifact(path: Path) -> dict[str, Any]:
    if not path.exists(): raise FileNotFoundError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

# HELPER: Read one approved JSON file.
def load_approved_json(path: Path, label: str) -> dict[str, Any]:
    payload = load_json_artifact(path)

    # Stop the workflow when the human approval flag is still false.
    if not bool(payload.get("approved", False)): raise ValueError(f"{label} is not approved: {path}")
    return payload

# HELPER: Hash one JSON artifact after normalizing key order and whitespace.
def json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    canonical_json = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()

# HELPER: Prevent accidental replacement of approved review files.
def refuse_approved_overwrite(path: Path, force: bool) -> None:
    if not path.exists() or force: return
    payload = load_json_artifact(path)
    if bool(payload.get("approved", False)):
        raise FileExistsError(f"{path} is already approved. Set FORCE = True to overwrite it.")

# ----------------------------------------------------------------------------------------------------------------------
# 2. CANONICAL DATA ADAPTATION

# HELPER: Normalize missing labels into the shared missing token.
def _normalize_token(value: object) -> str:
    if value is None or pd.isna(value): return MISSING_TOKEN
    text = str(value)
    return text if text and text != "nan" else MISSING_TOKEN

# HELPER: Convert missing categorical values into the shared missing token.
def _clean_string(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").fillna(MISSING_TOKEN)
    return cleaned.replace({"<NA>": MISSING_TOKEN, "nan": MISSING_TOKEN})

# HELPER: Convert one mapped field into the canonical representation.
def _coerce_field(field: str, values: pd.Series) -> pd.Series:
    if field in STRING_COLUMNS: return _clean_string(values)
    if field == OUTCOME: return values.astype("int64")
    if field in NUMERIC_COLUMNS: return values.astype("float64")
    return values

# HELPER: Apply a default value to a canonical field.
def _default_series(field: str, value: object, index: pd.Index) -> pd.Series:
    if value is None and field in NUMERIC_COLUMNS: return pd.Series(np.nan, index=index)
    return _coerce_field(field, pd.Series(value, index=index))

# Map processed split parquet into the canonical event schema.
def to_canonical_events(frame: pd.DataFrame, mapping_payload: dict[str, Any], dataset_id: str,
    client_id: str, split_name: str) -> pd.DataFrame:

    # dataset_mapping defines how one source parquet becomes canonical columns.
    dataset_mapping = mapping_payload["datasets"][dataset_id]
    column_mapping = dataset_mapping.get("column_mapping", {})
    defaults = dataset_mapping.get("default_values", {})

    # Start with fields known from the run context, not from the source parquet.
    out = pd.DataFrame(index=frame.index)
    out[DATASET_ID] = dataset_id
    out[CLIENT_ID] = str(client_id)
    out[SPLIT] = str(split_name)

    # Fill each canonical field from a mapped source column or from an approved default.
    for field in REQUIRED_COLUMNS:
        if field in {DATASET_ID, CLIENT_ID, SPLIT, EVENT_INDEX, TIME_SINCE_PREVIOUS, WEEKDAY_SIN, WEEKDAY_COS, HOUR_SIN, HOUR_COS}: continue
        source = column_mapping.get(field, "")
        if source: out[field] = _coerce_field(field, frame[source])
        elif field in defaults: out[field] = _default_series(field, defaults[field], frame.index)

    # case_id is mandatory because every later prefix operation groups by case.
    if CASE_ID not in out: raise ValueError(f"case id mapping is missing for {dataset_id}")
    case_source = column_mapping.get(CASE_ID, "")
    if not case_source: raise ValueError(f"case id source column is missing for {dataset_id}")

    # event_index is computed after loading so the source parquets stay untouched.
    out[EVENT_INDEX] = frame.groupby(case_source, sort=False).cumcount().astype("int64")

    # Derive the temporal model features from the canonical timestamp in the given trace order.
    event_timestamp = pd.to_datetime(out[TIMESTAMP], utc=True)

    # time_since_previous is the within case gap in seconds and the first event of each case has no gap.
    out[TIME_SINCE_PREVIOUS] = (event_timestamp.groupby(frame[case_source], sort=False)
                                .diff().dt.total_seconds().fillna(0.0).astype("float64"))

    # A negative gap proves the upstream parquet lost its case and time order, so the workflow must stop.
    if bool((out[TIME_SINCE_PREVIOUS] < 0).any()):
        raise ValueError(f"{dataset_id} events are not time-ordered within cases; temporal features would be invalid")

    # weekday and hour of the day are encoded as sin / cos pairs, so the model reads the calendar position as a cycle.
    weekday = event_timestamp.dt.weekday.astype("float64")
    hour = event_timestamp.dt.hour.astype("float64")
    out[WEEKDAY_SIN] = np.sin(2.0 * np.pi * weekday / 7.0)
    out[WEEKDAY_COS] = np.cos(2.0 * np.pi * weekday / 7.0)
    out[HOUR_SIN] = np.sin(2.0 * np.pi * hour / 24.0)
    out[HOUR_COS] = np.cos(2.0 * np.pi * hour / 24.0)

    # Ensure all offer fields exist even when a dataset does not provide them.
    for column in OFFER_NUMERICAL_COLUMNS:
        if column not in out: out[column] = np.nan

    # CreditScore equals zero is treated as missing and resets the previous offer state.
    if CREDIT_SCORE_VALUE in out: out[CREDIT_SCORE_VALUE] = out[CREDIT_SCORE_VALUE].mask(out[CREDIT_SCORE_VALUE] == 0.0)

    # Mark rows where at least one configured offer source value is present.
    offer_mask = pd.Series(False, index=frame.index)
    for column in dataset_mapping.get("offer_source_columns", []):
        values = out[column]
        offer_mask = pd.Series(
            np.logical_or(offer_mask.to_numpy(dtype=bool), (~values.isna()).to_numpy(dtype=bool)), index=frame.index,
        )

    # Datasets without offer sources receive inactive offer masks.
    if dataset_mapping.get("offer_source_columns", []):
        out[OFFER_FEATURE_MASK] = offer_mask.astype("int8")
        out[OFFER_PRESENT] = out[OFFER_FEATURE_MASK]
    else:
        out[OFFER_FEATURE_MASK] = 0
        out[OFFER_PRESENT] = 0

    validate_canonical_events(out.reset_index(drop=True), mapped=False)
    return out.reset_index(drop=True)

# Discover split parquets per bank for the selected run.
def discover_split_paths(mapping_payload: dict[str, Any], dataset_id: str,
                         heterogeneity: str, n_clients: int) -> dict[str, dict[str, Path]]:

    # Resolve the processed split folder and prepare one path block per client.
    dataset_mapping = mapping_payload["datasets"][dataset_id]
    root = Path(dataset_mapping["input_root"])
    prefix = str(dataset_mapping["split_prefix"])
    run_dir = root / f"{heterogeneity}_{n_clients}banks"
    paths: dict[str, dict[str, Path]] = {}

    # Discover clients from train files and derive validation and test paths from the same stem.
    for path in sorted(run_dir.glob(f"{prefix}_bank_*_train.parquet")):
        client_id = path.stem.split("_bank_", 1)[1].rsplit("_", 1)[0]
        paths[client_id] = {
            split: run_dir / f"{prefix}_bank_{client_id}_{split}.parquet"
            for split in SPLITS
        }

    # Stop when the expected split folder does not match the selected run matrix.
    if not paths: raise FileNotFoundError(f"no train split parquets found in {run_dir}")
    if len(paths) != int(n_clients): raise ValueError(f"expected {n_clients} clients in {run_dir}, found {len(paths)}")

    # Require complete train, validation and test triplets for every discovered client.
    missing = [
        str(split_path)
        for client_paths in paths.values()
        for split_path in client_paths.values()
        if not split_path.exists()
    ]
    if missing: raise FileNotFoundError(f"missing split parquet files: {missing}")
    return paths

# Load all split parquets for one configured dataset.
def load_dataset_events(config: Any, mapping_payload: dict[str, Any], dataset_id: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    # Convert each bank split into canonical events before concatenating clients.
    for client_id, split_paths in discover_split_paths(mapping_payload, dataset_id, config.heterogeneity, config.n_clients).items():
        for split_name, path in split_paths.items():
            raw = pd.read_parquet(path)
            frames.append(to_canonical_events(raw, mapping_payload, dataset_id, client_id, split_name))
    return pd.concat(frames, ignore_index=True)


# Load one dataset profile or the joint profile.
def load_run_events(config: Any, mapping_payload: dict[str, Any], schema_profile: dict[str, Any]) -> pd.DataFrame:

    # schema_profile decides which datasets participate in this run.
    frames = [
        load_dataset_events(config, mapping_payload, str(dataset_id))
        for dataset_id in schema_profile.get("datasets", [])
    ]
    if not frames: raise ValueError("schema profile does not list any datasets")
    return pd.concat(frames, ignore_index=True)

# Load all source-split parquets for one joint run and return one canonical event frame.
def load_joint_run_events(config: Any, mapping_payload: dict[str, Any], schema_profile_payload: dict[str, Any]) -> pd.DataFrame:
    run_name = f"{config.heterogeneity}_{config.n_clients}banks"
    spec = resolve_joint_run(run_name)
    expected_datasets = {source.dataset_id for source in spec.sources}
    profile_datasets = set(str(dataset_id) for dataset_id in schema_profile_payload.get("datasets", []))
    if not expected_datasets.issubset(profile_datasets):
        missing = sorted(expected_datasets - profile_datasets)
        raise ValueError(f"joint schema profile does not contain datasets: {missing}")

    frames: list[pd.DataFrame] = []
    for source in spec.sources:
        log.info(
            "Loading joint source %s %s_%sbanks", source.dataset_id, source.heterogeneity, source.n_clients,
        )
        split_paths = discover_split_paths(
            mapping_payload, source.dataset_id, source.heterogeneity, source.n_clients,
        )
        missing_banks = sorted(set(source.banks) - set(split_paths))
        if missing_banks: raise ValueError(f"joint source {source.dataset_id} is missing banks: {missing_banks}")

        for bank in source.banks:
            for split_name, path in split_paths[bank].items():
                raw = pd.read_parquet(path)
                frames.append(to_canonical_events(raw, mapping_payload, source.dataset_id, bank, split_name))

    if not frames: raise ValueError(f"joint run {run_name} did not load any source frames")
    return pd.concat(frames, ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------
# 3. VALIDATION

# Check that the canonical DataFrame contains the required columns.
def validate_required_columns(events: pd.DataFrame, mapped: bool = False) -> None:

    # mapped=True also requires canonical activity target columns.
    required = set(REQUIRED_COLUMNS)
    if mapped: required.update(MAPPED_COLUMNS)
    missing = sorted(required - set(events.columns))
    if missing: raise ValueError(f"canonical schema is missing required columns: {missing}")

# Check the chronological event_index order inside each case
def validate_event_order(events: pd.DataFrame) -> None:
    ordered = events.groupby([DATASET_ID, CLIENT_ID, SPLIT, CASE_ID], sort=False)[EVENT_INDEX].is_monotonic_increasing
    if not bool(ordered.all()):
        bad_cases = ordered.loc[~ordered].index.tolist()[:10]
        raise ValueError(f"event_index is not monotonic for cases: {bad_cases}")

# Check that outcome labels stay in the approved target space with three classes.
def validate_multiclass_outcome_labels(events: pd.DataFrame) -> None:
    if events[OUTCOME].isna().any(): raise ValueError("outcome contains missing labels")
    values = set(events[OUTCOME].astype(int).unique().tolist())
    if not values.issubset({0, 1, 2}): raise ValueError(f"outcome contains labels outside 0, 1, 2: {sorted(values)}")

# Run canonical input checks before fitting artifacts.
def validate_canonical_events(events: pd.DataFrame, mapped: bool = False) -> None:
    validate_required_columns(events, mapped=mapped)
    validate_event_order(events)
    validate_multiclass_outcome_labels(events)

# ----------------------------------------------------------------------------------------------------------------------
# 4. ACTIVITY MAPPING

# Load the reviewed dataset mapping file.
# Repository root of this checkout, taken from the module location because the workflows change the working directory.
ENCODING_REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# Resolve one stored processed-split root against this checkout, because the approved mapping records a fixed path.
# The longest suffix of the stored path that exists below the repository root wins, so a clone at any path resolves.
def resolve_input_root(stored: str) -> Path:
    stored_path = Path(stored)
    parts = stored_path.parts

    # Joining an absolute path resets the base, so the start index zero candidate is the stored path itself.
    # The local checkout must always win over an existing foreign path so that the candidate is never tested.
    first = 1 if stored_path.is_absolute() else 0
    for start in range(first, len(parts)):
        candidate = ENCODING_REPO_ROOT.joinpath(*parts[start:])
        if candidate.is_dir(): return candidate

    # Fall back to the stored value only when no suffix resolves, so the failure names the recorded path.
    return stored_path

# Load the approved dataset mapping and rewrite every processed-split root for this checkout.
# Every encoding consumer routes through here, so the runner, the POC and the training side share one resolution.
def load_dataset_mapping(path: Path, require_approved: bool = True) -> dict[str, Any]:
    payload = load_json_artifact(path)
    if require_approved and not bool(payload.get("approved", False)):
        raise ValueError(f"dataset mapping is not approved: {path}")
    if "activity_mapping" not in payload:
        raise ValueError("dataset mapping must contain activity_mapping")
    validate_activity_mapping_payload(payload)
    for dataset_mapping in payload.get("datasets", {}).values():
        stored = dataset_mapping.get("input_root")
        if stored: dataset_mapping["input_root"] = str(resolve_input_root(str(stored)))
    return payload

# Resolve the approved activity labels allowed in one mapping.
def allowed_activity_labels(mapping_payload: dict[str, Any]) -> set[str]:
    mapping = mapping_payload.get("activity_mapping", {})
    labels = mapping.get("allowed_canonical_activity_labels", CANONICAL_ACTIVITY_LABEL_NAMES)
    return {str(label) for label in labels}

# Check that mappings do not create new canonical activity labels.
def validate_activity_mapping_payload(mapping_payload: dict[str, Any]) -> None:

    # activity_mapping must preserve the reviewed grouping structure.
    mapping = mapping_payload.get("activity_mapping", {})
    if not isinstance(mapping, dict):
        raise ValueError("activity_mapping must be a dictionary")
    canonical_activities = mapping.get("canonical_activities", {})
    if not isinstance(canonical_activities, dict):
        raise ValueError("canonical_activities must be a dictionary")
    if OTHER_ACTIVITY_TOKEN in canonical_activities:
        raise ValueError("[OTHER_ACTIVITY] is the reserved fallback and must not be defined as a mapping group")

    # Allowed labels must stay inside the contract activity universe.
    allowed = allowed_activity_labels(mapping_payload)
    contract_labels = set(CANONICAL_ACTIVITY_LABEL_NAMES)
    unknown_allowed = sorted(allowed - contract_labels)
    if unknown_allowed:
        raise ValueError(f"unknown canonical activity labels in allowed list: {unknown_allowed}")
    unknown_groups = sorted(str(label) for label in canonical_activities if str(label) not in allowed)
    if unknown_groups:
        raise ValueError(f"unknown canonical activity labels in mapping groups: {unknown_groups}")

    # Raw A, O and W labels must not cross into another canonical origin namespace.
    _validate_activity_prefixes(canonical_activities)

# Return the A, O or W origin prefix from one activity label or token.
def _activity_origin_prefix(value: str) -> str:
    label = str(value).split("+", 1)[0]
    return label.split("_", 1)[0] if "_" in label else ""

# Enforce the source-origin boundary in reviewed activity mappings.
def _validate_activity_prefixes(canonical_activities: dict[str, Any]) -> None:
    mismatches: list[str] = []
    for canonical_label, group in canonical_activities.items():
        canonical_prefix = _activity_origin_prefix(str(canonical_label))
        labels_by_dataset = group.get("labels_by_dataset", {})
        for dataset_id, labels in labels_by_dataset.items():
            for raw_label in labels:
                raw_prefix = _activity_origin_prefix(str(raw_label))
                if raw_prefix and canonical_prefix and raw_prefix != canonical_prefix:
                    mismatches.append(f"{dataset_id}:{raw_label}->{canonical_label}")
        for raw_token, canonical_token in group.get("token_overrides", {}).items():
            raw_prefix = _activity_origin_prefix(str(raw_token))
            token_prefix = _activity_origin_prefix(str(canonical_token))
            if raw_prefix and token_prefix and raw_prefix != token_prefix:
                mismatches.append(f"{raw_token}->{canonical_token}")
    if mismatches: raise ValueError(f"cross-prefix activity mapping is not allowed: {sorted(mismatches)}")

# Flatten grouped mapping JSON into raw label and token override lookups.
def _flatten_activity_mapping(payload: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    validate_activity_mapping_payload(payload)
    label_map: dict[str, str] = {}
    token_overrides: dict[str, str] = {}
    mapping = payload.get("activity_mapping", payload)

    # label_map resolves raw activity labels, token_overrides resolve lifecycle-specific exceptions.
    for canonical_label, group in mapping["canonical_activities"].items():
        labels_by_dataset = group.get("labels_by_dataset", {})
        for labels in labels_by_dataset.values():
            for raw_label in labels:
                label_map[str(raw_label)] = str(canonical_label)
        for raw_token, canonical_token in group.get("token_overrides", {}).items():
            token_overrides[str(raw_token)] = str(canonical_token)
    return label_map, token_overrides

# Split lifecycle-aware raw tokens into label and lifecycle parts.
def _split_activity_token(token: str) -> tuple[str, str]:
    if token == END_TOKEN: return END_TOKEN, MISSING_TOKEN
    if "+" not in token: return token, MISSING_TOKEN
    label, lifecycle = token.rsplit("+", 1)
    return label, lifecycle or MISSING_TOKEN

# Build one canonical activity token from a raw label and lifecycle value.
def _canonical_token(raw_label: str, lifecycle: str, label_map: dict[str, str]) -> tuple[str, str]:
    canonical_label = label_map.get(raw_label, OTHER_ACTIVITY_TOKEN)
    if canonical_label == OTHER_ACTIVITY_TOKEN: return canonical_label, OTHER_ACTIVITY_TOKEN
    lifecycle_value = lifecycle if lifecycle != MISSING_TOKEN else MISSING_TOKEN
    return canonical_label, f"{canonical_label}+{lifecycle_value}"

# Apply the reviewed canonical activity mapping.
def apply_activity_mapping(events: pd.DataFrame, mapping_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:

    # Load the approved mapping and flatten it into fast lookup dictionaries.
    payload = load_dataset_mapping(mapping_path, require_approved=True)
    label_map, token_overrides = _flatten_activity_mapping(payload)
    mapped = events.copy()
    canonical_labels: list[str] = []
    canonical_tokens: list[str] = []
    target_tokens: list[str] = []
    fallback_labels: set[str] = set()

    # Map current activities and next activity targets without changing the source events.
    for _, row in mapped.iterrows():
        raw_label = _normalize_token(row[RAW_ACTIVITY])
        lifecycle = _normalize_token(row[LIFECYCLE])
        if raw_label not in label_map: fallback_labels.add(raw_label)
        canonical_label, canonical_token = _canonical_token(raw_label, lifecycle, label_map)
        raw_token = _normalize_token(row[RAW_ACTIVITY_TOKEN])
        canonical_tokens.append(token_overrides.get(raw_token, canonical_token))
        canonical_labels.append(canonical_label)

        # The cut final event receives [END] as the next activity target.
        next_raw = _normalize_token(row[NEXT_ACTIVITY_RAW])
        if next_raw == END_TOKEN:
            target_tokens.append(END_TOKEN)
            continue
        next_label, next_lifecycle = _split_activity_token(next_raw)
        if next_label not in label_map: fallback_labels.add(next_label)
        _, next_token = _canonical_token(next_label, next_lifecycle, label_map)
        target_tokens.append(token_overrides.get(next_raw, next_token))

    mapped[CANONICAL_ACTIVITY_LABEL] = canonical_labels
    mapped[CANONICAL_ACTIVITY_TOKEN] = canonical_tokens
    mapped[NEXT_ACTIVITY_TARGET] = target_tokens
    mapped[NEXT_ACTIVITY_MASK] = 1
    mapped[REMAINING_TIME_MASK] = 1

    # Store compact mapping provenance and unresolved fallback labels for review.
    mapping_hash = json_sha256(mapping_path)
    report = {
        "mapping_path": str(mapping_path),
        "mapping_sha256": mapping_hash,
        "mapping_modified_utc": datetime.fromtimestamp(mapping_path.stat().st_mtime, tz=timezone.utc).isoformat(),
        "approved": bool(payload.get("approved", False)),
        "schema_profile": payload.get("schema_profile"),
        "approved_count": len(label_map),
        "unresolved_labels": sorted(fallback_labels),
        "fallback_count": int(sum(label == OTHER_ACTIVITY_TOKEN for label in canonical_labels)),
    }
    return mapped, report

# Build the remaining time target from the selected E_04 target representation.
def build_remaining_time_target(events: pd.DataFrame, target_repr: Optional[RemainingTimeTargetRepr] = None) -> pd.Series:
    values = events[REMAINING_TIME].astype(float)
    if (values < 0).any(): raise ValueError("remaining_time contains negative values")
    if target_repr is None: return values.copy()
    return transform_remaining_time_target(values, target_repr)

# Apply the selected RT target transform before fitting or encoding.
def _remaining_time_apply_transform(values: pd.Series, transform: str) -> pd.Series:
    if transform == "raw": return values.astype("float64")
    if transform == "log": return np.log1p(values.astype("float64"))
    raise ValueError(f"unknown remaining-time transform: {transform}")

# Return the median as the lower of the two middle values, unlike np.median, which averages them on an even count.
# The median is used as a scale divisor and as the baseline prediction, so it stays a value the data actually contains.
def _lower_median(values: pd.Series) -> float:
    ordered = np.sort(values.astype("float64").to_numpy())
    if ordered.size == 0: return 0.0
    return float(ordered[(ordered.size - 1) // 2])

# Convert one target representation into a JSON-safe dictionary.
def remaining_time_target_repr_to_dict(target_repr: RemainingTimeTargetRepr) -> dict[str, Union[str, float, bool, int]]:
    return {
        "transform": target_repr.transform,
        "scaling": target_repr.scaling,
        "center": float(target_repr.center),
        "scale": float(target_repr.scale),
        "use_softplus": bool(target_repr.use_softplus),
        "median_model_units": float(target_repr.median_model_units),
        "train_median_seconds": float(target_repr.train_median_seconds),
        "train_value_count": int(target_repr.train_value_count),
    }

# Convert a stored JSON target representation back into the runtime dataclass.
def remaining_time_target_repr_from_dict(payload: Optional[dict[str, Any]]) -> RemainingTimeTargetRepr:
    if payload is None: return RemainingTimeTargetRepr("raw", "raw", 0.0, 1.0, True, 0.0, 0.0, 0)
    return RemainingTimeTargetRepr(
        transform=str(payload.get("transform", "raw")),
        scaling=str(payload.get("scaling", "raw")),
        center=float(payload.get("center", 0.0)),
        scale=max(float(payload.get("scale", 1.0)), 1e-6),
        use_softplus=bool(payload.get("use_softplus", str(payload.get("scaling", "raw")) != "zscore")),
        median_model_units=float(payload.get("median_model_units", 0.0)),
        train_median_seconds=float(payload.get("train_median_seconds", 0.0)),
        train_value_count=int(payload.get("train_value_count", 0)),
    )

# Fit the RT target representation from positive masked train target values.
def fit_remaining_time_target_repr(values: pd.Series, transform: str, scaling: str,
    mask: Optional[pd.Series] = None) -> RemainingTimeTargetRepr:

    # Validate representation knobs before fitting train-only statistics
    if transform not in REMAINING_TIME_TRANSFORMS: raise ValueError(f"unknown remaining-time transform: {transform}")
    if scaling not in REMAINING_TIME_SCALINGS: raise ValueError(f"unknown remaining-time scaling: {scaling}")

    # Keep the same fitting population as the previous training-side representation.
    valid = pd.to_numeric(values, errors="coerce")
    if mask is not None: valid = valid.loc[mask.astype(bool)]
    valid = valid.dropna().astype("float64")
    if (valid < 0).any(): raise ValueError("remaining_time contains negative values")
    valid = valid.loc[valid > 0.0]
    if valid.empty: return RemainingTimeTargetRepr(transform, scaling, 0.0, 1.0, scaling != "zscore", 0.0, 0.0, 0)

    # Fit center and scale in the selected transform space.
    transformed = _remaining_time_apply_transform(valid, transform)
    median_transformed = _lower_median(transformed)
    if scaling == "raw":
        center, scale = 0.0, 1.0
    elif scaling == "median":
        center, scale = 0.0, max(median_transformed, 1.0)
    else:
        center = float(transformed.mean())
        scale = max(float(transformed.std(ddof=0)), 1e-6)

    # Store the scaled median only for the non-zscore heads. Z-score heads start at the centered zero bias.
    median_model_units = 0.0 if scaling == "zscore" else (median_transformed - center) / scale
    train_median_seconds = 0.0 if scaling == "zscore" else _lower_median(valid)
    return RemainingTimeTargetRepr(
        transform=transform, scaling=scaling, center=float(center), scale=float(scale),
        use_softplus=scaling != "zscore", median_model_units=float(median_model_units),
        train_median_seconds=float(train_median_seconds), train_value_count=int(valid.shape[0]),
    )

# Collect raw train target values from prefix references with the RT mask applied.
def remaining_time_values_from_prefixes(events: pd.DataFrame, prefix_index: list[Any]) -> pd.Series:
    ordered = (events.sort_values([DATASET_ID, CLIENT_ID, SPLIT, CASE_ID, EVENT_INDEX], kind="mergesort")
               .reset_index(drop=True))
    case_events = {
        (str(dataset), str(client), str(split), str(case_id)): group.reset_index(drop=True)
        for (dataset, client, split, case_id), group in ordered.groupby(
            [DATASET_ID, CLIENT_ID, SPLIT, CASE_ID], sort=False,
        )
    }
    values: list[float] = []
    for row in prefix_index:
        if str(row.split) != "train": continue
        case = case_events[(row.dataset_id, row.client_id, row.split, row.case_id)]
        if int(row.label_pos) == len(case) - 1: continue
        values.append(float(case.iloc[int(row.label_pos)][REMAINING_TIME]))
    return pd.Series(values, dtype="float64")

# Fit the E_04 RT target representation from train prefixes only.
def fit_remaining_time_target_from_prefixes(events: pd.DataFrame, prefix_index: list[Any], transform: str,
    scaling: str) -> RemainingTimeTargetRepr:

    values = remaining_time_values_from_prefixes(events, prefix_index)
    return fit_remaining_time_target_repr(values, transform, scaling)

# Encode raw seconds into model units with the corresponding target representation.
def transform_remaining_time_target(values: pd.Series, target_repr: RemainingTimeTargetRepr) -> pd.Series:
    if (values.astype(float) < 0).any(): raise ValueError("remaining_time contains negative values")
    transformed = _remaining_time_apply_transform(values.astype("float64"), target_repr.transform)
    encoded = (transformed - float(target_repr.center)) / max(float(target_repr.scale), 1e-6)
    return encoded.fillna(0.0).astype("float32")

# Invert model units back into raw seconds for diagnostics and training metrics.
def inverse_remaining_time_target(values: pd.Series, target_repr: RemainingTimeTargetRepr) -> pd.Series:
    transformed = values.astype("float64") * float(target_repr.scale) + float(target_repr.center)
    seconds = np.expm1(transformed) if target_repr.transform == "log" else transformed
    return pd.Series(seconds, index=values.index).astype("float64")

# ----------------------------------------------------------------------------------------------------------------------
# 5. VOCABULARIES AND SCALERS

# Build one categorical vocabulary from training values only.
def build_vocabulary(values: pd.Series, include_end_token: bool = False, include_other_activity: bool = False) -> dict[str, int]:
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, MISSING_TOKEN: 2}
    if include_other_activity: vocab[OTHER_ACTIVITY_TOKEN] = len(vocab)
    if include_end_token: vocab[END_TOKEN] = len(vocab)
    cleaned = values.map(_normalize_token)
    for value in sorted(cleaned.unique().tolist()):
        if value not in vocab: vocab[value] = len(vocab)
    return vocab

# Build model-facing vocabularies from all client train splits.
def build_all_vocabularies(train_events: pd.DataFrame, categorical_columns: Optional[list[str]] = None,
    include_resource: bool = True, include_static: bool = True) -> dict[str, dict[str, int]]:

    # Default columns keep backward compatibility for tests and callers.
    if categorical_columns is None:
        categorical_columns = [CANONICAL_ACTIVITY_TOKEN]
        if include_resource: categorical_columns.append(RESOURCE)
        if include_static: categorical_columns.extend([LOAN_GOAL, APPLICATION_TYPE])

    # Next activity gets its own target vocabulary with [END] and [OTHER_ACTIVITY].
    vocabs: dict[str, dict[str, int]] = {
        NEXT_ACTIVITY_TARGET: build_vocabulary(
            train_events[NEXT_ACTIVITY_TARGET], include_end_token=True, include_other_activity=True,
        ),
    }

    # Input vocabularies are fit only from train values.
    for column in categorical_columns:
        if column not in train_events or column == NEXT_ACTIVITY_TARGET: continue
        vocabs[column] = build_vocabulary(train_events[column], include_other_activity=column == CANONICAL_ACTIVITY_TOKEN)
    return vocabs

# Encode categories with [UNK] for unseen values.
def encode_categorical_series(values: pd.Series, vocabulary: dict[str, int]) -> pd.Series:
    unk_index = vocabulary[UNK_TOKEN]
    return values.map(_normalize_token).map(lambda value: vocabulary.get(value, unk_index)).astype("int64")

# Fit StandardScaler parameters from training values.
def fit_scaler(values: pd.Series, mask: Optional[pd.Series] = None) -> dict[str, float]:
    valid = values.astype("float64")
    if mask is not None: valid = valid.loc[mask.astype(bool)]
    valid = valid.dropna()
    if valid.empty: return {"mean": 0.0, "std": 1.0}
    mean = float(valid.mean())
    std = float(valid.std(ddof=0))
    return {"mean": mean, "std": std if std > 0 else 1.0}

# Apply one fitted scaler and encode missing values as zero.
def transform_with_scaler(values: pd.Series, scaler: dict[str, float]) -> pd.Series:
    transformed = (values.astype("float64") - scaler["mean"]) / scaler["std"]
    return transformed.fillna(0.0).astype("float32")

# Fit all numerical scalers from train splits only.
def fit_all_scalers(train_events: pd.DataFrame, numerical_columns: Optional[list[str]] = None,
    offer_numerical_columns: Optional[list[str]] = None, include_static: bool = True, include_offers: bool = True,
    ) -> dict[str, dict[str, float]]:

    # Explicit columns come from the selected schema profile during normal runner use.
    explicit_numerical_columns = numerical_columns is not None
    if numerical_columns is None:
        numerical_columns = [TIME_DELTA]
        if include_static: numerical_columns.append(REQUESTED_AMOUNT_VALUE)
    if offer_numerical_columns is None:
        offer_numerical_columns = [] if explicit_numerical_columns else list(OFFER_NUMERICAL_COLUMNS) if include_offers else []
    scalers: dict[str, dict[str, float]] = {}

    # RequestedAmount uses its availability mask when present.
    for column in numerical_columns:
        if column in CYCLICAL_NUMERICAL_COLUMNS:
            # Cyclical features are already in the sin and cos range, so an identity scaler leaves them unchanged.
            scalers[column] = {"mean": 0.0, "std": 1.0}
        elif column == REQUESTED_AMOUNT_VALUE and REQUESTED_AMOUNT_MASK in train_events:
            scalers[column] = fit_scaler(train_events[column], train_events[REQUESTED_AMOUNT_MASK])
        else:
            scalers[column] = fit_scaler(train_events[column])

    # Offer scalers use only rows with active offer features.
    for column in offer_numerical_columns:
        scalers[column] = fit_scaler(train_events[column], train_events[OFFER_FEATURE_MASK])
    return scalers

# ----------------------------------------------------------------------------------------------------------------------
# 6. PREFIX INDEX

# Build compact prefix references without truncating source events.
def build_prefix_index(events: pd.DataFrame, max_prefix_length: Optional[int]) -> tuple[list[Any], int]:
    sort_columns = [DATASET_ID, CLIENT_ID, SPLIT, CASE_ID, EVENT_INDEX]
    ordered = events.sort_values(sort_columns, kind="mergesort")
    rows: list[Any] = []
    observed_max = 0

    # Build prefix references from length 1 to the configured cap for every retained case.
    for (dataset_id, client_id, split, case_id), case_events in ordered.groupby(
        [DATASET_ID, CLIENT_ID, SPLIT, CASE_ID], sort=False,
    ):
        n_events = int(len(case_events))
        observed_max = max(observed_max, n_events)
        n_prefixes = n_events if max_prefix_length is None else min(n_events, max_prefix_length)
        for label_pos in range(n_prefixes):
            rows.append(
                PrefixIndexRow(
                    dataset_id=str(dataset_id), case_id=str(case_id), client_id=str(client_id), split=str(split),
                    prefix_length=label_pos + 1, label_pos=label_pos,
                )
            )
    return rows, observed_max

# ----------------------------------------------------------------------------------------------------------------------
# 7. OFFER STATE

# Return the offer state visible at the end of one prefix.
def build_offer_state(prefix_events: pd.DataFrame) -> dict[str, Union[float, int]]:

    # The Initial state is empty until the first valid offer row appears.
    state: dict[str, Union[float, int]] = {"offer_present": 0}
    for column in OFFER_NUMERICAL_COLUMNS: state[column] = 0.0

    # Each valid offer row overwrites the visible state for later prefix events.
    for _, row in prefix_events.iterrows():
        if int(row.get(OFFER_FEATURE_MASK, 0)) != 1: continue
        state["offer_present"] = 1
        for column in OFFER_NUMERICAL_COLUMNS:
            value = row.get(column, np.nan)
            state[column] = 0.0 if pd.isna(value) else float(value)
    return state

# Build the visible offer sequence with forward fill within each prefix.
def _build_offer_sequence(prefix_events: pd.DataFrame, scalers: dict[str, dict[str, float]], pad_length: int,
    offer_numerical_columns: list[str]) -> tuple[torch.Tensor, torch.Tensor]:

    # BPIC 2012 profiles without offer features emit empty offer tensors.
    if not offer_numerical_columns:
        return torch.zeros((pad_length, 0), dtype=torch.float32), torch.zeros(pad_length, dtype=torch.int8)
    state = {column: np.nan for column in offer_numerical_columns}
    present = 0
    values: list[list[float]] = []
    masks: list[int] = []

    # Forward fill the current offer state only inside the visible prefix.
    for _, row in prefix_events.iterrows():
        if int(row.get(OFFER_FEATURE_MASK, 0)) == 1:
            present = 1
            for column in offer_numerical_columns: state[column] = row.get(column, np.nan)
        row_values: list[float] = []
        for column in offer_numerical_columns:
            raw_value = state[column]
            if present == 0 or pd.isna(raw_value):
                row_values.append(0.0)
            else:
                scaler = scalers.get(column, {"mean": 0.0, "std": 1.0})
                row_values.append(float((float(raw_value) - scaler["mean"]) / scaler["std"]))
        values.append(row_values)
        masks.append(present)

    # Pad offer sequences to the static prefix length with inactive masks.
    while len(values) < pad_length:
        values.append([0.0] * len(offer_numerical_columns))
        masks.append(0)
    return torch.tensor(values, dtype=torch.float32), torch.tensor(masks, dtype=torch.int8)

# ----------------------------------------------------------------------------------------------------------------------
# 8. PREFIX DATASET

# Encode prefix samples on demand.
class PrefixDataset:
    def __init__(self, events: pd.DataFrame, prefix_index: list[Any], vocabularies: dict[str, dict[str, int]],
        scalers: dict[str, dict[str, float]], static_padding_length: int,
        sequence_categorical_columns: Optional[list[str]] = None, sequence_numerical_columns: Optional[list[str]] = None,
        offer_numerical_columns: Optional[list[str]] = None, include_offer_features: bool = True,
        remaining_time_target_repr: Optional[Union[RemainingTimeTargetRepr, dict[str, Any]]] = None,) -> None:

        validate_canonical_events(events, mapped=True)

        # Keep events sorted once so every prefix lookup uses stable case order.
        self.events = events.sort_values([CLIENT_ID, SPLIT, CASE_ID, EVENT_INDEX], kind="mergesort").reset_index(drop=True)
        self.prefix_index = prefix_index
        self.vocabularies = vocabularies
        self.scalers = scalers
        self.static_padding_length = int(static_padding_length)
        self.sequence_categorical_columns = sequence_categorical_columns or [
            column for column in vocabularies if column != NEXT_ACTIVITY_TARGET
        ]
        self.sequence_numerical_columns = sequence_numerical_columns or list(scalers)
        self.offer_numerical_columns = offer_numerical_columns or []
        self.include_offer_features = bool(include_offer_features)
        self.remaining_time_target_repr = (
            remaining_time_target_repr
            if isinstance(remaining_time_target_repr, RemainingTimeTargetRepr)
            else remaining_time_target_repr_from_dict(remaining_time_target_repr)
        )

        # Cache case groups so __getitem__ can encode one prefix without scanning the full table.
        self.case_events = {
            (str(dataset), str(client), str(split), str(case_id)): group.reset_index(drop=True)
            for (dataset, client, split, case_id), group in self.events.groupby(
                [DATASET_ID, CLIENT_ID, SPLIT, CASE_ID], sort=False,
            )
        }

    def __len__(self) -> int: return len(self.prefix_index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # PrefixIndexRow identifies one prefix sample without storing tensors on the disk.
        row = self.prefix_index[index]
        case = self.case_events[(row.dataset_id, row.client_id, row.split, row.case_id)]
        prefix = case.iloc[: row.prefix_length].copy()
        label_event = case.iloc[row.label_pos]
        pad_length = self.static_padding_length

        # Encode categorical input columns and pad with [PAD] index 0.
        cat_blocks: list[np.ndarray] = []
        for column in self.sequence_categorical_columns:
            ids = encode_categorical_series(prefix[column], self.vocabularies[column]).to_numpy()
            cat_blocks.append(np.pad(ids, (0, pad_length - len(ids)), constant_values=0))
        categorical_ids = np.stack(cat_blocks, axis=1) if cat_blocks else np.zeros((pad_length, 0), dtype=np.int64)

        # Scale numerical input columns with train-only scaler parameters.
        num_blocks: list[np.ndarray] = []
        for column in self.sequence_numerical_columns:
            if column in self.offer_numerical_columns: continue
            values = transform_with_scaler(prefix[column], self.scalers[column]).to_numpy()
            num_blocks.append(np.pad(values, (0, pad_length - len(values)), constant_values=0.0))
        numerical = np.stack(num_blocks, axis=1) if num_blocks else np.zeros((pad_length, 0), dtype=np.float32)

        # Build offer features separately because they use within-prefix forward fill.
        if self.include_offer_features:
            offer_numerical, offer_mask = _build_offer_sequence(prefix, self.scalers, pad_length, self.offer_numerical_columns)
        else:
            offer_numerical = torch.zeros((pad_length, 0), dtype=torch.float32)
            offer_mask = torch.zeros(pad_length, dtype=torch.int8)

        # Padding mask uses 1 for real prefix events and 0 for static padding.
        padding_mask = np.array([1] * row.prefix_length + [0] * (pad_length - row.prefix_length), dtype=np.int8)

        # Unknown next activity targets are masked out instead of trained as [UNK].
        next_vocab = self.vocabularies[NEXT_ACTIVITY_TARGET]
        next_target = _normalize_token(label_event[NEXT_ACTIVITY_TARGET])
        next_label = next_vocab.get(next_target, next_vocab[UNK_TOKEN])
        next_mask = 0 if next_target not in next_vocab else 1

        # Final decision prefixes have no remaining-time loss contribution.
        remaining_time = float(build_remaining_time_target(case.iloc[[row.label_pos]], self.remaining_time_target_repr).iloc[0])
        remaining_mask = 0 if row.label_pos == len(case) - 1 else 1

        return {
            "categorical_ids": torch.tensor(categorical_ids, dtype=torch.long),
            "numerical": torch.tensor(numerical, dtype=torch.float32),
            "offer_numerical": offer_numerical,
            "offer_feature_mask": offer_mask,
            "padding_mask": torch.tensor(padding_mask, dtype=torch.int8),
            "prefix_length": torch.tensor(row.prefix_length, dtype=torch.long),
            "outcome_label": torch.tensor(int(label_event[OUTCOME]), dtype=torch.long),
            "next_activity_label": torch.tensor(int(next_label), dtype=torch.long),
            "next_activity_mask": torch.tensor(int(next_mask), dtype=torch.int8),
            "remaining_time_label": torch.tensor(remaining_time, dtype=torch.float32),
            "remaining_time_mask": torch.tensor(int(remaining_mask), dtype=torch.int8),
        }

# Assemble a static padded training batch.
def collate_prefix_batch(samples: list[dict[str, torch.Tensor]]) -> EncodedBatch:
    return EncodedBatch(
        categorical_ids=torch.stack([sample["categorical_ids"] for sample in samples]),
        numerical=torch.stack([sample["numerical"] for sample in samples]),
        offer_numerical=torch.stack([sample["offer_numerical"] for sample in samples]),
        offer_feature_mask=torch.stack([sample["offer_feature_mask"] for sample in samples]),
        padding_mask=torch.stack([sample["padding_mask"] for sample in samples]),
        prefix_length=torch.stack([sample["prefix_length"] for sample in samples]),
        outcome_label=torch.stack([sample["outcome_label"] for sample in samples]),
        next_activity_label=torch.stack([sample["next_activity_label"] for sample in samples]),
        next_activity_mask=torch.stack([sample["next_activity_mask"] for sample in samples]),
        remaining_time_label=torch.stack([sample["remaining_time_label"] for sample in samples]),
        remaining_time_mask=torch.stack([sample["remaining_time_mask"] for sample in samples]),
    )

# ----------------------------------------------------------------------------------------------------------------------
# 9. ARTIFACTS

# Return deterministic compact JSON artifact paths.
def build_artifact_paths(root: Path, dataset_name: str, heterogeneity: str, n_clients: int, joint: bool = False) -> Any:

    # Prefixes keep thesis artifact names aligned with BPIC 2017, BPIC 2012 and joint runs.
    if joint:
        dataset_dir = "joint"
        prefix = "J_04_"
    elif dataset_name == "bpic2017":
        dataset_dir = "bpic2017"
        prefix = "A_04_"
    elif dataset_name == "bpic2012":
        dataset_dir = "bpic2012"
        prefix = "B_04_"
    else: raise ValueError(f"unknown dataset for artifacts: {dataset_name}")
    run_root = Path(root) / dataset_dir / f"{heterogeneity}_{n_clients}banks"
    return ArtifactPaths(
        root=run_root,
        encoding_spec=run_root / f"{prefix}encoding_spec.json",
        vocabulary=run_root / f"{prefix}vocabulary.json",
        scaler=run_root / f"{prefix}scaler.json",
        mapping_report=run_root / f"{prefix}mapping_report.json",
    )

# Write the four approved E_04 JSON artifacts.
def save_encoding_artifacts(paths: Any, spec: dict[str, Any], vocabulary: dict[str, Any], scaler: dict[str, Any],
                            mapping_report: dict[str, Any]) -> None:
    save_json_artifact(paths.encoding_spec, spec)
    save_json_artifact(paths.vocabulary, vocabulary)
    save_json_artifact(paths.scaler, scaler)
    save_json_artifact(paths.mapping_report, mapping_report)

# Fold run validation content into the encoding spec JSON.
def build_encoding_spec(config: Any, events: pd.DataFrame, prefix_index: list[Any], observed_max_trace_length: int,
                        vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]],
                        mapping_report: dict[str, Any],
                        remaining_time_target_repr: Optional[RemainingTimeTargetRepr] = None) -> dict[str, Any]:
    per_bank = {}
    group_columns = [DATASET_ID, CLIENT_ID] if config.dataset == "joint" else [CLIENT_ID]
    group_key = group_columns if len(group_columns) > 1 else group_columns[0]

    # Summarize cases, events and generated prefix references per client
    for group_value, group in events.groupby(group_key, sort=True):
        if config.dataset == "joint":
            dataset_id, client_id = group_value
            group_name = f"{dataset_id}:{client_id}"
            client_prefixes = [
                row for row in prefix_index if row.dataset_id == str(dataset_id) and row.client_id == str(client_id)
            ]
        else:
            client_id = str(group_value)
            group_name = client_id
            client_prefixes = [row for row in prefix_index if row.client_id == client_id]
        per_bank[group_name] = {
            "case_count": int(group[CASE_ID].nunique()),
            "event_count": int(len(group)),
            "prefix_count": int(len(client_prefixes)),
        }
    train = events.loc[events[SPLIT] == "train"]

    # Count cases by dataset for joint runs to avoid case-id collisions.
    case_count = (
        int(events.drop_duplicates([DATASET_ID, CASE_ID]).shape[0])
        if config.dataset == "joint"
        else int(events[CASE_ID].nunique())
    )

    # Store the full recipe training needs to recreate on-the-fly tensors.
    return {
        "run": {
            "dataset": config.dataset,
            "schema_profile": config.schema_profile,
            "heterogeneity": config.heterogeneity,
            "n_clients": config.n_clients,
            "random_seed": config.random_seed,
        },
        "hyperparameters": {
            "max_prefix_length_for_encoding": config.max_prefix_length_for_encoding,
            "outcome_target_mode": config.outcome_target_mode,
            "remaining_time_transform": getattr(config, "remaining_time_transform", DEFAULT_REMAINING_TIME_TRANSFORM),
            "remaining_time_scaling": getattr(config, "remaining_time_scaling", DEFAULT_REMAINING_TIME_SCALING),
        },
        "counts": {
            "per_bank": per_bank,
            "aggregate": {
                "case_count": case_count,
                "event_count": int(len(events)),
                "prefix_count": int(len(prefix_index)),
            },
            "train_outcome_counts": {
                str(key): int(value)
                for key, value in train.drop_duplicates([DATASET_ID, CLIENT_ID, CASE_ID])[OUTCOME]
                .value_counts()
                .sort_index()
                .items()
            },
        },
        "prefix": {
            "cap": config.max_prefix_length_for_encoding,
            "observed_max_trace_length": int(observed_max_trace_length),
            "static_padding_length": int(config.max_prefix_length_for_encoding or observed_max_trace_length),
        },
        "vocabulary_sizes": {name: len(vocab) for name, vocab in vocabularies.items()},
        "reserved_token_indices": {
            name: {token: vocab[token] for token in vocab if token.startswith("[")}
            for name, vocab in vocabularies.items()
        },
        "scalers": scalers,
        "target_scalers": {
            "remaining_time": remaining_time_target_repr_to_dict(
                remaining_time_target_repr
                if remaining_time_target_repr is not None
                else RemainingTimeTargetRepr("raw", "raw", 0.0, 1.0, True, 0.0, 0.0, 0)
            )
        },
        "mapping_summary": mapping_report,
        "validation_summary": {
            "outcome_labels_valid": True,
            "source_prefix_tensors_saved": False,
            "prefix_index_file_saved": False,
        },
    }

__all__ = [name for name in globals() if name.isupper()] + [
    "PrefixIndexRow",
    "ArtifactPaths",
    "EncodedBatch",
    "RemainingTimeTargetRepr",
    "load_approved_json",
    "load_dataset_mapping",
    "allowed_activity_labels",
    "validate_activity_mapping_payload",
    "to_canonical_events",
    "load_run_events",
    "apply_activity_mapping",
    "build_all_vocabularies",
    "fit_all_scalers",
    "fit_remaining_time_target_repr",
    "fit_remaining_time_target_from_prefixes",
    "remaining_time_target_repr_to_dict",
    "remaining_time_target_repr_from_dict",
    "transform_remaining_time_target",
    "inverse_remaining_time_target",
    "remaining_time_values_from_prefixes",
    "build_prefix_index",
    "PrefixDataset",
    "collate_prefix_batch",
    "build_artifact_paths",
    "save_encoding_artifacts",
]

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────