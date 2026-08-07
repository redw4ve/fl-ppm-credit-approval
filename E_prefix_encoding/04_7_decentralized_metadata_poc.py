"""
Step 4.7: Validate decentralized E_04 metadata creation.

Run this proof-of-concept (POC) after the normal E_04 runner has created central metadata under `encoded_metadata/`.
It simulates clients on one machine: Each bank contributes local aggregate statistics, then a server step rebuilds
global vocabularies, scalers and run counts from those aggregates. Afterward, the decentralized metadata is compared
to central metadata. This script is not the default training metadata path. It is evidence for the claim that central
event-log access is not required to create shared E_04 encoding metadata.

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved canonical schema
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved dataset mapping
    E_prefix_encoding/encoded_metadata/*/*/*_encoding_spec.json: central runner metadata for comparison
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: central vocabulary metadata for comparison
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: central scaler metadata for comparison
    E_main_BPIC_2017/data/processed/*/*.parquet: BPIC 2017 split parquets
    E_ablation_BPIC_2012/data/processed/*/*.parquet: BPIC 2012 split parquets

CREATED FILES (one file per run, banks nested as top-level keys where applicable):
    E_prefix_encoding/decentralized_poc/local_stats/<dataset>/<run>.json
    E_prefix_encoding/decentralized_poc/secure_aggregation_messages/<dataset>/<run>.json
    E_prefix_encoding/decentralized_poc/server_aggregation/<dataset>/<run>.json
    E_prefix_encoding/decentralized_poc/comparison_reports/<dataset>/<run>.json
    E_prefix_encoding/decentralized_poc/04_07_DECENTRALIZED_poc_summary.json
"""

# IMPORTS
from __future__ import annotations
import argparse
import importlib
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

# Allow direct script execution from the repository root.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the frozen E_04 runtime pieces without changing the production runner.
encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")
runner = importlib.import_module("E_prefix_encoding.04_4_runner")

# Reuse the encoder helpers so the POC normalizes and serializes exactly like the central runner.
# The underscore access is deliberate, so the next line silences the PyCharm protected member warning.
# noinspection PyProtectedMember
json_default, normalize_token = encoding._json_default, encoding._normalize_token

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent                      # Folder that contains this script
MAPPING_ROOT: Path = SCRIPT_DIR / "mappings"                            # Folder with the approved manual mapping files
SCHEMA_PATH: Path = MAPPING_ROOT / "MANUAL_canonical_schemas.json"      # Approved schema used by central E_04
MAPPING_PATH: Path = MAPPING_ROOT / "MANUAL_dataset_mapping.json"       # Approved mapping used locally by each client
CENTRAL_METADATA_ROOT: Path = SCRIPT_DIR / "encoded_metadata"           # Central metadata comparison target
POC_OUTPUT_ROOT: Path = SCRIPT_DIR / "decentralized_poc"                # Generated POC evidence
SCHEMA_PROFILE: str = "all"                                             # Run the same matrix as the freeze workflow
USE_SECURE_AGGREGATION_SIMULATION: bool = True                          # Mask additive client stats
SECURE_AGGREGATION_SEED: int = 42                                       # Reproducible masks for the one-machine POC

# Module logger and its default level.
LOG_LEVEL: str = "INFO"
log = logging.getLogger("E_04_decentralized_metadata_poc")

# ----------------------------------------------------------------------------------------------------------------------
# 1. JSON AND CLI HELPERS

# Configure script logging once for manual and workflow runs.
def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, str(level).upper()), format="%(asctime)s %(levelname)s %(message)s")

# HELPER: Parse optional automation (e.g., from workflows) while keeping defaults defined in the CONFIGURATION section.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E_04 decentralized metadata proof of concept.")
    parser.add_argument("--schema-profile", default=SCHEMA_PROFILE)
    parser.add_argument("--schema-path", type=Path, default=SCHEMA_PATH)
    parser.add_argument("--mapping-path", type=Path, default=MAPPING_PATH)
    parser.add_argument("--central-metadata-root", type=Path, default=CENTRAL_METADATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=POC_OUTPUT_ROOT)
    parser.add_argument("--no-secure-aggregation-simulation", action="store_true")
    parser.add_argument("--secure-aggregation-seed", type=int, default=SECURE_AGGREGATION_SEED)
    parser.add_argument("--log-level", default=LOG_LEVEL)
    return parser.parse_args(argv)

# HELPER: Write one generated POC artifact with stable formatting.
def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=json_default), encoding="utf-8")

# ----------------------------------------------------------------------------------------------------------------------
# 2. LOCAL CLIENT STATISTICS

# HELPER: Add one numeric train vector into count, sum and squared-sum statistics.
def _add_numeric_values(target: dict[str, dict[str, float]], column: str, values: pd.Series) -> None:

    # Convert one numeric feature to clean finite values before local aggregation.
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty: return

    # Store sufficient statistics needed to rebuild a global StandardScaler.
    stats = target.setdefault(column, {"count": 0.0, "sum": 0.0, "sum_squared": 0.0})
    stats["count"] += float(clean.shape[0])
    stats["sum"] += float(clean.sum())
    stats["sum_squared"] += float((clean * clean).sum())

# Store categorical train counts without validation or test categories.
def _categorical_train_counts(train: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for column in columns:
        if column not in train: continue

        # Normalize categories (exactly like the central encoder) before counting train values.
        values = train[column].map(normalize_token)
        counts[column] = {token: int(count) for token, count in sorted(Counter(values.tolist()).items())}
    return counts

# Store only train numeric statistics with the same masks as the central scaler fitting.
def _numeric_train_statistics(train: pd.DataFrame, numerical_columns: list[str],
                            offer_columns: list[str]) -> dict[str, dict[str, Union[int, float]]]:

    # Accumulate sufficient statistics per available numeric feature.
    stats: dict[str, dict[str, float]] = {}
    for column in numerical_columns:
        if column not in train: continue

        # Respect feature masks so unavailable static values do not influence global scaling.
        if column == encoding.REQUESTED_AMOUNT_VALUE and encoding.REQUESTED_AMOUNT_MASK in train:
            values = train.loc[train[encoding.REQUESTED_AMOUNT_MASK].astype(bool), column]
        else: values = train[column]
        _add_numeric_values(stats, column, values)

    # Use only prefixes where offer state values are actually available
    for column in offer_columns:
        if column not in train: continue
        values = train.loc[train[encoding.OFFER_FEATURE_MASK].astype(bool), column]
        _add_numeric_values(stats, column, values)

    # Convert the internal float accumulator into compact JSON-safe metadata.
    return {
        column: {
            "count": int(values["count"]),
            "sum": float(values["sum"]),
            "sum_squared": float(values["sum_squared"]),
        }
        for column, values in sorted(stats.items())
    }

# Store local additive RT target statistics in the selected transform space.
def remaining_time_target_statistics_from_events(events: pd.DataFrame, prefix_index: list[Any], transform: str,
    scaling: str) -> dict[str, Union[str, int, float, bool]]:

    # Median scaling is intentionally not claimed as decentralized because exact medians are not additive.
    if scaling == "median":
        return {
            "transform": transform,
            "scaling": scaling,
            "decentralized_supported": False,
            "count": 0,
            "sum": 0.0,
            "sum_squared": 0.0,
            "median": 0.0,
            "median_seconds": 0.0,
        }

    # Reuse prefix masking so target fitting sees the same positive train prefix population.
    values = encoding.remaining_time_values_from_prefixes(events, prefix_index)
    if (values < 0.0).any(): raise ValueError("remaining_time contains negative values")
    values = values.loc[values > 0.0]
    target_repr = encoding.fit_remaining_time_target_repr(values, transform, "raw")
    transformed = encoding.transform_remaining_time_target(values, target_repr)
    transformed = transformed[np.isfinite(transformed)].astype("float64")

    # Store additive moments for raw and z-score reconstruction.
    return {
        "transform": transform,
        "scaling": scaling,
        "decentralized_supported": True,
        "count": int(transformed.shape[0]),
        "sum": float(transformed.sum()),
        "sum_squared": float((transformed * transformed).sum()),
        "median": float(target_repr.median_model_units),
        "median_seconds": float(target_repr.train_median_seconds),
    }

# Collect one simulated client report from already mapped canonical events.
def collect_local_stats_from_events(events: pd.DataFrame, bank_id: str, run_name: str, dataset_id: str,
    categorical_columns: list[str], numerical_columns: list[str], offer_columns: list[str], max_prefix_length: int,
    remaining_time_transform: str = encoding.DEFAULT_REMAINING_TIME_TRANSFORM,
    remaining_time_scaling: str = encoding.DEFAULT_REMAINING_TIME_SCALING,
    ) -> dict[str, Any]:

    # Keep train separate because vocabularies and scalers must not use validation or test events.
    train = events.loc[events[encoding.SPLIT] == "train"].copy()

    # Build local prefix counts with the same cap as the central metadata runner.
    prefix_index, observed_max = encoding.build_prefix_index(events, max_prefix_length)
    case_counts = events.groupby(encoding.CASE_ID, sort=False).size()

    # Count local metadata only. Case identifiers and rows are discarded after aggregation.
    counts = {
        "case_count": int(case_counts.shape[0]),
        "event_count": int(events.shape[0]),
        "prefix_count": int(len(prefix_index)),
        "observed_max_trace_length": int(observed_max),
    }

    # Count one train outcome per case to mirror the central encoding spec.
    train_cases = train.drop_duplicates([encoding.CASE_ID])
    train_outcomes = Counter(train_cases[encoding.OUTCOME].astype(int).astype(str).tolist())

    # Include next activity targets so the server can rebuild the target vocabulary.
    vocab_columns = list(categorical_columns)
    if encoding.NEXT_ACTIVITY_TARGET not in vocab_columns: vocab_columns.append(encoding.NEXT_ACTIVITY_TARGET)

    # Return aggregated metadata per bank (the privacy scope is stated once in the POC summary).
    return {
        "dataset": dataset_id,
        "run": run_name,
        "bank": str(bank_id),
        "counts": counts,
        "train_outcome_counts": {label: int(count) for label, count in sorted(train_outcomes.items())},
        "categorical_train_counts": _categorical_train_counts(train, vocab_columns),
        "numeric_train_statistics": _numeric_train_statistics(train, numerical_columns, offer_columns),
        "remaining_time_target_statistics": remaining_time_target_statistics_from_events(
            events, prefix_index, remaining_time_transform, remaining_time_scaling,
        ),
    }

# Load one bank locally and create its private aggregate report.
def collect_local_stats_for_bank(config: Any, mapping_payload: dict[str, Any], mapping_path: Path,
    schema_profile_payload: dict[str, Any], bank_id: str) -> dict[str, Any]:

    # Resolve the local split files for this simulated bank.
    frames: list[pd.DataFrame] = []
    all_split_paths = encoding.discover_split_paths(mapping_payload, config.dataset, config.heterogeneity, config.n_clients)
    split_paths = all_split_paths[bank_id]

    # Each simulated client reads only its own train, validation and test files.
    for split_name, path in split_paths.items():
        raw = pd.read_parquet(path)
        frames.append(encoding.to_canonical_events(raw, mapping_payload, config.dataset, bank_id, split_name))

    # Apply the local activity mapping before any metadata leaves the client boundary.
    local_events = pd.concat(frames, ignore_index=True)
    mapped, _ = encoding.apply_activity_mapping(local_events, mapping_path)

    # Return only aggregate metadata, not mapped event.
    return collect_local_stats_from_events(
        mapped,
        bank_id=bank_id,
        run_name=f"{config.heterogeneity}_{config.n_clients}banks",
        dataset_id=config.dataset,
        categorical_columns=list(schema_profile_payload["sequence_categorical_columns"]),
        numerical_columns=list(schema_profile_payload["sequence_numerical_columns"]),
        offer_columns=list(schema_profile_payload["offer_numerical_columns"]),
        max_prefix_length=int(config.max_prefix_length_for_encoding),
        remaining_time_transform=getattr(config, "remaining_time_transform", encoding.DEFAULT_REMAINING_TIME_TRANSFORM),
        remaining_time_scaling=getattr(config, "remaining_time_scaling", encoding.DEFAULT_REMAINING_TIME_SCALING),
    )

# Load one qualified joint client and create its private aggregate report.
def collect_local_stats_for_joint_bank(config: Any, mapping_payload: dict[str, Any], mapping_path: Path,
    schema_profile_payload: dict[str, Any], qualified_bank_id: str) -> dict[str, Any]:

    # Resolve the source dataset and plain bank id from the dataset-qualified client name.
    spec = encoding.resolve_joint_run(f"{config.heterogeneity}_{config.n_clients}banks")
    if ":" not in qualified_bank_id: raise ValueError(f"joint bank id must be dataset-qualified: {qualified_bank_id}")
    source_dataset, bank_id = str(qualified_bank_id).split(":", 1)
    source = next((item for item in spec.sources if item.dataset_id == source_dataset), None)
    if source is None: raise ValueError(f"joint run {spec.run_id} does not contain source dataset: {source_dataset}")
    if bank_id not in source.banks: raise ValueError(f"joint run {spec.run_id} does not contain bank: {qualified_bank_id}")

    # Resolve only the source split files because the mapping has no joint dataset block.
    frames: list[pd.DataFrame] = []
    all_split_paths = encoding.discover_split_paths(
        mapping_payload, source.dataset_id, source.heterogeneity, source.n_clients
    )
    split_paths = all_split_paths[bank_id]

    # Each simulated joint client reads one source bank and keeps the source dataset id in event rows.
    for split_name, path in split_paths.items():
        raw = pd.read_parquet(path)
        frames.append(encoding.to_canonical_events(raw, mapping_payload, source.dataset_id, bank_id, split_name))

    # Apply the local activity mapping before any metadata leaves the client boundary.
    local_events = pd.concat(frames, ignore_index=True)
    mapped, _ = encoding.apply_activity_mapping(local_events, mapping_path)

    # Return aggregate metadata under the dataset-qualified client name.
    return collect_local_stats_from_events(
        mapped,
        bank_id=qualified_bank_id,
        run_name=spec.run_id,
        dataset_id=config.dataset,
        categorical_columns=list(schema_profile_payload["sequence_categorical_columns"]),
        numerical_columns=list(schema_profile_payload["sequence_numerical_columns"]),
        offer_columns=list(schema_profile_payload["offer_numerical_columns"]),
        max_prefix_length=int(config.max_prefix_length_for_encoding),
        remaining_time_transform=getattr(config, "remaining_time_transform", encoding.DEFAULT_REMAINING_TIME_TRANSFORM),
        remaining_time_scaling=getattr(config, "remaining_time_scaling", encoding.DEFAULT_REMAINING_TIME_SCALING),
    )

# ----------------------------------------------------------------------------------------------------------------------
# 3. SERVER AGGREGATION PRIMITIVES

# Convert local count, sum and squared-sum records into one StandardScaler record.
def aggregate_numeric_statistics(stats: list[dict[str, float]]) -> dict[str, float]:

    # Combine local sufficient statistics before computing the global mean.
    count = float(sum(item.get("count", 0.0) for item in stats))
    if count <= 0: return {"mean": 0.0, "std": 1.0}
    total = float(sum(item.get("sum", 0.0) for item in stats))
    total_squared = float(sum(item.get("sum_squared", 0.0) for item in stats))
    mean = total / count

    # Reconstruct population variance from the global sum of squares.
    variance = max((total_squared / count) - (mean * mean), 0.0)
    std = math.sqrt(variance)
    return {"mean": float(mean), "std": float(std if std > 0 else 1.0)}

# Build one deterministic vocabulary from aggregated train-category counts.
def build_vocabulary_from_counts(counts: dict[str, int], include_end_token: bool = False,
    include_other_activity: bool = False) -> dict[str, int]:

    # Reserve stable token ids so every simulated client uses the same padding semantics.
    vocab = {encoding.PAD_TOKEN: 0, encoding.UNK_TOKEN: 1, encoding.MISSING_TOKEN: 2}
    if include_other_activity: vocab[encoding.OTHER_ACTIVITY_TOKEN] = len(vocab)
    if include_end_token: vocab[encoding.END_TOKEN] = len(vocab)

    # Add observed train categories in sorted order for deterministic global metadata.
    for token in sorted(str(value) for value in counts if str(value) not in vocab): vocab[token] = len(vocab)
    return vocab

# ----------------------------------------------------------------------------------------------------------------------
# 4. ADDITIVE AND SECURE AGGREGATION

# Keep only additive fields that can be masked and summed.
def local_stats_to_additive_payload(local_stats: dict[str, Any]) -> dict[str, Any]:

    # Counts, category frequencies and numeric sufficient statistics can be securely summed.
    return {
        "counts": {
            "case_count": float(local_stats["counts"]["case_count"]),
            "event_count": float(local_stats["counts"]["event_count"]),
            "prefix_count": float(local_stats["counts"]["prefix_count"]),
        },
        "train_outcome_counts": {
            str(label): float(count)
            for label, count in local_stats.get("train_outcome_counts", {}).items()
        },
        "categorical_train_counts": {
            str(column): {str(token): float(count) for token, count in counts.items()}
            for column, counts in local_stats.get("categorical_train_counts", {}).items()
        },
        "numeric_train_statistics": {
            str(column): {str(key): float(value) for key, value in values.items()}
            for column, values in local_stats.get("numeric_train_statistics", {}).items()
        },
        "remaining_time_target_statistics": {
            str(key): float(value)
            for key, value in local_stats.get("remaining_time_target_statistics", {}).items()
            if isinstance(value, (int, float))
        },
    }

# Add nested numeric payloads while preserving the union of keys.
def _add_nested(left: Any, right: Any, sign: float = 1.0) -> Any:

    # Recurse through nested dictionaries so category and feature keys stay aligned.
    if isinstance(left, dict) or isinstance(right, dict):
        left_dict = left if isinstance(left, dict) else {}
        right_dict = right if isinstance(right, dict) else {}
        keys = sorted(set(left_dict).union(set(right_dict)))
        return {
            key: _add_nested(left_dict.get(key, 0.0), right_dict.get(key, 0.0), sign)
            for key in keys
        }

    # Leaf values are numeric counts, sums or mask.
    return float(left) + (sign * float(right))

# HELPER: Aggregate additive payloads exactly as the server would after masks cancel.
def aggregate_additive_payloads(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {}

    # Sum every nested payload into one global additive result.
    for payload in payloads: total = _add_nested(total, payload)
    return total

# HELPER: Create one random mask with the same nested structure as a client payload.
def _random_mask_like(payload: Any, rng: np.random.Generator) -> Any:

    # Preserve the payload shape so masks can be added field by field.
    if isinstance(payload, dict): return {key: _random_mask_like(value, rng) for key, value in payload.items()}
    return float(rng.integers(-1_000_000, 1_000_000))

# HELPER: Build masked client messages whose masks cancel when the server sums all messages.
def build_masked_client_messages(local_stats: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    if not local_stats: return []

    # Convert each readable local report into the additive part that can be masked.
    rng = np.random.default_rng(seed)
    payloads = [local_stats_to_additive_payload(stats) for stats in local_stats]
    cumulative_mask: dict[str, Any] = {}
    messages: list[dict[str, Any]] = []
    for index, stats in enumerate(local_stats):
        payload = payloads[index]

        # Add a random mask to every client except the last one.
        if index < len(local_stats) - 1:
            mask = _random_mask_like(payload, rng)
            cumulative_mask = _add_nested(cumulative_mask, mask)
            masked_payload = _add_nested(payload, mask)

        # The last client sends the negative cumulative mask, so all masks cancel globally.
        else: masked_payload = _add_nested(payload, cumulative_mask, sign=-1.0)

        # Store only the masked payload that a server would receive in this simulation.
        messages.append({"bank": str(stats.get("bank")), "masked_payload": masked_payload})
    return messages

# Recover the global additive payload from masked client messages.
def aggregate_masked_messages(masked_messages: list[dict[str, Any]]) -> dict[str, Any]:

    # The server sees masked messages, not the original local reports.
    payloads = [message["masked_payload"] for message in masked_messages]
    return aggregate_additive_payloads(payloads)

# Rebuild vocabulary inputs from the aggregated categorical counts.
def _vocabularies_from_additive_payload(payload: dict[str, Any],
                                        categorical_columns: list[str]) -> dict[str, dict[str, int]]:

    # Rebuild the target vocabulary from recovered global target counts.
    categorical_counts = payload.get("categorical_train_counts", {})
    vocabs = {
        encoding.NEXT_ACTIVITY_TARGET: build_vocabulary_from_counts(
            categorical_counts.get(encoding.NEXT_ACTIVITY_TARGET, {}),
            include_end_token=True,
            include_other_activity=True,
        ),
    }

    # Rebuild model input vocabularies from recovered global feature counts.
    for column in categorical_columns:
        if column in categorical_counts:
            vocabs[column] = build_vocabulary_from_counts(
                categorical_counts[column], include_other_activity=column == encoding.CANONICAL_ACTIVITY_TOKEN,
            )
    return vocabs

# Rebuild scaler inputs from the aggregated numeric count, sum and squared sum.
def _scalers_from_additive_payload(payload: dict[str, Any]) -> dict[str, dict[str, float]]:
    numeric_stats = payload.get("numeric_train_statistics", {})

    # Cyclical features carry an identity scaler in central fitting, so the POC mirrors it instead of aggregating.
    return {
        column: (
            {"mean": 0.0, "std": 1.0}
            if column in encoding.CYCLICAL_NUMERICAL_COLUMNS
            else aggregate_numeric_statistics([values])
        )
        for column, values in sorted(numeric_stats.items())
    }

# Rebuild the E_04 RT target scaler from the recovered additive payload.
def remaining_time_target_from_additive_payload(payload: dict[str, Any], transform: str, scaling: str) -> dict[str, Any]:
    record = payload.get("remaining_time_target_statistics", {})
    if scaling == "median":
        raise ValueError("remaining-time median scaling is not decentralized through additive statistics")
    if not record:
        return encoding.remaining_time_target_repr_to_dict(
            encoding.RemainingTimeTargetRepr(transform, "raw", 0.0, 1.0, True, 0.0, 0.0, 0)
        )
    count = int(round(float(record.get("count", 0.0))))
    if scaling == "raw":
        return encoding.remaining_time_target_repr_to_dict(
            encoding.RemainingTimeTargetRepr(transform, scaling, 0.0, 1.0, True, 0.0, 0.0, count)
        )

    # Reconstruct z-score center and scale from recovered count, sum and squared sum.
    scaler = aggregate_numeric_statistics([
        {
            "count": float(record.get("count", 0.0)),
            "sum": float(record.get("sum", 0.0)),
            "sum_squared": float(record.get("sum_squared", 0.0)),
        }
    ])
    return encoding.remaining_time_target_repr_to_dict(
        encoding.RemainingTimeTargetRepr(
            transform, scaling, float(scaler["mean"]), float(scaler["std"]), False, 0.0, 0.0, count,
        )
    )

# Rebuild the RT target scaler from local reports through the additive path.
def remaining_time_target_from_local_stats(local_stats: list[dict[str, Any]]) -> dict[str, Any]:
    records = [stats.get("remaining_time_target_statistics", {}) for stats in local_stats]
    records = [record for record in records if record]
    if not records:
        return encoding.remaining_time_target_repr_to_dict(
            encoding.RemainingTimeTargetRepr("raw", "raw", 0.0, 1.0, True, 0.0, 0.0, 0)
        )
    transform = str(records[0].get("transform", encoding.DEFAULT_REMAINING_TIME_TRANSFORM))
    scaling = str(records[0].get("scaling", encoding.DEFAULT_REMAINING_TIME_SCALING))
    if any(str(record.get("transform", transform)) != transform or str(record.get("scaling", scaling)) != scaling for record in records):
        raise ValueError("local remaining-time target statistics use inconsistent representations")
    recovered = aggregate_additive_payloads([
        {
            "remaining_time_target_statistics": {
                str(key): float(value)
                for key, value in record.items()
                if isinstance(value, (int, float))
            }
        }
        for record in records
    ])
    return remaining_time_target_from_additive_payload(recovered, transform, scaling)

# ----------------------------------------------------------------------------------------------------------------------
# 5. SERVER METADATA BUILDING

# Build the server-side aggregate metadata from local reports through one additive path.
# Plain and secure runs produce identical sums because the masks cancel, so both reuse the same recovered payload.
def aggregate_local_stats(local_stats: list[dict[str, Any]], dataset: str, run_name: str, categorical_columns: list[str],
    cap: int, use_secure_aggregation: bool, seed: int,) -> tuple[dict[str, Any], list[dict[str, Any]]]:

    payloads = [local_stats_to_additive_payload(stats) for stats in local_stats]

    # The secure path recovers the global sums from masked messages; the plain path sums them directly.
    if use_secure_aggregation:
        masked_messages = build_masked_client_messages(local_stats, seed)
        recovered = aggregate_masked_messages(masked_messages)
    else:
        masked_messages = []
        recovered = aggregate_additive_payloads(payloads)

    # Keep counts per bank only as compatibility metadata for the central comparison.
    per_bank = {
        str(stats["bank"]): {
            "case_count": int(stats["counts"]["case_count"]),
            "event_count": int(stats["counts"]["event_count"]),
            "prefix_count": int(stats["counts"]["prefix_count"]),
        }
        for stats in local_stats
    }

    # Keep observed max outside the additive sum because max is not additive.
    observed_max = max(int(stats["counts"]["observed_max_trace_length"]) for stats in local_stats)

    # Convert recovered additive floats back to integer count records.
    aggregate_counts = {key: int(round(value)) for key, value in recovered.get("counts", {}).items()}
    train_outcomes = {
        str(label): int(round(count))
        for label, count in sorted(recovered.get("train_outcome_counts", {}).items())
    }
    remaining_stats = local_stats[0].get("remaining_time_target_statistics", {})
    remaining_transform = str(remaining_stats.get("transform", encoding.DEFAULT_REMAINING_TIME_TRANSFORM))
    remaining_scaling = str(remaining_stats.get("scaling", encoding.DEFAULT_REMAINING_TIME_SCALING))

    # Store the reconstructed server metadata. The privacy claim is stated once in the POC summary.
    aggregated = {
        "dataset": dataset,
        "run": run_name,
        "secure_aggregation": bool(use_secure_aggregation),
        "counts": {
            "aggregate": aggregate_counts,
            "per_bank": per_bank,
        },
        "train_outcome_counts": train_outcomes,
        "prefix": {
            "cap": int(cap),
            "observed_max_trace_length": int(observed_max),
            "static_padding_length": int(cap),
        },
        "vocabularies": _vocabularies_from_additive_payload(recovered, categorical_columns),
        "scalers": _scalers_from_additive_payload(recovered),
        "target_scalers": {
            "remaining_time": remaining_time_target_from_additive_payload(
                recovered,
                remaining_transform,
                remaining_scaling,
            ),
        },
    }
    return aggregated, masked_messages

# ----------------------------------------------------------------------------------------------------------------------
# 6. CENTRAL COMPARISON

# Compare floating records with a small tolerance for reconstruction error.
def _float_records_close(aggregated: dict[str, Any], central: dict[str, Any]) -> bool:
    if set(aggregated) != set(central): return False
    return all(
        math.isclose(float(aggregated[column][key]), float(central[column][key]), rel_tol=1e-9, abs_tol=1e-9)
        for column in aggregated
        for key in set(aggregated[column]).intersection(set(central[column]))
        if isinstance(aggregated[column][key], (int, float)) and isinstance(central[column][key], (int, float))
    )

# Compare target scaler records including string fields and numeric tolerances.
def _target_scalers_close(aggregated: dict[str, Any], central: dict[str, Any]) -> bool:
    if set(aggregated) != set(central): return False
    for target, record in aggregated.items():
        central_record = central[target]
        if set(record) != set(central_record): return False
        for key, value in record.items():
            other = central_record[key]
            if isinstance(value, (int, float)) and isinstance(other, (int, float)):
                if not math.isclose(float(value), float(other), rel_tol=1e-9, abs_tol=1e-9): return False
            elif value != other: return False
    return True

# Compare server aggregation against the existing central E_04 metadata.
def compare_aggregated_to_central(aggregated: dict[str, Any], central_spec: dict[str, Any],
    central_vocab: dict[str, Any], central_scaler: dict[str, Any]) -> dict[str, Any]:

    # One boolean per metadata family. The list of differences names the families that did not match.
    checks = {
        "counts_match": aggregated["counts"]["aggregate"] == central_spec["counts"]["aggregate"],
        "per_bank_counts_match": aggregated["counts"].get("per_bank") == central_spec["counts"].get("per_bank"),
        "train_outcome_counts_match": aggregated["train_outcome_counts"] == central_spec["counts"].get("train_outcome_counts"),
        "vocabularies_match": aggregated["vocabularies"] == central_vocab,
        "prefix_settings_match": aggregated["prefix"] == central_spec["prefix"],
        "scalers_match": _float_records_close(aggregated["scalers"], central_scaler),
        "target_scalers_match": _target_scalers_close(
            aggregated.get("target_scalers", {}), central_spec.get("target_scalers", {}),
        ),
    }
    differences = sorted(name for name, passed in checks.items() if not passed)
    return {
        "matches_central_metadata": all(checks.values()),
        "checks": checks,
        "differences": differences,
    }

# Load central runner artifacts for one run.
def load_central_artifacts(central_root: Path, dataset: str, heterogeneity: str,
    n_clients: int) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:

    # Reuse the artifact path helper so the POC compares against the right run folder.
    paths = encoding.build_artifact_paths(central_root, dataset, heterogeneity, n_clients, joint=dataset == "joint")
    return (
        encoding.load_json_artifact(paths.encoding_spec),
        encoding.load_json_artifact(paths.vocabulary),
        encoding.load_json_artifact(paths.scaler),
    )

# ----------------------------------------------------------------------------------------------------------------------
# 7. OUTPUTS

# Write the local reports, masked messages, server aggregation and central comparison for one run.
def write_poc_outputs(root: Path, dataset: str, run_name: str, local_stats: list[dict[str, Any]],
    aggregated: dict[str, Any], comparison: dict[str, Any], masked_messages: Optional[list[dict[str, Any]]] = None,
    ) -> list[Path]:

    # Store written paths.
    written: list[Path] = []

    # One combined local report per run, banks nested under their bank id.
    local_path = root / "local_stats" / dataset / f"{run_name}.json"
    save_json(local_path, {f"bank_{stats['bank']}": stats for stats in local_stats})
    written.append(local_path)

    # One combined server-visible message file per run, banks nested under their bank id.
    if masked_messages:
        message_path = root / "secure_aggregation_messages" / dataset / f"{run_name}.json"
        save_json(message_path, {f"bank_{message['bank']}": message["masked_payload"] for message in masked_messages})
        written.append(message_path)

    # The reconstructed server metadata and the central comparison report.
    aggregation_path = root / "server_aggregation" / dataset / f"{run_name}.json"
    comparison_path = root / "comparison_reports" / dataset / f"{run_name}.json"
    save_json(aggregation_path, aggregated)
    save_json(comparison_path, comparison)
    written.extend([aggregation_path, comparison_path])
    return written

# ----------------------------------------------------------------------------------------------------------------------
# 8. SUMMARY OUTPUT

# Build one thesis summary from the run records in memory
def build_poc_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    check_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"passed": 0, "failed": 0})
    datasets: dict[str, Any] = {}

    # Fold every run record into per-check tallies and per-dataset blocks.
    for record in records:
        for check_name, passed in record["checks"].items():
            check_counts[str(check_name)]["passed" if passed else "failed"] += 1

        matched = bool(record["matches_central_metadata"])
        entry = datasets.setdefault(
            record["dataset"],
            {"runs": {}, "run_count": 0, "matched_runs": 0, "mismatched_runs": 0, "local_client_reports": 0},
        )
        entry["run_count"] += 1
        entry["matched_runs"] += int(matched)
        entry["mismatched_runs"] += int(not matched)
        entry["local_client_reports"] += int(record["bank_count"])
        entry["runs"][record["run"]] = {
            "matches_central_metadata": matched,
            "bank_count": int(record["bank_count"]),
            "aggregate_counts": record["aggregate_counts"],
            "differences": record["differences"],
        }
    for entry in datasets.values(): entry["all_runs_match_central_metadata"] = entry["mismatched_runs"] == 0

    total_runs = len(records)
    matched_runs = sum(1 for record in records if record["matches_central_metadata"])
    secure_flags = [bool(record["secure_enabled"]) for record in records]
    return {
        "generated_by": "04_7_decentralized_metadata_poc.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Summarize decentralized E_04 metadata reconstruction evidence.",
        "run_summary": {
            "total_runs": total_runs,
            "matched_runs": matched_runs,
            "mismatched_runs": total_runs - matched_runs,
            "all_runs_match_central_metadata": matched_runs == total_runs,
        },
        "secure_aggregation_summary": {
            "enabled_in_all_runs": bool(secure_flags) and all(secure_flags),
            "production_protocol_implemented": False,
        },
        "privacy_scope": {
            "server_received_event_rows": False,
            "server_received_prefix_tensors": False,
            "server_received_case_ids": False,
            "local_dataset_mapping_can_stay_client_side": True,
        },
        "check_summary": dict(sorted(check_counts.items())),
        "datasets": dict(sorted(datasets.items())),
        "limitations": [
            "One-machine simulation, not real networked clients.",
            "No production key exchange, authentication or dropout handling.",
            "No differential privacy.",
        ],
    }

# Write one JSON summary for the whole POC.
def write_poc_summary(root: Path, records: list[dict[str, Any]]) -> Path:
    summary = build_poc_summary(records)
    json_path = root / "04_07_DECENTRALIZED_poc_summary.json"
    save_json(json_path, summary)
    return json_path

# ----------------------------------------------------------------------------------------------------------------------
# 9. RUN MATRIX

# Run one decentralized POC matrix entry and return its compact summary record.
def run_one_poc(config: Any, schema_profile_payload: dict[str, Any], mapping_payload: dict[str, Any], mapping_path: Path,
    central_root: Path, output_root: Path, use_secure_aggregation: bool, secure_aggregation_seed: int) -> dict[str, Any]:

    # Create the entry record.
    run_name = f"{config.heterogeneity}_{config.n_clients}banks"
    log.info("Running decentralized metadata POC for %s %s", config.dataset, run_name)

    # Let each simulated bank produce its own local aggregate report.
    if config.dataset == "joint":
        joint_spec = encoding.resolve_joint_run(run_name)
        local_stats = [
            collect_local_stats_for_joint_bank(config, mapping_payload, mapping_path, schema_profile_payload, bank_id)
            for bank_id in joint_spec.qualified_client_ids
        ]
    else:
        # Discover the simulated banks for this dataset, heterogeneity level and client count.
        split_paths = encoding.discover_split_paths(mapping_payload, config.dataset, config.heterogeneity, config.n_clients)
        local_stats = [
            collect_local_stats_for_bank(config, mapping_payload, mapping_path, schema_profile_payload, bank_id)
            for bank_id in sorted(split_paths)
        ]
    categorical_columns = list(schema_profile_payload["sequence_categorical_columns"])

    # Rebuild the server metadata through the additive path.
    aggregated, masked_messages = aggregate_local_stats(
        local_stats, config.dataset, run_name, categorical_columns, config.max_prefix_length_for_encoding,
        use_secure_aggregation, secure_aggregation_seed,
    )

    # Load central runner artifacts for the same run and compare all metadata families.
    central_spec, central_vocab, central_scaler = load_central_artifacts(
        central_root, config.dataset, config.heterogeneity, config.n_clients,
    )
    comparison = compare_aggregated_to_central(aggregated, central_spec, central_vocab, central_scaler)

    # Persist all POC evidence so the privacy claim can be inspected after the run.
    write_poc_outputs(output_root, config.dataset, run_name, local_stats, aggregated, comparison, masked_messages)
    if not comparison["matches_central_metadata"]:
        log.warning(
            "Decentralized POC differs from central metadata for %s %s: %s", config.dataset, run_name,
            comparison["differences"],
        )

    # Hand the summary builder everything it needs (without re-reading written files).
    return {
        "dataset": config.dataset,
        "run": run_name,
        "matches_central_metadata": comparison["matches_central_metadata"],
        "checks": comparison["checks"],
        "differences": comparison["differences"],
        "secure_enabled": bool(use_secure_aggregation),
        "bank_count": len(local_stats),
        "aggregate_counts": aggregated["counts"]["aggregate"],
    }

# Run the selected POC matrix.
def run_poc_matrix(schema_profile: str, schema_path: Path, mapping_path: Path, central_root: Path, output_root: Path,
    use_secure_aggregation: bool, secure_aggregation_seed: int) -> list[dict[str, Any]]:

    # Load the approved workflow inputs used by the normal central runner.
    schema_payload = encoding.load_approved_json(schema_path, "canonical schema")
    mapping_payload = encoding.load_dataset_mapping(mapping_path, require_approved=True)
    profiles = schema_payload["schema_profiles"]
    records: list[dict[str, Any]] = []

    # Reuse the frozen runner matrix so the POC covers the same configurations.
    for dataset, heterogeneity, n_clients, cap in runner.matrix_for_profile(schema_profile):
        run_profile = dataset if schema_profile == "all" else schema_profile

        # Build the same run config that the central runner used for metadata creation.
        config = runner.build_config(run_profile, dataset, heterogeneity, n_clients, cap, central_root)

        # Run one local client simulation and collect its summary record
        records.append(
            run_one_poc(
                config, profiles[run_profile], mapping_payload, mapping_path, central_root, output_root,
                use_secure_aggregation, secure_aggregation_seed,
            )
        )
    return records

# ----------------------------------------------------------------------------------------------------------------------
# 10. MAIN

# Execute the decentralized metadata POC.
def main(argv: Optional[list[str]] = None) -> None:

    args = parse_args(argv)
    _configure_logging(args.log_level)

    # Execute all selected decentralized metadata simulations.
    records = run_poc_matrix(
        args.schema_profile, args.schema_path, args.mapping_path, args.central_metadata_root, args.output_root,
        not bool(args.no_secure_aggregation_simulation), int(args.secure_aggregation_seed),
    )

    # Write one compact summary file for thesis and meeting evidence.
    summary_path = write_poc_summary(args.output_root, records)
    log.info("Wrote decentralized POC summary to %s", summary_path)

    # Stop the workflow if any reconstructed server metadata differs from the central runner.
    failures = [record for record in records if not record["matches_central_metadata"]]
    if failures: raise RuntimeError(f"decentralized metadata POC found {len(failures)} mismatching runs")

    # Log the number of matched runs as freeze evidence.
    log.info("Decentralized metadata POC matched central metadata for %d runs", len(records))

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────