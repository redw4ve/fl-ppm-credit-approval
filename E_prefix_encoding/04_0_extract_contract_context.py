"""
Step 4.0: Extract E_04 contract context without exporting event rows.

Run this script before contract design when a federation engineer needs to understand which source schemas,
activity labels and aggregate statistics are available. The script reads processed A_02 and B_02 split parquets
inside the data owner boundary and writes JSON schema summaries.

The output is not used by the current BPIC runner. It documents the privacy-oriented deployment path:
Real banks can share contract context instead of raw logs, while canonical schemas, dataset mappings and encoding
can be created locally.

REQUIRED FILES:
    E_main_BPIC_2017/data/processed/*/A_02 split parquets: BPIC 2017 processed split files
    E_ablation_BPIC_2012/data/processed/*/B_02 split parquets: BPIC 2012 processed split files

CREATED FILES:
    E_prefix_encoding/decentralized_poc/contract_context/04_00_bpic2017_contract_context.json: BPIC 2017 summary
    E_prefix_encoding/decentralized_poc/contract_context/04_00_bpic2012_contract_context.json: BPIC 2012 summary
    E_prefix_encoding/decentralized_poc/contract_context/04_00_federation_contract_context.json: combined summary
"""

# IMPORTS
from __future__ import annotations
import argparse
import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union
import numpy as np
import pandas as pd

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent
REPO_ROOT: Path = SCRIPT_DIR.parent
OUTPUT_ROOT: Path = SCRIPT_DIR / "decentralized_poc" / "contract_context"   # Destination for generated POC context
DATASETS_TO_SUMMARIZE: tuple[str, ...] = ("bpic2017", "bpic2012")           # Source datasets included in the export
HETEROGENEITY_TO_SCAN: tuple[str, ...] = ("iid_3banks",)                    # One run is enough (schema is stable)
SPLITS_TO_SCAN: tuple[str, ...] = ("train", "val", "test")                  # Scan all split files for completeness
INCLUDE_ACTIVITY_COUNTS: bool = False                                       # Keep false for more metadata disclosure
INCLUDE_MISSINGNESS: bool = True                                            # Missingness rates help field selection
INCLUDE_NUMERIC_TRAIN_STATS: bool = True                                    # Train-only sums support scaler framing

DATASET_INPUTS: dict[str, dict[str, Any]] = {
    "bpic2017": {
        "input_root": REPO_ROOT / "E_main_BPIC_2017" / "data" / "processed",
        "split_prefix": "A_02",
    },
    "bpic2012": {
        "input_root": REPO_ROOT / "E_ablation_BPIC_2012" / "data" / "processed",
        "split_prefix": "B_02",
    },
}

# Configure Logger.
LOG_LEVEL: str = "INFO"
LOGGER = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# 1. JSON AND CLI HELPERS

# Convert local Python and numpy values into JSON safe values.
def _json_default(value: object) -> object:
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    return value

# Write one generated context artifact with stable formatting.
def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

# Parse comma separated workflow values for optional automation.
def _parse_csv(value: str) -> tuple[str, ...]: return tuple(item.strip() for item in value.split(",") if item.strip())

# Keep command line overrides optional for bash scripts. For normal use edit the configuration block above.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract E_04 schema-level contract context.")
    parser.add_argument("--datasets", default=",".join(DATASETS_TO_SUMMARIZE))
    parser.add_argument("--heterogeneities", default=",".join(HETEROGENEITY_TO_SCAN))
    parser.add_argument("--splits", default=",".join(SPLITS_TO_SCAN))
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--include-activity-counts", action="store_true", default=INCLUDE_ACTIVITY_COUNTS)
    parser.add_argument("--no-missingness", action="store_true")
    parser.add_argument("--no-numeric-train-stats", action="store_true")
    parser.add_argument("--log-level", default=LOG_LEVEL)
    return parser.parse_args(argv)

# ----------------------------------------------------------------------------------------------------------------------
# 2. SOURCE FILE DISCOVERY

# Find client split parquets for the selected run names and split names.
def find_split_files(input_root: Path, split_prefix: str, heterogeneities: tuple[str, ...],
    splits: tuple[str, ...]) -> list[Path]:
    paths: list[Path] = []
    for heterogeneity in heterogeneities:
        run_dir = input_root / heterogeneity
        for split_name in splits: paths.extend(sorted(run_dir.glob(f"{split_prefix}_bank_*_{split_name}.parquet")))
    if not paths: raise FileNotFoundError(f"no split parquet files found under {input_root}")
    return paths

# Read the preprocessing metadata keys that are useful for contract context.
def _load_selected_metadata(input_root: Path, split_prefix: str, heterogeneities: tuple[str, ...]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for heterogeneity in heterogeneities:
        path = input_root / heterogeneity / f"{split_prefix}_metadata.json"
        if not path.exists(): continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config", {})
        metadata[heterogeneity] = {
            "max_prefix_length_for_encoding": config.get("MAX_PREFIX_LENGTH_FOR_ENCODING"),
            "remaining_time_unit": config.get("REMAINING_TIME_UNIT"),
            "remaining_time_log_transform_stage": config.get("REMAINING_TIME_LOG_TRANSFORM_STAGE"),
        }
    return metadata

# ----------------------------------------------------------------------------------------------------------------------
# 3. SUMMARY BUILDING

# Return one percentile for non-empty numeric vectors.
def _percentile(values: list[int], q: float) -> Optional[float]:
    if not values: return None
    return float(np.percentile(np.asarray(values, dtype=float), q))

# Summarize trace lengths without exporting case identifiers.
def _trace_length_summary(lengths: list[int]) -> dict[str, Any]:
    if not lengths:
        return {
            "case_count": 0,
            "event_count": 0,
            "min": None,
            "p50": None,
            "p95": None,
            "p98": None,
            "max": None,
            "mean": None,
        }
    return {
        "case_count": int(len(lengths)),
        "event_count": int(sum(lengths)),
        "min": int(min(lengths)),
        "p50": _percentile(lengths, 50),
        "p95": _percentile(lengths, 95),
        "p98": _percentile(lengths, 98),
        "max": int(max(lengths)),
        "mean": float(np.mean(np.asarray(lengths, dtype=float))),
    }

# Store aggregated train statistics for one numeric column.
def _update_numeric_stats(stats: dict[str, dict[str, float]], column: str, values: pd.Series) -> None:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty: return
    target = stats.setdefault(column, {"count": 0.0, "sum": 0.0, "sum_squared": 0.0})
    target["count"] += float(clean.shape[0])
    target["sum"] += float(clean.sum())
    target["sum_squared"] += float((clean * clean).sum())

# Convert aggregated numeric statistics into JSON safe integer counts.
def _finalize_numeric_stats(stats: dict[str, dict[str, float]]) -> dict[str, dict[str, Union[float, int]]]:
    finalized: dict[str, dict[str, Union[float, int]]] = {}
    for column, values in sorted(stats.items()):
        finalized[column] = {
            "count": int(values["count"]),
            "sum": float(values["sum"]),
            "sum_squared": float(values["sum_squared"]),
        }
    return finalized

# Build one dataset context from local parquets without storing row examples.
def summarize_dataset_context(dataset_id: str, input_root: Path, split_prefix: str,
    heterogeneities: tuple[str, ...], splits: tuple[str, ...], include_activity_counts: bool,
    include_missingness: bool, include_numeric_train_stats: bool) -> dict[str, Any]:

    paths = find_split_files(input_root, split_prefix, heterogeneities, splits)
    dtypes: dict[str, set[str]] = defaultdict(set)
    missing: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    activity_counts: Counter[str] = Counter()
    lifecycle_values: set[str] = set()
    outcome_counts: Counter[str] = Counter()
    trace_lengths: list[int] = []
    numeric_stats: dict[str, dict[str, float]] = {}

    for path in paths:
        frame = pd.read_parquet(path)

        # Record source columns and dtypes without preserving source rows.
        for column, dtype in frame.dtypes.items():
            dtypes[str(column)].add(str(dtype))
            totals[str(column)] += int(frame.shape[0])
            if include_missingness: missing[str(column)] += int(frame[column].isna().sum())

        # Record raw activity labels because the contract needs a canonical activity universe.
        if "concept:name" in frame.columns:
            activity_counts.update(frame["concept:name"].dropna().astype(str).tolist())
        if "lifecycle:transition" in frame.columns:
            lifecycle_values.update(frame["lifecycle:transition"].dropna().astype(str).unique().tolist())
        if "outcome" in frame.columns:
            outcome_counts.update(frame["outcome"].dropna().astype(int).astype(str).tolist())

        # Count events per trace and discard the case identifiers after aggregation.
        if "case:concept:name" in frame.columns:
            trace_lengths.extend(frame.groupby("case:concept:name", sort=False).size().astype(int).tolist())

        # Use train splits only for scaler compatible local statistics.
        if include_numeric_train_stats and path.name.endswith("_train.parquet"):
            for column in frame.select_dtypes(include=[np.number]).columns:
                if str(column) != "outcome": _update_numeric_stats(numeric_stats, str(column), frame[column])

    columns = sorted(dtypes)
    missingness = {
        column: {
            "missing_count": int(missing[column]),
            "total_count": int(totals[column]),
            "missing_rate": float(missing[column] / totals[column]) if totals[column] else None,
        }
        for column in columns
    }

    summary: dict[str, Any] = {
        "dataset_id": dataset_id,
        "generated_by": "04_0_extract_contract_context.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "privacy_scope": {
            "exports_event_rows": False,
            "exports_case_ids": False,
            "exports_timestamps": False,
            "exports_customer_values": False,
            "exports_resource_values": False,
        },
        "source": {
            "input_root": str(input_root),
            "split_prefix": split_prefix,
            "heterogeneities_scanned": list(heterogeneities),
            "splits_scanned": list(splits),
            "files_scanned": [str(path.relative_to(input_root)) for path in paths],
        },
        "columns": {
            "names": columns,
            "dtypes": {column: sorted(values) for column, values in sorted(dtypes.items())},
        },
        "activity_labels": sorted(activity_counts),
        "lifecycle_values": sorted(lifecycle_values),
        "outcome_counts": {label: int(count) for label, count in sorted(outcome_counts.items())},
        "trace_length_scope": "processed split files after preprocessing trace cut",
        "trace_length_summary": _trace_length_summary(trace_lengths),
        "preprocessing_metadata": _load_selected_metadata(input_root, split_prefix, heterogeneities),
        "numeric_train_statistics": _finalize_numeric_stats(numeric_stats),
    }

    # Add missingness only when the disclosure policy allows aggregate rates.
    if include_missingness:
        summary["missingness"] = missingness
    if include_activity_counts:
        summary["activity_counts"] = {label: int(count) for label, count in sorted(activity_counts.items())}
    return summary

# ----------------------------------------------------------------------------------------------------------------------
# 4. FEDERATION SUMMARY

# Combine dataset summaries into one overview for the contract engineer.
def build_federation_context(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    datasets = sorted(str(summary["dataset_id"]) for summary in summaries)
    column_sets = {
        str(summary["dataset_id"]): set(summary.get("columns", {}).get("names", []))
        for summary in summaries
    }
    shared_columns = sorted(set.intersection(*column_sets.values())) if column_sets else []
    dataset_specific = {
        dataset_id: sorted(columns - set(shared_columns))
        for dataset_id, columns in sorted(column_sets.items())
    }
    dataset_overview = {
        str(summary["dataset_id"]): {
            "column_count": len(summary.get("columns", {}).get("names", [])),
            "activity_label_count": len(summary.get("activity_labels", [])),
            "case_count": summary.get("trace_length_summary", {}).get("case_count"),
            "event_count": summary.get("trace_length_summary", {}).get("event_count"),
        }
        for summary in summaries
    }
    return {
        "generated_by": "04_0_extract_contract_context.py",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "datasets": datasets,
        "privacy_scope": {
            "exports_event_rows": False,
            "exports_case_ids": False,
            "exports_timestamps": False,
            "exports_customer_values": False,
            "exports_resource_values": False,
        },
        "dataset_overview": dataset_overview,
        "column_overlap": {
            "shared_columns": shared_columns,
            "dataset_specific_columns": dataset_specific,
        },
        "deployment_note": (
            "Use this file to design the central federation contract. "
            "Canonical schemas and dataset mappings can be created inside the data owner boundary."
        ),
    }

# Write one file per dataset plus one combined federation context file.
def write_contract_context_files(output_root: Path, summaries: list[dict[str, Any]]) -> list[Path]:
    written: list[Path] = []
    for summary in summaries:
        path = output_root / f"04_00_{summary['dataset_id']}_contract_context.json"
        _write_json(path, summary)
        written.append(path)

    federation_path = output_root / "04_00_federation_contract_context.json"
    _write_json(federation_path, build_federation_context(summaries))
    written.append(federation_path)
    return written

# ----------------------------------------------------------------------------------------------------------------------
# 5. SCRIPT ENTRY POINT

# Extract all configured dataset contexts and write generated JSON summaries.
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper()), format="%(asctime)s %(levelname)s %(message)s")
    summaries: list[dict[str, Any]] = []
    for dataset_id in _parse_csv(args.datasets):
        if dataset_id not in DATASET_INPUTS: raise ValueError(f"unknown dataset for contract context: {dataset_id}")
        dataset_input = DATASET_INPUTS[dataset_id]
        LOGGER.info("Summarizing contract context for %s", dataset_id)
        summaries.append(
            summarize_dataset_context(
                dataset_id=dataset_id,
                input_root=Path(dataset_input["input_root"]),
                split_prefix=str(dataset_input["split_prefix"]),
                heterogeneities=_parse_csv(args.heterogeneities),
                splits=_parse_csv(args.splits),
                include_activity_counts=bool(args.include_activity_counts),
                include_missingness=not bool(args.no_missingness),
                include_numeric_train_stats=not bool(args.no_numeric_train_stats),
            )
        )
    paths = write_contract_context_files(Path(args.output_root), summaries)
    for path in paths: LOGGER.info("Wrote contract context to %s", path)

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────