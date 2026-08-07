"""
Step 4.4: Run the approved E_04 metadata matrix.

Run this script after `04_2_create_canonical_schema.py` and `04_3_create_dataset_mapping.py`. It reads the approved
canonical schema profiles and the approved dataset mapping, then creates compact encoding metadata for the selected
profile. With `SCHEMA_PROFILE = "all"`, it runs the full BPIC 2017 and BPIC 2012 matrix from one reviewed mapping file.
The metadata is the "recipe" later training needs to rebuild on-the-fly tensors from the original parquets.

The runner writes no prefix tensors.
Later training reloads the processed parquets and these metadata files, then PrefixDataset creates tensors on demand.
Due to very slow on demand creation, later processes cache the tensors for faster access in training.

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved canonical schema
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved dataset mapping
    E_main_BPIC_2017/data/processed/*/*.parquet: BPIC 2017 split parquets
    E_ablation_BPIC_2012/data/processed/*/*.parquet: BPIC 2012 split parquets

CREATED FILES:
    E_prefix_encoding/encoded_metadata/*/*/*_encoding_spec.json: run config, counts, prefix cap and checks
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: categorical token indices only for the train split
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: numeric means and standard deviations for the train split
    E_prefix_encoding/encoded_metadata/*/*/*_mapping_report.json: mapping hash, fallback count and unresolved labels
"""

# IMPORTS
from __future__ import annotations
import argparse
import importlib
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Allow direct script execution from the repository root.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the reusable E_04 encoder library.
encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent                            # Folder that contains this script
MAPPING_ROOT: Path = SCRIPT_DIR / "mappings"                                  # Folder for approved schema and mapping
SCHEMA_PROFILE: str = "all"                                                   # bpic2017 | bpic2012 | joint | all
ARTIFACT_ROOT: Path = SCRIPT_DIR / "encoded_metadata"                         # Root folder for metadata artifacts
OUTCOME_TARGET_MODE: str = "every_prefix"                                     # Broadcast the outcome to every prefix
REMAINING_TIME_TRANSFORM: str = "raw"                                         # RT target transform: raw | log
REMAINING_TIME_SCALING: str = "zscore"                                        # RT target scaling: raw | median | zscore
RANDOM_SEED: int = 42                                                         # Seed stored in run metadata
CANONICAL_SCHEMA_PATH: Path = MAPPING_ROOT / "MANUAL_canonical_schemas.json"  # Approved schema input
DATASET_MAPPING_PATH: Path = MAPPING_ROOT / "MANUAL_dataset_mapping.json"     # Approved dataset mapping input

# Configure the script logger.
log = logging.getLogger("E_04_prefix_encoding")
def _configure_logging() -> None: logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# Define run matrices with a fix prefix length cap.
RUN_MATRICES: dict[str, list[tuple[str, str, int, int]]] = {
    "bpic2017": [
        ("bpic2017", "iid", 3, 83),
        ("bpic2017", "weak", 3, 83),
        ("bpic2017", "medium", 3, 83),
        ("bpic2017", "strong", 3, 83),
        ("bpic2017", "medium", 5, 83),
        ("bpic2017", "strong", 5, 83),
    ],
    "bpic2012": [
        ("bpic2012", "iid", 3, 42),
        ("bpic2012", "weak", 3, 42),
        ("bpic2012", "medium", 3, 42),
    ],
    "joint": [
        ("joint", "iid", 6, 83),
        ("joint", "weak", 6, 83),
        ("joint", "medium", 6, 83),
        ("joint", "medium", 8, 83),
    ],
}

# "all" is the normal freeze run across the separate BPIC 2017, BPIC 2012 and joint matrices.
RUN_MATRICES["all"] = RUN_MATRICES["bpic2017"] + RUN_MATRICES["bpic2012"] + RUN_MATRICES["joint"]

# Store one concrete metadata run configuration.
@dataclass(frozen=True)
class RunConfig:
    dataset: str
    heterogeneity: str
    n_clients: int
    max_prefix_length_for_encoding: int
    outcome_target_mode: str
    remaining_time_transform: str
    remaining_time_scaling: str
    schema_profile: str
    artifact_root: Path
    random_seed: int

# ----------------------------------------------------------------------------------------------------------------------
# 1. CLI OVERRIDES

# Parse optional automation arguments while keeping script defaults for WORKFLOW_run_encoding.sh.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E_04 encoding metadata matrix.")
    parser.add_argument("--schema-profile", default=SCHEMA_PROFILE)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--canonical-schema-path", type=Path, default=CANONICAL_SCHEMA_PATH)
    parser.add_argument("--dataset-mapping-path", type=Path, default=DATASET_MAPPING_PATH)
    parser.add_argument("--remaining-time-transform", choices=list(encoding.REMAINING_TIME_TRANSFORMS),
                        default=REMAINING_TIME_TRANSFORM)
    parser.add_argument("--remaining-time-scaling", choices=list(encoding.REMAINING_TIME_SCALINGS),
                        default=REMAINING_TIME_SCALING)
    return parser.parse_args(argv)

# ----------------------------------------------------------------------------------------------------------------------
# 2. MATRIX CONFIGURATION

# Return the approved E_04 run matrix for one schema profile.
def matrix_for_profile(schema_profile: str) -> list[tuple[str, str, int, int]]:
    # Refuse undeclared matrices so stale or ad hoc configurations do not run silently.
    if schema_profile in RUN_MATRICES: return list(RUN_MATRICES[schema_profile])
    raise ValueError(f"unknown schema profile: {schema_profile}")

# Build one concrete run configuration.
def build_config(schema_profile: str, dataset: str, heterogeneity: str, n_clients: int, cap: int,
    artifact_root: Path, remaining_time_transform: str = REMAINING_TIME_TRANSFORM,
    remaining_time_scaling: str = REMAINING_TIME_SCALING) -> Any:

    # Store the dataset, split setup, prefix cap and output path for one runner matrix entry.
    return RunConfig(
        dataset=dataset, schema_profile=schema_profile, heterogeneity=heterogeneity, n_clients=n_clients,
        max_prefix_length_for_encoding=cap, outcome_target_mode=OUTCOME_TARGET_MODE,
        remaining_time_transform=remaining_time_transform, remaining_time_scaling=remaining_time_scaling,
        artifact_root=artifact_root, random_seed=RANDOM_SEED,
    )

# ----------------------------------------------------------------------------------------------------------------------
# 3. RUN ENCODING

# Run one selected matrix entry and write compact artifacts.
def run_one(config: Any, schema_profile_payload: dict[str, Any], mapping_payload: dict[str, Any],
    dataset_mapping_path: Path) -> Any:
    log.info("Running %s %s_%sbanks", config.dataset, config.heterogeneity, config.n_clients)

    # Load original processed parquets and adapt them into canonical events.
    events = (
        encoding.load_joint_run_events(config, mapping_payload, schema_profile_payload)
        if config.dataset == "joint"
        else encoding.load_run_events(config, mapping_payload, schema_profile_payload)
    )

    # Apply the approved raw label mapping before fitting vocabularies.
    mapped, mapping_report = encoding.apply_activity_mapping(events, dataset_mapping_path)
    encoding.validate_canonical_events(mapped, mapped=True)

    # Fit vocabularies and scalers only on train events to avoid split leakage.
    train = mapped.loc[mapped[encoding.SPLIT] == "train"].copy()
    categorical_columns = list(schema_profile_payload["sequence_categorical_columns"])
    numerical_columns = list(schema_profile_payload["sequence_numerical_columns"])
    offer_columns = list(schema_profile_payload["offer_numerical_columns"])
    vocabularies = encoding.build_all_vocabularies(train, categorical_columns=categorical_columns)
    scalers = encoding.fit_all_scalers(train, numerical_columns=numerical_columns, offer_numerical_columns=offer_columns)

    # Build compact prefix references and fold all run checks into the spec.
    prefix_index, observed_max = encoding.build_prefix_index(mapped, config.max_prefix_length_for_encoding)
    remaining_time_target_repr = encoding.fit_remaining_time_target_from_prefixes(
        mapped, prefix_index, config.remaining_time_transform, config.remaining_time_scaling,
    )
    spec = encoding.build_encoding_spec(
        config, mapped, prefix_index, observed_max, vocabularies, scalers, mapping_report, remaining_time_target_repr,
    )

    # Save compact JSON artifacts (not full prefix tensors).
    paths = encoding.build_artifact_paths(
        config.artifact_root, config.dataset, config.heterogeneity, config.n_clients, joint=config.dataset == "joint",
    )
    encoding.save_encoding_artifacts(paths, spec, vocabularies, scalers, mapping_report)
    log.info("Wrote compact E_04 artifacts to %s", paths.root)
    return paths

# Run the full metadata matrix for one approved schema profile.
def run_full_matrix(schema_profile: str, artifact_root: Path, canonical_schema_path: Path, dataset_mapping_path: Path,
    remaining_time_transform: str = REMAINING_TIME_TRANSFORM,
    remaining_time_scaling: str = REMAINING_TIME_SCALING) -> list[Any]:

    # Load only approved workflow inputs before the metadata matrix starts.
    schema_payload = encoding.load_approved_json(canonical_schema_path, "canonical schema")
    mapping_payload = encoding.load_dataset_mapping(dataset_mapping_path, require_approved=True)

    # The dataset mapping must belong to the selected schema profile or the full review of all profiles.
    mapping_profile = mapping_payload.get("schema_profile")
    if mapping_profile not in {schema_profile, "all"}:
        raise ValueError("MANUAL_dataset_mapping.json schema_profile does not match SCHEMA_PROFILE")

    # The mapping hash must match the approved canonical schema on disk.
    schema_hash = encoding.json_sha256(canonical_schema_path)
    if mapping_payload.get("schema_sha256") != schema_hash:
        raise ValueError("MANUAL_dataset_mapping.json schema hash does not match the canonical schema")

    # Check that the approved schema contains every profile needed by this run.
    profiles = schema_payload.get("schema_profiles", {})
    selected_profiles = [schema_profile] if schema_profile in {"bpic2017", "bpic2012", "joint"} else ["bpic2017", "bpic2012", "joint"]
    missing_profiles = [profile for profile in selected_profiles if profile not in profiles]
    if missing_profiles:
        raise ValueError(f"canonical schema is missing schema profiles: {missing_profiles}")
    if schema_profile != "all" and schema_profile not in profiles:
        raise ValueError(f"canonical schema does not contain schema profile: {schema_profile}")

    # Check that every dataset named by the selected schema has a reviewed mapping block.
    missing_datasets = [
        dataset_id
        for profile_name in selected_profiles
        for dataset_id in profiles[profile_name].get("datasets", [])
        if dataset_id not in mapping_payload.get("datasets", {})
    ]
    if missing_datasets: raise ValueError(f"MANUAL_dataset_mapping.json is missing datasets: {missing_datasets}")

    paths: list[Any] = []
    for dataset, heterogeneity, n_clients, cap in matrix_for_profile(schema_profile):
        # all dispatches each dataset to its own profile, explicit profiles use their selected profile.
        run_profile = dataset if schema_profile == "all" else schema_profile
        schema_profile_payload = profiles[run_profile]

        # Build one run configuration and write its four JSON artifacts.
        config = build_config(
            run_profile, dataset, heterogeneity, n_clients, cap, artifact_root, remaining_time_transform,
            remaining_time_scaling,
        )
        paths.append(run_one(config, schema_profile_payload, mapping_payload, dataset_mapping_path))
    return paths

# ----------------------------------------------------------------------------------------------------------------------
# 4. MAIN

# Run the configured E_04 metadata matrix.
def main(argv: Optional[list[str]] = None) -> None:
    _configure_logging()
    args = parse_args(argv)
    run_full_matrix(
        args.schema_profile, args.artifact_root, args.canonical_schema_path, args.dataset_mapping_path,
        args.remaining_time_transform, args.remaining_time_scaling,
    )

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────