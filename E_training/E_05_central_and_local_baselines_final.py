"""
Step 5: Train multitask LSTM centralized baseline per federation and local baseline per bank using E_04 metadata.

Pipeline:
- Loads the approved E_04 encoding spec, vocabularies, scalers, schema profile and dataset mapping.
- Loads or builds the np memory-mapped prefix tensor cache for the train, validation and test splits.
- Trains the multitask LSTM with class-weighted outcome CE, masked next activity CE and masked Huber remaining time.
- Applies per head AdamW groups, cosine schedule over LR_SCHEDULER_T_MAX, early stopping on val_loss_total.
- Restores the best checkpoint, evaluates validation and test once and exports predictions including summaries.
- For centralized runs, evaluates the global model on every banks test split for the client fairness comparison.

Checkpoint selection: Restoration of the best checkpoint uses the lowest `val_loss_total` epoch only.
    -> The best epoch per head records are diagnostic. They are not used for checkpoint selection.

Run: WORKFLOW_run_baseline_final.sh for the full matrix, or call this script directly for one run.
The CONFIGURATION block below lists every environment override.
The shared model, loss, metric, cache and device logic are imported from E_training.training_core_final.

REQUIRED FILES:
    E_prefix_encoding/encoded_metadata/*/*/*_encoding_spec.json: frozen E_04 run config and prefix cap
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: train only categorical token indices
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: train only numeric means and standard deviations
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved schema profiles
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved dataset mapping
    E_main_BPIC_2017/data/processed/**/*.parquet: BPIC 2017 split parquets
    E_ablation_BPIC_2012/data/processed/**/*.parquet: BPIC 2012 split parquets

CREATED FILES (compact profile, per run directory <OUTPUT_ROOT>/baselines/<dataset>_<run>/<run_kind>_seed_<seed>_lr_<lr>/):
    E_05_run_report.json: Consolidated configuration, metrics, diagnostics, timing, curves and artifact manifest
    E_05_model_best.pt: Best checkpoint with model state, config payload and diagnostics
    E_05_loss_curves.png: Total train and validation loss curves
    E_05_task_loss_curves.png: Validation loss curves per task head
    E_05_outcome_macro_f1_curve.png: Validation outcome macro-F1 curve
    E_05_next_activity_accuracy_curve.png: Validation next activity top-1 accuracy curve
    E_05_remaining_time_mae_curve.png: Validation RT MAE curve
    predictions/E_05_train_log.csv: Losses per epoch, metrics and learning rates
    predictions/E_05_predictions_{val,test}.parquet: Predictions per prefix with case provenance

    Centralized runs additionally create:
    E_05_centralized_summary_test.{csv,json}: Flat global and centralized test summary per bank
    predictions/E_05_predictions_test_bank_<X>.parquet: Test predictions per bank

    Local runs additionally create, once all bank runs exist:
    <OUTPUT_ROOT>/baselines/<dataset>_<run>/E_05_local_summary_seed_<seed>_lr_<lr>.{csv,json}
"""

# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import argparse
import copy
from contextlib import contextmanager
import hashlib
import importlib
import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Optional

# Make direct script execution imports work.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Make Matplotlib use a safe writable cache/config directory.
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# Import the training core with helpers for warm caches.
from E_training import training_core_final as core
from E_training import training_reporting

# Render plots directly to PNG files without opening a display window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load the E_04 runtime without editing prefix-encoding files.
encoding: Any = importlib.import_module("E_prefix_encoding.04_5_encoding")

# HYPERPARAMETERS AND CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------

# Graph style and color palette.
PRIMARY_BLUE = "#0065BF"
REFERENCE_RED = "#C8102E"
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#666666"
LEGEND_EDGE = "#bfbfbf"
TEXT_DARK = "#222222"
NEUTRAL_GREY = "#cccccc"
TRAINING_CURVE_COLORS = [PRIMARY_BLUE, "#003E7A", "#5A9DDC", NEUTRAL_GREY]
plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Restrict baseline training to the two supported execution regimes.
Regime = Literal["centralized", "local"]

# WORKFLOW_run_baseline_final.sh overrides these environment variables for matrix runs.
OUTCOME_CLASSES: tuple[int, int, int] = (0, 1, 2)                                   # 0:canceled | 1:denied | 2:accepted
BANK_NAMES: tuple[str, ...] = ("A", "B", "C", "D", "E")                             # simulated bank labels
SCRIPT_ID: str = "E_05"                                                             # artifact filename prefix
PREDICTIONS_DIR_NAME: str = "predictions"                                           # prediction and log subfolder
PARQUET_FINGERPRINT_CHUNK_BYTES: int = 1 << 20                                      # read size per hash update
_PARQUET_FINGERPRINTS: dict[str, str] = {}                                          # parquet fingerprint per process

# File paths from the script directory.
SCRIPT_DIR: Path = Path(__file__).resolve().parent                                  # E_training folder
REPO_ROOT: Path = SCRIPT_DIR.parent                                                 # repository root
ARTIFACT_ROOT: Path = Path(os.environ.get("ARTIFACT_ROOT", REPO_ROOT / "E_prefix_encoding" / "encoded_metadata"))
MAPPING_ROOT: Path = REPO_ROOT / "E_prefix_encoding" / "mappings"                   # approved E_04 mappings
CANONICAL_SCHEMA_PATH: Path = MAPPING_ROOT / "MANUAL_canonical_schemas.json"        # approved schema input
DATASET_MAPPING_PATH: Path = MAPPING_ROOT / "MANUAL_dataset_mapping.json"           # approved mapping input
OUTPUT_ROOT: Path = SCRIPT_DIR / "training_outputs"                        # output root folder
CACHE_ROOT: Path = SCRIPT_DIR / "prefix_tensor_cache"                               # persistent tensor cache root

# Select the dataset, split and training regime from the environment (second value is default).
DATASET: str = os.environ.get("DATASET", "bpic2017")                                # bpic2017 | bpic2012
HETEROGENEITY: str = os.environ.get("HETEROGENEITY", "medium")                      # iid | weak | medium | strong
N_CLIENTS: int = int(os.environ.get("N_CLIENTS", "3"))                              # 3 | 5
BANK: Optional[str] = os.environ.get("BANK") or None                                # A/B/C/D/E for local mode
SEED: int = int(os.environ.get("SEED", "42"))                                       # fixed random seed

# Environment variables are plain strings, so the later config validation enforces the literal values.
REGIME: Regime = os.environ.get("REGIME", "centralized")                            # type: ignore[assignment]

# Training protocol.
MAX_EPOCHS: int = int(os.environ.get("MAX_EPOCHS", "40"))                           # Epoch budget, early stop ends runs
EARLY_STOPPING_PATIENCE: int = int(os.environ.get("EARLY_STOPPING_PATIENCE", "7"))  # Uniform patience across the matrix
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "512"))                          # Tested in Sweep, can be scaled
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "0"))                          # 0 is most stable on macOS

# Optimizer and learning rate schedule.
LEARNING_RATE: float = float(os.environ.get("LEARNING_RATE", "2.5e-4"))             # base AdamW LR (trial and error)
WEIGHT_DECAY: float = float(os.environ.get("WEIGHT_DECAY", "1e-4"))                 # AdamW weight decay
GRADIENT_CLIP_NORM: float = float(os.environ.get("GRADIENT_CLIP_NORM", "1.0"))      # clip gradients above this norm
LR_SCHEDULER: str = os.environ.get("LR_SCHEDULER", "cosine").lower()                # "cosine" or "none"
LR_SCHEDULER_MIN_LR: float = float(os.environ.get("LR_SCHEDULER_MIN_LR", "1e-6"))   # scheduler LR minimum
LR_SCHEDULER_T_MAX: int = int(os.environ.get("LR_SCHEDULER_T_MAX", "15"))           # Reach min LR before early stopping
DEVICE: str = os.environ.get("DEVICE", "auto")                                      # Device: auto | mps | cuda | cpu
PROGRESS_BARS: bool = os.environ.get("PROGRESS_BARS", "true").lower() not in {"0", "false", "no"}
REPORTING_PROFILE: str = training_reporting.normalize_reporting_profile(os.environ.get("REPORTING_PROFILE", "compact"))

# Model architecture.
# Model size locked at 128 hidden units and 2 LSTM layers after the larger variant did not improve test metrics.
HIDDEN_SIZE: int = int(os.environ.get("HIDDEN_SIZE", "128"))                        # LSTM hidden state size
NUM_LAYERS: int = int(os.environ.get("NUM_LAYERS", "2"))                            # number of stacked LSTM layers
DROPOUT: float = float(os.environ.get("DROPOUT", "0.30"))                           # dropout trunk model layers
HEAD_HIDDEN_SIZE: int = int(os.environ.get("HEAD_HIDDEN_SIZE", "64"))               # hidden size of task heads

# Label smoothing reduces confident wrong predictions on minority classes (production reg_medium value).
OUTCOME_LABEL_SMOOTHING: float = float(os.environ.get("OUTCOME_LABEL_SMOOTHING", "0.10"))

# Square-root inverse frequency weighting balances minority classes (default 0.5 | none 0.0 | full 1.0).
OUTCOME_CLASS_WEIGHT_POWER: float = float(os.environ.get("OUTCOME_CLASS_WEIGHT_POWER", "0.5"))

# AdamW LR multipliers per head on top of the overall LR -> slow overfitting in the outcome head.
OUTCOME_LR_SCALE: float = float(os.environ.get("OUTCOME_LR_SCALE", "0.3"))                      # OUTCOME is 0.3
NEXT_ACTIVITY_LR_SCALE: float = float(os.environ.get("NEXT_ACTIVITY_LR_SCALE", "1.0"))          # NEXT_ACTIVITY is 1.0
REMAINING_TIME_LR_SCALE: float = float(os.environ.get("REMAINING_TIME_LR_SCALE", "1.0"))        # RT is 1.0

# Multitask loss weights set each task's contribution. Outcome 0.7 is available if it dominates early stopping.
OUTCOME_LOSS_WEIGHT: float = float(os.environ.get("OUTCOME_LOSS_WEIGHT", "1.0"))                # OUTCOME weight
NEXT_ACTIVITY_LOSS_WEIGHT: float = float(os.environ.get("NEXT_ACTIVITY_LOSS_WEIGHT", "0.5"))    # NEXT_ACTIVITY weight
REMAINING_TIME_LOSS_WEIGHT: float = float(os.environ.get("REMAINING_TIME_LOSS_WEIGHT", "0.5"))  # RT weight

# RT scaling and transformation are selected in E_04 and validated here.
REMAINING_TIME_TRANSFORM: str = os.environ.get("REMAINING_TIME_TRANSFORM", "raw").lower()       # Transform: raw | log
REMAINING_TIME_SCALING: str = os.environ.get("REMAINING_TIME_SCALING", "zscore").lower()        # raw | median | zscore

# A smaller beta makes RT loss closer to MAE. A larger beta expands the quadratic error application (Default used 0.1).
REMAINING_TIME_HUBER_BETA: float = float(os.environ.get("REMAINING_TIME_HUBER_BETA", "0.1"))    # 0 behaves like MAE

# Outcome-head dropout overrides the global DROPOUT at the reg_medium value. An explicit empty value disables it.
_OUTCOME_HEAD_DROPOUT_ENV: str = os.environ.get("OUTCOME_HEAD_DROPOUT", "0.45")
OUTCOME_HEAD_DROPOUT: Optional[float] = float(_OUTCOME_HEAD_DROPOUT_ENV) if _OUTCOME_HEAD_DROPOUT_ENV else None

# Freeze the resolved baseline configuration so every artifact and checkpoint records the same run state.
@dataclass(frozen=True)
class BaselineRunConfig:
    dataset: str = DATASET
    heterogeneity: str = HETEROGENEITY
    n_clients: int = N_CLIENTS
    regime: Regime = REGIME
    bank: Optional[str] = BANK
    seed: int = SEED
    max_epochs: int = MAX_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    lr_scheduler: str = LR_SCHEDULER
    lr_scheduler_min_lr: float = LR_SCHEDULER_MIN_LR
    lr_scheduler_t_max: int = LR_SCHEDULER_T_MAX
    device: str = DEVICE
    progress_bars: bool = PROGRESS_BARS
    reporting_profile: str = REPORTING_PROFILE
    hidden_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS
    dropout: float = DROPOUT
    head_hidden_size: int = HEAD_HIDDEN_SIZE
    outcome_label_smoothing: float = OUTCOME_LABEL_SMOOTHING
    outcome_class_weight_power: float = OUTCOME_CLASS_WEIGHT_POWER
    outcome_lr_scale: float = OUTCOME_LR_SCALE
    next_activity_lr_scale: float = NEXT_ACTIVITY_LR_SCALE
    remaining_time_lr_scale: float = REMAINING_TIME_LR_SCALE
    outcome_loss_weight: float = OUTCOME_LOSS_WEIGHT
    next_activity_loss_weight: float = NEXT_ACTIVITY_LOSS_WEIGHT
    remaining_time_loss_weight: float = REMAINING_TIME_LOSS_WEIGHT
    remaining_time_transform: str = REMAINING_TIME_TRANSFORM
    remaining_time_scaling: str = REMAINING_TIME_SCALING
    remaining_time_huber_beta: float = REMAINING_TIME_HUBER_BETA
    outcome_head_dropout: Optional[float] = OUTCOME_HEAD_DROPOUT
    output_root: Path = OUTPUT_ROOT
    cache_root: Path = CACHE_ROOT
    output_dir_override: Optional[Path] = None
    artifact_root: Path = ARTIFACT_ROOT
    canonical_schema_path: Path = CANONICAL_SCHEMA_PATH
    dataset_mapping_path: Path = DATASET_MAPPING_PATH

    @property
    def run_name(self) -> str: return f"{self.heterogeneity}_{self.n_clients}banks"

# Configure the logger.
log = logging.getLogger("E_05")
def _configure_logging() -> None: logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# HELPER: Wall clock timing per pipeline phase, written to E_05_timing.json (analysis).
@contextmanager
def _timed(timings: dict[str, float], name: str) -> Any:
    start = time.perf_counter()
    try: yield
    finally: timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start)

# Compute inverse frequency CE weights from training outcome labels.
def compute_outcome_class_weights(labels: pd.Series, power: float = 1.0) -> dict[int, float]:

    # Count labels over prefixes because the training loss is averaged over prefix samples.
    counts = labels.astype(int).value_counts().reindex(OUTCOME_CLASSES, fill_value=0)

    # Stop when a class is absent because the weighted CE loss would be not well-defined.
    if (counts == 0).any():
        missing = counts.loc[counts == 0].index.tolist()
        raise ValueError(f"cannot compute class weights because classes are missing: {missing}")

    # Compute balanced CE weights: count[class] * weight[class] is equalized among outcome classes.
    total = float(counts.sum())
    n_classes = float(len(OUTCOME_CLASSES))
    weights = {label: float(total / (n_classes * float(counts[label]))) for label in OUTCOME_CLASSES}

    # Temper the balancing strength: 1.0 keeps full inverse frequency weighting, 0.0 returns all ones.
    return {label: float(weight**power) for label, weight in weights.items()}

# Encode a float for folder names without dots or minus signs.
def format_float_for_path(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")

# Convert dataset-qualified joint bank ids into file system safe tokens.
def _safe_bank(bank: str) -> str: return encoding.safe_client_id(str(bank))

# The prefix artifact filename with the script identifier. Keep prefixed names stable when helpers are nested.
def script_artifact_name(filename: str) -> str:
    if filename.startswith(f"{SCRIPT_ID}_"): return filename
    return f"{SCRIPT_ID}_{filename}"

# Resolve a prefixed artifact path in the run output directory.
def run_artifact_path(output_dir: Path, filename: str) -> Path: return output_dir / script_artifact_name(filename)

# Resolve a prefixed prediction artifact path in the prediction subdirectory.
def prediction_artifact_path(output_dir: Path, filename: str) -> Path:
    return output_dir / PREDICTIONS_DIR_NAME / script_artifact_name(filename)

# Resolve the CSV and JSON summary shared by all local bank runs with the same dataset, split config, seed and LR.
def local_summary_paths(config: BaselineRunConfig) -> tuple[Path, Path]:
    summary_root = config.output_root / "baselines" / f"{config.dataset}_{config.run_name}"
    filename_stem = f"local_summary_seed_{config.seed}_lr_{format_float_for_path(config.learning_rate)}"
    return (
        summary_root / script_artifact_name(f"{filename_stem}.csv"),
        summary_root / script_artifact_name(f"{filename_stem}.json"),
    )

# Disable progress bars when explicitly requested or when no TTY is attached (TTY = teletypewriter = Terminal).
def should_disable_progress(config: BaselineRunConfig, is_tty: Optional[bool] = None) -> bool:
    if not config.progress_bars: return True
    if is_tty is None: is_tty = sys.stderr.isatty()
    return not is_tty

# Wrap iterables in the configured progress bar behavior for interactive and batch runs (centralized tqdm settings).
def progress_iter(iterable: Any, config: BaselineRunConfig, description: str, total: Optional[int] = None,
                  leave: bool = False, unit: str = "it") -> Any:
    return tqdm(iterable, desc=description, total=total, leave=leave, unit=unit, dynamic_ncols=True,
                disable=should_disable_progress(config))

# Separate centralized and per bank cache trees because their prefix populations differ.
def cache_run_name(config: BaselineRunConfig) -> str:
    rt_suffix = f"rt_{config.remaining_time_transform}_{config.remaining_time_scaling}"

    # Centralized caches contain the pooled prefix population for one split config.
    if config.regime == "centralized": return f"centralized_{rt_suffix}"

    # Local caches must name the bank because each bank has a different prefix set.
    if not config.bank: raise ValueError("local cache requires BANK")
    return f"local_bank_{_safe_bank(config.bank)}_{rt_suffix}"

# Hash deterministic JSON serialization so equal cache payloads always produce equal hashes.
def stable_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=_json_default, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

# Fingerprint one processed parquet by its own bytes, so a cache can be traced back to the data that produced it.
# The hash covers file content only, never a path or a modification time, because both differ on a fresh clone.

# The parquet writer is deterministic, so regenerating a split reproduces the same fingerprint.
# One streamed read costs about 15 milliseconds for the 24 MB BPIC 2017 train split and is memoized per process.
def parquet_content_fingerprint(path: Path) -> str:
    resolved = str(Path(path).resolve())
    cached = _PARQUET_FINGERPRINTS.get(resolved)
    if cached is not None: return cached
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(PARQUET_FINGERPRINT_CHUNK_BYTES), b""): digest.update(chunk)
    _PARQUET_FINGERPRINTS[resolved] = digest.hexdigest()
    return _PARQUET_FINGERPRINTS[resolved]

# Resolve the processed parquets one split of one run reads, so the cache identity can cover their content.
def split_input_parquet_paths(config: BaselineRunConfig, mapping: dict[str, Any], split_name: str) -> list[Path]:

    # A joint run has no pooled parquet, so a centralized joint split reads every source bank split.
    if config.dataset == "joint":
        banks = bank_names_for_config(config) if config.regime == "centralized" else (str(config.bank),)
        return [bank_split_parquet_path(config, mapping, bank, split_name)[1] for bank in banks]

    # A single-dataset run reads the pooled centralized split or the one-bank split it trains on.
    dataset_mapping = mapping["datasets"][config.dataset]
    input_root = Path(dataset_mapping["input_root"])
    split_prefix = str(dataset_mapping["split_prefix"])
    if config.regime == "centralized":
        return [input_root / "centralized" / config.run_name / f"{split_prefix}_{split_name}.parquet"]
    if not config.bank: raise ValueError("local cache identity requires BANK")
    return [input_root / config.run_name / f"{split_prefix}_bank_{config.bank}_{split_name}.parquet"]

# Fingerprint every processed parquet one split reads, keyed by file name so the payload stays path independent.
def split_input_fingerprints(config: BaselineRunConfig, mapping: dict[str, Any],
                             split_name: str) -> dict[str, str]:
    fingerprints: dict[str, str] = {}
    for path in split_input_parquet_paths(config, mapping, split_name):
        if not path.exists(): raise FileNotFoundError(f"missing split parquet for the cache identity: {path}")
        fingerprints[f"{path.parent.name}/{path.name}"] = parquet_content_fingerprint(path)
    return fingerprints

# Collect every data parameter that defines one cache. Any change to these must invalidate the cache.
def cache_build_payload(config: BaselineRunConfig, split_name: str, spec: dict[str, Any],
    vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any],
    mapping: dict[str, Any],
    ) -> dict[str, Any]:

    # Include schema, mapping, vocabulary and scaler inputs so any tensor-changing input invalidates the cache.
    # The tensor layout version, the E_04 spec hash and the input fingerprints make the cache provable provenance.
    return {
        "cache_schema_version": int(core.PREFIX_TENSOR_CACHE_VERSION),
        "dataset": config.dataset,
        "run_name": config.run_name,
        "cache_run_name": cache_run_name(config),
        "split_name": split_name,
        "prefix_cap": int(spec["prefix"]["cap"]),
        "static_padding_length": int(spec["prefix"]["static_padding_length"]),
        "sequence_categorical_columns": list(schema_profile["sequence_categorical_columns"]),
        "sequence_numerical_columns": list(schema_profile["sequence_numerical_columns"]),
        "offer_numerical_columns": list(schema_profile["offer_numerical_columns"]),
        "canonical_schema_sha256": encoding.json_sha256(config.canonical_schema_path),
        "dataset_mapping_sha256": encoding.json_sha256(config.dataset_mapping_path),
        "encoding_spec_sha256": encoding.json_sha256(encoding_spec_path(config)),
        "input_parquet_sha256": split_input_fingerprints(config, mapping, split_name),
        "vocabularies": vocabularies,
        "scalers": scalers,
        "target_scalers": spec.get("target_scalers", {}),
    }

# Store the hash and the full payload in the cache metadata -> stale caches are explainable.
def cache_context(config: BaselineRunConfig, split_name: str, spec: dict[str, Any],
    vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any],
    mapping: dict[str, Any],
    ) -> dict[str, Any]:

    # Build the full payload first so the short hash can be stored beside it.
    payload = cache_build_payload(config, split_name, spec, vocabularies, scalers, schema_profile, mapping)
    return {"cache_build_hash": stable_json_hash(payload), "cache_build_payload": payload}

# Build a warm cache path without requiring the source PrefixDataset.
def prefix_cache_dir(config: BaselineRunConfig, split_name: str, spec: dict[str, Any]) -> Path:
    static_length = int(spec["prefix"]["static_padding_length"])
    return config.cache_root / config.dataset / config.run_name / cache_run_name(config) / f"{split_name}_pad{static_length}"

# Return a cached PrefixDataset when all tensor arrays and prefix index rows exist.
def load_cached_prefix_dataset(config: BaselineRunConfig, split_name: str, spec: dict[str, Any],
    vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any],
    mapping: dict[str, Any],
    ) -> Optional[Any]:

    # Resolve the expected hash before asking the shared core for a cache hit.
    cache_dir = prefix_cache_dir(config, split_name, spec)
    expected = cache_context(config, split_name, spec, vocabularies, scalers, schema_profile,
                             mapping)["cache_build_hash"]
    cached = core.try_load_prefix_tensor_cache(cache_dir, expected_cache_hash=expected)

    # Treat incomplete cache directories as misses so callers rebuild them from parquets.
    if cached is None: return None
    if not getattr(cached, "prefix_index", None): return None
    return cached

# Convert one E_04 PrefixDataset split into the reusable memory-mapped tensor cache with its cache identity.
def cache_prefix_dataset(config: BaselineRunConfig, dataset: Any, split_name: str, spec: dict[str, Any],
    vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any],
    mapping: dict[str, Any],
    ) -> Any:

    # Route cache progress through the configured workflow progress bar helper (see above).
    def cache_progress(iterable: Any, total: int) -> Any:
        return progress_iter(iterable, config, f"cache {split_name}", total=total, unit="prefix")

    # Let the shared core reuse a valid cache or store this split as disk-backed tensor arrays.
    return core.load_or_build_prefix_tensor_cache(
        dataset, prefix_cache_dir(config, split_name, spec), overwrite=False, progress_iter=cache_progress,
        cache_context=cache_context(config, split_name, spec, vocabularies, scalers, schema_profile, mapping),
    )

# Encode seed and learning rate in the run-directory name, so reruns at other rates never collide.
def output_dir_for_run(root: Path, config: BaselineRunConfig) -> Path:

    # Honor explicit output directories (used by the run test workflow).
    if config.output_dir_override is not None: return config.output_dir_override

    # Build the normal matrix output location from dataset, split and training variant.
    base = root / "baselines" / f"{config.dataset}_{config.run_name}"
    variant = f"seed_{config.seed}_lr_{format_float_for_path(config.learning_rate)}"
    if config.regime == "centralized": return base / f"centralized_{variant}"

    # Local baseline output folders must name the simulated bank.
    if not config.bank: raise ValueError("local baseline requires BANK")
    return base / f"local_bank_{_safe_bank(config.bank)}_{variant}"

# ----------------------------------------------------------------------------------------------------------------------
# 1. CLI OVERRIDES

# Parse optional automation arguments while keeping script defaults for workflow runs.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E_05 centralized and local baselines.")

    # Select the dataset, partition configuration and baseline regime.
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--heterogeneity", default=HETEROGENEITY)
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS)
    parser.add_argument("--regime", choices=["centralized", "local"], default=REGIME)
    parser.add_argument("--bank", default=BANK)

    # Configure reproducibility, epoch budget and data loading.
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--max-epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--early-stopping-patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    # Configure optimizer and LR schedule.
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=GRADIENT_CLIP_NORM)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default=LR_SCHEDULER)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=LR_SCHEDULER_MIN_LR)
    parser.add_argument("--lr-scheduler-t-max", type=int, default=LR_SCHEDULER_T_MAX)

    # Select device and progress behavior for interactive and batch runs.
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-progress-bars", action="store_false", dest="progress_bars", default=PROGRESS_BARS)
    parser.add_argument("--reporting-profile", choices=["compact", "debug"], default=REPORTING_PROFILE)

    # Configure the shared multitask LSTM architecture.
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--head-hidden-size", type=int, default=HEAD_HIDDEN_SIZE)

    # Configure outcome weighting, LR per head and multitask loss weights.
    parser.add_argument("--outcome-label-smoothing", type=float, default=OUTCOME_LABEL_SMOOTHING)
    parser.add_argument("--outcome-class-weight-power", type=float, default=OUTCOME_CLASS_WEIGHT_POWER)
    parser.add_argument("--outcome-lr-scale", type=float, default=OUTCOME_LR_SCALE)
    parser.add_argument("--next-activity-lr-scale", type=float, default=NEXT_ACTIVITY_LR_SCALE)
    parser.add_argument("--remaining-time-lr-scale", type=float, default=REMAINING_TIME_LR_SCALE)
    parser.add_argument("--outcome-loss-weight", type=float, default=OUTCOME_LOSS_WEIGHT)
    parser.add_argument("--next-activity-loss-weight", type=float, default=NEXT_ACTIVITY_LOSS_WEIGHT)
    parser.add_argument("--remaining-time-loss-weight", type=float, default=REMAINING_TIME_LOSS_WEIGHT)

    # Validate the E_04 RT representation selected for this run.
    parser.add_argument("--remaining-time-transform", choices=list(core.REMAINING_TIME_TRANSFORMS),
                        default=REMAINING_TIME_TRANSFORM)
    parser.add_argument("--remaining-time-scaling", choices=list(core.REMAINING_TIME_SCALINGS),
                        default=REMAINING_TIME_SCALING)
    parser.add_argument("--remaining-time-huber-beta", type=float, default=REMAINING_TIME_HUBER_BETA)
    parser.add_argument("--outcome-head-dropout", type=float, default=OUTCOME_HEAD_DROPOUT)

    # Configure output roots used by full matrix and test workflow calls.
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)

# Convert parsed CLI values into the immutable run configuration used by the pipeline (simple assignment).
def config_from_args(args: argparse.Namespace) -> BaselineRunConfig:
    return BaselineRunConfig(
        dataset=args.dataset, heterogeneity=args.heterogeneity, n_clients=args.n_clients, regime=args.regime,
        bank=args.bank, seed=args.seed, max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
        batch_size=args.batch_size, num_workers=args.num_workers, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, gradient_clip_norm=args.gradient_clip_norm, lr_scheduler=args.lr_scheduler,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr, lr_scheduler_t_max=args.lr_scheduler_t_max, device=args.device,
        progress_bars=args.progress_bars, reporting_profile=args.reporting_profile, hidden_size=args.hidden_size,
        num_layers=args.num_layers, dropout=args.dropout, head_hidden_size=args.head_hidden_size,
        outcome_label_smoothing=args.outcome_label_smoothing,
        outcome_class_weight_power=args.outcome_class_weight_power, outcome_lr_scale=args.outcome_lr_scale,
        next_activity_lr_scale=args.next_activity_lr_scale, remaining_time_lr_scale=args.remaining_time_lr_scale,
        outcome_loss_weight=args.outcome_loss_weight, next_activity_loss_weight=args.next_activity_loss_weight,
        remaining_time_loss_weight=args.remaining_time_loss_weight,
        remaining_time_transform=args.remaining_time_transform,
        remaining_time_scaling=args.remaining_time_scaling, remaining_time_huber_beta=args.remaining_time_huber_beta,
        outcome_head_dropout=args.outcome_head_dropout, output_root=args.output_root, cache_root=args.cache_root,
        artifact_root=args.artifact_root, output_dir_override=args.output_dir,
    )

# ----------------------------------------------------------------------------------------------------------------------
# 2. JSON AND METADATA HELPERS

# HELPER: Convert values that are not JSON native and are used in artifacts into JSON safe primitives.
def _json_default(value: object) -> object:
    # Convert filesystem and numpy values into JSON-safe primitive values.
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    return value

# Create parent folders and write one stable JSON artifact with sorted keys.
def save_json(path: Path, payload: dict[str, Any]) -> None: training_reporting.save_json(path, payload)

# Write one legacy fragment only when debug reporting is active.
def save_legacy_json(config: BaselineRunConfig, path: Path, payload: dict[str, Any]) -> None:
    if training_reporting.should_write_legacy_reports(config.reporting_profile): save_json(path, payload)

# Select the script artifact prefix that belongs to the active dataset.
def _artifact_prefix(dataset: str) -> str:
    if dataset == "bpic2017": return "A_04"
    if dataset == "bpic2012": return "B_04"
    if dataset == "joint": return "J_04"
    raise ValueError(f"unknown dataset: {dataset}")

# Resolve one stored processed split root against this checkout because the frozen mapping records a fixed path.
# The longest suffix of the stored path that exists below the repository root wins.
def resolve_input_root(stored: str) -> Path:
    # REPO_ROOT comes from this module's location, never from the working directory, which the workflows change.
    stored_path = Path(stored)
    parts = stored_path.parts

    # Joining an absolute path resets the base, so the start index zero candidate is the stored path itself.
    # The local checkout must always win over an existing foreign path so that the candidate is never tested.
    first = 1 if stored_path.is_absolute() else 0
    for start in range(first, len(parts)):
        candidate = REPO_ROOT.joinpath(*parts[start:])
        if candidate.is_dir(): return candidate

    # Fall back to the stored value only when no suffix resolves, so the failure names the recorded path.
    return stored_path

# Load the approved dataset mapping and rewrite every processed-split root for this checkout.
def load_dataset_mapping(path: Path, require_approved: bool = True) -> dict[str, Any]:
    mapping = encoding.load_dataset_mapping(path, require_approved=require_approved)
    for dataset_mapping in mapping.get("datasets", {}).values():
        stored = dataset_mapping.get("input_root")
        if stored: dataset_mapping["input_root"] = str(resolve_input_root(str(stored)))
    return mapping

# Resolve the frozen E_04 encoding spec of one dataset and split configuration.
def encoding_spec_path(config: BaselineRunConfig) -> Path:
    metadata_root = config.artifact_root / config.dataset / config.run_name
    return metadata_root / f"{_artifact_prefix(config.dataset)}_encoding_spec.json"

# Load the E_04 metadata, created by the runner, needed to rebuild prefix tensors.
def load_training_metadata(config: BaselineRunConfig) -> tuple[
    dict[str, Any],
    dict[str, dict[str, int]],
    dict[str, dict[str, float]],
    dict[str, Any],
    dict[str, Any],
]:
    # Resolve the reviewed metadata folder for the selected dataset and split config.
    prefix = _artifact_prefix(config.dataset)
    metadata_root = config.artifact_root / config.dataset / config.run_name

    # Load the E_04 spec, vocabulary and scaler artifacts.
    spec = encoding.load_json_artifact(metadata_root / f"{prefix}_encoding_spec.json")
    vocabularies = encoding.load_json_artifact(metadata_root / f"{prefix}_vocabulary.json")
    scalers = encoding.load_json_artifact(metadata_root / f"{prefix}_scaler.json")

    # Load approved schema and mapping inputs that define the training tensor layout.
    schemas = encoding.load_approved_json(config.canonical_schema_path, "canonical schema")
    schema_profile = schemas["schema_profiles"][config.dataset]
    mapping = load_dataset_mapping(config.dataset_mapping_path, require_approved=True)
    return spec, vocabularies, scalers, schema_profile, mapping

# Build the training-local next activity mask from each dataset's own observed target classes.
def build_next_activity_mask_context(config: BaselineRunConfig, train_dataset: Any,
                                     n_classes: int) -> core.NextActivityMaskContext:

    # A single-dataset run keeps the whole head visible.
    if config.dataset != "joint":
        return core.build_next_activity_mask_context(
            {config.dataset: np.ones(int(n_classes), dtype=bool)},
            single_dataset_noop=True,
        )
    # A joint run masks each dataset to the next activity classes it actually contains in training.
    return core.build_next_activity_mask_context(next_activity_presence_by_dataset(train_dataset, n_classes))

# Build the central next activity evaluation mask from the pooled train split.
# Validation and test then score over the joint class set.
def build_evaluation_next_activity_mask_context(config: BaselineRunConfig,
    training_mask_context: core.NextActivityMaskContext, spec: dict[str, Any], vocabularies: dict[str, dict[str, int]],
    scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any], mapping: dict[str, Any]
    ) -> core.NextActivityMaskContext:

    # Single dataset runs and joint centralized runs already build their mask from the evaluated class set.
    # The training mask therefore equals the evaluation mask for these regimes.
    if config.dataset != "joint" or config.regime == "centralized":
        return training_mask_context

    # Score a joint local baseline over the pooled joint class set.
    # Build its evaluation mask from centralized training data, not the bank's own classes.
    central_config = replace(config, regime="centralized", bank=None)
    central_train_dataset = load_cached_prefix_dataset(central_config, "train", spec, vocabularies, scalers,
                                                       schema_profile, mapping)
    if central_train_dataset is None:
        central_mapped = load_mapped_events(central_config, mapping)
        central_train_dataset = cache_prefix_dataset(
            central_config, build_prefix_dataset(central_mapped, "train", spec, vocabularies, scalers, schema_profile),
            "train", spec, vocabularies, scalers, schema_profile, mapping,
        )
    return build_next_activity_mask_context(central_config, central_train_dataset,
                                            len(vocabularies[encoding.NEXT_ACTIVITY_TARGET]))

# Load processed split parquets and map them into canonical E_04 events.
def load_mapped_events(config: BaselineRunConfig, mapping: dict[str, Any]) -> pd.DataFrame:
    if config.dataset == "joint": return load_joint_mapped_events(config, mapping)

    dataset_mapping = mapping["datasets"][config.dataset]
    input_root = Path(dataset_mapping["input_root"])
    split_prefix = str(dataset_mapping["split_prefix"])
    frames: list[pd.DataFrame] = []

    # Load train, validation and test splits through the same canonical mapping path.
    for split_name in progress_iter(encoding.SPLITS, config,f"load {config.dataset} {config.run_name}",
        total=len(encoding.SPLITS), unit="split"):
        if config.regime == "centralized":
            path = input_root / "centralized" / config.run_name / f"{split_prefix}_{split_name}.parquet"
            client_id = "centralized"
        else:
            if not config.bank: raise ValueError("local baseline requires BANK")
            path = input_root / config.run_name / f"{split_prefix}_bank_{config.bank}_{split_name}.parquet"
            client_id = config.bank

        # Stop early when the expected preprocessing output is missing.
        if not path.exists(): raise FileNotFoundError(f"missing split parquet: {path}")

        # Convert the raw split into the canonical E_04 event schema.
        raw = pd.read_parquet(path)
        frames.append(encoding.to_canonical_events(raw, mapping, config.dataset, client_id, split_name))

    # Apply the reviewed activity mapping after all splits share one canonical frame.
    events = pd.concat(frames, ignore_index=True)
    mapped, _ = encoding.apply_activity_mapping(events, config.dataset_mapping_path)
    encoding.validate_canonical_events(mapped, mapped=True)
    return mapped

# Resolve one dataset-qualified joint bank id into its source spec and plain bank id.
def _joint_source_for_bank(config: BaselineRunConfig, qualified_bank: str) -> tuple[Any, str]:
    if ":" not in str(qualified_bank): raise ValueError(f"joint bank id must be dataset-qualified: {qualified_bank}")
    source_dataset, bank = str(qualified_bank).split(":", 1)
    spec = encoding.resolve_joint_run(config.run_name)
    source = next((item for item in spec.sources if item.dataset_id == source_dataset), None)
    if source is None: raise ValueError(f"joint run {spec.run_id} does not contain source dataset: {source_dataset}")
    if bank not in source.banks: raise ValueError(f"joint run {spec.run_id} does not contain bank: {qualified_bank}")
    return source, bank

# Load one source bank split from the per-dataset preprocessing outputs.
def _load_joint_source_split(mapping: dict[str, Any], source: Any, bank: str, split_name: str) -> pd.DataFrame:
    dataset_mapping = mapping["datasets"][source.dataset_id]
    input_root = Path(dataset_mapping["input_root"])
    split_prefix = str(dataset_mapping["split_prefix"])
    run_name = f"{source.heterogeneity}_{source.n_clients}banks"
    path = input_root / run_name / f"{split_prefix}_bank_{bank}_{split_name}.parquet"
    if not path.exists(): raise FileNotFoundError(f"missing split parquet: {path}")
    raw = pd.read_parquet(path)
    return encoding.to_canonical_events(raw, mapping, source.dataset_id, bank, split_name)

# Load joint processed split parquets and map them into one canonical E_04 event frame.
def load_joint_mapped_events(config: BaselineRunConfig, mapping: dict[str, Any]) -> pd.DataFrame:
    spec = encoding.resolve_joint_run(config.run_name)
    frames: list[pd.DataFrame] = []

    # Centralized joint runs pool every resolved source bank because no pre-baked joint parquet exists.
    if config.regime == "centralized":
        for source in spec.sources:
            for bank in source.banks:
                for split_name in encoding.SPLITS:
                    frames.append(_load_joint_source_split(mapping, source, bank, split_name))
    else:
        if not config.bank: raise ValueError("local baseline requires BANK")
        source, bank = _joint_source_for_bank(config, config.bank)
        for split_name in encoding.SPLITS:
            frames.append(_load_joint_source_split(mapping, source, bank, split_name))

    # Apply the reviewed activity mapping after all source splits share one canonical frame.
    events = pd.concat(frames, ignore_index=True)
    mapped, _ = encoding.apply_activity_mapping(events, config.dataset_mapping_path)
    encoding.validate_canonical_events(mapped, mapped=True)
    return mapped

# Resolve the processed split parquet of one simulated bank, for single-dataset and joint runs alike.
def bank_split_parquet_path(config: BaselineRunConfig, mapping: dict[str, Any], bank: str,
                            split_name: str) -> tuple[str, Path]:
    if config.dataset == "joint":
        source, source_bank = _joint_source_for_bank(config, bank)
        dataset_id = str(source.dataset_id)
        run_name = f"{source.heterogeneity}_{source.n_clients}banks"
    else:
        dataset_id = config.dataset
        source_bank = bank
        run_name = config.run_name

    # Both branches resolve the same reviewed dataset mapping layout.
    dataset_mapping = mapping["datasets"][dataset_id]
    input_root = Path(dataset_mapping["input_root"])
    split_prefix = str(dataset_mapping["split_prefix"])
    return dataset_id, input_root / run_name / f"{split_prefix}_bank_{source_bank}_{split_name}.parquet"

# Build the case-to-bank lookup of one split from the processed parquets per bank.
# A centralized run pools every bank into one parquet without a bank column, so the bank is recovered by case id.
def bank_by_case_for_split(config: BaselineRunConfig, mapping: dict[str, Any],
                           split_name: str) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    for bank in bank_names_for_config(config):
        dataset_id, path = bank_split_parquet_path(config, mapping, bank, split_name)
        if not path.exists(): raise FileNotFoundError(f"missing bank split parquet: {path}")

        # Read only the case id column because the lookup needs nothing else from the split.
        case_column = str(mapping["datasets"][dataset_id]["column_mapping"][encoding.CASE_ID])
        cases = pd.read_parquet(path, columns=[case_column])[case_column].astype(str).unique()
        for case_id in cases: lookup[(dataset_id, str(case_id))] = bank
    return lookup

# Turn the case-to-bank lookup into one boolean prefix selector per simulated bank.
def bank_selectors_for_dataset(dataset: Any, banks: tuple[str, ...],
                               bank_by_case: dict[tuple[str, str], str]) -> dict[str, np.ndarray]:
    prefix_index = getattr(dataset, "prefix_index", None)
    if not prefix_index: raise ValueError("per-bank selectors require prefix_index rows")

    # Resolve the owning bank of every prefix and stop when a prefix belongs to no bank split.
    owners = np.array([bank_by_case.get((str(row.dataset_id), str(row.case_id)), "") for row in prefix_index])
    unmapped = int((owners == "").sum())
    if unmapped: raise ValueError(f"{unmapped} prefixes could not be assigned to a bank split")
    return {bank: np.asarray(owners == bank) for bank in banks}

# Build the E_04 PrefixDataset for one split.
def build_prefix_dataset(mapped: pd.DataFrame, split_name: str, spec: dict[str, Any],
    vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]], schema_profile: dict[str, Any],
    ) -> Any:

    # Select the requested split before building prefix references.
    split_events = mapped.loc[mapped[encoding.SPLIT] == split_name].copy()
    prefix_index, _ = encoding.build_prefix_index(split_events, spec["prefix"]["cap"])

    # Let E_04 create padded prefix tensors on demand with the approved schema columns.
    return encoding.PrefixDataset(
        split_events, prefix_index, vocabularies, scalers,
        static_padding_length=spec["prefix"]["static_padding_length"],
        sequence_categorical_columns=list(schema_profile["sequence_categorical_columns"]),
        sequence_numerical_columns=list(schema_profile["sequence_numerical_columns"]),
        offer_numerical_columns=list(schema_profile["offer_numerical_columns"]),
        include_offer_features=True,
        remaining_time_target_repr=spec.get("target_scalers", {}).get("remaining_time"),
    )

# ----------------------------------------------------------------------------------------------------------------------
# 3. DATASET, LOADER AND MODEL HELPERS

# Resolve the active simulated bank names from the configured client count.
def bank_names_for_config(config: BaselineRunConfig) -> tuple[str, ...]:
    if config.dataset == "joint": return tuple(encoding.resolve_joint_run(config.run_name).qualified_client_ids)
    return tuple(BANK_NAMES[: config.n_clients])

# Build a DataLoader with the E_04 collate function and reproducible shuffling.
def make_loader(dataset: Any, batch_size: int, shuffle: bool, seed: int, num_workers: int,
                next_activity_mask_context: Optional[core.NextActivityMaskContext] = None) -> DataLoader:

    # Seed the shuffling generator and wrap the dataset only when the bank-local next activity mask is active.
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_dataset = dataset
    collate_fn = encoding.collate_prefix_batch
    if next_activity_mask_context is not None and not next_activity_mask_context.is_noop:
        loader_dataset = core.DatasetIdEncodedDataset(dataset, next_activity_mask_context.dataset_code_by_id)
        collate_fn = core.collate_dataset_aware_batch

    # Keep every prefix by never dropping the last batch. Reuse the workers across epochs when workers are enabled.
    return DataLoader(
        loader_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False, num_workers=num_workers,
        pin_memory=False, persistent_workers=num_workers > 0, collate_fn=collate_fn, generator=generator,
    )

# Derive model dimensions from the approved schema profile and the train vocabularies.
def build_model(config: BaselineRunConfig, vocabularies: dict[str, dict[str, int]],
                schema_profile: dict[str, Any], *, lstm_cls: type[nn.Module] = nn.LSTM) -> core.MultitaskLSTM:

    # Build a vocabulary size map for the active categorical sequence fields.
    categorical_columns = list(schema_profile["sequence_categorical_columns"])
    categorical_vocab_sizes = {column: len(vocabularies[column]) for column in categorical_columns}

    # The linear head is used only for z-score targets, which are centered and can be negative.
    remaining_time_softplus = config.remaining_time_scaling != "zscore"

    # Instantiate shared E_05/E_06 architecture with dimensions from the approved schema.
    return core.MultitaskLSTM(
        categorical_vocab_sizes=categorical_vocab_sizes,
        numerical_dim=len(schema_profile["sequence_numerical_columns"]),
        offer_dim=len(schema_profile["offer_numerical_columns"]),
        next_activity_classes=len(vocabularies[encoding.NEXT_ACTIVITY_TARGET]),
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        head_hidden_size=config.head_hidden_size,
        remaining_time_softplus=remaining_time_softplus,
        outcome_head_dropout=config.outcome_head_dropout,
        lstm_cls=lstm_cls,
    )

# HELPER: Collect one sample field (e.g. outcome_label) across the entire split into a single numpy array.
def tensor_array_from_dataset(dataset: Any, key: str) -> np.ndarray:
    # Warm cache: the field already exists, so return it and skip work per sample.
    if hasattr(dataset, "arrays") and key in dataset.arrays: return np.asarray(dataset.arrays[key])

    # Uncached dataset: encode each prefix sample, extract the requested tensor field and stack values into one array.
    return np.stack([dataset[index][key].detach().cpu().numpy() for index in range(len(dataset))])

# HELPER: Class weights are fit on the train labels per prefix because the loss averages over prefixes.
def training_outcome_labels_from_dataset(dataset: Any) -> pd.Series:
    # Read labels as a flat integer series for class-frequency counting.
    labels = tensor_array_from_dataset(dataset, "outcome_label").reshape(-1)
    return pd.Series(labels.astype(int))

# HELPER: Report which next activity classes have local target support in one dataset.
def next_activity_class_presence_from_dataset(dataset: Any, n_classes: int) -> np.ndarray:
    labels = tensor_array_from_dataset(dataset, "next_activity_label").reshape(-1).astype(int)
    masks = tensor_array_from_dataset(dataset, "next_activity_mask").reshape(-1).astype(bool)
    presence = np.zeros(int(n_classes), dtype=bool)
    valid_labels = labels[masks]
    valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < int(n_classes))]
    if len(valid_labels) > 0: presence[np.unique(valid_labels)] = True
    return presence

# HELPER: Count valid next activity targets in one local prefix dataset.
def next_activity_target_counts_from_dataset(dataset: Any, n_classes: int) -> np.ndarray:
    labels = tensor_array_from_dataset(dataset, "next_activity_label").reshape(-1).astype(int)
    masks = tensor_array_from_dataset(dataset, "next_activity_mask").reshape(-1).astype(bool)
    valid_labels = labels[masks]
    valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < int(n_classes))]
    if len(valid_labels) == 0: return np.zeros(int(n_classes), dtype=np.float32)
    return np.bincount(valid_labels, minlength=int(n_classes)).astype(np.float32, copy=False)

# HELPER: Report each dataset's observed next activity classes from a pooled prefix dataset.
def next_activity_presence_by_dataset(dataset: Any, n_classes: int) -> dict[str, np.ndarray]:
    prefix_index = getattr(dataset, "prefix_index", None)
    if prefix_index is None: raise ValueError("per-dataset next-activity presence requires prefix_index rows")
    dataset_ids = np.array([str(row.dataset_id) for row in prefix_index])
    labels = tensor_array_from_dataset(dataset, "next_activity_label").reshape(-1).astype(int)
    masks = tensor_array_from_dataset(dataset, "next_activity_mask").reshape(-1).astype(bool)
    presence_by_dataset: dict[str, np.ndarray] = {}
    for dataset_id in sorted(set(dataset_ids.tolist())):
        selected = (dataset_ids == dataset_id) & masks
        valid_labels = labels[selected]
        valid_labels = valid_labels[(valid_labels >= 0) & (valid_labels < int(n_classes))]
        presence = np.zeros(int(n_classes), dtype=bool)
        if len(valid_labels) > 0: presence[np.unique(valid_labels)] = True
        if not bool(presence.any()): raise ValueError(f"next-activity presence for {dataset_id} contains no classes")
        presence_by_dataset[dataset_id] = presence
    return presence_by_dataset

# HELPER: Load the RT representation fitted by E_04 and validate the selected training mode.
def remaining_time_repr_from_spec(spec: dict[str, Any], config: BaselineRunConfig) -> core.RemainingTimeRepr:
    payload = spec.get("target_scalers", {}).get("remaining_time")
    if not isinstance(payload, dict): raise ValueError("E_04 encoding spec is missing remaining-time target metadata")
    target_repr = encoding.remaining_time_target_repr_from_dict(payload)
    if target_repr.transform != config.remaining_time_transform:
        raise ValueError("REMAINING_TIME_TRANSFORM does not match the E_04 encoding spec")
    if target_repr.scaling != config.remaining_time_scaling:
        raise ValueError("REMAINING_TIME_SCALING does not match the E_04 encoding spec")
    return core.RemainingTimeRepr(
        target_repr.transform, target_repr.scaling, target_repr.center, target_repr.scale,
        target_repr.use_softplus, target_repr.median_model_units,
    )

# HELPER: Build the outcome CE loss with train class weights and label smoothing.
def build_outcome_loss_from_labels(config: BaselineRunConfig, labels: pd.Series, device: torch.device) -> nn.Module:
    smoothing = float(config.outcome_label_smoothing)
    weights = compute_outcome_class_weights(labels, power=float(config.outcome_class_weight_power))

    # Move class weights to the training device before constructing the loss module.
    tensor = torch.tensor([weights[label] for label in OUTCOME_CLASSES], dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=tensor, reduction="none", label_smoothing=smoothing)

# HELPER: Invert model unit predictions to raw seconds through the fitted RT representation.
def remaining_time_model_units_to_seconds(values: np.ndarray,
                                          remaining_time_repr: core.RemainingTimeRepr) -> np.ndarray:
    # Convert numpy predictions to torch so the shared inverse helper is reused.
    tensor = torch.tensor(np.asarray(values, dtype=float), dtype=torch.float32)
    return core.remaining_time_model_units_to_seconds(tensor, remaining_time_repr).numpy()

# HELPER: Keep only masked finite RT targets for the diagnostics, filtering to valid masked positions first.
def _valid_remaining_time_values(labels: np.ndarray, masks: np.ndarray) -> np.ndarray:
    values = np.asarray(labels, dtype=float).reshape(-1)
    valid = np.asarray(masks).reshape(-1).astype(bool)
    values = values[valid]
    return values[np.isfinite(values)]

# HELPER: Compact quantile summary used by every RT diagnostic block.
def distribution_summary(values: np.ndarray) -> dict[str, float]:
    # Drop values that are not finite so diagnostics never fail on invalid predictions.
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]

    # Return an empty but schema stable summary when no valid values exist.
    if finite.size == 0: return {"count": 0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}

    # Record the central tendency and spread needed for target analysis.
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "p10": float(np.quantile(finite, 0.10)),
        "p25": float(np.quantile(finite, 0.25)),
        "p75": float(np.quantile(finite, 0.75)),
        "p90": float(np.quantile(finite, 0.90)),
    }

# Record the train target distribution so the bias initialization and representation stay auditable (RT head).
def remaining_time_target_diagnostics(train_dataset: Any,
                                      remaining_time_repr: core.RemainingTimeRepr) -> dict[str, Any]:

    # Read train model-unit labels from the encoded training split.
    labels = tensor_array_from_dataset(train_dataset, "remaining_time_label").reshape(-1)
    masks = tensor_array_from_dataset(train_dataset, "remaining_time_mask").reshape(-1)
    model_unit_values = _valid_remaining_time_values(labels, masks)

    # Invert model-unit labels back to raw seconds for diagnostics and baselines.
    seconds_values = remaining_time_model_units_to_seconds(model_unit_values, remaining_time_repr)

    # Store both seconds and model unit summaries.
    return {
        "remaining_time_transform": remaining_time_repr.transform,
        "remaining_time_scaling": remaining_time_repr.scaling,
        "remaining_time_center": float(remaining_time_repr.center),
        "remaining_time_scale": float(remaining_time_repr.scale),
        "remaining_time_use_softplus": bool(remaining_time_repr.use_softplus),
        "train_median_seconds": float(np.median(seconds_values)) if seconds_values.size else 0.0,
        "train_median_model_units": float(remaining_time_repr.median_model_units),
        "train_seconds_distribution": distribution_summary(seconds_values),
        "train_model_unit_distribution": distribution_summary(model_unit_values),
    }

# Detect prediction collapse: A nearly constant or zero-clamped head shows up here.
def remaining_time_prediction_diagnostics(predictions: pd.DataFrame) -> dict[str, float]:

    # Evaluate only prefixes where RT loss and metrics are valid.
    valid = predictions.loc[predictions["remaining_time_mask"].astype(bool)]
    if valid.empty: return {"pred_mean_seconds": 0.0, "pred_median_seconds": 0.0, "percent_clamped_to_zero": 0.0}

    # Count predictions that came out negative and were floored to zero seconds.
    # Softplus heads never trigger this. The linear z-score head can and a high share is a warning sign.
    raw_seconds = valid["remaining_time_pred_seconds_raw"].to_numpy(dtype=float)
    clamped = valid["remaining_time_pred_seconds_clamped"].to_numpy(dtype=float)
    return {
        "pred_mean_seconds": float(np.mean(clamped)),
        "pred_median_seconds": float(np.median(clamped)),
        "percent_clamped_to_zero": float((raw_seconds < 0.0).mean() * 100.0),
    }

# Anchor the RT head against the train median baseline and the test median "oracle" (for sanity).
def remaining_time_baseline_diagnostics(predictions: pd.DataFrame, train_median_seconds: float) -> dict[str, Any]:
    # Evaluate RT baselines only on prefixes with a valid target.
    valid = predictions.loc[predictions["remaining_time_mask"].astype(bool)].copy()
    if valid.empty:
        return {
            "train_median_baseline_mae_seconds": 0.0,
            "test_median_oracle_mae_seconds": 0.0,
            "true_seconds_distribution": distribution_summary(np.array([], dtype=float)),
            "pred_seconds_distribution": distribution_summary(np.array([], dtype=float)),
        }

    # Compare model predictions against train median and test median baselines.
    true_seconds = valid["remaining_time_label_seconds"].to_numpy(dtype=float)
    pred_seconds = valid["remaining_time_pred_seconds_clamped"].to_numpy(dtype=float)
    test_median = float(np.median(true_seconds))
    return {
        "train_median_seconds": float(train_median_seconds),
        "train_median_baseline_mae_seconds": float(np.mean(np.abs(true_seconds - train_median_seconds))),
        "test_median_oracle_seconds": test_median,
        "test_median_oracle_mae_seconds": float(np.mean(np.abs(true_seconds - test_median))),
        "true_seconds_distribution": distribution_summary(true_seconds),
        "pred_seconds_distribution": distribution_summary(pred_seconds),
        **remaining_time_prediction_diagnostics(predictions),
    }

# Reshape compute_prefix_bucket_metrics output into one flat row per bucket that is not empty.
def prefix_bucket_summary_rows(bucket_metrics: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    # Keep the exported CSV focused on the headline early-to-late metrics.
    columns = ("outcome_accuracy", "outcome_macro_f1", "outcome_weighted_f1",
               "outcome_balanced_accuracy", "remaining_time_mae_seconds")
    rows: list[dict[str, Any]] = []

    # Convert every non-empty bucket into one flat row for the prefix length summary table.
    for bucket_name, metrics in bucket_metrics.items():
        if int(metrics.get("n_prefixes", 0)) == 0: continue
        row: dict[str, Any] = {"prefix_bucket": bucket_name, "n_prefixes": int(metrics["n_prefixes"])}
        for column in columns: row[column] = float(metrics.get(column, 0.0))
        rows.append(row)
    return rows

# Write the early-to-late prefix bucket per run table as a flat CSV.
# The script_id keeps the artifact prefix correct when E_06 reuses this helper, defaulting to the E_05 prefix.
def write_prefix_bucket_summary_csv(output_dir: Path, test_predictions: pd.DataFrame,
                                    script_id: str = SCRIPT_ID) -> None:
    rows = prefix_bucket_summary_rows(compute_prefix_bucket_metrics(test_predictions))
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / f"{script_id}_prefix_bucket_summary_test.csv", index=False)

# Read one metric block out of a run report that a previous invocation already wrote.
def metrics_block_from_run_report(output_dir: Path, block: str) -> dict[str, Any]:
    # Return an empty block when no run report exists yet, so callers fall back to their in-memory payload.
    report_path = run_artifact_path(output_dir, "run_report.json")
    if not report_path.exists(): return {}
    report = json.loads(report_path.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})
    if not isinstance(metrics, dict): return {}
    resolved = metrics.get(block, {})
    return resolved if isinstance(resolved, dict) else {}

# Write a flat CSV and JSON for a centralized run. Reuse the local summary flattening to obtain a shared schema.
def write_centralized_summary(output_dir: Path, test_metrics: dict[str, Any],
                              per_bank_test_metrics: dict[str, Any]) -> None:
    # Start with the global pooled test row.
    rows: list[dict[str, Any]] = []
    global_row: dict[str, Any] = {"scope": "global_test"}
    global_row.update(flatten_local_summary_metrics(test_metrics))
    rows.append(global_row)

    # Append one row per bank from the in-memory fairness payload, or from an already written run report.
    per_bank = per_bank_test_metrics or metrics_block_from_run_report(output_dir, "per_bank_test")
    for bank_name, bank_metrics in per_bank.items():
        if not isinstance(bank_metrics, dict): continue
        bank_row: dict[str, Any] = {"scope": f"bank_{bank_name}"}
        bank_row.update(flatten_local_summary_metrics(bank_metrics))
        rows.append(bank_row)

    # Write both CSV and JSON versions from the same row set.
    frame = pd.DataFrame(rows)
    csv_path = run_artifact_path(output_dir, "centralized_summary_test.csv")
    json_path = run_artifact_path(output_dir, "centralized_summary_test.json")
    frame.to_csv(csv_path, index=False)
    save_json(json_path, {"rows": frame.to_dict(orient="records")})

# Compute outcome and RT headline metrics for fixed early-to-late prefix buckets.
def compute_prefix_bucket_metrics(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    bucket_specs = {
        "1": predictions["prefix_length"] == 1,
        "2-5": predictions["prefix_length"].between(2, 5),
        "6-10": predictions["prefix_length"].between(6, 10),
        "11-20": predictions["prefix_length"].between(11, 20),
        "21+": predictions["prefix_length"] >= 21,
    }
    probability_columns = [f"outcome_prob_{label}" for label in OUTCOME_CLASSES]
    metrics: dict[str, dict[str, float]] = {}

    # Compute outcome and RT metrics for each filled bucket.
    for name, selector in bucket_specs.items():
        bucket = predictions.loc[selector]
        if bucket.empty:
            metrics[name] = {"n_prefixes": 0}
            continue

        # Reconstruct outcome logits from probabilities because the prediction frame stores probabilities.
        outcome_logits = np.log(np.clip(bucket[probability_columns].to_numpy(dtype=float), 1e-12, 1.0))
        outcome = core.compute_outcome_metrics(bucket["outcome_label"].to_numpy(dtype=int), outcome_logits)

        # RT MAE uses only prefixes where the target is defined.
        remaining = bucket.loc[bucket["remaining_time_mask"].astype(bool)]
        rt_mae = 0.0
        if not remaining.empty:
            rt_mae = float(
                np.mean(
                    np.abs(
                        remaining["remaining_time_pred_seconds_clamped"].to_numpy(dtype=float)
                        - remaining["remaining_time_label_seconds"].to_numpy(dtype=float)
                    )
                )
            )
        metrics[name] = {
            "n_prefixes": int(len(bucket)),
            "outcome_accuracy": float((bucket["outcome_label"] == bucket["outcome_pred"]).mean()),
            "outcome_macro_f1": float(outcome["macro_f1"]),
            "outcome_weighted_f1": float(outcome["weighted_f1"]),
            "outcome_balanced_accuracy": float(outcome["balanced_accuracy"]),
            "remaining_time_mae_seconds": rt_mae,
        }
    return metrics

# ----------------------------------------------------------------------------------------------------------------------
# 4. EVALUATION AND PREDICTION EXPORT

# CLASS: Carry the per-prefix arrays of one evaluated split so any prefix subset is scored from the same forward pass.
@dataclass(frozen=True)
class SplitEvaluationArrays:
    outcome_labels: np.ndarray
    outcome_logits: np.ndarray
    next_activity_labels: np.ndarray
    next_activity_logits: np.ndarray
    next_activity_masks: np.ndarray
    remaining_time_true_seconds: np.ndarray
    remaining_time_pred_seconds: np.ndarray
    remaining_time_masks: np.ndarray
    outcome_loss_values: np.ndarray
    next_activity_loss_values: np.ndarray
    remaining_time_loss_values: np.ndarray

# HELPER: Average one loss value vector over the selected prefixes, restricted to the head's own valid positions.
def _subset_loss_mean(values: np.ndarray, selector: np.ndarray, valid: Optional[np.ndarray] = None) -> float:
    selected = selector if valid is None else (selector & valid)
    if not bool(selected.any()): return 0.0
    return float(np.asarray(values, dtype=float).reshape(-1)[selected].mean())

# Score one named prefix subset with the same metric implementations the pooled block uses.
def subset_evaluation_metrics(arrays: SplitEvaluationArrays, selector: np.ndarray,
                              config: BaselineRunConfig) -> dict[str, Any]:

    # Average the three head losses over the subset, each on the positions its own mask marks valid.
    loss_outcome = _subset_loss_mean(arrays.outcome_loss_values, selector)
    loss_next = _subset_loss_mean(arrays.next_activity_loss_values, selector, arrays.next_activity_masks)
    loss_remaining = _subset_loss_mean(arrays.remaining_time_loss_values, selector, arrays.remaining_time_masks)

    # Reuse the shared metric helpers, so a subset and the pooled split are never scored by different code.
    next_selected = selector & arrays.next_activity_masks
    return {
        "loss_total": core.weighted_total_loss(
            loss_outcome, loss_next, loss_remaining, config.outcome_loss_weight, config.next_activity_loss_weight,
            config.remaining_time_loss_weight,
        ),
        "loss_outcome": loss_outcome,
        "loss_next_activity": loss_next,
        "loss_remaining_time": loss_remaining,
        "n_prefixes": int(selector.sum()),
        "outcome": core.compute_outcome_metrics(arrays.outcome_labels[selector], arrays.outcome_logits[selector]),
        "next_activity": core.compute_next_activity_metrics(
            arrays.next_activity_labels[next_selected], arrays.next_activity_logits[next_selected]),
        "remaining_time": core.compute_remaining_time_metrics(
            torch.tensor(arrays.remaining_time_true_seconds[selector]),
            torch.tensor(arrays.remaining_time_pred_seconds[selector]),
            torch.tensor(arrays.remaining_time_masks[selector].astype(np.int8)),
        ),
    }

# Evaluate one split: Per epoch validation skips the prediction frame, final val and test build it.
# Passing subset selectors score named prefix subsets from the same forward pass and fills subset_metrics in place.
def evaluate_model(model: core.MultitaskLSTM, loader: DataLoader, outcome_loss: nn.Module, next_activity_loss: nn.Module,
    remaining_time_repr: core.RemainingTimeRepr, huber_beta: float, device: torch.device, config: BaselineRunConfig,
    dataset: Any, progress_label: str = "eval", collect_predictions: bool = False,
    next_activity_mask_context: Optional[core.NextActivityMaskContext] = None,
    subset_selectors: Optional[dict[str, np.ndarray]] = None,
    subset_metrics: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:

    # Initialize loss per head accumulators and prediction buffers.
    model.eval()
    loss_sums = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    loss_counts = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    outcome_logits: list[np.ndarray] = []
    outcome_labels: list[np.ndarray] = []
    next_logits: list[np.ndarray] = []
    next_labels: list[np.ndarray] = []
    next_masks: list[np.ndarray] = []
    remaining_pred_scaled: list[np.ndarray] = []
    remaining_true: list[np.ndarray] = []
    remaining_masks: list[np.ndarray] = []
    outcome_loss_values: list[np.ndarray] = []
    next_loss_values: list[np.ndarray] = []
    remaining_loss_values: list[np.ndarray] = []
    collect_subsets = subset_selectors is not None and subset_metrics is not None
    total_start = time.perf_counter()

    # Run evaluation without gradients while retaining logits and targets for final metrics.
    with torch.no_grad():
        batch_iter = progress_iter(loader, config, f"{progress_label} batches", total=len(loader), unit="batch")
        for batch in batch_iter:
            batch = core.move_batch_to_device(batch, device)
            outputs = model(batch)

            # Compute the same masked multitask losses used during training.
            losses = core.compute_multitask_loss(
                outputs, batch, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta,
                config.outcome_loss_weight, config.next_activity_loss_weight, config.remaining_time_loss_weight,
                next_activity_mask_context=next_activity_mask_context,
            )

            # Accumulate exact loss per sample, means for each head.
            batch_n = float(batch.outcome_label.shape[0])
            next_n = float(batch.next_activity_mask.float().sum().item())
            remaining_n = float(batch.remaining_time_mask.float().sum().item())
            loss_sums["outcome"] += float(losses.outcome.item()) * batch_n
            loss_sums["next_activity"] += float(losses.next_activity.item()) * next_n
            loss_sums["remaining_time"] += float(losses.remaining_time.item()) * remaining_n
            loss_counts["outcome"] += batch_n
            loss_counts["next_activity"] += next_n
            loss_counts["remaining_time"] += remaining_n
            if hasattr(batch_iter, "set_postfix"): batch_iter.set_postfix({"loss": f"{float(losses.total.item()):.4f}"})

            # Store CPU np arrays so the final metric computation does not hold device tensors.
            outcome_logits.append(outputs.outcome_logits.detach().cpu().numpy())
            outcome_labels.append(batch.outcome_label.detach().cpu().numpy())
            masked_next_logits = outputs.next_activity_logits
            if next_activity_mask_context is not None and not next_activity_mask_context.is_noop:
                if not hasattr(batch, "dataset_code"):
                    raise ValueError("dataset-aware evaluation requires dataset_code")
                masked_next_logits = core.mask_next_activity_logits(
                    outputs.next_activity_logits, batch.dataset_code, next_activity_mask_context
                )
            next_logits.append(masked_next_logits.detach().cpu().numpy())
            next_labels.append(batch.next_activity_label.detach().cpu().numpy())
            next_masks.append(batch.next_activity_mask.detach().cpu().numpy())
            remaining_pred_scaled.append(outputs.remaining_time_scaled.detach().cpu().numpy())
            remaining_true.append(batch.remaining_time_label.detach().cpu().numpy())
            remaining_masks.append(batch.remaining_time_mask.detach().cpu().numpy())

            # Keep the loss values per sample so named prefix subsets can be averaged without a second forward pass.
            if collect_subsets:
                outcome_loss_values.append(losses.outcome_values.detach().cpu().numpy())
                next_loss_values.append(losses.next_activity_values.detach().cpu().numpy())
                remaining_loss_values.append(losses.remaining_time_values.detach().cpu().numpy())

    # Reconstruct the weighted total loss from the means per head.
    loss_means = {key: loss_sums[key] / max(loss_counts[key], 1.0) for key in loss_sums}
    loss_total = core.weighted_total_loss(
        loss_means["outcome"], loss_means["next_activity"], loss_means["remaining_time"],
        config.outcome_loss_weight, config.next_activity_loss_weight, config.remaining_time_loss_weight,
    )

    # Concatenate all batch buffers into split arrays.
    outcome_logit_array = np.concatenate(outcome_logits)
    outcome_label_array = np.concatenate(outcome_labels)
    next_logit_array = np.concatenate(next_logits)
    next_label_array = np.concatenate(next_labels)
    next_mask_array = np.concatenate(next_masks).astype(bool)
    remaining_pred_model_units_array = np.concatenate(remaining_pred_scaled)
    remaining_true_model_units_array = np.concatenate(remaining_true)
    remaining_mask_array = np.concatenate(remaining_masks).astype(bool)

    # The cache stores encoded RT targets, so labels and predictions both need inversion for second-based metrics.
    remaining_true_seconds_array = remaining_time_model_units_to_seconds(remaining_true_model_units_array,
                                                                         remaining_time_repr)
    remaining_pred_seconds_array = remaining_time_model_units_to_seconds(remaining_pred_model_units_array,
                                                                         remaining_time_repr)

    # Compute the three task metric blocks in their reporting units.
    outcome_metrics = core.compute_outcome_metrics(outcome_label_array, outcome_logit_array)
    next_metrics = core.compute_next_activity_metrics(next_label_array[next_mask_array],
                                                      next_logit_array[next_mask_array])
    remaining_metrics = core.compute_remaining_time_metrics(
        torch.tensor(remaining_true_seconds_array),
        torch.tensor(remaining_pred_seconds_array),
        torch.tensor(remaining_mask_array.astype(np.int8)),
    )
    metrics = {
        "loss_total": loss_total,
        "loss_outcome": loss_means["outcome"],
        "loss_next_activity": loss_means["next_activity"],
        "loss_remaining_time": loss_means["remaining_time"],
        "n_prefixes": int(outcome_label_array.shape[0]),
        "timing_total_seconds": time.perf_counter() - total_start,
        "outcome": outcome_metrics,
        "next_activity": next_metrics,
        "remaining_time": remaining_metrics,
    }
    if not collect_predictions: return metrics, None

    # Build the final prediction frame for final validation, test and exports per bank.
    predictions = build_prediction_frame(
        dataset, outcome_label_array, outcome_logit_array, next_label_array, next_mask_array, next_logit_array,
        remaining_true_model_units_array, remaining_true_seconds_array, remaining_mask_array,
        remaining_pred_model_units_array,
        remaining_pred_seconds_array, remaining_time_repr,
    )
    metrics["remaining_time"].update(remaining_time_prediction_diagnostics(predictions))

    # Score every named prefix subset from the arrays of this same forward pass.
    if collect_subsets:
        arrays = SplitEvaluationArrays(
            outcome_labels=outcome_label_array,
            outcome_logits=outcome_logit_array,
            next_activity_labels=next_label_array,
            next_activity_logits=next_logit_array,
            next_activity_masks=next_mask_array,
            remaining_time_true_seconds=remaining_true_seconds_array,
            remaining_time_pred_seconds=remaining_pred_seconds_array,
            remaining_time_masks=remaining_mask_array,
            outcome_loss_values=np.concatenate(outcome_loss_values),
            next_activity_loss_values=np.concatenate(next_loss_values),
            remaining_time_loss_values=np.concatenate(remaining_loss_values),
        )
        for name, selector in (subset_selectors or {}).items():
            subset = subset_evaluation_metrics(arrays, np.asarray(selector, dtype=bool), config)
            subset["remaining_time"].update(remaining_time_prediction_diagnostics(predictions.loc[selector]))
            if subset_metrics is not None: subset_metrics[name] = subset
    return metrics, predictions

# Join the predictions per prefix with the cached references so every row keeps case provenance.
def build_prediction_frame(dataset: Any, outcome_labels: np.ndarray, outcome_logits: np.ndarray,
    next_labels: np.ndarray, next_masks: np.ndarray, next_logits: np.ndarray, remaining_true_model_units: np.ndarray,
    remaining_true_seconds: np.ndarray, remaining_masks: np.ndarray,
    remaining_pred_model_units: np.ndarray, remaining_pred_seconds: np.ndarray,
    remaining_time_repr: core.RemainingTimeRepr) -> pd.DataFrame:

    # Convert raw logits into exported probabilities and top-k next activity labels.
    outcome_probs = _softmax(outcome_logits)
    next_top_k = np.argsort(-next_logits, axis=1)[:, : min(3, next_logits.shape[1])]

    # Start the prediction frame from the cached prefix references.
    rows = [
        {
            "dataset_id": row.dataset_id,
            "client_id": row.client_id,
            "case_id": row.case_id,
            "split": row.split,
            "prefix_length": row.prefix_length,
            "label_pos": row.label_pos,
        }
        for row in dataset.prefix_index
    ]
    frame = pd.DataFrame(rows)

    # Store outcome labels, argmax predictions and one probability column per class.
    frame["outcome_label"] = outcome_labels.astype(int)
    frame["outcome_pred"] = outcome_probs.argmax(axis=1).astype(int)
    for label in OUTCOME_CLASSES: frame[f"outcome_prob_{label}"] = outcome_probs[:, label]

    # Store next activity labels, masks, top-1 prediction and top-3 prediction set.
    frame["next_activity_label"] = next_labels.astype(int)
    frame["next_activity_mask"] = next_masks.astype(int)
    frame["next_activity_pred"] = next_logits.argmax(axis=1).astype(int)
    frame["next_activity_top3"] = [",".join(str(int(value)) for value in row) for row in next_top_k]

    # Store RT representation metadata beside every prefix prediction.
    frame["remaining_time_transform"] = remaining_time_repr.transform
    frame["remaining_time_scaling"] = remaining_time_repr.scaling
    frame["remaining_time_center"] = float(remaining_time_repr.center)
    frame["remaining_time_scale"] = float(remaining_time_repr.scale)

    # Store RT labels and predictions in both model units and raw seconds.
    frame["remaining_time_label_model_units"] = remaining_true_model_units.astype(float)
    frame["remaining_time_mask"] = remaining_masks.astype(int)
    frame["remaining_time_pred_model_units"] = remaining_pred_model_units.astype(float)
    frame["remaining_time_label_seconds"] = remaining_true_seconds.astype(float)
    frame["remaining_time_pred_seconds_raw"] = remaining_pred_seconds.astype(float)
    frame["remaining_time_pred_seconds_clamped"] = np.maximum(remaining_pred_seconds, 0.0).astype(float)
    return frame

# HELPER: Numerically stable softmax for the exported probability columns.
def _softmax(logits: np.ndarray) -> np.ndarray:
    # Subtract the row max so the largest exponent is 0. Softmax remains unchanged, but exp cannot overflow.
    shifted = logits - logits.max(axis=1, keepdims=True)

    # Normalize exponentiated logits into class probabilities per row.
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)

# ----------------------------------------------------------------------------------------------------------------------
# 5. TRAINING LOOP

# Train one baseline end to end: data, model, epochs, restore the best checkpoint and artifact export.
def run_baseline_training(config: BaselineRunConfig, output_dir: Path) -> dict[str, Any]:

    # Initialize reproducibility, device and phase timing containers.
    timings: dict[str, float] = {}
    core.set_global_seed(config.seed)
    device = core.select_device(config.device)

    # Load the E_04 metadata artifacts that define tensor columns, vocabularies and scalers.
    with _timed(timings, "load_metadata"):
        spec, vocabularies, scalers, schema_profile, mapping = load_training_metadata(config)

    # Try all three split caches before touching processed parquet files.
    with _timed(timings, "load_cached_prefix_datasets"):
        train_dataset = load_cached_prefix_dataset(config, "train", spec, vocabularies, scalers, schema_profile,
                                                  mapping)
        val_dataset = load_cached_prefix_dataset(config, "val", spec, vocabularies, scalers, schema_profile,
                                                mapping)
        test_dataset = load_cached_prefix_dataset(config, "test", spec, vocabularies, scalers, schema_profile,
                                                 mapping)

    # Record cache hit flags for the final timing artifact.
    cache_hits = {"train": train_dataset is not None, "val": val_dataset is not None, "test": test_dataset is not None}

    # Rebuild only the missing splits from the processed parquets, then cache them for later runs.
    mapped: Optional[pd.DataFrame] = None
    if not all(cache_hits.values()):
        with _timed(timings, "load_mapped_events"): mapped = load_mapped_events(config, mapping)
        with _timed(timings, "build_and_cache_prefix_datasets"):
            if train_dataset is None:
                train_dataset = cache_prefix_dataset(
                    config, build_prefix_dataset(mapped, "train", spec, vocabularies, scalers, schema_profile),
                    "train", spec, vocabularies, scalers, schema_profile, mapping,
                )
            if val_dataset is None:
                val_dataset = cache_prefix_dataset(
                    config, build_prefix_dataset(mapped, "val", spec, vocabularies, scalers, schema_profile),
                    "val", spec, vocabularies, scalers, schema_profile, mapping,
                )
            if test_dataset is None:
                test_dataset = cache_prefix_dataset(
                    config, build_prefix_dataset(mapped, "test", spec, vocabularies, scalers, schema_profile),
                    "test", spec, vocabularies, scalers, schema_profile, mapping,
                )

    # Fail prominently if neither cache loading nor cache rebuilding produced all splits.
    if train_dataset is None or val_dataset is None or test_dataset is None:
        raise RuntimeError("prefix datasets were not loaded or built")

    # Build the bank-local next-activity training mask from the resolved training data, so joint runs mask per dataset.
    training_next_activity_mask_context = build_next_activity_mask_context(
        config, train_dataset, len(vocabularies[encoding.NEXT_ACTIVITY_TARGET])
    )

    # Build the central next-activity evaluation mask, so validation and test score over the pooled joint class set.
    evaluation_next_activity_mask_context = build_evaluation_next_activity_mask_context(
        config, training_next_activity_mask_context, spec, vocabularies, scalers, schema_profile, mapping
    )

    # Build deterministic train, validation and test loaders over the resolved prefix datasets.
    with _timed(timings, "build_loaders"):
        train_loader = make_loader(
            train_dataset, config.batch_size, True, config.seed, config.num_workers, training_next_activity_mask_context
        )
        val_loader = make_loader(
            val_dataset, config.batch_size, False, config.seed, config.num_workers,
            evaluation_next_activity_mask_context
        )
        test_loader = make_loader(
            test_dataset, config.batch_size, False, config.seed, config.num_workers,
            evaluation_next_activity_mask_context
        )

    # Build the multitask model and train outcome loss on the selected device.
    with _timed(timings, "build_model"):
        model = build_model(config, vocabularies, schema_profile).to(device)
        outcome_loss = build_outcome_loss_from_labels(config, training_outcome_labels_from_dataset(train_dataset),
                                                      device)

    # Construct the remaining loss objects and scalar Huber setting for both training and evaluation.
    next_activity_loss = nn.CrossEntropyLoss(reduction="none")
    huber_beta = float(config.remaining_time_huber_beta)

    # Load the E_04 RT representation and initialize the head bias.
    with _timed(timings, "load_remaining_time_repr"):
        remaining_time_repr = remaining_time_repr_from_spec(spec, config)
        target_diagnostics = remaining_time_target_diagnostics(train_dataset, remaining_time_repr)
        target_diagnostics["remaining_time_huber_beta"] = huber_beta
        target_diagnostics["outcome_head_dropout"] = (
            float(config.outcome_head_dropout) if config.outcome_head_dropout is not None else float(config.dropout)
        )
        initialized_model_units = core.initialize_remaining_time_head_bias(model, remaining_time_repr)
        target_diagnostics["remaining_time_head_bias_initialized_model_units"] = initialized_model_units

    # Build one AdamW over trunk and task head groups with the shared E_05/E_06 helper.
    optimizer = core.build_multitask_optimizer(
        model, config.learning_rate, config.weight_decay, config.outcome_lr_scale, config.next_activity_lr_scale,
        config.remaining_time_lr_scale
    )

    # Cosine LR decay over LR_SCHEDULER_T_MAX, then the step guard holds the floor LR.
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    if config.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(int(config.lr_scheduler_t_max), 1), eta_min=float(config.lr_scheduler_min_lr),
        )
    else: scheduler = None

    # Track the best validation loss checkpoint while retaining diagnostics per epoch.
    best_state = copy.deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    best_loss = float("inf")
    wait = 0
    history: list[dict[str, Any]] = []
    epoch_iter = progress_iter(
        range(1, config.max_epochs + 1), config,f"epochs {config.dataset} {config.run_name}",
        total=config.max_epochs, leave=True, unit="epoch"
    )
    with _timed(timings, "train_epochs"):
        for epoch in epoch_iter:
            # Train one epoch on the active prefix population.
            train_metrics = train_epoch(
                model, train_loader, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta,
                optimizer, device, config, progress_label=f"train epoch {epoch}",
                next_activity_mask_context=training_next_activity_mask_context,
            )

            # Validation per epoch: Metrics only, no prediction frame is built and discarded.
            val_metrics, _ = evaluate_model(
                model, val_loader, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta, device,
                config, val_dataset, progress_label=f"val epoch {epoch}", collect_predictions=False,
                next_activity_mask_context=evaluation_next_activity_mask_context,
            )

            # Show this epoch's val loss and outcome macro-F1 on the live progress bar when one is active.
            if hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(
                    {
                        "val_loss": f"{float(val_metrics['loss_total']):.4f}",
                        "macro_f1": f"{float(val_metrics['outcome']['macro_f1']):.3f}",
                    }
                )

            # Append one complete train log row for curves and diagnostics.
            epoch_record = flatten_epoch_metrics(epoch, train_metrics, val_metrics)

            # Record the active trunk and LRs per head by group name.
            group_lr_by_name = {str(group.get("group_name", "")): float(group["lr"])
                                for group in optimizer.param_groups}
            epoch_record["learning_rate"] = group_lr_by_name.get("trunk", float(optimizer.param_groups[0]["lr"]))
            epoch_record["learning_rate_outcome"] = group_lr_by_name.get("outcome", epoch_record["learning_rate"])
            epoch_record["learning_rate_next_activity"] = group_lr_by_name.get(
                "next_activity", epoch_record["learning_rate"])
            epoch_record["learning_rate_remaining_time"] = group_lr_by_name.get(
                "remaining_time", epoch_record["learning_rate"])
            history.append(epoch_record)

            # Early stopping on total validation loss with best state restoration.
            val_loss = float(val_metrics["loss_total"])
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                best_metrics = val_metrics
                wait = 0
            else:
                wait += 1
                if wait >= config.early_stopping_patience:
                    log.info("Early stopping at epoch %s", epoch)
                    break
            # Step only, while cosine decay is running, so LR holds the floor beyond LR_SCHEDULER_T_MAX.
            if scheduler is not None and epoch <= int(config.lr_scheduler_t_max): scheduler.step()

    # Restore the best validation loss checkpoint before final evaluation.
    model.load_state_dict(best_state)

    # Evaluate final validation with prediction export for traceability.
    with _timed(timings, "final_validation"):
        val_metrics, val_predictions = evaluate_model(
            model, val_loader, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta, device, config,
            val_dataset, progress_label="final val", collect_predictions=True,
            next_activity_mask_context=evaluation_next_activity_mask_context,
        )

    # Resolve the per-bank prefix subsets of the pooled test split, so fairness metrics need no second forward pass.
    per_bank: dict[str, Any] = {}
    bank_selectors: dict[str, np.ndarray] = {}
    if config.regime == "centralized":
        with _timed(timings, "resolve_bank_selectors"):
            bank_selectors = bank_selectors_for_dataset(
                test_dataset, bank_names_for_config(config), bank_by_case_for_split(config, mapping, "test")
            )

    # Evaluate the final test with prediction export for tables and diagnostics.
    with _timed(timings, "final_test"):
        test_metrics, test_predictions = evaluate_model(
            model, test_loader, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta, device, config,
            test_dataset, progress_label="test", collect_predictions=True,
            next_activity_mask_context=evaluation_next_activity_mask_context,
            subset_selectors=bank_selectors or None, subset_metrics=per_bank,
        )

    # Final evaluations must return prediction frames because downstream artifacts depend on them, else throw.
    if val_predictions is None or test_predictions is None:
        raise RuntimeError("final evaluation must collect prediction frames")

    # Write every artifact needed for reporting immediately after training.
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_artifact_path(output_dir, "train_log.csv").parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(prediction_artifact_path(output_dir, "train_log.csv"), index=False)
    val_predictions.to_parquet(prediction_artifact_path(output_dir, "predictions_val.parquet"), index=False)
    test_predictions.to_parquet(prediction_artifact_path(output_dir, "predictions_test.parquet"), index=False)
    prefix_bucket_metrics_val = compute_prefix_bucket_metrics(val_predictions)
    prefix_bucket_metrics_test = compute_prefix_bucket_metrics(test_predictions)
    remaining_time_baselines_val = remaining_time_baseline_diagnostics(
        val_predictions,
        target_diagnostics["train_median_seconds"],
    )
    remaining_time_baselines_test = remaining_time_baseline_diagnostics(
        test_predictions,
        target_diagnostics["train_median_seconds"],
    )
    save_legacy_json(config, run_artifact_path(output_dir, "validation_metrics.json"), val_metrics)
    save_legacy_json(config, run_artifact_path(output_dir, "test_metrics.json"), test_metrics)
    save_legacy_json(config, run_artifact_path(output_dir, "target_diagnostics.json"), target_diagnostics)
    save_legacy_json(config, run_artifact_path(output_dir, "prefix_bucket_metrics_val.json"), prefix_bucket_metrics_val)
    save_legacy_json(config, run_artifact_path(output_dir, "prefix_bucket_metrics_test.json"),
                     prefix_bucket_metrics_test)
    # Flat thesis-facing early-to-late prefix bucket table.
    write_prefix_bucket_summary_csv(output_dir, test_predictions)
    save_legacy_json(config, run_artifact_path(output_dir, "remaining_time_baselines_val.json"),
                     remaining_time_baselines_val)
    save_legacy_json(config, run_artifact_path(output_dir, "remaining_time_baselines_test.json"),
                     remaining_time_baselines_test)

    # Centralized runs export the per-bank slices of the pooled test split for the client fairness comparison.
    per_bank_prefix_bucket_metrics: dict[str, Any] = {}
    if config.regime == "centralized":
        export_per_bank_test_artifacts(
            output_dir, test_predictions, bank_selectors, per_bank_prefix_bucket_metrics
        )
        save_legacy_json(config, run_artifact_path(output_dir, "per_bank_test_metrics.json"), per_bank)
        write_centralized_summary(output_dir, test_metrics, per_bank)

    best_epoch_payload = best_epoch_diagnostics(pd.DataFrame(history))

    # Store the best model checkpoint with the resolved run config and best epoch analysis.
    torch.save(
        {
            "model_state_dict": best_state,
            "config": _config_payload(config, resolved_device=str(device)),
            "best_validation_loss": best_loss,
            "best_epoch_diagnostics": best_epoch_payload,
        },
        run_artifact_path(output_dir, "model_best.pt"),
    )
    save_legacy_json(config, run_artifact_path(output_dir, "best_epoch_diagnostics.json"), best_epoch_payload)
    plot_training_curves(pd.DataFrame(history), output_dir)

    # Write coarse phase timings and cache hit flags for runtime comparison.
    timing_payload = {
        **{f"{name}_seconds": float(seconds) for name, seconds in sorted(timings.items())},
        "cache_train_hit": bool(cache_hits["train"]),
        "cache_val_hit": bool(cache_hits["val"]),
        "cache_test_hit": bool(cache_hits["test"]),
    }
    save_legacy_json(config, run_artifact_path(output_dir, "timing.json"), timing_payload)

    diagnostics_payload: dict[str, Any] = {
        "target": target_diagnostics,
        "best_epoch": best_epoch_payload,
        "prefix_buckets": {
            "validation": prefix_bucket_metrics_val,
            "test": prefix_bucket_metrics_test,
            "test_by_bank": per_bank_prefix_bucket_metrics,
        },
        "remaining_time_baselines": {
            "validation": remaining_time_baselines_val,
            "test": remaining_time_baselines_test,
        },
        "curves": {
            "files": [
                script_artifact_name("loss_curves.png"),
                script_artifact_name("task_loss_curves.png"),
                script_artifact_name("outcome_macro_f1_curve.png"),
                script_artifact_name("next_activity_accuracy_curve.png"),
                script_artifact_name("remaining_time_mae_curve.png"),
            ]
        },
        "predictions": {
            "validation": (Path(PREDICTIONS_DIR_NAME) / script_artifact_name("predictions_val.parquet")).as_posix(),
            "test": (Path(PREDICTIONS_DIR_NAME) / script_artifact_name("predictions_test.parquet")).as_posix(),
        },
        "timing": timing_payload,
    }
    metrics_payload: dict[str, Any] = {
        "validation": val_metrics,
        "test": test_metrics,
        "per_bank_test": per_bank,
    }
    artifact_manifest = training_reporting.build_artifact_manifest(output_dir)
    artifact_manifest["files"] = sorted(set(artifact_manifest["files"] + [script_artifact_name("run_report.json")]))
    artifact_manifest["file_count"] = len(artifact_manifest["files"])
    report = training_reporting.build_run_report(
        SCRIPT_ID,
        _config_payload(config, resolved_device=str(device)),
        metrics_payload,
        diagnostics_payload,
        artifact_manifest,
    )
    training_reporting.write_run_report(output_dir, SCRIPT_ID, report)

    # Local runs join the shared local summary from the run reports, so this run's own report must exist first.
    if config.regime == "local": write_local_summary_if_complete(config)

    # Return the compact run payload used by tests and future orchestration.
    return {
        "best_validation_loss": best_loss,
        "best_validation_metrics": best_metrics,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "target_diagnostics": target_diagnostics,
        "batch_size": config.batch_size,
        "timing": timing_payload,
    }

# Export the per-bank slices of the pooled test predictions for the client fairness comparison.
# The slices come from the one pooled forward pass, so every per-bank table decomposes the global table exactly.
def export_per_bank_test_artifacts(output_dir: Path, test_predictions: pd.DataFrame,
                                   bank_selectors: dict[str, np.ndarray],
                                   per_bank_prefix_bucket_metrics: dict[str, Any]) -> None:

    # Write one prediction parquet and one prefix-bucket block per simulated bank.
    prediction_artifact_path(output_dir, "predictions_test.parquet").parent.mkdir(parents=True, exist_ok=True)
    for bank, selector in bank_selectors.items():
        bank_predictions = test_predictions.loc[selector]
        safe_bank = _safe_bank(bank)
        bank_predictions.to_parquet(
            prediction_artifact_path(output_dir, f"predictions_test_bank_{safe_bank}.parquet"), index=False)
        per_bank_prefix_bucket_metrics[bank] = compute_prefix_bucket_metrics(bank_predictions)

# Flatten one nested metric block into one summary row, including the columns per class.
def flatten_local_summary_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    row: dict[str, float] = {}

    # Copy top-level scalar metrics first.
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)): row[key] = float(value)

    # Flatten the outcome block into columns: scalars as outcome_<metric>, values as outcome_class_<label>_<metric>.
    outcome = metrics.get("outcome", {})
    if isinstance(outcome, dict):
        for key, value in outcome.items():
            if isinstance(value, (int, float, np.integer, np.floating)): row[f"outcome_{key}"] = float(value)
        per_class = outcome.get("per_class", {})
        if isinstance(per_class, dict):
            for class_label, class_metrics in per_class.items():
                if not isinstance(class_metrics, dict): continue
                for key, value in class_metrics.items():
                    if isinstance(value, (int, float, np.integer, np.floating)):
                        row[f"outcome_class_{class_label}_{key}"] = float(value)

    # Flatten the next activity and RT scalar sections.
    for section_name in ("next_activity", "remaining_time"):
        section = metrics.get(section_name, {})
        if not isinstance(section, dict): continue
        for key, value in section.items():
            if isinstance(value, (int, float, np.integer, np.floating)): row[f"{section_name}_{key}"] = float(value)
    return row

# Write the local bank table with prefix weighted and unweighted averages once every bank run exists.
def write_local_summary_if_complete(config: BaselineRunConfig) -> None:

    # One summary row per bank from its run report and predictions, written once every bank run finished.
    banks = bank_names_for_config(config)
    rows: list[dict[str, Any]] = []
    for bank in banks:
        bank_config = BaselineRunConfig(**{**asdict(config), "bank": bank, "output_dir_override": None})
        bank_output_dir = output_dir_for_run(config.output_root, bank_config)
        pred_path = prediction_artifact_path(bank_output_dir, "predictions_test.parquet")
        metrics = metrics_block_from_run_report(bank_output_dir, "test")
        if not pred_path.exists() or not metrics: return
        row: dict[str, Any] = {"bank": bank, "n_prefixes": int(len(pd.read_parquet(pred_path, columns=["case_id"])))}
        row.update(flatten_local_summary_metrics(metrics))
        rows.append(row)

    # Identify scalar metric columns and support columns before computing aggregate rows.
    frame = pd.DataFrame(rows)
    metric_columns = [column for column in frame.columns if column not in {"bank", "n_prefixes"}]
    support_columns = [column for column in metric_columns if column.endswith("_support")]
    unweighted: dict[str, Any] = {"bank": "unweighted_average", "n_prefixes": int(frame["n_prefixes"].sum())}
    weighted: dict[str, Any] = {"bank": "prefix_weighted_average", "n_prefixes": int(frame["n_prefixes"].sum())}
    weights = frame["n_prefixes"].to_numpy(dtype=float)

    # Average metric columns and sum support columns for reporting.
    for column in metric_columns:
        values = frame[column].to_numpy(dtype=float)
        if column in support_columns:
            unweighted[column] = float(values.sum())
            weighted[column] = float(values.sum())
        else:
            unweighted[column] = float(frame[column].mean())
            weighted[column] = float(np.average(values, weights=weights))

    # Write the rows per bank plus unweighted and prefix-weighted aggregate rows.
    summary = pd.concat([frame, pd.DataFrame([unweighted, weighted])], ignore_index=True)
    csv_path, json_path = local_summary_paths(config)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    save_json(json_path, {"rows": summary.to_dict(orient="records")})

# Train 1 epoch: one full pass over the split with forward, masked multitask loss and clipped AdamW step per batch.
def train_epoch(model: core.MultitaskLSTM, loader: DataLoader, outcome_loss: nn.Module, next_activity_loss: nn.Module,
    remaining_time_repr: core.RemainingTimeRepr, huber_beta: float, optimizer: torch.optim.Optimizer,
    device: torch.device, config: BaselineRunConfig, progress_label: str = "train",
    next_activity_mask_context: Optional[core.NextActivityMaskContext] = None) -> dict[str, float]:

    # Switch the model to training mode before iterating over prefix batches.
    model.train()

    # Loss accumulation per sample like evaluate_model.
    loss_sums = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    loss_counts = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    gradient_norm_sum = 0.0
    n_batches = 0
    total_start = time.perf_counter()
    batch_iter = progress_iter(loader, config, f"{progress_label} batches", total=len(loader), unit="batch")

    # Train over all batches of the current split once.
    for batch in batch_iter:
        batch = core.move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)

        # Compute the masked multitask objective in model units.
        losses = core.compute_multitask_loss(
            outputs, batch, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta,
            config.outcome_loss_weight, config.next_activity_loss_weight, config.remaining_time_loss_weight,
            next_activity_mask_context=next_activity_mask_context,
        )
        losses.total.backward()

        # Clip gradients before the AdamW step to match the E_05 training decision.
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        optimizer.step()

        # Accumulate exact loss means by task per sample.
        batch_n = float(batch.outcome_label.shape[0])
        next_n = float(batch.next_activity_mask.float().sum().item())
        remaining_n = float(batch.remaining_time_mask.float().sum().item())
        loss_sums["outcome"] += float(losses.outcome.item()) * batch_n
        loss_sums["next_activity"] += float(losses.next_activity.item()) * next_n
        loss_sums["remaining_time"] += float(losses.remaining_time.item()) * remaining_n
        loss_counts["outcome"] += batch_n
        loss_counts["next_activity"] += next_n
        loss_counts["remaining_time"] += remaining_n
        gradient_norm_sum += float(gradient_norm.item() if hasattr(gradient_norm, "item") else gradient_norm)
        n_batches += 1
        if hasattr(batch_iter, "set_postfix"): batch_iter.set_postfix({"loss": f"{float(losses.total.item()):.4f}"})

    # Rebuild the weighted total loss from the three means per task.
    loss_means = {key: loss_sums[key] / max(loss_counts[key], 1.0) for key in loss_sums}
    loss_total = (
        config.outcome_loss_weight * loss_means["outcome"]
        + config.next_activity_loss_weight * loss_means["next_activity"]
        + config.remaining_time_loss_weight * loss_means["remaining_time"]
    )

    # Return one compact epoch metric block for the train log.
    return {
        "loss_total": loss_total,
        "loss_outcome": loss_means["outcome"],
        "loss_next_activity": loss_means["next_activity"],
        "loss_remaining_time": loss_means["remaining_time"],
        "gradient_norm": gradient_norm_sum / max(n_batches, 1),
        "timing_total_seconds": time.perf_counter() - total_start,
    }

# Flatten one epoch into one train log row for the CSV and the curve plots.
def flatten_epoch_metrics(epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "train_loss_total": train_metrics["loss_total"],
        "train_loss_outcome": train_metrics["loss_outcome"],
        "train_loss_next_activity": train_metrics["loss_next_activity"],
        "train_loss_remaining_time": train_metrics["loss_remaining_time"],
        "train_gradient_norm": train_metrics.get("gradient_norm", 0.0),
        "train_seconds": train_metrics.get("timing_total_seconds", 0.0),
        "val_loss_total": val_metrics["loss_total"],
        "val_loss_outcome": val_metrics["loss_outcome"],
        "val_loss_next_activity": val_metrics["loss_next_activity"],
        "val_loss_remaining_time": val_metrics["loss_remaining_time"],
        "val_seconds": val_metrics.get("timing_total_seconds", 0.0),
        "val_outcome_macro_f1": val_metrics["outcome"]["macro_f1"],
        "val_next_activity_top1": val_metrics["next_activity"]["top1_accuracy"],
        "val_remaining_time_mae": val_metrics["remaining_time"]["mae"],
    }

# Record per head the best epochs for analysis. Checkpoint selection is done on val_loss_total only.
def best_epoch_diagnostics(history: pd.DataFrame) -> dict[str, Any]:
    # Return an empty payload when no epoch history was produced.
    if history.empty: return {}

    # Define which metric is minimized or maximized for each diagnostic entry.
    diagnostics: dict[str, Any] = {}
    specs = {
        "best_total_validation_loss": ("val_loss_total", True),
        "best_outcome_macro_f1": ("val_outcome_macro_f1", False),
        "best_remaining_time_mae": ("val_remaining_time_mae", True),
    }

    # Extract the epoch and value for each diagnostic metric that exists in the train log.
    for name, (column, minimize) in specs.items():
        if column not in history: continue
        index = history[column].idxmin() if minimize else history[column].idxmax()
        row = history.loc[index]
        diagnostics[name] = {"epoch": int(row["epoch"]), "metric": column, "value": float(row[column])}
    return diagnostics

# HELPER: Apply the thesis grid, margin and spine styling to one axes object (matches the thesis plot scripts).
def style_non_pie_axes(ax: plt.Axes, x_margin: float = 0.04) -> None:
    ax.grid(False)
    ax.grid(axis="x", linestyle="-", linewidth=0.6, color=GRID_COLOR, alpha=0.45)
    ax.grid(axis="y", linestyle=":", linewidth=0.8, color=GRID_COLOR, alpha=0.75)
    ax.set_axisbelow(True)
    ax.margins(x=x_margin)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.8)

# HELPER: Apply the thesis legend frame styling (keep styling consistent with the generated BPIC summary plots).
def style_training_legend(ax: plt.Axes) -> None:
    legend = ax.legend(loc="best", frameon=True)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_alpha(0.78)
    frame.set_linewidth(0.8)

# PLOT: Draw the total loss, loss per head and metric curves per head over epochs or rounds.
def plot_training_curves(history: pd.DataFrame, output_dir: Path,
                         artifact_path_fn: Callable[[Path, str], Path] = run_artifact_path,
                         x_label: str = "Epoch") -> None:
    # Skip plot generation for failed or empty training histories.
    if history.empty: return

    # Keep one figure per metric family, so I can reference curves individually.
    plot_specs = [
        ("loss_curves.png", ["train_loss_total", "val_loss_total"], "Loss"),
        ("task_loss_curves.png", ["val_loss_outcome", "val_loss_next_activity", "val_loss_remaining_time"], "Loss"),
        ("outcome_macro_f1_curve.png", ["val_outcome_macro_f1"], "Macro-F1"),
        ("next_activity_accuracy_curve.png", ["val_next_activity_top1"], "Top-1 Accuracy"),
        ("remaining_time_mae_curve.png", ["val_remaining_time_mae"], "MAE Seconds"),
    ]

    # Draw one compact line chart for each metric family.
    for filename, columns, ylabel in plot_specs:
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="white")
        for index, column in enumerate(columns):
            if column in history:
                color = TRAINING_CURVE_COLORS[index % len(TRAINING_CURVE_COLORS)]
                ax.plot(
                    history["epoch"], history[column], marker="o", markersize=4, linewidth=1.6, color=color,
                    markeredgecolor="white", markeredgewidth=0.6, label=column
                )
                if not column.startswith("train_"):
                    series = history[column].dropna()
                    if not series.empty:
                        best_index = series.idxmin() if "loss" in column or "mae" in column else series.idxmax()
                        best_row = history.loc[best_index]
                        ax.plot(
                            best_row["epoch"], best_row[column], marker="o", markersize=8.0,
                            markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4, zorder=6,
                            linestyle="none"
                        )

        # Apply the thesis plot styling and export the figure.
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.tick_params(colors=TEXT_DARK)
        style_non_pie_axes(ax)
        style_training_legend(ax)
        fig.tight_layout(pad=1.2)
        fig.savefig(artifact_path_fn(output_dir, filename), dpi=300)
        plt.close(fig)

# Store the complete resolved run configuration for later analysis.
def _config_payload(config: BaselineRunConfig, resolved_device: Optional[str] = None) -> dict[str, Any]:
    # Start from the dataclass fields so all workflow and CLI overrides are recorded.
    payload = asdict(config)
    payload["run_name"] = config.run_name

    # Record the actual torch device when the caller has already resolved it.
    if resolved_device is not None: payload["resolved_device"] = resolved_device
    return payload

# ----------------------------------------------------------------------------------------------------------------------
# 6. ENTRY POINT

# Resolve the run directory, persist the config payload and execute one baseline run.
def run_one_baseline(config: BaselineRunConfig) -> dict[str, Any]:

    # Resolve and create the output directory before any run artifact is written.
    output_dir = output_dir_for_run(config.output_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve the device once for the training log line and optional debug config artifact.
    resolved_device = core.select_device(config.device)
    save_legacy_json(config, run_artifact_path(output_dir, "config.json"),
                     _config_payload(config, resolved_device=str(resolved_device)))
    log.info("Training %s %s with batch size %s on %s", config.dataset, config.run_name, config.batch_size,
             resolved_device)

    # Execute the full baseline pipeline for this single configuration.
    return run_baseline_training(config, output_dir)

# MAIN (this is where things happen): one configured baseline run per invocation. The workflow script drives the matrix.
def main(argv: Optional[list[str]] = None) -> None:
    _configure_logging()
    config = config_from_args(parse_args(argv))
    run_one_baseline(config)

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────