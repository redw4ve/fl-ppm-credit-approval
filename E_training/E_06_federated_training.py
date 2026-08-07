"""
Step 6: Train one federated multitask LSTM run using the E_05 model and E_04 prefix caches.

Pipeline:
- Loads the approved E_04 encoding metadata through the E_05 baseline helpers
- Builds one simulated bank client per partition and one pooled validation and test context
- Trains FedAvg or FedProx with full client participation in a synchronous Flower loop
- Applies the final E_05 workflow hyperparameters and one E_04 RemainingTimeRepr
- Restores the best global checkpoint by pooled val_loss_total and exports E_06 artifacts

REQUIRED FILES:
    E_prefix_encoding/encoded_metadata/*/*/*_encoding_spec.json: frozen E_04 run config and prefix cap
    E_prefix_encoding/encoded_metadata/*/*/*_vocabulary.json: train only categorical token indices
    E_prefix_encoding/encoded_metadata/*/*/*_scaler.json: train only numeric means and standard deviations
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: approved schema profiles
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved dataset mapping
    E_main_BPIC_2017/data/processed/**/*.parquet: BPIC 2017 split parquets
    E_ablation_BPIC_2012/data/processed/**/*.parquet: BPIC 2012 split parquets

CREATED FILES (compact profile, per run directory <OUTPUT_ROOT>/federated/<dataset>_<run>/<strategy>_*):
    E_06_run_report.json: Consolidated configuration, metrics, diagnostics, timing, rounds and artifact manifest
    E_06_round_log.csv: Round-level train, validation and learning-rate diagnostics
    E_06_model_best.pt: Best global checkpoint with model state, config payload and diagnostics
    E_06_loss_curves.png: Total train and validation loss curves
    E_06_task_loss_curves.png: Validation loss curves per task head
    E_06_outcome_macro_f1_curve.png: Validation outcome macro-F1 curve
    E_06_next_activity_accuracy_curve.png: Validation next-activity top-1 accuracy curve
    E_06_remaining_time_mae_curve.png: Validation remaining-time MAE curve
    E_06_federated_summary_test.{csv,json}: Flat global and client test summary
    predictions/E_06_predictions_{val,test}.parquet: Predictions per prefix with case provenance
    predictions/E_06_predictions_test_bank_<X>.parquet: Test predictions per bank
"""

# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import importlib
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union, cast
from flwr.client import NumPyClient
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg, FedProx
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Make direct script execution imports work.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the baseline owner so E_06 reuses E_05 configuration and artifact conventions.
from E_training import E_05_central_and_local_baselines_final
from E_training import training_core_final as core
from E_training import training_reporting

baseline = E_05_central_and_local_baselines_final

# HYPERPARAMETERS AND CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------

# Restrict federated training to the two approved aggregation strategies.
FederatedStrategy = Literal["fedavg", "fedprox"]
NextActivityHeadAgg = Literal["sample", "equal", "contribution"]
NextActivityContributionPayload = dict[int, np.ndarray]

# Reuse the E_05 artifact roots without touching frozen E_04 files.
SCRIPT_ID: str = "E_06"  # artifact filename prefix
PREDICTIONS_DIR_NAME: str = "predictions"  # prediction and log subfolder
SCRIPT_DIR: Path = Path(__file__).resolve().parent  # E_training folder
REPO_ROOT: Path = SCRIPT_DIR.parent  # repository root
ARTIFACT_ROOT: Path = baseline.ARTIFACT_ROOT  # E_04 metadata root
CANONICAL_SCHEMA_PATH: Path = baseline.CANONICAL_SCHEMA_PATH  # approved schema input
DATASET_MAPPING_PATH: Path = baseline.DATASET_MAPPING_PATH  # approved mapping input
OUTPUT_ROOT: Path = SCRIPT_DIR / "training_outputs"  # final thesis output root
CACHE_ROOT: Path = SCRIPT_DIR / "prefix_tensor_cache"  # persistent tensor cache root

# Select the dataset, partition configuration and reproducibility controls.
DATASET: str = os.environ.get("DATASET", "bpic2017")  # bpic2017 | bpic2012
HETEROGENEITY: str = os.environ.get("HETEROGENEITY", "medium")  # iid | weak | medium | strong
N_CLIENTS: int = int(os.environ.get("N_CLIENTS", "3"))  # 3 or 5 simulated banks
SEED: int = int(os.environ.get("SEED", "42"))  # fixed thesis seed

# Configure the federated protocol, with LOCAL_EPOCHS=1 as the production setting.
_STRATEGY_ENV: str = os.environ.get("STRATEGY", "fedprox")  # raw strategy override
STRATEGY: FederatedStrategy = cast(FederatedStrategy, _STRATEGY_ENV)  # fedavg | fedprox

# Configure the joint next-activity head aggregation.
# The mode is sample, equal or contribution. The federated workflow selects equal for joint runs.
_NEXT_ACTIVITY_HEAD_AGG_ENV: str = os.environ.get("NEXT_ACTIVITY_HEAD_AGG", "sample").lower()
NEXT_ACTIVITY_HEAD_AGG: NextActivityHeadAgg = cast(NextActivityHeadAgg, _NEXT_ACTIVITY_HEAD_AGG_ENV)

# Configure the off-by-default secure aggregation simulation.
_SECURE_AGGREGATION_SIMULATION_ENV: str = os.environ.get("SECURE_AGGREGATION_SIMULATION", "false").lower()
SECURE_AGGREGATION_SIMULATION: bool = _SECURE_AGGREGATION_SIMULATION_ENV in {"1", "true", "yes"}  # pairwise masking
SECURE_AGGREGATION_SEED: int = int(os.environ.get("SECURE_AGGREGATION_SEED", "42"))  # mask seed
SECURE_AGGREGATION_TOLERANCE: float = 1e-6  # reconstr. tolerance

# Bound every pairwise mask by one public constant, because both pair members must draw the identical array.
# It sits above any weighted contribution and keeps the float64 cancellation near machine epsilon.
SECURE_AGGREGATION_MASK_BOUND: float = 1.0e6

# Give every masked channel its own generator index, so two channels of one tensor never share a mask.
SECURE_AGGREGATION_CHANNELS: dict[str, int] = {
    "contribution": 0, "weight": 1, "numerator": 2, "denominator": 3, "sample_contribution": 4, "sample_weight": 5,
}

# Fix the channel set each masked message kind must carry, so the server helper can reject anything else.
SECURE_AGGREGATION_MESSAGE_CHANNELS: dict[str, frozenset[str]] = {
    "weighted": frozenset({"contribution", "weight"}),
    "contribution": frozenset({"numerator", "denominator", "sample_contribution", "sample_weight"}),
}

# Configure the epochs, rounds, fedprox_mu and early stopping patience.
MAX_ROUNDS: int = int(os.environ.get("MAX_ROUNDS", "40"))  # uniform federated budget
LOCAL_EPOCHS: int = int(os.environ.get("LOCAL_EPOCHS", "1"))  # local epochs per round
FEDPROX_MU: float = float(os.environ.get("FEDPROX_MU", "1e-4"))  # fixed production FedProx mu
EARLY_STOPPING_PATIENCE: int = int(os.environ.get("EARLY_STOPPING_PATIENCE", "7"))  # uniform patience

# Configure the optional DP-SGD path, which is inactive unless USE_DP=true.
_USE_DP_ENV: str = os.environ.get("USE_DP", "false").lower()  # raw DP switch override
USE_DP: bool = _USE_DP_ENV in {"1", "true", "yes"}  # DP-SGD switch
DP_TARGET_EPSILON: float = float(os.environ.get("DP_TARGET_EPSILON", "10.0"))  # target epsilon per client
DP_DELTA: float = float(os.environ.get("DP_DELTA", "1e-6"))  # target delta per client
DP_MAX_GRAD_NORM: float = float(os.environ.get("DP_MAX_GRAD_NORM", "1.0"))  # clipping norm per sample
_DP_SMOKE_MAX_BATCHES_ENV: str = os.environ.get("DP_SMOKE_MAX_BATCHES", "")  # optional DP batch cap
DP_SMOKE_MAX_BATCHES: Optional[int] = (
    int(_DP_SMOKE_MAX_BATCHES_ENV) if _DP_SMOKE_MAX_BATCHES_ENV else None
)  # capped DP batches or None
_DP_WARNING_FILTERS_CONFIGURED: bool = False  # DP warning filter state.

# Configure execution and local data loading.
DEVICE: str = os.environ.get("DEVICE", "auto")  # auto | mps | cuda | cpu
_PROGRESS_BARS_ENV: str = os.environ.get("PROGRESS_BARS", "true").lower()  # raw progress override
PROGRESS_BARS: bool = _PROGRESS_BARS_ENV not in {"0", "false", "no"}  # show progress bars
REPORTING_PROFILE: str = training_reporting.normalize_reporting_profile(os.environ.get("REPORTING_PROFILE", "compact"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "512"))  # prefix samples per batch
NUM_WORKERS: int = int(os.environ.get("NUM_WORKERS", "0"))  # 0 is stable on macOS

# Configure optimizer and learning rate schedule from the final E_05 workflow.
LEARNING_RATE: float = float(os.environ.get("LEARNING_RATE", "2.5e-4"))  # base AdamW LR
WEIGHT_DECAY: float = float(os.environ.get("WEIGHT_DECAY", "1e-4"))  # AdamW weight decay
GRADIENT_CLIP_NORM: float = float(os.environ.get("GRADIENT_CLIP_NORM", "1.0"))  # no-DP global clip norm
_LR_SCHEDULER_ENV: str = os.environ.get("LR_SCHEDULER", "cosine").lower()  # raw scheduler override
LR_SCHEDULER: Literal["cosine", "none"] = cast(
    Literal["cosine", "none"], _LR_SCHEDULER_ENV
)  # cosine | none
LR_SCHEDULER_MIN_LR: float = float(os.environ.get("LR_SCHEDULER_MIN_LR", "1e-6"))  # cosine LR floor
LR_SCHEDULER_T_MAX: int = int(os.environ.get("LR_SCHEDULER_T_MAX", "15"))  # cosine period in rounds

# Configure the shared E_05 multitask LSTM architecture.
HIDDEN_SIZE: int = int(os.environ.get("HIDDEN_SIZE", "128"))  # LSTM hidden state size
NUM_LAYERS: int = int(os.environ.get("NUM_LAYERS", "2"))  # stacked LSTM layers
DROPOUT: float = float(os.environ.get("DROPOUT", "0.30"))  # shared trunk dropout
HEAD_HIDDEN_SIZE: int = int(os.environ.get("HEAD_HIDDEN_SIZE", "64"))  # task head hidden size

# Configure outcome regularization from the final baseline workflow.
OUTCOME_LABEL_SMOOTHING: float = float(os.environ.get("OUTCOME_LABEL_SMOOTHING", "0.10"))  # final CE smoothing
OUTCOME_CLASS_WEIGHT_POWER: float = float(os.environ.get("OUTCOME_CLASS_WEIGHT_POWER", "0.5"))  # class-weight power
_OUTCOME_HEAD_DROPOUT_ENV: str = os.environ.get("OUTCOME_HEAD_DROPOUT", "0.45")  # empty = trunk dropout
OUTCOME_HEAD_DROPOUT: Optional[float] = (  # outcome dropout
    float(_OUTCOME_HEAD_DROPOUT_ENV) if _OUTCOME_HEAD_DROPOUT_ENV else None
)

# Configure AdamW learning rate multipliers per task head.
OUTCOME_LR_SCALE: float = float(os.environ.get("OUTCOME_LR_SCALE", "0.3"))  # outcome head LR multiplier
NEXT_ACTIVITY_LR_SCALE: float = float(os.environ.get("NEXT_ACTIVITY_LR_SCALE", "1.0"))  # next head LR multiplier
REMAINING_TIME_LR_SCALE: float = float(os.environ.get("REMAINING_TIME_LR_SCALE", "1.0"))  # RT head LR multiplier

# Configure the fixed multitask loss contribution weights.
OUTCOME_LOSS_WEIGHT: float = float(os.environ.get("OUTCOME_LOSS_WEIGHT", "1.0"))  # OC loss contribution
NEXT_ACTIVITY_LOSS_WEIGHT: float = float(os.environ.get("NEXT_ACTIVITY_LOSS_WEIGHT", "0.5"))  # NA loss contribution
REMAINING_TIME_LOSS_WEIGHT: float = float(os.environ.get("REMAINING_TIME_LOSS_WEIGHT", "0.5"))  # RT loss contribution

# Validate the E_04 RemainingTimeRepr used by all clients.
_REMAINING_TIME_TRANSFORM_ENV: str = os.environ.get("REMAINING_TIME_TRANSFORM", "raw").lower()  # Transform
REMAINING_TIME_TRANSFORM: Literal["raw", "log"] = cast(Literal["raw", "log"],
                                                       _REMAINING_TIME_TRANSFORM_ENV)  # raw | log
_REMAINING_TIME_SCALING_ENV: str = os.environ.get("REMAINING_TIME_SCALING", "zscore").lower()  # Scaling
REMAINING_TIME_HUBER_BETA: float = float(os.environ.get("REMAINING_TIME_HUBER_BETA", "0.1"))  # Huber beta
REMAINING_TIME_SCALING: Literal["raw", "median", "zscore"] = cast(
    Literal["raw", "median", "zscore"], _REMAINING_TIME_SCALING_ENV  # raw | median | zscore
)


# HELPER: Build a dense RDP alpha grid for Opacus privacy accounting.
def build_dp_rdp_alphas() -> tuple[float, ...]:
    # Cover values close to one so loose privacy budgets do not hit the lower alpha boundary.
    low_orders = [1.0 + step / 100.0 for step in range(1, 100)]

    # Cover the middle range densely because it often contains the optimum for BPIC-sized clients.
    middle_orders = [2.0 + step / 10.0 for step in range(0, 100)]

    # Extend high orders so strict budgets do not hit the upper alpha boundary.
    high_orders = [float(order) for order in range(12, 513)]
    return tuple(sorted(set(low_orders + middle_orders + high_orders)))


# RDP orders (1.01 to 512).
DP_RDP_ALPHAS: tuple[float, ...] = build_dp_rdp_alphas()


# ----------------------------------------------------------------------------------------------------------------------
# 1. CONFIGURATION AND ARTIFACT HELPERS

# CLASS: Freeze the resolved federated configuration for one E_06 run.
@dataclass(frozen=True)
class FederatedRunConfig:
    dataset: str = DATASET
    heterogeneity: str = HETEROGENEITY
    n_clients: int = N_CLIENTS
    strategy: FederatedStrategy = STRATEGY
    next_activity_head_agg: NextActivityHeadAgg = NEXT_ACTIVITY_HEAD_AGG
    secure_aggregation_simulation: bool = SECURE_AGGREGATION_SIMULATION
    secure_aggregation_seed: int = SECURE_AGGREGATION_SEED
    max_rounds: int = MAX_ROUNDS
    local_epochs: int = LOCAL_EPOCHS
    early_stopping_patience: int = EARLY_STOPPING_PATIENCE
    fedprox_mu: float = FEDPROX_MU
    use_dp: bool = USE_DP
    dp_target_epsilon: float = DP_TARGET_EPSILON
    dp_delta: float = DP_DELTA
    dp_max_grad_norm: float = DP_MAX_GRAD_NORM
    dp_smoke_max_batches: Optional[int] = DP_SMOKE_MAX_BATCHES
    seed: int = SEED
    device: str = DEVICE
    progress_bars: bool = PROGRESS_BARS
    reporting_profile: str = REPORTING_PROFILE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    lr_scheduler: Literal["cosine", "none"] = LR_SCHEDULER
    lr_scheduler_min_lr: float = LR_SCHEDULER_MIN_LR
    lr_scheduler_t_max: int = LR_SCHEDULER_T_MAX
    batch_size: int = BATCH_SIZE
    num_workers: int = NUM_WORKERS
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
    remaining_time_transform: Literal["raw", "log"] = REMAINING_TIME_TRANSFORM
    remaining_time_scaling: Literal["raw", "median", "zscore"] = REMAINING_TIME_SCALING
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


# HELPER: Encode a float for folder names without dots or minus signs.
def format_float_for_path(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


# HELPER: Resolve the E_06 artifact filename with the script identifier.
def script_artifact_name(filename: str) -> str:
    if filename.startswith(f"{SCRIPT_ID}_"): return filename
    return f"{SCRIPT_ID}_{filename}"


# HELPER: Resolve a prefixed artifact path in the run output directory.
def run_artifact_path(output_dir: Path, filename: str) -> Path:
    return output_dir / script_artifact_name(filename)


# HELPER: Resolve a prefixed prediction artifact path in the prediction subdirectory.
def prediction_artifact_path(output_dir: Path, filename: str) -> Path:
    return output_dir / PREDICTIONS_DIR_NAME / script_artifact_name(filename)


# HELPER: Convert dataset-qualified joint bank ids into filesystem safe tokens.
def _safe_bank(bank: str) -> str:
    return baseline._safe_bank(str(bank))


# HELPER: Store the complete resolved run configuration for later analysis.
def _config_payload(config: FederatedRunConfig, resolved_device: Optional[str] = None) -> dict[str, Any]:
    # Start from the dataclass fields so workflow and CLI overrides are recorded.
    payload = asdict(config)
    payload["run_name"] = config.run_name

    # Record the actual torch device when the caller has already resolved it.
    if resolved_device is not None: payload["resolved_device"] = resolved_device
    if config.use_dp:
        payload["dp_batch_mode"] = "full" if config.dp_smoke_max_batches is None else str(config.dp_smoke_max_batches)
        payload["dp_rdp_alpha_min"] = float(min(DP_RDP_ALPHAS))
        payload["dp_rdp_alpha_max"] = float(max(DP_RDP_ALPHAS))
        payload["dp_rdp_alpha_count"] = int(len(DP_RDP_ALPHAS))
    return payload


# HELPER: Resolve the federated run directory and encode the tokens specific to a strategy.
def output_dir_for_run(root: Path, config: FederatedRunConfig) -> Path:
    # Honor explicit output directories used by tests and smoke runs.
    if config.output_dir_override is not None: return config.output_dir_override

    # Build the normal matrix output location from dataset, split and strategy.
    # Route DP and secure POC run separately, so analysis never mixes them with the federated matrix.
    if config.use_dp:
        subfolder = "differential_privacy"
    elif config.secure_aggregation_simulation:
        subfolder = "secure_aggregation"
    else:
        subfolder = "federated"
    base = root / subfolder / f"{config.dataset}_{config.run_name}"
    variant = (
        f"{config.strategy}_seed_{config.seed}_lr_{format_float_for_path(config.learning_rate)}"
        f"_rounds_{config.max_rounds}_le_{config.local_epochs}"
    )

    # Add the FedProx mu token only for FedProx runs.
    if config.strategy == "fedprox": variant = f"{variant}_mu_{format_float_for_path(config.fedprox_mu)}"

    # Add the head-aggregation token only for joint runs, where the mode is a real experiment axis.
    # Single dataset runs resolve every mode to sample, so a token there would rename paths without changing a run.
    if config.dataset == "joint": variant = f"{variant}_agg_{config.next_activity_head_agg}"

    # Add the DP token only for DP runs, including the batch mode, so capped and full DP outputs never collide.
    if config.use_dp:
        batch_mode = "full" if config.dp_smoke_max_batches is None else str(int(config.dp_smoke_max_batches))
        variant = (
            f"{variant}_dp_eps_{format_float_for_path(config.dp_target_epsilon)}"
            f"_dp_batches_{batch_mode}"
        )
    return base / variant


# HELPER: Write one stable JSON artifact with sorted keys.
def save_json(path: Path, payload: dict[str, Any]) -> None: training_reporting.save_json(path, payload)


# HELPER: Write one legacy fragment only when debug reporting is active.
def save_legacy_json(config: FederatedRunConfig, path: Path, payload: dict[str, Any]) -> None:
    if training_reporting.should_write_legacy_reports(config.reporting_profile): save_json(path, payload)


# HELPER: Convert a scalar artifact value to float, falling back to NaN when the metric is absent.
def _float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


# HELPER: Extract one nested scalar from a metric block.
def _nested_metric_float(metrics: dict[str, Any], block: str, key: str) -> float:
    nested = metrics.get(block, {})
    if not isinstance(nested, dict): return float("nan")
    return _float_or_nan(nested.get(key))


# HELPER: Convert federated round diagnostics into the E_05 train-log schema expected by the shared plotter.
def federated_curve_history(round_rows: list[dict[str, Any]], round_metrics: list[dict[str, Any]]) -> pd.DataFrame:
    train_loss_by_round = {
        int(row.get("round", 0)): _float_or_nan(row.get("train_loss_total"))
        for row in round_rows
    }
    rows: list[dict[str, float]] = []
    for record in round_metrics:
        round_idx = int(record.get("round", 0))
        pooled_validation = record.get("pooled_validation", {})
        metrics = pooled_validation if isinstance(pooled_validation, dict) else {}
        rows.append(
            {
                "epoch": float(round_idx),
                "train_loss_total": train_loss_by_round.get(round_idx, float("nan")),
                "val_loss_total": _float_or_nan(metrics.get("loss_total")),
                "val_loss_outcome": _float_or_nan(metrics.get("loss_outcome")),
                "val_loss_next_activity": _float_or_nan(metrics.get("loss_next_activity")),
                "val_loss_remaining_time": _float_or_nan(metrics.get("loss_remaining_time")),
                "val_outcome_macro_f1": _nested_metric_float(metrics, "outcome", "macro_f1"),
                "val_next_activity_top1": _nested_metric_float(metrics, "next_activity", "top1_accuracy"),
                "val_remaining_time_mae": _nested_metric_float(metrics, "remaining_time", "mae"),
            }
        )
    return pd.DataFrame(rows)


# HELPER: Write a flat pooled and per-client test summary that mirrors the E_05 centralized summary schema.
def write_federated_summary(output_dir: Path, test_metrics: dict[str, Any], per_bank: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    global_row: dict[str, Any] = {"scope": "global_test"}
    global_row.update(baseline.flatten_local_summary_metrics(test_metrics))
    rows.append(global_row)

    for bank_name, bank_metrics in per_bank.items():
        if not isinstance(bank_metrics, dict): continue
        bank_row: dict[str, Any] = {"scope": f"bank_{bank_name}"}
        bank_row.update(baseline.flatten_local_summary_metrics(bank_metrics))
        rows.append(bank_row)

    frame = pd.DataFrame(rows)
    frame.to_csv(run_artifact_path(output_dir, "federated_summary_test.csv"), index=False)
    save_json(run_artifact_path(output_dir, "federated_summary_test.json"), {"rows": frame.to_dict(orient="records")})


# ----------------------------------------------------------------------------------------------------------------------
# 2. FEDERATED DATA CONTEXT

# CLASS: Carry one simulated bank's datasets and loaders for local federated training.
@dataclass(frozen=True)
class FederatedClientContext:
    bank: str
    train_dataset: Any
    val_dataset: Any
    test_dataset: Any
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_prefix_count: int
    next_activity_class_counts: np.ndarray
    training_next_activity_mask_context: core.NextActivityMaskContext


# CLASS: Carry all E_04 metadata, client contexts and pooled evaluation objects for one run.
@dataclass(frozen=True)
class FederatedRunContext:
    clients: list[FederatedClientContext]
    pooled_train_dataset: Any
    pooled_val_dataset: Any
    pooled_test_dataset: Any
    pooled_val_loader: DataLoader
    pooled_test_loader: DataLoader
    pooled_train_labels: pd.Series
    remaining_time_repr: core.RemainingTimeRepr
    evaluation_next_activity_mask_context: core.NextActivityMaskContext
    spec: dict[str, Any]
    vocabularies: dict[str, dict[str, int]]
    scalers: dict[str, dict[str, float]]
    schema_profile: dict[str, Any]
    mapping: dict[str, Any]


# Build an E_05 baseline config that points at the same data population as this E_06 run.
def baseline_config_for_federated(config: FederatedRunConfig, regime: baseline.Regime,
                                  bank: Optional[str] = None) -> baseline.BaselineRunConfig:
    return baseline.BaselineRunConfig(
        dataset=config.dataset,
        heterogeneity=config.heterogeneity,
        n_clients=config.n_clients,
        regime=regime,
        bank=bank,
        seed=config.seed,
        max_epochs=config.max_rounds,
        early_stopping_patience=config.early_stopping_patience,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        gradient_clip_norm=config.gradient_clip_norm,
        lr_scheduler=config.lr_scheduler,
        lr_scheduler_min_lr=config.lr_scheduler_min_lr,
        lr_scheduler_t_max=config.lr_scheduler_t_max,
        device=config.device,
        progress_bars=config.progress_bars,
        reporting_profile=config.reporting_profile,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        head_hidden_size=config.head_hidden_size,
        outcome_label_smoothing=config.outcome_label_smoothing,
        outcome_class_weight_power=config.outcome_class_weight_power,
        outcome_lr_scale=config.outcome_lr_scale,
        next_activity_lr_scale=config.next_activity_lr_scale,
        remaining_time_lr_scale=config.remaining_time_lr_scale,
        outcome_loss_weight=config.outcome_loss_weight,
        next_activity_loss_weight=config.next_activity_loss_weight,
        remaining_time_loss_weight=config.remaining_time_loss_weight,
        remaining_time_transform=config.remaining_time_transform,
        remaining_time_scaling=config.remaining_time_scaling,
        remaining_time_huber_beta=config.remaining_time_huber_beta,
        outcome_head_dropout=config.outcome_head_dropout,
        output_root=config.output_root,
        cache_root=config.cache_root,
        artifact_root=config.artifact_root,
        canonical_schema_path=config.canonical_schema_path,
        dataset_mapping_path=config.dataset_mapping_path,
    )


# HELPER: Load one Opacus symbol only inside DP code paths.
def load_opacus_symbol(module_name: str, symbol_name: str) -> Any:
    # Fail with a workflow message when the DP dependency is missing.
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"Opacus is required for USE_DP=true but {module_name} could not be imported") from exc

    # Resolve the requested symbol dynamically so no-DP static analysis does not require Opacus imports.
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise RuntimeError(f"Opacus module {module_name} does not expose {symbol_name}") from exc


# HELPER: Log and suppress known DP warnings that otherwise repeat once per client and round.
def configure_dp_warning_filters() -> None:
    global _DP_WARNING_FILTERS_CONFIGURED

    # Keep unrelated warnings visible and install filters only once per process.
    if _DP_WARNING_FILTERS_CONFIGURED: return
    warnings.warn(
        "Known DP warnings are suppressed: Opacus secure RNG, "
        "Opacus RDP alpha boundary and PyTorch full backward hook.",
        UserWarning, stacklevel=2,
    )

    # Suppress the documented Opacus secure RNG warning because E_06 explicitly uses secure_mode=False for experiments.
    warnings.filterwarnings("ignore", message=r"Secure RNG turned off\..*", category=UserWarning)

    # Suppress RDP boundary warnings after expanding the alpha grid while keeping other accountant warnings visible.
    warnings.filterwarnings("ignore", message=r"Optimal order is the .* alpha\..*", category=UserWarning)

    # Suppress the known PyTorch hook warning emitted by Opacus DPLSTM grad samplers.
    warnings.filterwarnings(
        "ignore", message=r"Full backward hook is firing when gradients are computed.*", category=UserWarning)
    _DP_WARNING_FILTERS_CONFIGURED = True


# HELPER: Iterate with the E_05 progress helper through a BaselineRunConfig adapter.
def progress_iter_for_federated(iterable: Any, config: FederatedRunConfig, description: str,
                                total: Optional[int] = None, unit: str = "it", leave: bool = False) -> Any:
    return baseline.progress_iter(
        iterable, baseline_config_for_federated(config, "centralized"),
        description, total=total, leave=leave, unit=unit
    )


# HELPER: Evaluate an E_06 model through the E_05 metric implementation.
def evaluate_model_for_federated(model: core.MultitaskLSTM, loader: DataLoader, outcome_loss: nn.Module,
                                 next_activity_loss: nn.Module, remaining_time_repr: core.RemainingTimeRepr,
                                 huber_beta: float, device: torch.device, config: FederatedRunConfig,
                                 dataset: Any, progress_label: str = "eval",
                                 collect_predictions: bool = False,
                                 next_activity_mask_context: Optional[core.NextActivityMaskContext] = None,
                                 subset_selectors: Optional[dict[str, np.ndarray]] = None,
                                 subset_metrics: Optional[dict[str, Any]] = None,
                                 ) -> tuple[dict[str, Any], Optional[pd.DataFrame]]:
    # Reuse E_05 evaluation with a real BaselineRunConfig so static checks match the runtime contract.
    eval_config = baseline_config_for_federated(config, "centralized")
    return baseline.evaluate_model(
        model, loader, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta, device, eval_config,
        dataset, progress_label=progress_label, collect_predictions=collect_predictions,
        next_activity_mask_context=next_activity_mask_context,
        subset_selectors=subset_selectors, subset_metrics=subset_metrics,
    )


# Resolve the next activity vocabulary from current metadata.
def _next_activity_vocabulary(vocabularies: dict[str, dict[str, int]]) -> dict[str, int]:
    if baseline.encoding.NEXT_ACTIVITY_TARGET in vocabularies:
        return vocabularies[baseline.encoding.NEXT_ACTIVITY_TARGET]
    if "next_activity" in vocabularies: return vocabularies["next_activity"]
    raise KeyError(baseline.encoding.NEXT_ACTIVITY_TARGET)


# Resolve the dataset ids present in one local prefix dataset.
def _dataset_ids_from_prefix_dataset(dataset: Any) -> set[str]:
    prefix_index = getattr(dataset, "prefix_index", None)
    if prefix_index is None: raise ValueError("client-local next-activity masks require prefix_index rows")
    dataset_ids = {str(row.dataset_id) for row in prefix_index}
    if not dataset_ids: raise ValueError("client-local next-activity masks require at least one prefix row")
    return dataset_ids


# Build one client's training mask from its own observed next-activity classes, no shared vocabulary.
def build_client_next_activity_mask_context(local_config: baseline.BaselineRunConfig, train_dataset: Any,
                                            n_classes: int) -> core.NextActivityMaskContext:
    # A client with a single dataset keeps the whole head visible.
    if local_config.dataset != "joint":
        return core.build_next_activity_mask_context(
            {local_config.dataset: np.ones(int(n_classes), dtype=bool)},
            single_dataset_noop=True,
        )

    # A joint client masks to exactly the next activity classes it observed in its own local training data.
    dataset_ids = _dataset_ids_from_prefix_dataset(train_dataset)
    if len(dataset_ids) != 1:
        raise ValueError(f"one federated client must map to exactly one source dataset, got {sorted(dataset_ids)}")
    dataset_id = next(iter(dataset_ids))
    presence = baseline.next_activity_class_presence_from_dataset(train_dataset, n_classes)
    return core.build_next_activity_mask_context({dataset_id: presence})


# Load or rebuild all train, validation and test prefix datasets for one E_05 config.
def _load_split_datasets(local_config: baseline.BaselineRunConfig, spec: dict[str, Any],
                         vocabularies: dict[str, dict[str, int]], scalers: dict[str, dict[str, float]],
                         schema_profile: dict[str, Any], mapping: dict[str, Any]) -> dict[str, Any]:
    # Try warm caches first, so normal E_06 runs do not touch processed parquet files.
    datasets = {
        split_name: baseline.load_cached_prefix_dataset(
            local_config, split_name, spec, vocabularies, scalers, schema_profile, mapping)
        for split_name in ("train", "val", "test")
    }

    # Rebuild missing splits through the E_05 canonical mapping path.
    mapped: Optional[pd.DataFrame] = None
    for split_name, dataset in list(datasets.items()):
        if dataset is not None: continue
        if mapped is None: mapped = baseline.load_mapped_events(local_config, mapping)
        built = baseline.build_prefix_dataset(mapped, split_name, spec, vocabularies, scalers, schema_profile)
        datasets[split_name] = baseline.cache_prefix_dataset(
            local_config, built, split_name, spec, vocabularies, scalers, schema_profile, mapping
        )

    # Stop when cache loading and rebuilding did not produce every split.
    if any(dataset is None for dataset in datasets.values()):
        raise RuntimeError("federated prefix datasets were not loaded")
    return datasets


# Load all local client data plus the pooled train statistics and pooled evaluation splits.
def load_federated_context(config: FederatedRunConfig) -> FederatedRunContext:
    # Load metadata through the centralized E_05 config because vocabularies and scalers are pooled train artifacts.
    pooled_config = baseline_config_for_federated(config, "centralized")
    spec, vocabularies, scalers, schema_profile, mapping = baseline.load_training_metadata(pooled_config)
    next_activity_vocab = _next_activity_vocabulary(vocabularies)

    # Load pooled splits for global class weights plus pooled validation and test evaluation.
    pooled_splits = _load_split_datasets(pooled_config, spec, vocabularies, scalers, schema_profile, mapping)
    pooled_train_dataset = pooled_splits["train"]
    pooled_val_dataset = pooled_splits["val"]
    pooled_test_dataset = pooled_splits["test"]
    pooled_train_labels = baseline.training_outcome_labels_from_dataset(pooled_train_dataset)
    if config.remaining_time_scaling == "median":
        raise ValueError("median remaining-time scaling is not supported for federated training")
    remaining_time_repr = baseline.remaining_time_repr_from_spec(spec, pooled_config)

    # Evaluation masking is analysis only and per dataset, built from the observed pooled training classes.
    evaluation_next_activity_mask_context = baseline.build_next_activity_mask_context(
        pooled_config, pooled_train_dataset, len(next_activity_vocab)
    )

    # Evaluation masking is analysis only reporting on centralized validation and test splits.
    pooled_val_loader = baseline.make_loader(
        pooled_val_dataset, config.batch_size, False, config.seed, config.num_workers,
        evaluation_next_activity_mask_context
    )
    pooled_test_loader = baseline.make_loader(
        pooled_test_dataset, config.batch_size, False, config.seed, config.num_workers,
        evaluation_next_activity_mask_context
    )

    # Build one local client context per simulated bank.
    clients: list[FederatedClientContext] = []
    for bank in baseline.bank_names_for_config(pooled_config):
        local_config = baseline_config_for_federated(config, "local", bank=bank)
        local_splits = _load_split_datasets(local_config, spec, vocabularies, scalers, schema_profile, mapping)
        training_next_activity_mask_context = build_client_next_activity_mask_context(
            local_config, local_splits["train"], len(next_activity_vocab),
        )
        next_activity_class_counts = baseline.next_activity_target_counts_from_dataset(
            local_splits["train"], len(next_activity_vocab),
        )
        clients.append(
            FederatedClientContext(
                bank=bank,
                train_dataset=local_splits["train"],
                val_dataset=local_splits["val"],
                test_dataset=local_splits["test"],
                train_loader=baseline.make_loader(
                    local_splits["train"], config.batch_size, True, config.seed, config.num_workers,
                    training_next_activity_mask_context,
                ),
                val_loader=baseline.make_loader(
                    local_splits["val"], config.batch_size, False, config.seed, config.num_workers,
                    evaluation_next_activity_mask_context,
                ),
                test_loader=baseline.make_loader(
                    local_splits["test"], config.batch_size, False, config.seed, config.num_workers,
                    evaluation_next_activity_mask_context,
                ),
                train_prefix_count=int(len(local_splits["train"])),
                next_activity_class_counts=next_activity_class_counts,
                training_next_activity_mask_context=training_next_activity_mask_context,
            )
        )

    # Return one immutable context so later training never refits pooled statistics.
    return FederatedRunContext(
        clients=clients, pooled_train_dataset=pooled_train_dataset, pooled_val_dataset=pooled_val_dataset,
        pooled_test_dataset=pooled_test_dataset, pooled_val_loader=pooled_val_loader,
        pooled_test_loader=pooled_test_loader, pooled_train_labels=pooled_train_labels,
        remaining_time_repr=remaining_time_repr,
        evaluation_next_activity_mask_context=evaluation_next_activity_mask_context, spec=spec,
        vocabularies=vocabularies, scalers=scalers, schema_profile=schema_profile, mapping=mapping,
    )


# ----------------------------------------------------------------------------------------------------------------------
# 3. LOCAL FEDERATED TRAINING HELPERS

# Compute the cosine schedule ratio for one communication round.
def lr_ratio_for_round(config: FederatedRunConfig, round_idx: int) -> float:
    # Keep fixed base LRs when scheduling is disabled.
    if config.lr_scheduler == "none": return 1.0
    if config.lr_scheduler != "cosine": raise ValueError(f"unknown scheduler: {config.lr_scheduler}")

    # Match E_05's guarded cosine step over LR_SCHEDULER_T_MAX, then hold the floor.
    t_max = max(int(config.lr_scheduler_t_max), 1)
    step = min(max(int(round_idx) - 1, 0), t_max)
    return 0.5 * (1.0 + math.cos(math.pi * step / t_max))


# Resolve scheduled LRs for every AdamW parameter group.
def optimizer_lrs_for_round(config: FederatedRunConfig, round_idx: int) -> dict[str, float]:
    # Start from each group's own base LR so head groups decay to the shared floor independently.
    base_lrs = {
        "trunk": float(config.learning_rate),
        "outcome": float(config.learning_rate) * float(config.outcome_lr_scale),
        "next_activity": float(config.learning_rate) * float(config.next_activity_lr_scale),
        "remaining_time": float(config.learning_rate) * float(config.remaining_time_lr_scale),
    }
    if config.lr_scheduler == "none": return base_lrs

    # Apply PyTorch CosineAnnealingLR semantics per group with a shared eta_min.
    ratio = lr_ratio_for_round(config, round_idx)
    eta_min = float(config.lr_scheduler_min_lr)
    return {name: eta_min + (base_lr - eta_min) * ratio for name, base_lr in base_lrs.items()}


# Apply scheduled learning rates to an optimizer that uses named parameter groups.
def apply_optimizer_lrs(optimizer: torch.optim.Optimizer, lrs_by_group: dict[str, float]) -> None:
    # Fail loudly when an optimizer group lacks a schedule entry.
    for group in optimizer.param_groups:
        name = str(group.get("group_name", ""))
        if name not in lrs_by_group: raise ValueError(f"optimizer group is missing scheduled LR: {name}")
        group["lr"] = float(lrs_by_group[name])


# Train one client for one local epoch with the E_05 masked multitask objective plus optional FedProx.
def train_federated_epoch(model: core.MultitaskLSTM, loader: DataLoader, outcome_loss: nn.Module,
                          next_activity_loss: nn.Module, remaining_time_repr: core.RemainingTimeRepr,
                          huber_beta: float, optimizer: torch.optim.Optimizer,
                          reference_params: list[torch.Tensor], fedprox_mu: float, device: torch.device,
                          config: FederatedRunConfig, use_dp: bool = False,
                          progress_label: str = "client train",
                          next_activity_mask_context: Optional[core.NextActivityMaskContext] = None) -> dict[
    str, float]:
    # Switch the client model to training mode before iterating over local prefix batches.
    model.train()

    # Accumulate exact task loss means by valid sample counts, matching E_05.
    loss_sums = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    loss_counts = {"outcome": 0.0, "next_activity": 0.0, "remaining_time": 0.0}
    gradient_norm_sum = 0.0
    fedprox_sum = 0.0
    n_batches = 0
    total_start = time.perf_counter()
    progress_total = len(loader)
    if use_dp and config.dp_smoke_max_batches is not None:
        progress_total = min(progress_total, int(config.dp_smoke_max_batches))
    batch_iter = progress_iter_for_federated(
        loader, config, f"{progress_label} batches", total=progress_total, unit="batch"
    )

    # Run one full local pass over the client train prefixes.
    for batch_index, batch in enumerate(batch_iter, 1):
        batch = core.move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)

        # Compute E_05 task losses and add FedProx only to the backward objective.
        losses = core.compute_multitask_loss(
            outputs, batch, outcome_loss, next_activity_loss, remaining_time_repr, huber_beta,
            config.outcome_loss_weight, config.next_activity_loss_weight, config.remaining_time_loss_weight,
            next_activity_mask_context=next_activity_mask_context,
        )
        prox = fedprox_penalty(model, reference_params, fedprox_mu)
        backward_loss = losses.total + prox
        backward_loss.backward()

        # Use manual global clipping only for no-DP runs because Opacus owns clipping in DP mode.
        if use_dp:
            gradient_norm = torch.tensor(0.0, device=device)
        else:
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
        fedprox_sum += float(prox.item())
        n_batches += 1
        if hasattr(batch_iter, "set_postfix"): batch_iter.set_postfix({"loss": f"{float(backward_loss.item()):.4f}"})

        # Stop early only for explicit DP smoke checks, not for production runs.
        if use_dp and config.dp_smoke_max_batches is not None and batch_index >= int(config.dp_smoke_max_batches): break

    # Rebuild the weighted task loss from the three task means.
    loss_means = {key: loss_sums[key] / max(loss_counts[key], 1.0) for key in loss_sums}
    loss_total = (
            config.outcome_loss_weight * loss_means["outcome"]
            + config.next_activity_loss_weight * loss_means["next_activity"]
            + config.remaining_time_loss_weight * loss_means["remaining_time"]
    )
    avg_fedprox = fedprox_sum / max(n_batches, 1)

    # Return task loss separately from the FedProx-augmented optimization objective.
    return {
        "loss_total": loss_total,
        "loss_outcome": loss_means["outcome"],
        "loss_next_activity": loss_means["next_activity"],
        "loss_remaining_time": loss_means["remaining_time"],
        "loss_total_with_fedprox": loss_total + avg_fedprox,
        "train_fedprox_penalty": avg_fedprox,
        "gradient_norm": gradient_norm_sum / max(n_batches, 1),
        "n_batches": int(n_batches),
        "timing_total_seconds": time.perf_counter() - total_start,
    }


# ----------------------------------------------------------------------------------------------------------------------
# 4. FEDERATED PARAMETER HELPERS

# Return the sorted state-dict keys used for all E_06 parameter serialization.
def parameter_key_order(model: nn.Module) -> list[str]: return sorted(model.state_dict().keys())


# Convert model parameters to float32 numpy arrays in canonical order of sorted keys.
def model_parameters_to_numpy(model: nn.Module) -> list[np.ndarray]:
    # Read the state dict once, so indexing over sorted keys cannot observe a mutated state.
    state = model.state_dict()
    return [state[key].detach().cpu().numpy().astype(np.float32, copy=True) for key in parameter_key_order(model)]


# Load numpy parameters back into the model in the canonical order of sorted keys.
def load_numpy_parameters(model: nn.Module, params: list[np.ndarray]) -> None:
    # Stop when the server and client model state shapes cannot be paired safely.
    state = model.state_dict()
    keys = parameter_key_order(model)
    if len(params) != len(keys): raise ValueError(f"parameter count mismatch: {len(params)} != {len(keys)}")

    # Rebuild the state dict with tensor dtype and device matching the receiving model.
    new_state = {}
    for key, array in zip(keys, params):
        new_state[key] = torch.as_tensor(array, dtype=state[key].dtype, device=state[key].device)
    model.load_state_dict(new_state, strict=True)


# Average one tensor by train prefix count.
def _sample_weighted_tensor(results: list[tuple[list[np.ndarray], int]], tensor_idx: int, total: float) -> np.ndarray:
    weighted = sum(params[tensor_idx] * (count / total) for params, count in results)
    return np.asarray(weighted, dtype=np.float32)


# Average one tensor by equal client weights.
def _equal_weighted_tensor(results: list[tuple[list[np.ndarray], int]], tensor_idx: int) -> np.ndarray:
    averaged = sum(params[tensor_idx] for params, _ in results) / float(len(results))
    return np.asarray(averaged, dtype=np.float32)


# Decide next-activity class support against a threshold rather than against exact zero.
# The counts are integers, so a genuine count is at least one while a mask-cancellation residue is far below.
NEXT_ACTIVITY_SUPPORT_THRESHOLD: float = 0.5


# Return whether a tensor belongs to the next-activity output layer.
def _is_next_activity_output_slot(key: str) -> bool:
    return key in {"next_activity_head.3.weight", "next_activity_head.3.bias"}


# Build the weighted numerator payload on the client side for contribution aggregation.
def _client_next_activity_contribution_payload(params: list[np.ndarray], parameter_keys: list[str],
                                               class_counts: np.ndarray) -> NextActivityContributionPayload:
    # Weighting is applied before aggregation, so a real server would receive only masked additive messages.
    counts = np.asarray(class_counts, dtype=np.float32)
    if counts.ndim != 1: raise ValueError("next-activity class counts must be a one-dimensional vector")

    payload: NextActivityContributionPayload = {}
    for tensor_idx, key in enumerate(parameter_keys):
        if not _is_next_activity_output_slot(key): continue
        tensor = np.asarray(params[tensor_idx], dtype=np.float32)
        if tensor.shape[0] != counts.shape[0]:
            raise ValueError("next-activity class count length does not match output head shape")
        if tensor.ndim == 1:
            payload[tensor_idx] = tensor * counts
        else:
            payload[tensor_idx] = tensor * counts.reshape((-1,) + (1,) * (tensor.ndim - 1))
    return payload


# Average one next-activity output tensor through secure-sum-compatible contribution payloads.
def _contribution_weighted_tensor(results: list[tuple[list[np.ndarray], int]], tensor_idx: int, total: float,
                                  class_counts: list[np.ndarray],
                                  contribution_payloads: list[NextActivityContributionPayload]) -> np.ndarray:
    # Start from sample weighting so zero-denominator slots keep a defined FedAvg value.
    sample_weighted = _sample_weighted_tensor(results, tensor_idx, total)
    denominator = np.asarray(sum(np.asarray(counts, dtype=np.float32) for counts in class_counts), dtype=np.float32)
    numerator = np.asarray(
        sum(payload[tensor_idx] for payload in contribution_payloads),
        dtype=np.float32,
    )
    if sample_weighted.shape[0] != denominator.shape[0]:
        raise ValueError("next-activity contribution denominator does not match output head shape")

    # Divide only slots with federation-wide support; unsupported slots keep the sample-weighted fallback.
    supported = denominator >= NEXT_ACTIVITY_SUPPORT_THRESHOLD
    if not bool(supported.any()): return sample_weighted
    aggregated = sample_weighted.copy()
    if aggregated.ndim == 1:
        aggregated[supported] = numerator[supported] / denominator[supported]
    else:
        expanded_denominator = denominator.reshape((-1,) + (1,) * (aggregated.ndim - 1))
        aggregated[supported] = numerator[supported] / expanded_denominator[supported]
    return np.asarray(aggregated, dtype=np.float32)


# Validate the aggregation inputs shared by the plain and the secure path.
def _validate_head_aggregation(next_activity_head_agg: NextActivityHeadAgg, n_tensors: int,
                               parameter_keys: Optional[list[str]], is_joint_run: bool) -> NextActivityHeadAgg:
    if next_activity_head_agg not in {"sample", "equal", "contribution"}:
        raise ValueError(f"unknown next-activity head aggregation mode: {next_activity_head_agg}")
    effective_head_agg = cast(NextActivityHeadAgg, next_activity_head_agg if is_joint_run else "sample")
    if effective_head_agg != "sample" and parameter_keys is None:
        raise ValueError("next-activity head aggregation requires parameter keys")
    if parameter_keys is not None and len(parameter_keys) != n_tensors:
        raise ValueError("parameter key count does not match client tensors")
    return effective_head_agg


# Aggregate client model parameters by train prefix count, with an optional next-activity head weighting.
def aggregate_parameters(results: list[tuple[list[np.ndarray], int]],
                         next_activity_head_agg: NextActivityHeadAgg = "sample",
                         parameter_keys: Optional[list[str]] = None,
                         next_activity_class_counts: Optional[list[np.ndarray]] = None,
                         next_activity_contribution_payloads: Optional[list[NextActivityContributionPayload]] = None,
                         is_joint_run: bool = True) -> list[np.ndarray]:
    # Reject empty or non-positive aggregation weights because FedAvg needs a valid denominator.
    if not results: raise ValueError("cannot aggregate an empty result list")
    total = float(sum(count for _, count in results))
    if total <= 0.0: raise ValueError("cannot aggregate results with non-positive total weight")

    # Require every client to return the same number of tensors before weighted averaging.
    n_tensors = len(results[0][0])
    if any(len(params) != n_tensors for params, _ in results): raise ValueError("client parameter counts do not match")
    effective_head_agg = _validate_head_aggregation(next_activity_head_agg, n_tensors, parameter_keys, is_joint_run)
    if effective_head_agg == "contribution":
        if next_activity_class_counts is None:
            raise ValueError("contribution aggregation requires client next-activity class counts")
        if len(next_activity_class_counts) != len(results):
            raise ValueError("client class-count vector count does not match aggregation results")
        if next_activity_contribution_payloads is None:
            raise ValueError("contribution aggregation requires client-side contribution payloads")
        if len(next_activity_contribution_payloads) != len(results):
            raise ValueError("client contribution payload count does not match aggregation results")

    # Average each tensor independently using prefix counts as FedAvg weights.
    aggregated = []
    for tensor_idx in range(n_tensors):
        key = "" if parameter_keys is None else parameter_keys[tensor_idx]
        if effective_head_agg == "equal" and key.startswith("next_activity_head."):
            aggregated.append(_equal_weighted_tensor(results, tensor_idx))
        elif effective_head_agg == "contribution" and _is_next_activity_output_slot(key):
            # Contribution mode mirrors two secure sums: the weighted client numerator and the target count denominator.
            # The aggregator reconstructs only federation-wide sums per output slot, never a per-bank value.
            aggregated.append(
                _contribution_weighted_tensor(
                    results, tensor_idx, total, next_activity_class_counts or [],
                                                next_activity_contribution_payloads or [],
                )
            )
        else:
            aggregated.append(_sample_weighted_tensor(results, tensor_idx, total))
    return aggregated


# CLASS: Carry one client's masked message for one model tensor.
# The orchestrator therefore never handles a raw update, which is a message-flow property and not secrecy.
@dataclass(frozen=True)
class SecureTensorMessage:
    kind: str
    dtype: np.dtype
    channels: dict[str, np.ndarray]


# Derive one deterministic generator for one unordered client pair, round, tensor and channel.
# Both pair members derive the identical generator, so their masks cancel exactly in the sum.

# The seed is recorded in the run report, so this construction hides nothing from a reader of that report.

def _secure_pair_rng(secure_aggregation_seed: int, round_index: int, tensor_index: int, channel_index: int,
                     lower_client: int, upper_client: int) -> np.random.Generator:
    return np.random.default_rng(
        (int(secure_aggregation_seed), int(round_index), int(tensor_index), int(channel_index),
         int(lower_client), int(upper_client))
    )


# Build the additive mask one client adds to one channel array.
# A client adds the pair mask for every higher-indexed partner and subtracts it for every lower one.
def secure_client_mask(shape: tuple[int, ...], client_index: int, n_clients: int, secure_aggregation_seed: int,
                       round_index: int, tensor_index: int, channel_index: int) -> np.ndarray:
    if not 0 <= int(client_index) < int(n_clients):
        raise ValueError(f"client index {client_index} is outside the participant range {n_clients}")

    # A single participant has no partner, so its mask is zero and the sum is the client's own value.
    mask = np.zeros(shape, dtype=np.float64)
    for partner in range(int(n_clients)):
        if partner == int(client_index): continue
        lower, upper = sorted((int(client_index), partner))
        rng = _secure_pair_rng(secure_aggregation_seed, round_index, tensor_index, channel_index, lower, upper)
        pair_mask = rng.uniform(-SECURE_AGGREGATION_MASK_BOUND, SECURE_AGGREGATION_MASK_BOUND, size=shape)
        mask = mask + pair_mask if partner > int(client_index) else mask - pair_mask
    return mask


# Add the client's own pairwise mask to one channel array before it leaves the client.
def _secure_mask_channel(array: np.ndarray, channel_index: int, client_index: int, n_clients: int,
                         secure_aggregation_seed: int, round_index: int, tensor_index: int) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    mask = secure_client_mask(values.shape, client_index, n_clients, secure_aggregation_seed, round_index,
                              tensor_index, channel_index)
    return values + mask


# Recover the federation-wide sum of one channel from the masked client messages only.
def _secure_recover_sum(messages: list[SecureTensorMessage], channel: str) -> np.ndarray:
    total = np.zeros_like(messages[0].channels[channel])
    for message in messages: total = total + message.channels[channel]
    return total


# Build one client's masked message list from its own update, its own weight and the protocol inputs.
# It needs only its index, the participant count, the round index and the shared seed.
def build_secure_client_message(params: list[np.ndarray], train_prefix_count: int, client_index: int, n_clients: int,
                                next_activity_head_agg: NextActivityHeadAgg = "sample",
                                parameter_keys: Optional[list[str]] = None,
                                next_activity_class_counts: Optional[np.ndarray] = None,
                                is_joint_run: bool = True, round_index: int = 1,
                                secure_aggregation_seed: int = SECURE_AGGREGATION_SEED,
                                ) -> list[SecureTensorMessage]:
    # Reject inputs the aggregation could not combine, using the same rules as the plain path.
    if int(n_clients) < 1: raise ValueError("secure aggregation needs at least one participant")
    if float(train_prefix_count) <= 0.0: raise ValueError("secure aggregation needs a positive client weight")
    effective_head_agg = _validate_head_aggregation(next_activity_head_agg, len(params), parameter_keys, is_joint_run)
    if effective_head_agg == "contribution" and next_activity_class_counts is None:
        raise ValueError("contribution aggregation requires the client next-activity class counts")

    # Mask one channel of one tensor with the client's own pairwise mask.
    def masked(array: np.ndarray, channel: str, tensor_idx_temp: int) -> np.ndarray:
        return _secure_mask_channel(array, SECURE_AGGREGATION_CHANNELS[channel], client_index, n_clients,
                                    secure_aggregation_seed, round_index, tensor_idx_temp)

    # Form this client's contribution and weight per tensor, then mask both before they leave the client.
    count = float(train_prefix_count)
    messages: list[SecureTensorMessage] = []
    for tensor_idx, tensor in enumerate(params):
        key = "" if parameter_keys is None else parameter_keys[tensor_idx]
        values = np.asarray(tensor, dtype=np.float64)
        dtype = np.asarray(tensor).dtype
        if effective_head_agg == "equal" and key.startswith("next_activity_head."):
            # Equal mode contributes the raw head tensor under a unit weight, so every bank counts once.
            channels = {
                "contribution": masked(values, "contribution", tensor_idx),
                "weight": masked(np.asarray(1.0), "weight", tensor_idx),
            }
            messages.append(SecureTensorMessage(kind="weighted", dtype=dtype, channels=channels))
        elif effective_head_agg == "contribution" and _is_next_activity_output_slot(key):
            # Contribution mode carries the class-count weighted numerator, its denominator and the sample fallback.
            counts = np.asarray(next_activity_class_counts, dtype=np.float64)
            payload = _client_next_activity_contribution_payload([tensor], [key], counts)
            channels = {
                "numerator": masked(np.asarray(payload[0], dtype=np.float64), "numerator", tensor_idx),
                "denominator": masked(counts, "denominator", tensor_idx),
                "sample_contribution": masked(values * count, "sample_contribution", tensor_idx),
                "sample_weight": masked(np.asarray(count), "sample_weight", tensor_idx),
            }
            messages.append(SecureTensorMessage(kind="contribution", dtype=dtype, channels=channels))
        else:
            # Sample mode contributes the prefix-count weighted tensor under the prefix count as its weight.
            channels = {
                "contribution": masked(values * count, "contribution", tensor_idx),
                "weight": masked(np.asarray(count), "weight", tensor_idx),
            }
            messages.append(SecureTensorMessage(kind="weighted", dtype=dtype, channels=channels))
    return messages


# Reject anything that is not a list of masked client messages, so the server helper cannot consume raw updates.
def _validate_masked_messages(client_messages: list[list[SecureTensorMessage]]) -> None:
    if not client_messages: raise ValueError("secure aggregation received no client messages")
    n_tensors = len(client_messages[0])
    for messages in client_messages:
        if not isinstance(messages, list) or len(messages) != n_tensors:
            raise ValueError("every client must send one masked message per model tensor")
        for message in messages:
            if not isinstance(message, SecureTensorMessage):
                raise TypeError("secure aggregation accepts only masked client messages, not raw client updates")
            expected = SECURE_AGGREGATION_MESSAGE_CHANNELS.get(message.kind)
            if expected is None: raise ValueError(f"unknown masked message kind: {message.kind}")
            if set(message.channels) != expected:
                raise ValueError(f"masked {message.kind} message must carry exactly the channels {sorted(expected)}")


# Recover the aggregated parameters from masked client messages only, never from a raw update or a raw count.
def secure_aggregate_from_masked_messages(client_messages: list[list[SecureTensorMessage]]) -> list[np.ndarray]:
    _validate_masked_messages(client_messages)

    # Combine the per-tensor messages of every client and divide the recovered sums into the aggregate.
    aggregated: list[np.ndarray] = []
    for tensor_idx in range(len(client_messages[0])):
        messages = [messages_for_client[tensor_idx] for messages_for_client in client_messages]
        model_dtype = messages[0].dtype
        if messages[0].kind == "contribution":
            # Recover the federation-wide numerator, denominator and sample fallback from the masked sums.
            numerator = _secure_recover_sum(messages, "numerator")
            denominator = _secure_recover_sum(messages, "denominator")
            sample_weighted = (_secure_recover_sum(messages, "sample_contribution")
                               / _secure_recover_sum(messages, "sample_weight"))
            supported = denominator >= NEXT_ACTIVITY_SUPPORT_THRESHOLD
            if not bool(supported.any()):
                tensor = sample_weighted
            else:
                tensor = sample_weighted.copy()
                if tensor.ndim == 1:
                    tensor[supported] = numerator[supported] / denominator[supported]
                else:
                    expanded_denominator = denominator.reshape((-1,) + (1,) * (tensor.ndim - 1))
                    tensor[supported] = numerator[supported] / expanded_denominator[supported]
        else:
            # Divide the recovered weighted sum by the recovered weight sum for the sample and equal modes.
            tensor = _secure_recover_sum(messages, "contribution") / _secure_recover_sum(messages, "weight")
        aggregated.append(np.asarray(tensor, dtype=model_dtype))
    return aggregated


# Measure the reconstruction deviation of the masking construction without reading any client update.
# Synthetic tensors of the run's own shapes are driven through the masked path and compared against the plain one.
def secure_construction_deviation(parameter_shapes: list[tuple[int, ...]], parameter_keys: list[str], n_clients: int,
                                  next_activity_head_agg: NextActivityHeadAgg, is_joint_run: bool,
                                  secure_aggregation_seed: int) -> float:
    # Read the next-activity class count from the output head itself, so the synthetic counts always fit the tensors.
    output_slots = [index for index, key in enumerate(parameter_keys) if _is_next_activity_output_slot(key)]
    n_next_activity_classes = int(parameter_shapes[output_slots[0]][0]) if output_slots else 1

    # Draw reproducible synthetic client updates and weights that never touch the training data.
    rng = np.random.default_rng(int(secure_aggregation_seed))
    synthetic_params = [
        [rng.normal(0.0, 1.0, size=shape).astype(np.float32) for shape in parameter_shapes]
        for _ in range(int(n_clients))
    ]
    synthetic_counts = [int(1000 * (index + 1)) for index in range(int(n_clients))]
    synthetic_class_counts = [
        rng.integers(0, 100, size=n_next_activity_classes).astype(np.float32) for _ in range(int(n_clients))
    ]

    # Aggregate the synthetic input once through the masked path and once through the plain path.
    results = list(zip(synthetic_params, synthetic_counts))
    payloads = [
        _client_next_activity_contribution_payload(params, parameter_keys, counts)
        for params, counts in zip(synthetic_params, synthetic_class_counts)
    ]
    secure = secure_aggregate_from_masked_messages([
        build_secure_client_message(
            params, count, client_index, int(n_clients), next_activity_head_agg=next_activity_head_agg,
            parameter_keys=parameter_keys, next_activity_class_counts=class_counts, is_joint_run=is_joint_run,
            round_index=1, secure_aggregation_seed=secure_aggregation_seed,
        )
        for client_index, (params, count, class_counts) in enumerate(
            zip(synthetic_params, synthetic_counts, synthetic_class_counts))
    ])
    plain = aggregate_parameters(
        results, next_activity_head_agg=next_activity_head_agg, parameter_keys=parameter_keys,
        next_activity_class_counts=synthetic_class_counts, next_activity_contribution_payloads=payloads,
        is_joint_run=is_joint_run,
    )
    return max(
        (float(np.max(np.abs(np.asarray(secure_tensor, dtype=np.float64) - np.asarray(plain_tensor, dtype=np.float64))))
         for secure_tensor, plain_tensor in zip(secure, plain)),
        default=0.0,
    )


# Compute the FedProx proximal penalty against module-order reference parameters.
def fedprox_penalty(model: nn.Module, reference_params: list[torch.Tensor], mu: float) -> torch.Tensor:
    # Return a differentiable zero on the right device when FedProx is disabled.
    first_param = next(model.parameters())
    if mu <= 0.0: return first_param.sum() * 0.0

    # Accumulate squared drift in module parameter order, not sorted serialization order.
    penalty = first_param.sum() * 0.0
    for param, reference in zip(model.parameters(), reference_params):
        penalty = penalty + torch.sum((param - reference.to(param.device, dtype=param.dtype)) ** 2)
    return 0.5 * float(mu) * penalty


# ----------------------------------------------------------------------------------------------------------------------
# 5. FLOWER CLIENT AND STRATEGY HELPERS

# Build the E_05 model for one federated client or server evaluation.
def build_model_for_federated(config: FederatedRunConfig, vocabularies: dict[str, dict[str, int]],
                              schema_profile: dict[str, Any], use_dp: bool = False) -> core.MultitaskLSTM:
    # Keep the no-DP path identical to E_05 by omitting the LSTM hook.
    model_config = baseline_config_for_federated(config, "centralized")
    if not use_dp: return baseline.build_model(model_config, vocabularies, schema_profile)

    # Import Opacus only for DP runs, so no-DP training stays on the plain E_05 path.
    dp_lstm_cls = cast(type[nn.Module], load_opacus_symbol("opacus.layers", "DPLSTM"))
    module_validator = load_opacus_symbol("opacus.validators", "ModuleValidator")

    # Swap only the LSTM trunk and validate the resulting Opacus-compatible model.
    model = baseline.build_model(model_config, vocabularies, schema_profile, lstm_cls=dp_lstm_cls)
    validation_errors = module_validator.validate(model, strict=False)
    if validation_errors: raise RuntimeError(f"DP model is not Opacus-compatible: {validation_errors}")
    return model


# Resolve the Opacus noise multiplier for one client privacy budget.
def resolve_dp_noise_multiplier(sample_rate: float, target_epsilon: float, target_delta: float, epochs: int) -> float:
    # Reject invalid accountant inputs before calling Opacus.
    if sample_rate <= 0.0 or sample_rate > 1.0: raise ValueError(f"sample_rate must be in (0, 1], got {sample_rate}")
    if target_epsilon <= 0.0: raise ValueError(f"target_epsilon must be positive, got {target_epsilon}")
    if target_delta <= 0.0: raise ValueError(f"target_delta must be positive, got {target_delta}")

    # Use Opacus' RDP accountant utility so the helper matches PrivacyEngine calibration.
    get_noise_multiplier = load_opacus_symbol("opacus.accountants.utils", "get_noise_multiplier")

    return float(
        get_noise_multiplier(
            target_epsilon=float(target_epsilon), target_delta=float(target_delta), sample_rate=float(sample_rate),
            epochs=max(int(epochs), 1), accountant="rdp", alphas=list(DP_RDP_ALPHAS),
        )
    )


# Return the underlying model when Opacus wraps it for per-sample gradients.
def _dp_base_model(model: nn.Module) -> nn.Module:
    # GradSampleModule stores the actual PyTorch module on _module.
    if hasattr(model, "_module"): return getattr(model, "_module")
    return model


# Fail before wrapping a model twice with Opacus.
def _reject_already_wrapped(model: nn.Module) -> None:
    # Import the wrapper type lazily so no-DP runs do not touch Opacus.
    wrapper_cls = cast(type, load_opacus_symbol("opacus.grad_sample", "AbstractGradSampleModule"))

    if isinstance(model, wrapper_cls): raise RuntimeError("DP model is already wrapped by Opacus")


# Reset AdamW moment buffers while preserving the DP optimizer wrapper and accountant.
def _reset_optimizer_state(optimizer: torch.optim.Optimizer) -> None:
    # Clear the public optimizer state used by plain AdamW and Opacus DPOptimizer.
    optimizer.state.clear()

    # Clear the wrapped base optimizer state when Opacus keeps it separately.
    original_optimizer = getattr(optimizer, "original_optimizer", None)
    if original_optimizer is not None: original_optimizer.state.clear()


# CLASS: Train one simulated bank through the Flower NumPyClient contract.
class FederatedBankClient(NumPyClient):
    # Store static client inputs and shared losses for synchronous local fit calls.
    # The client index, the participant count and the key order are the public inputs its own masks need.
    def __init__(self, config: FederatedRunConfig, client_context: FederatedClientContext,
                 run_context: FederatedRunContext, device: torch.device, outcome_loss: nn.Module,
                 next_activity_loss: nn.Module, client_index: int, n_clients: int,
                 parameter_keys: list[str]) -> None:
        self.config = config
        self.client_context = client_context
        self.run_context = run_context
        self.device = device
        self.outcome_loss = outcome_loss
        self.next_activity_loss = next_activity_loss
        self.client_index = int(client_index)
        self.n_clients = int(n_clients)
        self.parameter_keys = list(parameter_keys)
        self._dp_model: Optional[nn.Module] = None
        self._dp_optimizer: Optional[torch.optim.Optimizer] = None
        self._dp_loader: Optional[DataLoader] = None
        self._privacy_engine: Any = None

    # Build and wrap the persistent Opacus client state on the first DP fit.
    def _ensure_dp_state(self, parameters: list[np.ndarray]) -> tuple[nn.Module, torch.optim.Optimizer, DataLoader]:
        # Reuse the wrapped DP state after the first round.
        if self._dp_model is not None and self._dp_optimizer is not None and self._dp_loader is not None:
            load_numpy_parameters(_dp_base_model(self._dp_model), parameters)
            return self._dp_model, self._dp_optimizer, self._dp_loader

        # Build the DPLSTM client model, load server parameters and guard against double wrapping.
        model = build_model_for_federated(
            self.config, self.run_context.vocabularies, self.run_context.schema_profile, use_dp=True
        ).to(self.device)
        load_numpy_parameters(model, parameters)
        _reject_already_wrapped(model)
        optimizer = core.build_multitask_optimizer(
            model, self.config.learning_rate, self.config.weight_decay, self.config.outcome_lr_scale,
            self.config.next_activity_lr_scale, self.config.remaining_time_lr_scale
        )

        # Wrap the client loader once with PrivacyEngine and keep secure_mode false as the documented limitation.
        configure_dp_warning_filters()
        privacy_engine_cls = load_opacus_symbol("opacus", "PrivacyEngine")

        privacy_engine = privacy_engine_cls(accountant="rdp", secure_mode=False)
        model, optimizer, loader = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=self.client_context.train_loader,
            target_epsilon=float(self.config.dp_target_epsilon),
            target_delta=float(self.config.dp_delta),
            epochs=max(int(self.config.max_rounds) * int(self.config.local_epochs), 1),
            max_grad_norm=float(self.config.dp_max_grad_norm),
            alphas=list(DP_RDP_ALPHAS),
        )
        self._dp_model = model
        self._dp_optimizer = optimizer
        self._dp_loader = loader
        self._privacy_engine = privacy_engine
        return model, optimizer, loader

    # Fit one client from server parameters and return the updated tensors with prefix-count weight.
    def fit(self, parameters: list[np.ndarray],
            fit_config: dict[str, Any]) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        # Build or reuse the client model and load the current global server parameters.
        round_idx = int(fit_config.get("server_round", fit_config.get("round", 1)))
        proximal_mu = float(fit_config.get("proximal_mu", 0.0))
        if self.config.use_dp:
            model, optimizer, train_loader = self._ensure_dp_state(parameters)
        else:
            model = build_model_for_federated(
                self.config, self.run_context.vocabularies, self.run_context.schema_profile, use_dp=False
            ).to(self.device)
            load_numpy_parameters(model, parameters)
            optimizer = core.build_multitask_optimizer(
                model, self.config.learning_rate, self.config.weight_decay, self.config.outcome_lr_scale,
                self.config.next_activity_lr_scale, self.config.remaining_time_lr_scale
            )
            train_loader = self.client_context.train_loader

        # Reset AdamW moments per client fit so federation does not carry optimizer memory across rounds.
        _reset_optimizer_state(optimizer)

        # Snapshot reference parameters in module order for the FedProx proximal term.
        reference_params = [param.detach().clone() for param in model.parameters()]

        # Apply the round learning rates after loading server parameters and before local fitting.
        apply_optimizer_lrs(optimizer, optimizer_lrs_for_round(self.config, round_idx))

        # Run the configured number of local epochs, one in production for E_05 comparability.
        train_metrics: dict[str, float] = {}
        for local_epoch in range(1, self.config.local_epochs + 1):
            train_metrics = train_federated_epoch(
                model, train_loader, self.outcome_loss, self.next_activity_loss,
                self.run_context.remaining_time_repr, self.config.remaining_time_huber_beta, optimizer,
                reference_params, proximal_mu, self.device, self.config, use_dp=self.config.use_dp,
                progress_label=f"bank {self.client_context.bank} round {round_idx} epoch {local_epoch}",
                next_activity_mask_context=self.client_context.training_next_activity_mask_context,
            )

        # Evaluate the client validation split through the E_05 metric path without building prediction frames.
        val_metrics, _ = evaluate_model_for_federated(
            model, self.client_context.val_loader, self.outcome_loss, self.next_activity_loss,
            self.run_context.remaining_time_repr, self.config.remaining_time_huber_beta, self.device, self.config,
            self.client_context.val_dataset, progress_label=f"bank {self.client_context.bank} val",
            collect_predictions=False,
            next_activity_mask_context=self.run_context.evaluation_next_activity_mask_context,
        )

        # Return only scalar metrics because Flower metrics are flat scalar dictionaries.
        # The train prefix count travels on this reporting channel, which weights the round-log client averages.
        metrics: dict[str, Any] = {
            "round": round_idx,
            "bank": self.client_context.bank,
            "train_prefix_count": int(self.client_context.train_prefix_count),
            "train_loss_total": float(train_metrics["loss_total"]),
            "train_loss_total_with_fedprox": float(train_metrics["loss_total_with_fedprox"]),
            "train_fedprox_penalty": float(train_metrics["train_fedprox_penalty"]),
            "n_batches": int(train_metrics["n_batches"]),
            "val_loss_total": float(val_metrics["loss_total"]),
            "val_outcome_macro_f1": float(val_metrics["outcome"]["macro_f1"]),
            "val_next_activity_top1": float(val_metrics["next_activity"]["top1_accuracy"]),
            "val_remaining_time_mae": float(val_metrics["remaining_time"]["mae"]),
        }
        if self.config.use_dp and self._privacy_engine is not None:
            metrics["dp_epsilon_spent"] = float(
                self._privacy_engine.accountant.get_epsilon(
                    delta=float(self.config.dp_delta), alphas=list(DP_RDP_ALPHAS)
                )
            )
            metrics["dp_noise_multiplier"] = float(getattr(optimizer, "noise_multiplier", 0.0))
        return model_parameters_to_numpy(_dp_base_model(model)), int(self.client_context.train_prefix_count), metrics

    # Fit one client and return its already masked message list for the secure-aggregation simulation.
    # The raw update and the raw weight stay inside this method, so only masked arrays cross the boundary.
    def fit_secure(self, parameters: list[np.ndarray],
                   fit_config: dict[str, Any]) -> tuple[list[SecureTensorMessage], dict[str, Any]]:
        round_idx = int(fit_config.get("server_round", fit_config.get("round", 1)))
        params, train_prefix_count, metrics = self.fit(parameters, fit_config)
        message = build_secure_client_message(
            params,
            train_prefix_count,
            self.client_index,
            self.n_clients,
            next_activity_head_agg=self.config.next_activity_head_agg,
            parameter_keys=self.parameter_keys,
            next_activity_class_counts=self.client_context.next_activity_class_counts,
            is_joint_run=self.config.dataset == "joint",
            round_index=round_idx,
            secure_aggregation_seed=self.config.secure_aggregation_seed,
        )
        return message, metrics

    # Evaluate one client validation split when Flower calls the optional evaluate-contract.
    def evaluate(self, parameters: list[np.ndarray], eval_config: dict[str, Any]) -> tuple[float, int, dict[str, Any]]:
        # Load server parameters into a fresh E_05 model before metrics-only validation.
        model = build_model_for_federated(
            self.config, self.run_context.vocabularies, self.run_context.schema_profile, use_dp=False
        ).to(self.device)
        load_numpy_parameters(model, parameters)
        metrics, _ = evaluate_model_for_federated(
            model, self.client_context.val_loader, self.outcome_loss, self.next_activity_loss,
            self.run_context.remaining_time_repr, self.config.remaining_time_huber_beta, self.device, self.config,
            self.client_context.val_dataset, progress_label=f"bank {self.client_context.bank} eval",
            collect_predictions=False,
            next_activity_mask_context=self.run_context.evaluation_next_activity_mask_context,
        )
        return (float(metrics["loss_total"]), int(self.client_context.train_prefix_count),
                {"bank": self.client_context.bank})


# CLASS: Store the bookkeeping of the synchronous loop while retaining the Flower FedAvg strategy type.
class SavingFedAvgStrategy(FedAvg):
    # Initialize Flower FedAvg and the local trackers for the best global state.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latest_params: Optional[list[np.ndarray]] = None
        self.best_params: Optional[list[np.ndarray]] = None
        self.best_round: Optional[int] = None
        self.best_validation_loss: float = float("inf")
        self.round_history: list[dict[str, Any]] = []

    # Track pooled validation and update the best server parameters by val_loss_total.
    def record_server_validation(self, round_idx: int, params: list[np.ndarray], metrics: dict[str, Any]) -> None:
        self.latest_params = [np.asarray(array, dtype=np.float32).copy() for array in params]
        validation_loss = float(metrics["loss_total"])
        record = {"round": int(round_idx), "val_loss_total": validation_loss}
        self.round_history.append(record)
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.best_round = int(round_idx)
            self.best_params = [np.asarray(array, dtype=np.float32).copy() for array in params]


# CLASS: Store the bookkeeping of the synchronous loop while retaining the Flower FedProx strategy type.
class SavingFedProxStrategy(FedProx):
    # Initialize Flower FedProx and the local trackers for the best global state.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.latest_params: Optional[list[np.ndarray]] = None
        self.best_params: Optional[list[np.ndarray]] = None
        self.best_round: Optional[int] = None
        self.best_validation_loss: float = float("inf")
        self.round_history: list[dict[str, Any]] = []

    # Track pooled validation and update the best server parameters by val_loss_total.
    def record_server_validation(self, round_idx: int, params: list[np.ndarray], metrics: dict[str, Any]) -> None:
        self.latest_params = [np.asarray(array, dtype=np.float32).copy() for array in params]
        validation_loss = float(metrics["loss_total"])
        record = {"round": int(round_idx), "val_loss_total": validation_loss}
        self.round_history.append(record)
        if validation_loss < self.best_validation_loss:
            self.best_validation_loss = validation_loss
            self.best_round = int(round_idx)
            self.best_params = [np.asarray(array, dtype=np.float32).copy() for array in params]


# Build the Flower strategy wrapper for parameter bookkeeping in the synchronous loop.
def build_strategy(config: FederatedRunConfig, initial_params: list[np.ndarray],
                   evaluate_fn: Callable[[int, list[np.ndarray], dict[str, Any]], tuple[float, dict[str, Any]]]) -> Any:
    # Share one fit config function so every round receives its index and FedProx mu when active.
    def fit_config(round_idx: int) -> dict[str, Any]:
        payload: dict[str, Any] = {"server_round": int(round_idx)}
        if config.strategy == "fedprox": payload["proximal_mu"] = float(config.fedprox_mu)
        return payload

    # Use Flower strategy classes for strategy identity and fit config, while aggregation stays in the synchronous loop.
    common_kwargs = {
        "fraction_fit": 1.0,
        "fraction_evaluate": 1.0,
        "min_fit_clients": int(config.n_clients),
        "min_evaluate_clients": int(config.n_clients),
        "min_available_clients": int(config.n_clients),
        "evaluate_fn": evaluate_fn,
        "on_fit_config_fn": fit_config,
        "initial_parameters": ndarrays_to_parameters(initial_params),
    }
    if config.strategy == "fedavg": return SavingFedAvgStrategy(**common_kwargs)
    if config.strategy == "fedprox": return SavingFedProxStrategy(**common_kwargs, proximal_mu=float(config.fedprox_mu))
    raise ValueError(f"unknown federated strategy: {config.strategy}")


# ----------------------------------------------------------------------------------------------------------------------
# 6. ROUND LOOP AND FINAL ARTIFACTS

# Aggregate one scalar metric from client fit metrics by client prefix count.
def _weighted_client_metric(client_metrics: list[dict[str, Any]], key: str) -> float:
    # Return zero when no client reports the optional metric.
    available = [metrics for metrics in client_metrics if key in metrics]
    if not available: return 0.0

    # Weight client train metrics by the local train prefix counts they report.
    total = float(sum(float(metrics["train_prefix_count"]) for metrics in available))
    if total <= 0.0: return 0.0
    return float(sum(float(metrics[key]) * (float(metrics["train_prefix_count"]) / total) for metrics in available))


# Aggregate one optional scalar metric as the minimum across reporting clients.
def _min_client_metric(client_metrics: list[dict[str, Any]], key: str) -> float:
    # Return zero when no client reports the optional metric.
    values = [float(metrics[key]) for metrics in client_metrics if key in metrics]
    if not values: return 0.0
    return min(values)


# Aggregate one optional scalar metric as the maximum across reporting clients.
def _max_client_metric(client_metrics: list[dict[str, Any]], key: str) -> float:
    # Return zero when no client reports the optional metric.
    values = [float(metrics[key]) for metrics in client_metrics if key in metrics]
    if not values: return 0.0
    return max(values)


# Build a compact RT target diagnostic block from the pooled representation.
def target_diagnostics_from_context(config: FederatedRunConfig, context: FederatedRunContext) -> dict[str, Any]:
    # Use the full E_05 diagnostic helper when the pooled train dataset exposes tensor arrays.
    dataset = context.pooled_train_dataset
    if hasattr(dataset, "arrays") or hasattr(dataset, "samples"):
        diagnostics = baseline.remaining_time_target_diagnostics(dataset, context.remaining_time_repr)
    else:
        diagnostics = {
            "remaining_time_transform": context.remaining_time_repr.transform,
            "remaining_time_scaling": context.remaining_time_repr.scaling,
            "remaining_time_center": float(context.remaining_time_repr.center),
            "remaining_time_scale": float(context.remaining_time_repr.scale),
            "remaining_time_use_softplus": bool(context.remaining_time_repr.use_softplus),
            "train_median_model_units": float(context.remaining_time_repr.median_model_units),
        }

    # Record the E_06 loss and outcome regularization knobs in addition to the target representation.
    diagnostics["remaining_time_huber_beta"] = float(config.remaining_time_huber_beta)
    diagnostics["outcome_head_dropout"] = (
        float(config.outcome_head_dropout) if config.outcome_head_dropout is not None else float(config.dropout)
    )
    return diagnostics


# Write final validation, test, per-bank and checkpoint artifacts for one E_06 run.
def write_final_artifacts(config: FederatedRunConfig, output_dir: Path, model: core.MultitaskLSTM,
                          context: FederatedRunContext, outcome_loss: nn.Module, next_activity_loss: nn.Module,
                          device: torch.device, round_rows: list[dict[str, Any]], round_metrics: list[dict[str, Any]],
                          strategy: Union[SavingFedAvgStrategy, SavingFedProxStrategy],
                          timings: dict[str, float],
                          secure_aggregation: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    # Evaluate final validation and pooled test with prediction frames for traceability.
    val_metrics, val_predictions = evaluate_model_for_federated(
        model, context.pooled_val_loader, outcome_loss, next_activity_loss, context.remaining_time_repr,
        config.remaining_time_huber_beta, device, config, context.pooled_val_dataset,
        progress_label="final federated val", collect_predictions=True,
        next_activity_mask_context=context.evaluation_next_activity_mask_context,
    )

    # Resolve the per-client prefix subsets of the pooled test split before the single pooled test pass.
    pooled_config = baseline_config_for_federated(config, "centralized")
    bank_selectors = baseline.bank_selectors_for_dataset(
        context.pooled_test_dataset,
        tuple(client.bank for client in context.clients),
        baseline.bank_by_case_for_split(pooled_config, context.mapping, "test"),
    )
    per_bank: dict[str, Any] = {}
    test_metrics, test_predictions = evaluate_model_for_federated(
        model, context.pooled_test_loader, outcome_loss, next_activity_loss, context.remaining_time_repr,
        config.remaining_time_huber_beta, device, config, context.pooled_test_dataset,
        progress_label="federated test", collect_predictions=True,
        next_activity_mask_context=context.evaluation_next_activity_mask_context,
        subset_selectors=bank_selectors, subset_metrics=per_bank,
    )
    if val_predictions is None or test_predictions is None:
        raise RuntimeError("final E_06 evaluation must export predictions")

    # Write pooled prediction and metric artifacts.
    prediction_artifact_path(output_dir, "predictions_val.parquet").parent.mkdir(parents=True, exist_ok=True)
    val_predictions.to_parquet(prediction_artifact_path(output_dir, "predictions_val.parquet"), index=False)
    test_predictions.to_parquet(prediction_artifact_path(output_dir, "predictions_test.parquet"), index=False)
    prefix_bucket_metrics_val = baseline.compute_prefix_bucket_metrics(val_predictions)
    prefix_bucket_metrics_test = baseline.compute_prefix_bucket_metrics(test_predictions)
    save_legacy_json(config, run_artifact_path(output_dir, "validation_metrics.json"), val_metrics)
    save_legacy_json(config, run_artifact_path(output_dir, "test_metrics.json"), test_metrics)
    save_legacy_json(config, run_artifact_path(output_dir, "prefix_bucket_metrics_val.json"), prefix_bucket_metrics_val)
    save_legacy_json(config, run_artifact_path(output_dir, "prefix_bucket_metrics_test.json"),
                     prefix_bucket_metrics_test)
    baseline.write_prefix_bucket_summary_csv(output_dir, test_predictions, script_id=SCRIPT_ID)

    # Write remaining-time baselines with the train median when it is available.
    target_diagnostics = target_diagnostics_from_context(config, context)
    train_median_seconds = float(target_diagnostics.get("train_median_seconds", context.remaining_time_repr.scale))
    remaining_time_baselines_val = baseline.remaining_time_baseline_diagnostics(val_predictions, train_median_seconds)
    remaining_time_baselines_test = baseline.remaining_time_baseline_diagnostics(test_predictions, train_median_seconds)
    save_legacy_json(config, run_artifact_path(output_dir, "target_diagnostics.json"), target_diagnostics)
    save_legacy_json(config, run_artifact_path(output_dir, "remaining_time_baselines_val.json"),
                     remaining_time_baselines_val)
    save_legacy_json(config, run_artifact_path(output_dir, "remaining_time_baselines_test.json"),
                     remaining_time_baselines_test)

    # Export the per-bank slices of the pooled test predictions, so every client table decomposes the global table.
    for bank, selector in bank_selectors.items():
        test_predictions.loc[selector].to_parquet(
            prediction_artifact_path(output_dir, f"predictions_test_bank_{_safe_bank(bank)}.parquet"), index=False
        )
    save_legacy_json(config, run_artifact_path(output_dir, "per_bank_test_metrics.json"), per_bank)

    # Write round diagnostics, checkpoint and coarse timing artifacts.
    pd.DataFrame(round_rows).to_csv(run_artifact_path(output_dir, "round_log.csv"), index=False)
    save_legacy_json(config, run_artifact_path(output_dir, "round_metrics.json"), {"rounds": round_metrics})
    curve_files = [
        script_artifact_name("loss_curves.png"),
        script_artifact_name("task_loss_curves.png"),
        script_artifact_name("outcome_macro_f1_curve.png"),
        script_artifact_name("next_activity_accuracy_curve.png"),
        script_artifact_name("remaining_time_mae_curve.png"),
    ]
    baseline.plot_training_curves(
        federated_curve_history(round_rows, round_metrics),
        output_dir,
        artifact_path_fn=run_artifact_path,
        x_label="Round",
    )
    write_federated_summary(output_dir, test_metrics, per_bank)
    best_round_diagnostics: dict[str, Any] = {
        "best_round": strategy.best_round,
        "best_validation_loss": strategy.best_validation_loss,
    }
    if config.use_dp and round_rows:
        best_round_diagnostics["dp_epsilon_spent"] = max(float(row.get("dp_epsilon_spent", 0.0)) for row in round_rows)
    save_legacy_json(config, run_artifact_path(output_dir, "best_round_diagnostics.json"), best_round_diagnostics)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": _config_payload(config, resolved_device=str(device)),
            "best_validation_loss": strategy.best_validation_loss,
            "best_round": strategy.best_round,
        },
        run_artifact_path(output_dir, "model_best.pt"),
    )
    timing_payload = {key: float(value) for key, value in timings.items()}
    save_legacy_json(config, run_artifact_path(output_dir, "timing.json"), timing_payload)
    diagnostics_payload: dict[str, Any] = {
        "target": target_diagnostics,
        "best_round": best_round_diagnostics,
        "prefix_buckets": {
            "validation": prefix_bucket_metrics_val,
            "test": prefix_bucket_metrics_test,
        },
        "remaining_time_baselines": {
            "validation": remaining_time_baselines_val,
            "test": remaining_time_baselines_test,
        },
        "predictions": {
            "validation": (Path(PREDICTIONS_DIR_NAME) / script_artifact_name("predictions_val.parquet")).as_posix(),
            "test": (Path(PREDICTIONS_DIR_NAME) / script_artifact_name("predictions_test.parquet")).as_posix(),
        },
        "curves": {"files": curve_files},
        "rounds": {
            "history": round_metrics,
            "log": script_artifact_name("round_log.csv"),
        },
        "timing": timing_payload,
    }
    if secure_aggregation is not None:
        diagnostics_payload["secure_aggregation"] = secure_aggregation
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
    return test_metrics


# Train one federated run end to end and write the E_06 artifact family.
def run_one_federated(config: FederatedRunConfig) -> dict[str, Any]:
    # Reject combining the secure aggregation simulation with the DP-SGD path, which this run does not support.
    if config.secure_aggregation_simulation and config.use_dp:
        raise ValueError("secure aggregation simulation and DP-SGD cannot be combined in one run")

    # Resolve output, seed and device before data loading so artifacts record failed-run context early.
    timings: dict[str, float] = {}
    start = time.perf_counter()
    output_dir = output_dir_for_run(config.output_root, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    core.set_global_seed(config.seed)
    device = core.select_device(config.device)
    save_legacy_json(config, run_artifact_path(output_dir, "config.json"),
                     _config_payload(config, resolved_device=str(device)))

    # Load federated context and construct the shared global objective from pooled train statistics.
    context = load_federated_context(config)
    global_model = build_model_for_federated(
        config, context.vocabularies, context.schema_profile, use_dp=config.use_dp).to(device)
    outcome_loss = baseline.build_outcome_loss_from_labels(
        baseline_config_for_federated(config, "centralized"), context.pooled_train_labels, device
    )
    next_activity_loss = nn.CrossEntropyLoss(reduction="none")
    core.initialize_remaining_time_head_bias(global_model, context.remaining_time_repr)
    current_params = model_parameters_to_numpy(global_model)

    # Build Flower clients and a saving strategy for the bookkeeping of the synchronous loop.
    # Each client receives its index, the participant count and the key order, the public inputs its masks need.
    parameter_keys = parameter_key_order(global_model)
    clients = [
        FederatedBankClient(
            config=config,
            client_context=client_context,
            run_context=context,
            device=device,
            outcome_loss=outcome_loss,
            next_activity_loss=next_activity_loss,
            client_index=client_index,
            n_clients=len(context.clients),
            parameter_keys=parameter_keys,
        )
        for client_index, client_context in enumerate(context.clients)
    ]
    strategy = build_strategy(config, current_params, lambda _round, _params, _config: (0.0, {}))

    # Measure the reconstruction deviation of the masking construction once, on synthetic tensors of this run's shapes.
    # The check never reads a client update, so it can run inside a secure run without weakening the boundary.
    secure_aggregation_deviation = 0.0
    if config.secure_aggregation_simulation:
        secure_aggregation_deviation = secure_construction_deviation(
            [tuple(array.shape) for array in current_params],
            parameter_keys,
            len(context.clients),
            config.next_activity_head_agg,
            config.dataset == "joint",
            config.secure_aggregation_seed,
        )

    # Run full-participation synchronous rounds and stop on pooled val_loss_total patience.
    wait = 0
    early_stopped = False
    round_rows: list[dict[str, Any]] = []
    round_metrics: list[dict[str, Any]] = []
    round_iter = progress_iter_for_federated(
        range(1, config.max_rounds + 1), config, f"rounds {config.dataset} {config.run_name}",
        total=config.max_rounds, unit="round", leave=True,
    )
    for round_idx in round_iter:
        fit_config: dict[str, Any] = {"server_round": round_idx}
        if config.strategy == "fedprox": fit_config["proximal_mu"] = float(config.fedprox_mu)
        if config.secure_aggregation_simulation:
            # Every client masks its own contribution and weight, so the loop only ever holds masked messages.
            secure_results = [client.fit_secure(current_params, fit_config) for client in clients]
            client_metrics = [metrics for _, metrics in secure_results]
            current_params = secure_aggregate_from_masked_messages([message for message, _ in secure_results])
        else:
            client_results = [client.fit(current_params, fit_config) for client in clients]
            client_metrics = [metrics for _, _, metrics in client_results]
            next_activity_class_counts = [client.client_context.next_activity_class_counts for client in clients]
            next_activity_contribution_payloads = None
            if config.dataset == "joint" and config.next_activity_head_agg == "contribution":
                next_activity_contribution_payloads = [
                    _client_next_activity_contribution_payload(params, parameter_keys, class_counts)
                    for (params, _, _), class_counts in zip(client_results, next_activity_class_counts)
                ]
            current_params = aggregate_parameters(
                [(params, count) for params, count, _ in client_results],
                next_activity_head_agg=config.next_activity_head_agg,
                parameter_keys=parameter_keys,
                next_activity_class_counts=next_activity_class_counts,
                next_activity_contribution_payloads=next_activity_contribution_payloads,
                is_joint_run=config.dataset == "joint",
            )
        load_numpy_parameters(global_model, current_params)

        # Evaluate pooled validation after aggregation and update best global parameters.
        val_metrics, _ = evaluate_model_for_federated(
            global_model, context.pooled_val_loader, outcome_loss, next_activity_loss, context.remaining_time_repr,
            config.remaining_time_huber_beta, device, config, context.pooled_val_dataset,
            progress_label=f"federated val round {round_idx}", collect_predictions=False,
            next_activity_mask_context=context.evaluation_next_activity_mask_context,
        )
        previous_best = float(strategy.best_validation_loss)
        strategy.record_server_validation(round_idx, current_params, val_metrics)

        # Store one flat round-log row and the nested metrics for auditability.
        lrs = optimizer_lrs_for_round(config, round_idx)
        row = {
            "round": int(round_idx),
            "val_loss_total": float(val_metrics["loss_total"]),
            "val_outcome_macro_f1": float(val_metrics["outcome"]["macro_f1"]),
            "val_next_activity_top1": float(val_metrics["next_activity"]["top1_accuracy"]),
            "val_remaining_time_mae": float(val_metrics["remaining_time"]["mae"]),
            "learning_rate": float(lrs["trunk"]),
            "learning_rate_outcome": float(lrs["outcome"]),
            "learning_rate_next_activity": float(lrs["next_activity"]),
            "learning_rate_remaining_time": float(lrs["remaining_time"]),
            "train_loss_total": _weighted_client_metric(client_metrics, "train_loss_total"),
            "train_loss_total_with_fedprox": _weighted_client_metric(client_metrics, "train_loss_total_with_fedprox"),
            "train_fedprox_penalty": _weighted_client_metric(client_metrics, "train_fedprox_penalty"),
            "train_n_batches_weighted": _weighted_client_metric(client_metrics, "n_batches"),
            "train_n_batches_min": _min_client_metric(client_metrics, "n_batches"),
            "train_n_batches_max": _max_client_metric(client_metrics, "n_batches"),
        }
        if config.use_dp:
            row["dp_epsilon_spent"] = _max_client_metric(client_metrics, "dp_epsilon_spent")
            row["dp_noise_multiplier"] = _max_client_metric(client_metrics, "dp_noise_multiplier")
        round_rows.append(row)
        round_metrics.append({"round": int(round_idx), "pooled_validation": val_metrics,
                              "clients": client_metrics})
        if hasattr(round_iter, "set_postfix"):
            round_iter.set_postfix({
                "val_loss": f"{float(row['val_loss_total']):.4f}",
                "next_top1": f"{float(row['val_next_activity_top1']):.3f}",
            })

        # Apply E_05-style patience on pooled val_loss_total.
        if float(val_metrics["loss_total"]) < previous_best:
            wait = 0
        else:
            wait += 1
            if wait >= config.early_stopping_patience:
                early_stopped = True
                break

    # Record the secure aggregation evidence block only when the simulation ran.
    # It claims to mask cancellation plus the message flow. It denies input privacy because the seed is recorded.
    secure_aggregation_summary: Optional[dict[str, Any]] = None
    if config.secure_aggregation_simulation:
        secure_aggregation_summary = {
            "enabled": True,
            "seed": int(config.secure_aggregation_seed),
            "masking_construction": "pairwise_additive",
            "mask_bound": float(SECURE_AGGREGATION_MASK_BOUND),
            "client_side_masking": True,
            "production_protocol_implemented": False,
            "orchestrator_receives_masked_messages_only": True,
            "provides_cryptographic_input_privacy": False,
            "pair_seed_derivation": "recorded public seed, for bit-reproducibility rather than secrecy",
            "deviation_source": "construction check on synthetic tensors of this run's shapes, no client update read",
            "reconstruction_matches_plain_within_tolerance": bool(
                secure_aggregation_deviation <= SECURE_AGGREGATION_TOLERANCE),
            "max_reconstruction_abs_deviation": float(secure_aggregation_deviation),
            "limitations": [
                "one-machine simulation",
                "no key exchange",
                "no authentication",
                "no dropout handling",
                "no differential privacy",
                "per-client scalar training metrics travel unmasked on the reporting channel",
                "the pair seeds derive from the recorded public secure_aggregation_seed, so anyone holding this "
                "report can regenerate any client mask and recover that client's individual update; the simulation "
                "demonstrates mask cancellation and the structural client-to-server message boundary, not "
                "cryptographic input privacy",
                "a deployment derives the pair secrets through Diffie-Hellman key agreement between clients, which "
                "the server never learns; that key agreement is not implemented here",
            ],
        }

    # Restore the best global parameters before final validation and test exports.
    if strategy.best_params is None: strategy.best_params = current_params
    load_numpy_parameters(global_model, strategy.best_params)
    timings["run_total_seconds"] = time.perf_counter() - start
    test_metrics = write_final_artifacts(
        config, output_dir, global_model, context, outcome_loss, next_activity_loss, device,
        round_rows, round_metrics, strategy, timings, secure_aggregation=secure_aggregation_summary
    )
    return {
        "best_validation_loss": float(strategy.best_validation_loss),
        "best_round": int(strategy.best_round or 0),
        "rounds_completed": len(round_rows),
        "early_stopped": bool(early_stopped),
        "test_metrics": test_metrics,
    }


# ----------------------------------------------------------------------------------------------------------------------
# 7. CLI OVERRIDES

# Parse one E_06 federated run invocation.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one E_06 federated model.")

    # Select the dataset, partition configuration and federated strategy.
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--heterogeneity", default=HETEROGENEITY)
    parser.add_argument("--n-clients", type=int, default=N_CLIENTS)
    parser.add_argument("--strategy", choices=["fedavg", "fedprox"], default=STRATEGY)
    parser.add_argument("--next-activity-head-agg", choices=["sample", "equal", "contribution"],
                        default=NEXT_ACTIVITY_HEAD_AGG)
    parser.add_argument("--secure-aggregation-simulation", action="store_true",
                        dest="secure_aggregation_simulation", default=SECURE_AGGREGATION_SIMULATION)
    parser.add_argument("--secure-aggregation-seed", type=int, default=SECURE_AGGREGATION_SEED)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--local-epochs", type=int, default=LOCAL_EPOCHS)
    parser.add_argument("--fedprox-mu", type=float, default=FEDPROX_MU)

    # Configure the optional DP-SGD path and reproducibility.
    parser.add_argument("--use-dp", action="store_true", dest="use_dp", default=USE_DP)
    parser.add_argument("--no-use-dp", action="store_false", dest="use_dp")
    parser.add_argument("--dp-target-epsilon", type=float, default=DP_TARGET_EPSILON)
    parser.add_argument("--dp-delta", type=float, default=DP_DELTA)
    parser.add_argument("--dp-max-grad-norm", type=float, default=DP_MAX_GRAD_NORM)
    parser.add_argument("--dp-smoke-max-batches", type=int, default=DP_SMOKE_MAX_BATCHES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--no-progress-bars", action="store_false", dest="progress_bars", default=PROGRESS_BARS)
    parser.add_argument("--reporting-profile", choices=["compact", "debug"], default=REPORTING_PROFILE)

    # Configure optimizer, scheduler and data loading.
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--gradient-clip-norm", type=float, default=GRADIENT_CLIP_NORM)
    parser.add_argument("--lr-scheduler", choices=["cosine", "none"], default=LR_SCHEDULER)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=LR_SCHEDULER_MIN_LR)
    parser.add_argument("--lr-scheduler-t-max", type=int, default=LR_SCHEDULER_T_MAX)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)

    # Configure model architecture and task losses.
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--num-layers", type=int, default=NUM_LAYERS)
    parser.add_argument("--dropout", type=float, default=DROPOUT)
    parser.add_argument("--head-hidden-size", type=int, default=HEAD_HIDDEN_SIZE)
    parser.add_argument("--outcome-label-smoothing", type=float, default=OUTCOME_LABEL_SMOOTHING)
    parser.add_argument("--outcome-class-weight-power", type=float, default=OUTCOME_CLASS_WEIGHT_POWER)
    parser.add_argument("--outcome-lr-scale", type=float, default=OUTCOME_LR_SCALE)
    parser.add_argument("--next-activity-lr-scale", type=float, default=NEXT_ACTIVITY_LR_SCALE)
    parser.add_argument("--remaining-time-lr-scale", type=float, default=REMAINING_TIME_LR_SCALE)
    parser.add_argument("--outcome-loss-weight", type=float, default=OUTCOME_LOSS_WEIGHT)
    parser.add_argument("--next-activity-loss-weight", type=float, default=NEXT_ACTIVITY_LOSS_WEIGHT)
    parser.add_argument("--remaining-time-loss-weight", type=float, default=REMAINING_TIME_LOSS_WEIGHT)
    parser.add_argument("--remaining-time-transform", choices=["raw", "log"], default=REMAINING_TIME_TRANSFORM)
    parser.add_argument("--remaining-time-scaling", choices=["raw", "median", "zscore"], default=REMAINING_TIME_SCALING)
    parser.add_argument("--remaining-time-huber-beta", type=float, default=REMAINING_TIME_HUBER_BETA)
    parser.add_argument("--outcome-head-dropout", type=float, default=OUTCOME_HEAD_DROPOUT)

    # Configure output and cache roots for matrix, smoke and test workflow calls.
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--cache-root", type=Path, default=CACHE_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=ARTIFACT_ROOT)
    return parser.parse_args(argv)


# Convert parsed CLI values into one immutable federated config.
def config_from_args(args: argparse.Namespace) -> FederatedRunConfig:
    return FederatedRunConfig(
        dataset=args.dataset,
        heterogeneity=args.heterogeneity,
        n_clients=args.n_clients,
        strategy=args.strategy,
        next_activity_head_agg=args.next_activity_head_agg,
        secure_aggregation_simulation=args.secure_aggregation_simulation,
        secure_aggregation_seed=args.secure_aggregation_seed,
        max_rounds=args.max_rounds,
        local_epochs=args.local_epochs,
        fedprox_mu=args.fedprox_mu,
        use_dp=args.use_dp,
        dp_target_epsilon=args.dp_target_epsilon,
        dp_delta=args.dp_delta,
        dp_max_grad_norm=args.dp_max_grad_norm,
        dp_smoke_max_batches=args.dp_smoke_max_batches,
        seed=args.seed,
        device=args.device,
        progress_bars=args.progress_bars,
        reporting_profile=args.reporting_profile,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
        lr_scheduler=args.lr_scheduler,
        lr_scheduler_min_lr=args.lr_scheduler_min_lr,
        lr_scheduler_t_max=args.lr_scheduler_t_max,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        head_hidden_size=args.head_hidden_size,
        outcome_label_smoothing=args.outcome_label_smoothing,
        outcome_class_weight_power=args.outcome_class_weight_power,
        outcome_lr_scale=args.outcome_lr_scale,
        next_activity_lr_scale=args.next_activity_lr_scale,
        remaining_time_lr_scale=args.remaining_time_lr_scale,
        outcome_loss_weight=args.outcome_loss_weight,
        next_activity_loss_weight=args.next_activity_loss_weight,
        remaining_time_loss_weight=args.remaining_time_loss_weight,
        remaining_time_transform=args.remaining_time_transform,
        remaining_time_scaling=args.remaining_time_scaling,
        remaining_time_huber_beta=args.remaining_time_huber_beta,
        outcome_head_dropout=args.outcome_head_dropout,
        output_root=args.output_root,
        cache_root=args.cache_root,
        artifact_root=args.artifact_root,
    )


# ----------------------------------------------------------------------------------------------------------------------
# 8. ENTRY POINT

# MAIN: Execute one configured E_06 federated run per invocation.
def main(argv: Optional[list[str]] = None) -> None:
    run_one_federated(config_from_args(parse_args(argv)))


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────