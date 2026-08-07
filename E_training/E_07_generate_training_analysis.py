"""
Step 7: Generate a compact training analysis from the E_05 and E_06 outputs.

Pipeline:
- Discovers every run below the output root: baselines, federated and differential-privacy runs
- Reads compact run reports first and falls back to the legacy per-file JSON fragments
- Flattens each run into one summary row plus per-bank rows with a shared metric column set
- Checks the observed runs against the expected no-DP matrix and records data-quality warnings
- Builds the comparison tables: federated versus centralized, FedProx versus FedAvg, joint source balance and fairness
- Writes the Markdown report, the JSON payload and the two CSV summaries into the analysis root

Run: WORKFLOW_run_FULL_PIPELINE.sh runs this as the analysis stage or call the script directly with both roots.
The analysis is read-only over the training outputs and never modifies a run directory.

REQUIRED FILES:
    <output_root>/baselines/**/E_05_run_report.json: compact baseline run reports
    <output_root>/federated/**/E_06_run_report.json: compact federated run reports
    <output_root>/differential_privacy/**/E_06_run_report.json: compact DP run reports when the DP experiment exists
    <output_root>/**/E_0*_test_metrics.json: legacy fragments, read only when a compact report is missing

CREATED FILES:
    <analysis_root>/E_07_training_analysis.md: sectioned Markdown analysis for thesis inspection
    <analysis_root>/E_07_training_analysis.json: full payload with all rows and warnings
    <analysis_root>/E_07_run_summary.csv: one row per run with headline metrics and spreads
    <analysis_root>/E_07_per_bank_summary.csv: one row per bank and run for the fairness analysis
    <analysis_root>/DP_privacy_utility_curves.png: headline privacy-utility curves per head against the no-DP baseline
    <analysis_root>/DP_utility_drop_by_head.png: relative utility loss per head and target epsilon against the baseline
    <analysis_root>/DP_convergence_rounds.png: pooled validation loss per round for the DP runs and the no-DP baseline
"""

# IMPORTS
from __future__ import annotations
import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

# ----------------------------------------------------------------------------------------------------------------------
# 1. EXPECTED MATRIX AND COLUMN CONFIGURATION

# Script identifiers shared by the run reports and the matrix keys.
SCRIPT_ID_BASELINE = "E_05"
SCRIPT_ID_FEDERATED = "E_06"

# The three output subtrees this analysis reads, plus the POC subtree it deliberately leaves out.
MATRIX_SUBFOLDERS: tuple[str, ...] = ("baselines", "federated", "differential_privacy")
SECURE_AGGREGATION_SUBFOLDER: str = "secure_aggregation"

# Expected single-dataset splits per dataset family as (heterogeneity, client count) pairs.
SINGLE_RUNS: dict[str, tuple[tuple[str, int], ...]] = {
    "bpic2017": (
        ("iid", 3),
        ("weak", 3),
        ("medium", 3),
        ("strong", 3),
        ("medium", 5),
        ("strong", 5),
    ),
    "bpic2012": (
        ("iid", 3),
        ("weak", 3),
        ("medium", 3),
    ),
}

# Expected joint runs with their dataset-qualified bank lists.
JOINT_RUN_BANKS: dict[str, tuple[str, ...]] = {
    "iid_6banks": ("bpic2017:A", "bpic2017:B", "bpic2017:C", "bpic2012:A", "bpic2012:B", "bpic2012:C"),
    "weak_6banks": ("bpic2017:A", "bpic2017:B", "bpic2017:C", "bpic2012:A", "bpic2012:B", "bpic2012:C"),
    "medium_6banks": ("bpic2017:A", "bpic2017:B", "bpic2017:C", "bpic2012:A", "bpic2012:B", "bpic2012:C"),
    "medium_8banks": (
        "bpic2017:A",
        "bpic2017:B",
        "bpic2017:C",
        "bpic2017:D",
        "bpic2017:E",
        "bpic2012:A",
        "bpic2012:B",
        "bpic2012:C",
    ),
}

# Column order of the run summary CSV.
RUN_SUMMARY_COLUMNS: list[str] = [
    "script_id",
    "dataset",
    "run_name",
    "heterogeneity",
    "n_clients",
    "regime",
    "strategy",
    "bank",
    "seed",
    "learning_rate",
    "fedprox_mu",
    "use_dp",
    "dp_target_epsilon",
    "dp_delta",
    "dp_max_grad_norm",
    "dp_epsilon_spent",
    "max_epochs",
    "max_rounds",
    "local_epochs",
    "best_epoch",
    "best_round",
    "n_prefixes",
    "outcome_macro_f1",
    "outcome_weighted_f1",
    "outcome_balanced_accuracy",
    "next_activity_top1_accuracy",
    "remaining_time_mae_seconds",
    "remaining_time_rmse_seconds",
    "worst_bank_outcome_macro_f1",
    "best_bank_outcome_macro_f1",
    "bank_outcome_macro_f1_spread",
    "worst_bank_remaining_time_mae_seconds",
    "best_bank_remaining_time_mae_seconds",
    "bank_remaining_time_mae_spread_seconds",
    "run_dir",
    "report_path",
]

# Column order of the per-bank summary CSV.
PER_BANK_COLUMNS: list[str] = [
    "script_id",
    "dataset",
    "run_name",
    "regime",
    "strategy",
    "bank",
    "source_dataset",
    "outcome_macro_f1",
    "outcome_weighted_f1",
    "outcome_balanced_accuracy",
    "next_activity_top1_accuracy",
    "remaining_time_mae_seconds",
    "remaining_time_rmse_seconds",
    "n_prefixes",
    "run_dir",
]

# Fixed section order of the Markdown report.
MARKDOWN_SECTIONS: tuple[str, ...] = (
    "Run inventory",
    "Matrix completeness",
    "Best centralized baselines",
    "Best local baselines",
    "Best federated runs",
    "Joint versus single-dataset comparison",
    "Federated versus centralized comparison",
    "FedProx versus FedAvg comparison",
    "Client fairness and worst-client risks",
    "Remaining-time error profile",
    "Next-activity performance profile",
    "Convergence and training stability",
    "Differential privacy profile",
    "Warnings",
    "Artifact index",
)

# ----------------------------------------------------------------------------------------------------------------------
# 2. CLI AND FILE IO HELPERS

# Parse the output root, the analysis root and the strict switch from the CLI.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E_07 training analysis.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)

# HELPER: Read one JSON artifact into a dictionary.
def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

# HELPER: Write one JSON artifact with the stable key order.
def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

# HELPER: Write one CSV summary with the fixed column order and empty cells for missing keys.
def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})

# ----------------------------------------------------------------------------------------------------------------------
# 3. EXPECTED MATRIX AND LEGACY CONFIG PARSING

# HELPER: Build the canonical run name from heterogeneity and client count.
def run_name_from_parts(heterogeneity: str, n_clients: int) -> str:
    return f"{heterogeneity}_{n_clients}banks"

# Build the full set of expected no-DP matrix keys for the completeness check.
def expected_matrix_keys() -> set[str]:
    keys: set[str] = set()

    # Expect one centralized run, one local run per bank and both strategies for every single-dataset split.
    for dataset, specs in SINGLE_RUNS.items():
        for heterogeneity, n_clients in specs:
            run_name = run_name_from_parts(heterogeneity, n_clients)
            keys.add(matrix_key(SCRIPT_ID_BASELINE, dataset, run_name, "centralized", "", ""))
            for bank_index in range(n_clients):
                bank = chr(ord("A") + bank_index)
                keys.add(matrix_key(SCRIPT_ID_BASELINE, dataset, run_name, "local", "", bank))
            for strategy in ("fedavg", "fedprox"):
                keys.add(matrix_key(SCRIPT_ID_FEDERATED, dataset, run_name, "", strategy, ""))

    # Expect the same regime coverage for every joint run with its dataset-qualified banks.
    for run_name, banks in JOINT_RUN_BANKS.items():
        keys.add(matrix_key(SCRIPT_ID_BASELINE, "joint", run_name, "centralized", "", ""))
        for bank in banks:
            keys.add(matrix_key(SCRIPT_ID_BASELINE, "joint", run_name, "local", "", bank))
        for strategy in ("fedavg", "fedprox"):
            keys.add(matrix_key(SCRIPT_ID_FEDERATED, "joint", run_name, "", strategy, ""))
    return keys

# HELPER: Build one stable matrix key from the run identity fields.
# The DP flag is part of the key, so a present DP run can never satisfy a missing no-DP expectation.
def matrix_key(script_id: str, dataset: str, run_name: str, regime: str, strategy: str, bank: str,
               use_dp: bool = False) -> str:
    return "|".join([script_id, dataset, run_name, regime, strategy, bank, "dp" if use_dp else "nodp"])

# HELPER: Split a run folder name into the dataset id and the run name.
def parse_dataset_run(folder_name: str) -> tuple[str, str]:
    for dataset in ("bpic2017", "bpic2012", "joint"):
        prefix = f"{dataset}_"
        if folder_name.startswith(prefix):
            return dataset, folder_name[len(prefix):]
    return "", folder_name

# HELPER: Recover one numeric token from a run directory name, where p encodes the decimal point and m the minus sign.
def parse_float_token(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    if not match:
        return ""
    return match.group(1).replace("p", ".").replace("m", "-")

# Reconstruct the run configuration from a legacy directory name when no report or config file exists.
def config_from_legacy_dir(script_id: str, run_dir: Path, output_root: Path) -> dict[str, Any]:
    # Read dataset, run name and split shape from the parent folder name.
    parent = run_dir.parent.name
    dataset, run_name = parse_dataset_run(parent)
    heterogeneity = run_name.rsplit("_", 1)[0] if "_" in run_name else ""
    n_clients_match = re.search(r"_(\d+)banks$", run_name)
    config: dict[str, Any] = {
        "dataset": dataset,
        "run_name": run_name,
        "heterogeneity": heterogeneity,
        "n_clients": int(n_clients_match.group(1)) if n_clients_match else "",
        "seed": int(parse_float_token(run_dir.name, r"seed_(\d+)") or 42),
        "learning_rate": parse_float_token(run_dir.name, r"lr_([0-9pm]+)"),
    }

    # Baseline directories encode the regime and, for local runs, the bank in the folder name.
    if script_id == SCRIPT_ID_BASELINE:
        config["regime"] = "centralized" if run_dir.name.startswith("centralized_") else "local"
        if config["regime"] == "local":
            bank = run_dir.name.split("local_bank_", 1)[1].split("_seed_", 1)[0]

            # Restore the dataset-qualified colon form for joint banks from the file-safe underscore form.
            if dataset == "joint":
                parts = bank.split("_", 1)
                config["bank"] = f"{parts[0]}:{parts[1]}" if len(parts) == 2 else bank
            else:
                config["bank"] = bank

    # Federated directories encode strategy, FedProx mu, the DP token and the round budget.
    else:
        config["strategy"] = run_dir.name.split("_", 1)[0]
        config["fedprox_mu"] = parse_float_token(run_dir.name, r"mu_([0-9pm]+)")
        config["use_dp"] = "_dp_eps_" in run_dir.name
        config["dp_target_epsilon"] = parse_float_token(run_dir.name, r"dp_eps_([0-9pm]+)")
        rounds_match = re.search(r"_rounds_(\d+)_", run_dir.name)
        local_epochs_match = re.search(r"_le_(\d+)", run_dir.name)
        config["max_rounds"] = int(rounds_match.group(1)) if rounds_match else ""
        config["local_epochs"] = int(local_epochs_match.group(1)) if local_epochs_match else ""
    config["run_dir"] = str(run_dir.relative_to(output_root))
    return config

# ----------------------------------------------------------------------------------------------------------------------
# 4. METRIC EXTRACTION HELPERS

# HELPER: Read one metric from a section block and return empty when absent.
def metric_value(metrics: dict[str, Any], section: str, key: str) -> Any:
    block = metrics.get(section, {})
    return block.get(key, "") if isinstance(block, dict) else ""

# Collect the shared headline metrics of one metric block, accepting both compact and legacy key names.
def metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "outcome_macro_f1": metric_value(metrics, "outcome", "macro_f1"),
        "outcome_weighted_f1": metric_value(metrics, "outcome", "weighted_f1"),
        "outcome_balanced_accuracy": metric_value(metrics, "outcome", "balanced_accuracy"),
        "next_activity_top1_accuracy": metric_value(metrics, "next_activity", "top1_accuracy"),
        "remaining_time_mae_seconds": metric_value(metrics, "remaining_time", "mae_seconds")
        or metric_value(metrics, "remaining_time", "mae"),
        "remaining_time_rmse_seconds": metric_value(metrics, "remaining_time", "rmse_seconds")
        or metric_value(metrics, "remaining_time", "rmse"),
        "n_prefixes": prefix_count(metrics),
    }

# Resolve the prefix count of one metric block, falling back to the confusion-matrix total.
# Reports written before the count became part of the metric block still carry the matrix.
def prefix_count(metrics: dict[str, Any]) -> Any:
    declared = metrics.get("n_prefixes", "")
    if declared not in {"", None}: return declared
    matrix = metric_value(metrics, "outcome", "confusion_matrix")
    if isinstance(matrix, list) and matrix:
        return int(sum(int(cell) for row in matrix for cell in row))
    return ""

# Resolve the run regime for reports whose config does not carry it.
# An E_06 report has no regime key, so a federated row would otherwise print an empty identity column.
def resolve_regime(config: dict[str, Any]) -> str:
    regime = str(config.get("regime", "") or "")
    if regime: return regime
    if str(config.get("script_id", "")) == SCRIPT_ID_FEDERATED:
        return "federated_dp" if is_truthy(config.get("use_dp")) else "federated"
    return regime

# HELPER: Convert one value to a finite float or None.
def numeric_or_none(value: Any) -> Optional[float]:
    if value in {"", None}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

# HELPER: Interpret report booleans that may arrive as strings from legacy configs.
def is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}

# HELPER: Average one metric over rows while skipping empty and non-finite values.
def metric_mean(rows: Iterable[dict[str, Any]], metric: str) -> Any:
    values = [numeric_or_none(row.get(metric)) for row in rows]
    clean = [value for value in values if value is not None]
    return sum(clean) / len(clean) if clean else ""

# HELPER: Format one table cell with compact numeric precision.
def format_cell(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    number = numeric_or_none(value)
    if number is None:
        return str(value)
    if abs(number) >= 100:
        return f"{number:.1f}"

    # Four decimals would render a DP delta of 1e-6 or a learning rate of 2.5e-4 as zero.
    if number != 0.0 and abs(number) < 1e-3:
        return f"{number:.3g}"
    return f"{number:.4f}".rstrip("0").rstrip(".")

# Read the checkpoint epoch from the best-epoch diagnostics block.
def best_epoch_from_diagnostics(diagnostics: dict[str, Any]) -> Any:
    block = diagnostics.get("best_epoch", diagnostics.get("best_epoch_diagnostics", {}))
    if isinstance(block, dict):
        total = block.get("best_total_validation_loss", {})
        if isinstance(total, dict):
            return total.get("epoch", "")
    return ""

# Read the checkpoint round from the diagnostics block or the rounds block as fallback.
def best_round_from_report(report: dict[str, Any], diagnostics: dict[str, Any]) -> Any:
    block = diagnostics.get("best_round", diagnostics.get("best_round_diagnostics", {}))
    if isinstance(block, dict) and block.get("best_round", "") != "":
        return block.get("best_round", "")
    rounds = report.get("rounds", {})
    if isinstance(rounds, dict):
        best = rounds.get("best_round", "")
        if best != "":
            return best
    return ""

# Read the spent DP epsilon recorded at the best round.
def dp_epsilon_from_diagnostics(diagnostics: dict[str, Any]) -> Any:
    block = diagnostics.get("best_round", diagnostics.get("best_round_diagnostics", {}))
    if isinstance(block, dict) and block.get("dp_epsilon_spent", "") != "":
        return block.get("dp_epsilon_spent", "")
    return ""

# Flatten the per-bank test metrics of one run into per-bank summary rows.
def per_bank_rows_for_run(config: dict[str, Any], metrics: dict[str, Any], run_dir: Path) -> list[dict[str, Any]]:
    # Prefer the per-bank block and fall back to the pooled test block for local baselines, which train one bank.
    per_bank = metrics.get("per_bank_test", {})
    rows: list[dict[str, Any]] = []
    if isinstance(per_bank, dict) and per_bank:
        items = per_bank.items()
    elif config.get("regime") == "local" and config.get("bank"):
        items = [(str(config["bank"]), metrics.get("test", {}))]
    else:
        items = []

    # Build one row per bank and derive the source dataset from the qualified bank id for joint runs.
    for bank, bank_metrics in items:
        if not isinstance(bank_metrics, dict):
            continue
        summary = metric_summary(bank_metrics)
        source_dataset = str(bank).split(":", 1)[0] if ":" in str(bank) else config.get("dataset", "")
        rows.append(
            {
                "script_id": config.get("script_id", ""),
                "dataset": config.get("dataset", ""),
                "run_name": config.get("run_name", ""),
                "regime": resolve_regime(config),
                "strategy": config.get("strategy", ""),
                "bank": bank,
                "source_dataset": source_dataset,
                **summary,
                "run_dir": str(run_dir),
            }
        )
    return rows

# Compute the worst, best and spread values across banks for the fairness columns of one run.
def bank_spreads(per_bank_rows: list[dict[str, Any]]) -> dict[str, Any]:
    f1_values = [numeric_or_none(row.get("outcome_macro_f1")) for row in per_bank_rows]
    rt_values = [numeric_or_none(row.get("remaining_time_mae_seconds")) for row in per_bank_rows]
    f1_clean = [value for value in f1_values if value is not None]
    rt_clean = [value for value in rt_values if value is not None]
    return {
        "worst_bank_outcome_macro_f1": min(f1_clean) if f1_clean else "",
        "best_bank_outcome_macro_f1": max(f1_clean) if f1_clean else "",
        "bank_outcome_macro_f1_spread": (max(f1_clean) - min(f1_clean)) if f1_clean else "",
        "worst_bank_remaining_time_mae_seconds": max(rt_clean) if rt_clean else "",
        "best_bank_remaining_time_mae_seconds": min(rt_clean) if rt_clean else "",
        "bank_remaining_time_mae_spread_seconds": (max(rt_clean) - min(rt_clean)) if rt_clean else "",
    }

# ----------------------------------------------------------------------------------------------------------------------
# 5. RUN REPORT LOADERS AND DISCOVERY

# Load one compact run report and normalize its identity fields for row building.
def load_compact_report(path: Path, output_root: Path) -> dict[str, Any]:
    report = read_json(path)
    config = dict(report.get("config", {}))
    config.update({key: value for key, value in report.get("run_identity", {}).items() if key not in config})
    config["script_id"] = str(report.get("script_id", ""))
    config["report_path"] = str(path.relative_to(output_root))
    config["run_dir"] = str(path.parent.relative_to(output_root))
    return {
        "config": config,
        "metrics": dict(report.get("metrics", {})),
        "diagnostics": dict(report.get("diagnostics", {})),
        "rounds": report.get("rounds", {}),
        "report": report,
        "run_dir": path.parent,
        "source": "compact",
    }

# Load one legacy E_05 run from its per-file JSON fragments.
def load_legacy_baseline(run_dir: Path, output_root: Path) -> dict[str, Any]:
    config_path = run_dir / "E_05_config.json"
    config = (read_json(config_path) if config_path.exists()
              else config_from_legacy_dir(SCRIPT_ID_BASELINE, run_dir, output_root))
    config["script_id"] = SCRIPT_ID_BASELINE
    config.setdefault("report_path", "")
    config.setdefault("run_dir", str(run_dir.relative_to(output_root)))
    metrics = {
        "test": read_json(run_dir / "E_05_test_metrics.json") if (run_dir / "E_05_test_metrics.json").exists() else {},
        "validation": read_json(run_dir / "E_05_validation_metrics.json")
        if (run_dir / "E_05_validation_metrics.json").exists() else {},
        "per_bank_test": read_json(run_dir / "E_05_per_bank_test_metrics.json")
        if (run_dir / "E_05_per_bank_test_metrics.json").exists() else {},
    }
    diagnostics = {
        "best_epoch": read_json(run_dir / "E_05_best_epoch_diagnostics.json")
        if (run_dir / "E_05_best_epoch_diagnostics.json").exists() else {},
        "target": read_json(run_dir / "E_05_target_diagnostics.json")
        if (run_dir / "E_05_target_diagnostics.json").exists() else {},
    }
    return {"config": config, "metrics": metrics, "diagnostics": diagnostics, "rounds": {}, "report": {},
            "run_dir": run_dir, "source": "legacy"}

# Load one legacy E_06 run from its per-file JSON fragments including the round metrics.
def load_legacy_federated(run_dir: Path, output_root: Path) -> dict[str, Any]:
    config_path = run_dir / "E_06_config.json"
    config = (read_json(config_path) if config_path.exists()
              else config_from_legacy_dir(SCRIPT_ID_FEDERATED, run_dir, output_root))
    config["script_id"] = SCRIPT_ID_FEDERATED
    config.setdefault("report_path", "")
    config.setdefault("run_dir", str(run_dir.relative_to(output_root)))
    metrics = {
        "test": read_json(run_dir / "E_06_test_metrics.json") if (run_dir / "E_06_test_metrics.json").exists() else {},
        "validation": read_json(run_dir / "E_06_validation_metrics.json")
        if (run_dir / "E_06_validation_metrics.json").exists() else {},
        "per_bank_test": read_json(run_dir / "E_06_per_bank_test_metrics.json")
        if (run_dir / "E_06_per_bank_test_metrics.json").exists() else {},
    }
    diagnostics = {
        "best_round": read_json(run_dir / "E_06_best_round_diagnostics.json")
        if (run_dir / "E_06_best_round_diagnostics.json").exists() else {},
        "target": read_json(run_dir / "E_06_target_diagnostics.json")
        if (run_dir / "E_06_target_diagnostics.json").exists() else {},
    }
    rounds = read_json(run_dir / "E_06_round_metrics.json") if (run_dir / "E_06_round_metrics.json").exists() else {}
    return {"config": config, "metrics": metrics, "diagnostics": diagnostics, "rounds": rounds, "report": {},
            "run_dir": run_dir, "source": "legacy"}

# Discover every run below the output root, compact reports first and legacy fallbacks second.
# Only the MATRIX_SUBFOLDERS are read because every secure POC run is a duplicate of a federated run.
def discover_runs(output_root: Path, warnings: list[dict[str, str]]) -> list[dict[str, Any]]:
    # Collect every compact run report from the matrix subtrees, one script identifier per subtree.
    script_ids = {"baselines": SCRIPT_ID_BASELINE}
    runs: list[dict[str, Any]] = []
    for subfolder in MATRIX_SUBFOLDERS:
        script_id = script_ids.get(subfolder, SCRIPT_ID_FEDERATED)
        for path in sorted((output_root / subfolder).glob(f"**/{script_id}_run_report.json")):
            runs.append(load_compact_report(path, output_root))

    # Fall back to legacy fragments only for directories without a compact report and record an info warning.
    compact_dirs = {run["run_dir"] for run in runs}
    loaders = {SCRIPT_ID_BASELINE: load_legacy_baseline, SCRIPT_ID_FEDERATED: load_legacy_federated}
    for subfolder in MATRIX_SUBFOLDERS:
        script_id = script_ids.get(subfolder, SCRIPT_ID_FEDERATED)
        for path in sorted((output_root / subfolder).glob(f"**/{script_id}_test_metrics.json")):
            if path.parent in compact_dirs:
                continue
            warnings.append(
                {"severity": "info", "message": f"missing report file, used legacy fallback: {path.parent}"})
            runs.append(loaders[script_id](path.parent, output_root))

    # Assert the exclusion instead of trusting the glob roots, so a future subtree rename cannot leak POC runs in.
    excluded = f"{SECURE_AGGREGATION_SUBFOLDER}/"
    leaked = [run for run in runs if excluded in Path(run["run_dir"]).as_posix()]
    if leaked: raise RuntimeError(f"secure-aggregation POC runs must stay out of the matrix analysis: {leaked[0]}")
    return runs

# ----------------------------------------------------------------------------------------------------------------------
# 6. ROW BUILDING AND WARNING COLLECTION

# Build the run summary rows and the per-bank rows from every discovered run.
def build_rows(runs: list[dict[str, Any]], output_root: Path,
               warnings: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_rows: list[dict[str, Any]] = []
    bank_rows: list[dict[str, Any]] = []
    for run in runs:
        # Normalize the config and guard against runs without test metrics.
        config = dict(run["config"])
        metrics = run["metrics"]
        diagnostics = run["diagnostics"]
        run_dir = Path(run["run_dir"])
        config["script_id"] = config.get("script_id", run.get("script_id", ""))
        test_metrics = metrics.get("test", {})
        if not isinstance(test_metrics, dict) or not test_metrics:
            warnings.append({"severity": "warning", "message": f"missing test metrics: {run_dir}"})
            test_metrics = {}

        # Collect the per-bank rows and derive the fairness spreads before assembling the summary row.
        per_bank = per_bank_rows_for_run(config, metrics, run_dir)
        bank_rows.extend(per_bank)
        spreads = bank_spreads(per_bank)
        summary = metric_summary(test_metrics)
        row = {
            "script_id": config.get("script_id", ""),
            "dataset": config.get("dataset", ""),
            "run_name": config.get("run_name", ""),
            "heterogeneity": config.get("heterogeneity", ""),
            "n_clients": config.get("n_clients", ""),
            "regime": resolve_regime(config),
            "strategy": config.get("strategy", ""),
            "bank": config.get("bank", ""),
            "seed": config.get("seed", ""),
            "learning_rate": config.get("learning_rate", ""),
            "fedprox_mu": config.get("fedprox_mu", ""),
            "use_dp": config.get("use_dp", False),
            "dp_target_epsilon": config.get("dp_target_epsilon", ""),
            "dp_delta": config.get("dp_delta", ""),
            "dp_max_grad_norm": config.get("dp_max_grad_norm", ""),
            "dp_epsilon_spent": dp_epsilon_from_diagnostics(diagnostics),
            "max_epochs": config.get("max_epochs", ""),
            "max_rounds": config.get("max_rounds", ""),
            "local_epochs": config.get("local_epochs", ""),
            "best_epoch": best_epoch_from_diagnostics(diagnostics),
            "best_round": best_round_from_report(run.get("report", {}), diagnostics),
            **summary,
            **spreads,
            "run_dir": config.get(
                "run_dir", str(run_dir.relative_to(output_root)) if run_dir.is_absolute() else str(run_dir)),
            "report_path": config.get("report_path", ""),
        }
        run_rows.append(row)
        collect_run_warnings(row, per_bank, run_dir, warnings)
    collect_matrix_warnings(run_rows, warnings)
    return run_rows, bank_rows

# Record the data-quality warnings for one run row.
def collect_run_warnings(row: dict[str, Any], per_bank: list[dict[str, Any]], run_dir: Path,
                         warnings: list[dict[str, str]]) -> None:
    # Flag non-finite metrics because they would silently corrupt thesis tables.
    for column in RUN_SUMMARY_COLUMNS:
        value = row.get(column, "")
        if isinstance(value, float) and not math.isfinite(value):
            warnings.append({"severity": "warning", "message": f"NaN or infinite metric in {column}: {run_dir}"})

    # Flag missing round logs and missing per-bank coverage for the runs that must have them.
    if row["script_id"] == SCRIPT_ID_FEDERATED and not (run_dir / "E_06_round_log.csv").exists():
        warnings.append({"severity": "warning", "message": f"E06 run has no round log: {run_dir}"})
    if (row["regime"] != "local" and row["script_id"] in {SCRIPT_ID_BASELINE, SCRIPT_ID_FEDERATED}
            and not per_bank):
        warnings.append({"severity": "warning", "message": f"missing per-bank metrics: {run_dir}"})

    # A joint local baseline trains exactly one bank, so only pooled joint runs must cover both source datasets.
    if row["dataset"] == "joint" and row["regime"] != "local" and per_bank:
        sources = {str(item.get("source_dataset", "")) for item in per_bank}
        if not {"bpic2017", "bpic2012"}.issubset(sources):
            warnings.append(
                {"severity": "warning", "message": f"joint per-source bank metrics are incomplete: {run_dir}"})

    # Flag degenerate per-bank blocks, empty prefix populations and near-zero next-activity accuracy.
    for bank_row in per_bank:
        n_prefixes = numeric_or_none(bank_row.get("n_prefixes"))
        if n_prefixes == 0:
            warnings.append(
                {"severity": "warning", "message": f"zero-prefix metric block: {run_dir} {bank_row.get('bank')}"})
        top1 = numeric_or_none(bank_row.get("next_activity_top1_accuracy"))
        if top1 is not None and top1 < 0.05:
            warnings.append(
                {"severity": "warning",
                 "message": f"suspiciously low next-activity accuracy below 0.05: {run_dir} {bank_row.get('bank')}"})

# Compare the observed runs against the expected no-DP matrix and warn once per missing entry.
def collect_matrix_warnings(run_rows: list[dict[str, Any]], warnings: list[dict[str, str]]) -> None:
    observed = set()
    for row in run_rows:
        # A DP run keys separately, so it never fills the slot with a missing no-DP federated run.
        use_dp = is_truthy(row.get("use_dp"))
        if row["script_id"] == SCRIPT_ID_BASELINE:
            observed.add(matrix_key(row["script_id"], row["dataset"], row["run_name"], row["regime"], "",
                                    str(row.get("bank", "")), use_dp))
            if row["regime"] == "centralized":
                observed.add(matrix_key(row["script_id"], row["dataset"], row["run_name"], "centralized", "", "",
                                        use_dp))
        if row["script_id"] == SCRIPT_ID_FEDERATED:
            observed.add(matrix_key(row["script_id"], row["dataset"], row["run_name"], "", row["strategy"], "",
                                    use_dp))
    missing = sorted(expected_matrix_keys() - observed)
    for key in missing:
        warnings.append({"severity": "warning", "message": f"missing expected matrix run: {key}"})

# ----------------------------------------------------------------------------------------------------------------------
# 7. TABLE SELECTION AND COMPARISON ROWS

# HELPER: Select the best rows by one metric under a row predicate.
def top_rows(rows: list[dict[str, Any]], predicate: Any, metric: str, limit: int = 5,
             reverse: bool = True) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(row) and numeric_or_none(row.get(metric)) is not None]
    return sorted(selected, key=lambda row: float(row[metric]), reverse=reverse)[:limit]

# Render rows as one GFM table with padded columns.
def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    # Wrap the block in blank lines and pad every column so a GFM table renders and the raw text stays aligned.
    if not rows:
        return "\nNo rows available.\n"
    cell_rows = [[format_cell(row.get(column, "")) for column in columns] for row in rows]
    widths = [len(column) for column in columns]
    for cells in cell_rows:
        for index, value in enumerate(cells):
            widths[index] = max(widths[index], len(value))
    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(
            temp_value.ljust(widths[temp_index]) for temp_index, temp_value in enumerate(values)) + " |"
    lines = [render_row(columns), "|" + "|".join("-" * (widths[index] + 2) for index in range(len(columns))) + "|"]
    for cells in cell_rows:
        lines.append(render_row(cells))
    return "\n" + "\n".join(lines) + "\n"

# HELPER: Return the single best row by one metric.
def best_row(rows: list[dict[str, Any]], predicate: Any, metric: str, reverse: bool = True) -> Optional[dict[str, Any]]:
    selected = top_rows(rows, predicate, metric, limit=1, reverse=reverse)
    return selected[0] if selected else None

# HELPER: Keep the federated rows that belong to the no-DP matrix.
def non_dp_federated_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in run_rows
        if row.get("script_id") == SCRIPT_ID_FEDERATED and not is_truthy(row.get("use_dp"))
    ]

# Pair each centralized baseline with its best no-DP federated run for the delta table.
def federated_vs_centralized_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Index the centralized baselines and group the federated runs per split.
    centralized = {
        (row.get("dataset"), row.get("run_name")): row
        for row in run_rows
        if row.get("script_id") == SCRIPT_ID_BASELINE and row.get("regime") == "centralized"
    }
    federated_by_run: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in non_dp_federated_rows(run_rows):
        federated_by_run.setdefault((row.get("dataset"), row.get("run_name")), []).append(row)

    # Emit one delta row per split where both regimes exist.
    rows: list[dict[str, Any]] = []
    for key, central_row in sorted(centralized.items()):
        candidates = federated_by_run.get(key, [])
        best_fed = best_row(candidates, lambda row: True, "outcome_macro_f1")
        if best_fed is None:
            continue
        central_f1 = numeric_or_none(central_row.get("outcome_macro_f1"))
        fed_f1 = numeric_or_none(best_fed.get("outcome_macro_f1"))
        central_rt = numeric_or_none(central_row.get("remaining_time_mae_seconds"))
        fed_rt = numeric_or_none(best_fed.get("remaining_time_mae_seconds"))
        rows.append(
            {
                "dataset": key[0],
                "run_name": key[1],
                "best_federated_strategy": best_fed.get("strategy", ""),
                "centralized_macro_f1": central_f1 if central_f1 is not None else "",
                "federated_macro_f1": fed_f1 if fed_f1 is not None else "",
                "delta_macro_f1": (fed_f1 - central_f1) if central_f1 is not None and fed_f1 is not None else "",
                "centralized_rt_mae": central_rt if central_rt is not None else "",
                "federated_rt_mae": fed_rt if fed_rt is not None else "",
                "delta_rt_mae": (fed_rt - central_rt) if central_rt is not None and fed_rt is not None else "",
            }
        )
    return rows

# Pair FedAvg and FedProx per split for the strategy delta table at the locked mu.
def fedprox_vs_fedavg_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group the no-DP federated rows per split and strategy.
    by_key: dict[tuple[Any, Any], dict[str, dict[str, Any]]] = {}
    for row in non_dp_federated_rows(run_rows):
        by_key.setdefault((row.get("dataset"), row.get("run_name")), {})[str(row.get("strategy", ""))] = row

    # Emit one delta row per split where both strategies exist.
    rows: list[dict[str, Any]] = []
    for key, strategies in sorted(by_key.items()):
        fedavg = strategies.get("fedavg")
        fedprox = strategies.get("fedprox")
        if fedavg is None or fedprox is None:
            continue
        avg_f1 = numeric_or_none(fedavg.get("outcome_macro_f1"))
        prox_f1 = numeric_or_none(fedprox.get("outcome_macro_f1"))
        avg_rt = numeric_or_none(fedavg.get("remaining_time_mae_seconds"))
        prox_rt = numeric_or_none(fedprox.get("remaining_time_mae_seconds"))
        rows.append(
            {
                "dataset": key[0],
                "run_name": key[1],
                "fedavg_macro_f1": avg_f1 if avg_f1 is not None else "",
                "fedprox_macro_f1": prox_f1 if prox_f1 is not None else "",
                "FedProx minus FedAvg": (prox_f1 - avg_f1) if avg_f1 is not None and prox_f1 is not None else "",
                "fedavg_rt_mae": avg_rt if avg_rt is not None else "",
                "fedprox_rt_mae": prox_rt if prox_rt is not None else "",
                "FedProx minus FedAvg RT": (prox_rt - avg_rt) if avg_rt is not None and prox_rt is not None else "",
            }
        )
    return rows

# Aggregate the joint per-bank rows per source dataset to expose the BPIC 2017 versus BPIC 2012 balance.
def joint_source_balance_rows(bank_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Group the joint bank rows by run, regime, strategy and source dataset.
    grouped: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for row in bank_rows:
        if row.get("dataset") != "joint":
            continue
        key = (row.get("run_name"), row.get("regime"), row.get("strategy"), row.get("source_dataset"))
        grouped.setdefault(key, []).append(row)

    # Sum the prefix counts per run so the source share has a denominator.
    totals: dict[tuple[Any, ...], float] = {}
    for key, rows in grouped.items():
        run_key = key[:3]
        totals[run_key] = totals.get(run_key, 0.0) + sum(numeric_or_none(row.get("n_prefixes")) or 0.0 for row in rows)

    # Emit one balance row per source with its prefix share and mean headline metrics.
    balance_rows: list[dict[str, Any]] = []
    for key, rows in sorted(grouped.items()):
        run_key = key[:3]
        prefix_total = sum(numeric_or_none(row.get("n_prefixes")) or 0.0 for row in rows)
        total = totals.get(run_key, 0.0)
        balance_rows.append(
            {
                "run_name": key[0],
                "regime": key[1],
                "strategy": key[2],
                "source_dataset": key[3],
                "banks": len(rows),
                "n_prefixes": prefix_total,
                "prefix_share": (prefix_total / total) if total else "",
                "mean_outcome_macro_f1": metric_mean(rows, "outcome_macro_f1"),
                "mean_next_activity_top1": metric_mean(rows, "next_activity_top1_accuracy"),
                "mean_rt_mae_seconds": metric_mean(rows, "remaining_time_mae_seconds"),
            }
        )
    return balance_rows

# Rank the runs by bank spread to surface the worst-client fairness risks.
def fairness_risk_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in run_rows
        if numeric_or_none(row.get("bank_outcome_macro_f1_spread")) is not None
        or numeric_or_none(row.get("bank_remaining_time_mae_spread_seconds")) is not None
    ]
    return sorted(
        candidates,
        key=lambda row: (
            numeric_or_none(row.get("bank_outcome_macro_f1_spread")) or 0.0,
            numeric_or_none(row.get("bank_remaining_time_mae_spread_seconds")) or 0.0,
        ),
        reverse=True,
    )[:10]

# HELPER: Keep the DP-SGD runs for the privacy profile section.
def dp_profile_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in run_rows
        if row.get("script_id") == SCRIPT_ID_FEDERATED and is_truthy(row.get("use_dp"))
    ]

# HELPER: Drop the DP runs from a profile, because DP-SGD trades utility for privacy and is reported separately.
def without_dp_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in run_rows if not is_truthy(row.get("use_dp"))]

# Select the runs that stopped at their budget ceiling instead of early stopping, ordered by dataset and split.
# The deterministic rule keeps the table stable across reruns and names the possibly schedule-limited runs.
def budget_limited_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in without_dp_rows(run_rows):
        checkpoint = numeric_or_none(row.get("best_round")) or numeric_or_none(row.get("best_epoch"))
        budget = numeric_or_none(row.get("max_rounds")) or numeric_or_none(row.get("max_epochs"))
        if checkpoint is None or budget is None: continue
        if checkpoint >= budget: selected.append(row)
    return sorted(
        selected,
        key=lambda item: (str(item.get("dataset", "")), str(item.get("run_name", "")), str(item.get("regime", "")),
                          str(item.get("strategy", "")), str(item.get("bank", ""))),
    )

# ----------------------------------------------------------------------------------------------------------------------
# 8. MARKDOWN REPORT

# Build the sectioned Markdown report from the rows and warnings.
def build_markdown(run_rows: list[dict[str, Any]], bank_rows: list[dict[str, Any]],
                   warnings: list[dict[str, str]], output_root: Path) -> str:
    # Open with an executive summary naming the strongest runs and the warning count.
    lines = ["# E_07 Training Analysis", ""]
    best_central = best_row(run_rows, lambda row: row.get("regime") == "centralized", "outcome_macro_f1")
    best_fed = best_row(non_dp_federated_rows(run_rows), lambda row: True, "outcome_macro_f1")
    missing_count = sum(1 for warning in warnings if warning["message"].startswith("missing expected matrix run"))
    lines.append("## Executive summary")
    lines.append(f"Runs discovered: {len(run_rows)}. Per-bank metric rows discovered: {len(bank_rows)}.")
    lines.append(f"Expected matrix entries: {len(expected_matrix_keys())}. Missing expected entries: {missing_count}.")
    if best_central:
        lines.append(
            "Best centralized baseline by outcome macro-F1: "
            f"{best_central.get('dataset')} {best_central.get('run_name')} "
            f"with {format_cell(best_central.get('outcome_macro_f1'))}."
        )
    if best_fed:
        lines.append(
            "Best non-DP federated run by outcome macro-F1: "
            f"{best_fed.get('dataset')} {best_fed.get('run_name')} {best_fed.get('strategy')} "
            f"with {format_cell(best_fed.get('outcome_macro_f1'))}."
        )
    if warnings:
        lines.append("Warnings require review before using these numbers in the thesis: "
                     f"{len(warnings)} warnings recorded.")
    else:
        lines.append("No warnings were recorded.")
    lines.append("")

    # Render every section in the fixed thesis order with its specialized table.
    for section in MARKDOWN_SECTIONS:
        lines.append(f"## {section}")
        lines.append("")
        if section == "Run inventory":
            lines.append(f"Runs discovered: {len(run_rows)}")
            lines.append(f"Per-bank rows discovered: {len(bank_rows)}")
            lines.append("Subtrees read: " + ", ".join(MATRIX_SUBFOLDERS) + ".")
            lines.append(
                f"The {SECURE_AGGREGATION_SUBFOLDER}/ subtree is excluded by design. Each secure-aggregation "
                "POC run has an exact plain FedProx counterpart in federated/ and is reported separately."
            )
        elif section == "Matrix completeness":
            lines.append(f"Expected matrix entries: {len(expected_matrix_keys())}")
            lines.append(f"Missing expected entries: {missing_count}")
            if missing_count:
                lines.append("The analysis is useful for diagnostics, "
                             "but the matrix is not complete enough for final thesis tables.")
            else:
                lines.append("The expected no-DP baseline and federated matrix is complete.")
        elif section == "Best centralized baselines":
            lines.append(
                markdown_table(
                    top_rows(run_rows, lambda row: row.get("regime") == "centralized", "outcome_macro_f1"),
                    ["dataset", "run_name", "outcome_macro_f1", "next_activity_top1_accuracy",
                     "remaining_time_mae_seconds"],
                )
            )
        elif section == "Best local baselines":
            lines.append(
                markdown_table(
                    top_rows(run_rows, lambda row: row.get("regime") == "local", "outcome_macro_f1"),
                    ["dataset", "run_name", "bank", "outcome_macro_f1", "next_activity_top1_accuracy",
                     "remaining_time_mae_seconds"],
                )
            )
        elif section == "Best federated runs":
            lines.append(
                markdown_table(
                    top_rows(non_dp_federated_rows(run_rows), lambda row: True, "outcome_macro_f1"),
                    ["dataset", "run_name", "strategy", "outcome_macro_f1", "next_activity_top1_accuracy",
                     "remaining_time_mae_seconds"],
                )
            )
        elif section == "Joint versus single-dataset comparison":
            no_dp_rows = without_dp_rows(run_rows)
            joint_rows = [row for row in no_dp_rows if row.get("dataset") == "joint"]
            single_rows = [row for row in no_dp_rows if row.get("dataset") in {"bpic2017", "bpic2012"}]
            lines.append(
                "Joint rows are interpreted against the single-dataset matrix, "
                "but they use one shared vocabulary and one shared scaler."
            )
            lines.append(
                "Both means pool the centralized, local and federated regimes of the no-DP matrix. "
                "The DP runs are excluded here and reported in their own section."
            )
            lines.append(
                markdown_table(
                    [
                        {
                            "scope": "joint",
                            "runs": len(joint_rows),
                            "mean_outcome_macro_f1": metric_mean(joint_rows, "outcome_macro_f1"),
                            "mean_next_activity_top1": metric_mean(joint_rows, "next_activity_top1_accuracy"),
                            "mean_rt_mae_seconds": metric_mean(joint_rows, "remaining_time_mae_seconds"),
                        },
                        {
                            "scope": "single-dataset",
                            "runs": len(single_rows),
                            "mean_outcome_macro_f1": metric_mean(single_rows, "outcome_macro_f1"),
                            "mean_next_activity_top1": metric_mean(single_rows, "next_activity_top1_accuracy"),
                            "mean_rt_mae_seconds": metric_mean(single_rows, "remaining_time_mae_seconds"),
                        },
                    ],
                    ["scope", "runs", "mean_outcome_macro_f1", "mean_next_activity_top1", "mean_rt_mae_seconds"],
                )
            )
            lines.append("Dataset balance for joint runs.")
            lines.append(
                markdown_table(
                    joint_source_balance_rows(bank_rows),
                    [
                        "run_name",
                        "regime",
                        "strategy",
                        "source_dataset",
                        "banks",
                        "n_prefixes",
                        "prefix_share",
                        "mean_outcome_macro_f1",
                        "mean_next_activity_top1",
                        "mean_rt_mae_seconds",
                    ],
                )
            )
        elif section == "Warnings":
            if warnings:
                lines.extend(f"- {warning['severity']}: {warning['message']}" for warning in warnings[:200])
            else:
                lines.append("No warnings.")
        elif section == "Artifact index":
            lines.append(f"Output root: {output_root}")
            lines.append("Files: E_07_training_analysis.md, E_07_training_analysis.json, "
                         "E_07_run_summary.csv, E_07_per_bank_summary.csv")
        elif section == "Federated versus centralized comparison":
            lines.append(
                "Positive macro-F1 deltas mean the best non-DP federated run "
                "beat the centralized baseline on the same split."
            )
            lines.append(
                markdown_table(
                    federated_vs_centralized_rows(run_rows),
                    [
                        "dataset",
                        "run_name",
                        "best_federated_strategy",
                        "centralized_macro_f1",
                        "federated_macro_f1",
                        "delta_macro_f1",
                        "centralized_rt_mae",
                        "federated_rt_mae",
                        "delta_rt_mae",
                    ],
                )
            )
        elif section == "FedProx versus FedAvg comparison":
            lines.append("FedProx minus FedAvg shows whether the proximal term helped at the locked mu.")
            lines.append(
                markdown_table(
                    fedprox_vs_fedavg_rows(run_rows),
                    [
                        "dataset",
                        "run_name",
                        "fedavg_macro_f1",
                        "fedprox_macro_f1",
                        "FedProx minus FedAvg",
                        "fedavg_rt_mae",
                        "fedprox_rt_mae",
                        "FedProx minus FedAvg RT",
                    ],
                )
            )
        elif section == "Client fairness and worst-client risks":
            lines.append("Rows with the largest bank spreads are the first candidates for thesis fairness discussion.")
            lines.append("The ranking covers the no-DP matrix only, because DP noise changes every spread.")
            lines.append(
                markdown_table(
                    fairness_risk_rows(without_dp_rows(run_rows)),
                    [
                        "dataset",
                        "run_name",
                        "regime",
                        "strategy",
                        "bank_outcome_macro_f1_spread",
                        "worst_bank_outcome_macro_f1",
                        "best_bank_outcome_macro_f1",
                        "bank_remaining_time_mae_spread_seconds",
                    ],
                )
            )
        elif section == "Remaining-time error profile":
            lines.append("Lower remaining-time MAE is better. The table keeps the strongest and weakest rows visible.")
            lines.append("The DP runs are excluded, because their utility loss is the subject of the DP section.")
            no_dp_rows = without_dp_rows(run_rows)
            strongest = top_rows(no_dp_rows, lambda row: True, "remaining_time_mae_seconds", reverse=False)
            weakest = top_rows(no_dp_rows, lambda row: True, "remaining_time_mae_seconds", reverse=True)
            lines.append("Lowest remaining-time MAE.")
            lines.append(markdown_table(
                strongest, ["dataset", "run_name", "regime", "strategy",
                            "remaining_time_mae_seconds", "remaining_time_rmse_seconds"]))
            lines.append("Highest remaining-time MAE.")
            lines.append(markdown_table(
                weakest, ["dataset", "run_name", "regime", "strategy",
                          "remaining_time_mae_seconds", "remaining_time_rmse_seconds"]))
        elif section == "Next-activity performance profile":
            lines.append("Low next-activity accuracy often indicates cross-dataset vocabulary pressure "
                         "or client-specific activity drift.")
            lines.append("The DP runs are excluded, because their utility loss is the subject of the DP section.")
            lines.append(
                markdown_table(
                    top_rows(without_dp_rows(run_rows), lambda row: True, "next_activity_top1_accuracy",
                             reverse=False),
                    ["dataset", "run_name", "regime", "strategy", "next_activity_top1_accuracy", "outcome_macro_f1"],
                )
            )
        elif section == "Convergence and training stability":
            budget_limited = budget_limited_rows(run_rows)
            lines.append("Selection rule: every no-DP run whose checkpoint epoch or round reached its own budget "
                         "ceiling, ordered by dataset, split, regime, strategy and bank.")
            lines.append("These runs did not early-stop, so their headline numbers may be schedule-limited rather "
                         f"than converged. Runs listed: {len(budget_limited)}.")
            lines.append(
                markdown_table(
                    budget_limited,
                    ["dataset", "run_name", "regime", "strategy", "bank", "best_epoch", "best_round",
                     "max_epochs", "max_rounds", "local_epochs"],
                )
            )
        elif section == "Differential privacy profile":
            rows = dp_profile_rows(run_rows)
            lines.append("DP-SGD runs are interpreted as privacy accounting experiments "
                         "and are separate from the no-DP thesis matrix.")
            lines.append(
                markdown_table(
                    rows,
                    [
                        "dataset",
                        "run_name",
                        "strategy",
                        "dp_target_epsilon",
                        "dp_delta",
                        "dp_max_grad_norm",
                        "dp_epsilon_spent",
                        "outcome_macro_f1",
                        "remaining_time_mae_seconds",
                    ],
                )
            )
        else:
            lines.append("No specialized renderer is registered for this section.")
        lines.append("")
    return "\n".join(lines)

# ----------------------------------------------------------------------------------------------------------------------
# 9. DP FIGURES

# Thesis graph style constants, mirrored from the E_05 and E_06 curve exports so the DP figures match them exactly.
PRIMARY_BLUE = "#0065BF"
REFERENCE_RED = "#C8102E"
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#666666"
LEGEND_EDGE = "#bfbfbf"
TEXT_DARK = "#222222"

# Series palettes taken from the thesis bank palette, read from dark to light as the privacy budget grows.
DP_SPLIT_COLORS: dict[str, str] = {"iid": "#003E7A", "medium": PRIMARY_BLUE, "strong": "#5A9DDC"}
DP_EPSILON_COLORS: dict[float, str] = {1.0: "#003E7A", 5.0: PRIMARY_BLUE, 10.0: "#5A9DDC", 50.0: "#99C2E5"}
DP_HEAD_COLORS: tuple[str, ...] = ("#003E7A", PRIMARY_BLUE, "#5A9DDC")

# Target epsilon levels of the DP grid and the heterogeneity order the panels follow.
DP_EPSILON_LEVELS: tuple[float, ...] = (1.0, 5.0, 10.0, 50.0)
DP_SPLIT_ORDER: tuple[str, ...] = ("iid", "medium", "strong")

# The three heads as (row metric, axis label, lower-is-better flag), so one loop can render every panel.
DP_HEAD_SPECS: tuple[tuple[str, str, bool], ...] = (
    ("outcome_macro_f1", "Outcome Macro-F1", False),
    ("next_activity_top1_accuracy", "Next-Activity Top-1 Accuracy", False),
    ("remaining_time_mae_seconds", "Remaining-Time MAE Seconds", True),
)

# HELPER: Apply the thesis rcParams to the lazily imported pyplot module (matches the E_05 and E_06 plot exports).
def configure_dp_plot_style(plt: Any) -> None:
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

# HELPER: Apply the thesis grid, margin and spine styling to one axes object (matches the thesis plot scripts).
def style_non_pie_axes(ax: Any, x_margin: float = 0.04) -> None:
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
def style_training_legend(ax: Any, loc: str = "best", ncol: int = 1) -> None:
    legend = ax.legend(loc=loc, frameon=True, ncol=ncol)
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_alpha(0.78)
    frame.set_linewidth(0.8)

# HELPER: Index the no-DP FedProx federated rows per split so each DP curve finds its epsilon-infinity baseline.
def dp_baseline_by_split(run_rows: list[dict[str, Any]]) -> dict[tuple[Any, Any], dict[str, Any]]:
    return {
        (row.get("dataset"), row.get("run_name")): row
        for row in non_dp_federated_rows(run_rows)
        if str(row.get("strategy", "")) == "fedprox"
    }

# HELPER: Group the DP rows per split and sort each group by target epsilon so the curves read left to right.
def dp_rows_by_split(run_rows: list[dict[str, Any]]) -> dict[tuple[Any, Any], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in dp_profile_rows(run_rows):
        grouped.setdefault((row.get("dataset"), row.get("run_name")), []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row_temp: numeric_or_none(row_temp.get("dp_target_epsilon")) or 0.0)
    return grouped

# HELPER: Map the relative run directory of every discovered run back to its absolute path for the round-log read.
def dp_run_dir_index(runs: list[dict[str, Any]], output_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for run in runs:
        run_dir = Path(run["run_dir"])
        key = str(run_dir.relative_to(output_root)) if run_dir.is_absolute() else str(run_dir)
        index[key] = run_dir
    return index

# HELPER: Read the round number and the pooled validation loss from one E_06 round log.
def read_round_log_validation(run_dir: Path) -> tuple[list[float], list[float]]:
    log_path = run_dir / "E_06_round_log.csv"
    if not log_path.exists():
        return [], []
    rounds: list[float] = []
    losses: list[float] = []
    with log_path.open(newline="", encoding="utf-8") as handle:
        for record in csv.DictReader(handle):
            round_value = numeric_or_none(record.get("round"))
            loss_value = numeric_or_none(record.get("val_loss_total"))
            if round_value is None or loss_value is None:
                continue
            rounds.append(round_value)
            losses.append(loss_value)
    return rounds, losses

# HELPER: Express the loss of one DP metric against its baseline as a positive percentage of the baseline value.
def dp_relative_loss_percent(baseline: Optional[float], observed: Optional[float], lower_is_better: bool) -> Optional[float]:
    if baseline is None or observed is None or baseline == 0.0:
        return None
    delta = (observed - baseline) if lower_is_better else (baseline - observed)
    return delta / abs(baseline) * 100.0

# HELPER: Order the grouped DP rows by the fixed heterogeneity panel order.
def _dp_split_order_key(item: tuple[Any, list[dict[str, Any]]]) -> int:
    return DP_SPLIT_ORDER.index(str(item[1][0].get("heterogeneity")))

# PLOT: Draw the headline privacy-utility curves, one panel per head and one line per split over the target epsilon.
def plot_dp_privacy_utility_curves(plt: Any, grouped: dict[tuple[Any, Any], list[dict[str, Any]]],
                                   baselines: dict[tuple[Any, Any], dict[str, Any]], output_path: Path) -> None:
    fig, axes = plt.subplots(1, len(DP_HEAD_SPECS), figsize=(13.5, 4.4), facecolor="white")

    # Draw one panel per head and inside it one DP curve plus one dashed baseline per split.
    for axis, (metric, ylabel, _lower_is_better) in zip(axes, DP_HEAD_SPECS):
        baseline_labelled = False
        for key, rows in sorted(grouped.items(), key=_dp_split_order_key):
            heterogeneity = str(rows[0].get("heterogeneity", ""))
            color = DP_SPLIT_COLORS.get(heterogeneity, PRIMARY_BLUE)
            epsilons = [numeric_or_none(row.get("dp_target_epsilon")) for row in rows]
            values = [numeric_or_none(row.get(metric)) for row in rows]
            points = [(eps, value) for eps, value in zip(epsilons, values) if eps is not None and value is not None]
            if points:
                axis.plot(
                    [point[0] for point in points], [point[1] for point in points], marker="o", markersize=4,
                    linewidth=1.6, color=color, markeredgecolor="white", markeredgewidth=0.6, label=heterogeneity
                )

            # Add the epsilon-infinity FedProx reference as a red dashed line, one encoding across the figure set.
            # It is named once per panel because the split color of the solid curve already identifies the split.
            baseline_row = baselines.get(key)
            baseline_value = numeric_or_none(baseline_row.get(metric)) if baseline_row else None
            if baseline_value is not None:
                axis.axhline(
                    baseline_value, color=REFERENCE_RED, linewidth=1.0, linestyle="--",
                    label=None if baseline_labelled else "no DP"
                )
                baseline_labelled = True

        # Apply the thesis plot styling and the log epsilon axis with the four grid levels as explicit ticks.
        axis.set_xscale("log")
        axis.set_xticks(list(DP_EPSILON_LEVELS))
        axis.set_xticklabels([f"{int(level)}" for level in DP_EPSILON_LEVELS])
        axis.set_xlabel("Target epsilon")
        axis.set_ylabel(ylabel)
        axis.tick_params(colors=TEXT_DARK)
        style_non_pie_axes(axis)
        style_training_legend(axis, loc="best", ncol=2)

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# PLOT: Draw the relative utility loss per head as grouped bars over the four target epsilon levels.
def plot_dp_utility_drop_by_head(plt: Any, grouped: dict[tuple[Any, Any], list[dict[str, Any]]],
                                 baselines: dict[tuple[Any, Any], dict[str, Any]], output_path: Path) -> None:
    # Average the relative loss of every head over the splits so one bar group summarizes one epsilon level.
    means: dict[str, list[Optional[float]]] = {}
    for metric, _ylabel, lower_is_better in DP_HEAD_SPECS:
        per_level: list[Optional[float]] = []
        for level in DP_EPSILON_LEVELS:
            collected: list[float] = []
            for key, rows in grouped.items():
                baseline_row = baselines.get(key)
                if baseline_row is None:
                    continue
                for row in rows:
                    if numeric_or_none(row.get("dp_target_epsilon")) != level:
                        continue
                    percent = dp_relative_loss_percent(
                        numeric_or_none(baseline_row.get(metric)), numeric_or_none(row.get(metric)), lower_is_better)
                    if percent is not None:
                        collected.append(percent)
            per_level.append(sum(collected) / len(collected) if collected else None)
        means[metric] = per_level

    # Draw one bar per head inside each epsilon group and label every bar with its percentage.
    fig, axis = plt.subplots(figsize=(9, 4.2), facecolor="white")
    positions = list(range(len(DP_EPSILON_LEVELS)))
    width = 0.8 / len(DP_HEAD_SPECS)
    for index, (metric, ylabel, _lower_is_better) in enumerate(DP_HEAD_SPECS):
        offset = (index - (len(DP_HEAD_SPECS) - 1) / 2) * width
        values = [value if value is not None else 0.0 for value in means[metric]]
        bars = axis.bar(
            [position + offset for position in positions], values, width, color=DP_HEAD_COLORS[index],
            edgecolor="white", linewidth=0.5, label=ylabel
        )
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.6, f"{value:.1f}", ha="center", va="bottom",
                fontsize=7, color=TEXT_DARK
            )

    # Apply the thesis plot styling and keep the epsilon order identical to the privacy-utility curves.
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{int(level)}" for level in DP_EPSILON_LEVELS])
    axis.set_xlabel("Target epsilon")
    axis.set_ylabel("Relative utility loss (%)")
    axis.tick_params(colors=TEXT_DARK)
    style_non_pie_axes(axis)
    style_training_legend(axis, loc="upper right")
    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# PLOT: Draw the pooled validation loss per round, one panel per split, for the DP runs and the no-DP baseline.
def plot_dp_convergence_rounds(plt: Any, grouped: dict[tuple[Any, Any], list[dict[str, Any]]],
                               baselines: dict[tuple[Any, Any], dict[str, Any]],
                               run_dirs: dict[str, Path], output_path: Path) -> None:
    ordered = sorted(grouped.items(), key=_dp_split_order_key)
    fig, axes = plt.subplots(1, len(ordered), figsize=(13.5, 4.4), facecolor="white")
    axes = axes if len(ordered) > 1 else [axes]

    # Draw one panel per split with the four DP curves and the no-DP FedProx reference curve.
    for axis, (key, rows) in zip(axes, ordered):
        heterogeneity = str(rows[0].get("heterogeneity", ""))
        for row in rows:
            level = numeric_or_none(row.get("dp_target_epsilon"))
            run_dir = run_dirs.get(str(row.get("run_dir", "")))
            if level is None or run_dir is None:
                continue
            round_values, loss_values = read_round_log_validation(run_dir)
            if not round_values:
                continue
            color = DP_EPSILON_COLORS.get(level, PRIMARY_BLUE)
            axis.plot(round_values, loss_values, linewidth=1.6, color=color, label=f"epsilon {int(level)}")

            # Mark the checkpoint round of the curve, so a run that never early-stops is visible at the budget cap.
            best_index = min(range(len(loss_values)), key=lambda position: loss_values[position])
            axis.plot(
                round_values[best_index], loss_values[best_index], marker="o", markersize=8.0,
                markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.4, zorder=6, linestyle="none"
            )

        # Add the no-DP FedProx run of the same split as the red-dashed reference curve.
        baseline_row = baselines.get(key)
        baseline_dir = run_dirs.get(str(baseline_row.get("run_dir", ""))) if baseline_row else None
        if baseline_dir is not None:
            round_values, loss_values = read_round_log_validation(baseline_dir)
            if round_values:
                axis.plot(round_values, loss_values, linewidth=1.4, linestyle="--", color=REFERENCE_RED, label="no DP")
                best_index = min(range(len(loss_values)), key=lambda position: loss_values[position])
                axis.plot(
                    round_values[best_index], loss_values[best_index], marker="o", markersize=8.0,
                    markerfacecolor="white", markeredgecolor=REFERENCE_RED, markeredgewidth=1.4, zorder=6,
                    linestyle="none"
                )

        # Apply the thesis plot styling and name the split in a short bold panel title.
        axis.set_title(heterogeneity, fontsize=11, fontweight="bold")
        axis.set_xlabel("Round")
        axis.set_ylabel("Pooled validation loss")
        axis.tick_params(colors=TEXT_DARK)
        style_non_pie_axes(axis)
        style_training_legend(axis, loc="upper right")

    fig.tight_layout(pad=1.2)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

# Write the three DP figures into the analysis root and return without importing matplotlib when no DP run exists.
def write_dp_figures(run_rows: list[dict[str, Any]], runs: list[dict[str, Any]], output_root: Path,
                     analysis_root: Path) -> None:
    # Keep the script a strict no-op on DP-less output roots so the import footprint stays unchanged there.
    grouped = dp_rows_by_split(run_rows)
    if not grouped:
        return

    # Import matplotlib only now and select the file backend before pyplot, so no display is required.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configure_dp_plot_style(plt)

    # Render the three figures from the DP rows, their no-DP references and the round logs of both.
    baselines = dp_baseline_by_split(run_rows)
    run_dirs = dp_run_dir_index(runs, output_root)
    plot_dp_privacy_utility_curves(plt, grouped, baselines, analysis_root / "DP_privacy_utility_curves.png")
    plot_dp_utility_drop_by_head(plt, grouped, baselines, analysis_root / "DP_utility_drop_by_head.png")
    plot_dp_convergence_rounds(plt, grouped, baselines, run_dirs, analysis_root / "DP_convergence_rounds.png")

# ----------------------------------------------------------------------------------------------------------------------
# 10. ANALYSIS ENTRY POINT

# Run the full analysis and write the four artifacts and the DP figures into the analysis root.
def run_analysis(output_root: Path,
                 analysis_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, str]]]:
    # Discover the runs and flatten them into summary and per-bank rows.
    warnings: list[dict[str, str]] = []
    runs = discover_runs(output_root, warnings)
    run_rows, bank_rows = build_rows(runs, output_root, warnings)

    # Write the two CSV summaries, the JSON payload and the Markdown report.
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_csv(analysis_root / "E_07_run_summary.csv", run_rows, RUN_SUMMARY_COLUMNS)
    write_csv(analysis_root / "E_07_per_bank_summary.csv", bank_rows, PER_BANK_COLUMNS)
    payload = {
        "schema_version": 1,
        "run_count": len(run_rows),
        "per_bank_row_count": len(bank_rows),
        "expected_matrix_count": len(expected_matrix_keys()),
        "warnings": warnings,
        "run_summary": run_rows,
        "per_bank_summary": bank_rows,
    }
    write_json(analysis_root / "E_07_training_analysis.json", payload)
    (analysis_root / "E_07_training_analysis.md").write_text(
        build_markdown(run_rows, bank_rows, warnings, output_root),
        encoding="utf-8",
    )

    # Add the three DP figures, which stay a no-op when the output root carries no DP run.
    write_dp_figures(run_rows, runs, output_root, analysis_root)
    return run_rows, bank_rows, warnings

# Return a nonzero exit code in strict mode when any warning was recorded.
def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    _run_rows, _bank_rows, warnings = run_analysis(args.output_root, args.analysis_root)
    return 1 if args.strict and warnings else 0

if __name__ == "__main__":
    sys.exit(main())

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────