"""
Shared reporting helpers for the E_05 baseline and the E_06 federated training runs.

The module owns everything that keeps the run reports of both training scripts identical in shape:
- The reporting profile switch between the compact production report and the legacy debug fragments
- JSON serialization that converts Path and numpy values into stable JSON-safe primitives
- The artifact manifest that lists generated files without reading prediction or checkpoint contents
- The environment block that records the interpreter, the platform and the library versions of one run
- The canonical compact run report with one fixed top-level layout for E_05 and E_06

E_05 and E_06 import this module, so a report schema change happens in exactly one place.
E_07 and the analysis builders read the reports this module writes, so the top-level keys are a stable contract.
The module has no dependency on the training scripts and imports no torch code.

Run: python -m E_training.training_reporting --output-root <root> to record one environment snapshot per invocation.

REQUIRED FILES:
    none directly (the calling script passes config, metrics, diagnostics and artifact payloads in)

CREATED FILES:
    <run_dir>/E_05_run_report.json or <run_dir>/E_06_run_report.json: the canonical compact run report
    <output_root>/E_environment.json: interpreter, platform and library versions of one workflow invocation
"""

# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Optional

# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------
REPORTING_PROFILES: tuple[str, str] = ("compact", "debug")   # allowed reporting profiles (compact is default)

# Libraries whose version decides whether a rerun can reproduce a reported number.
ENVIRONMENT_PACKAGES: tuple[str, ...] = ("torch", "numpy", "pandas", "scikit-learn", "matplotlib", "pyarrow", "flwr")

# Opacus only matters for the DP path, so it is recorded when a run activates differential privacy.
DP_ENVIRONMENT_PACKAGES: tuple[str, ...] = ("opacus",)

# ----------------------------------------------------------------------------------------------------------------------
# 1. PROFILE AND JSON HELPERS

# Convert non-native values used in reporting payloads into JSON-safe primitives.
def _json_default(value: object) -> object:
    if isinstance(value, Path): return str(value)
    if hasattr(value, "tolist") and value.__class__.__module__.startswith("numpy"): return value.tolist()
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"): return value.item()
    return value

# Normalize the reporting profile selected by the workflow.
def normalize_reporting_profile(value: Optional[str]) -> str:
    profile = "compact" if value is None else str(value).strip().lower()
    if profile not in REPORTING_PROFILES: raise ValueError(f"unknown reporting profile: {value}")
    return profile

# Debug mode keeps the legacy JSON fragments besides the compact report.
def should_write_legacy_reports(profile: str) -> bool:
    return normalize_reporting_profile(profile) == "debug"

# Write one stable JSON artifact with the same formatting used by E_05 and E_06.
def save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")

# ----------------------------------------------------------------------------------------------------------------------
# 2. ENVIRONMENT CAPTURE

# Read one installed distribution version, reporting an absent package instead of failing the run.
def _package_version(name: str) -> str:
    try: return str(importlib.metadata.version(name))
    except importlib.metadata.PackageNotFoundError: return "not installed"

# Record the interpreter, the machine and the library versions a reported number depends on.
# A clone-and-reproduce claim needs this block, because hyperparameters alone do not pin a result.
def build_environment_block(resolved_device: Optional[str] = None, use_dp: bool = False) -> dict[str, Any]:
    packages = ENVIRONMENT_PACKAGES + (DP_ENVIRONMENT_PACKAGES if use_dp else ())
    block: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {name: _package_version(name) for name in packages},
    }
    if resolved_device is not None: block["resolved_device"] = str(resolved_device)
    return block

# Write one environment snapshot beside the output root, so a whole workflow invocation is traceable as well.
def write_environment_snapshot(output_root: Path, use_dp: bool = False) -> Path:
    path = Path(output_root) / "E_environment.json"
    payload = {"created_at": datetime.now(timezone.utc).isoformat(), **build_environment_block(use_dp=use_dp)}
    save_json(path, payload)
    return path

# ----------------------------------------------------------------------------------------------------------------------
# 3. COMPACT RUN REPORT

# List generated artifacts without reading prediction or checkpoint contents.
def build_artifact_manifest(output_dir: Path) -> dict[str, Any]:
    files = [
        path.relative_to(output_dir).as_posix()
        for path in sorted(output_dir.rglob("*"))
        if path.is_file()
    ]
    return {"root": str(output_dir), "file_count": len(files), "files": files}

# Extract stable identity fields from the resolved run configuration.
def _run_identity(config: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "dataset", "run_name", "heterogeneity", "n_clients", "regime", "strategy", "bank", "seed", "learning_rate",
        "fedprox_mu", "max_epochs", "max_rounds", "local_epochs",
    )
    return {key: config[key] for key in keys if key in config and config[key] is not None}

# Build the canonical compact report for one E_05 or E_06 run.
def build_run_report(
    script_id: str,
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:

    # Copy the diagnostics so the pops below do not mutate the caller's payload.
    diagnostics_payload = dict(diagnostics)

    # Lift curves, predictions and timing out of diagnostics into their own top-level report keys.
    report: dict[str, Any] = {
        "schema_version": 1,
        "script_id": str(script_id),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_identity": _run_identity(config),
        "environment": build_environment_block(
            resolved_device=config.get("resolved_device"), use_dp=bool(config.get("use_dp", False))),
        "config": dict(config),
        "metrics": dict(metrics),
        "diagnostics": diagnostics_payload,
        "curves": diagnostics_payload.pop("curves", {}),
        "predictions": diagnostics_payload.pop("predictions", {}),
        "artifacts": dict(artifacts),
        "timing": diagnostics_payload.pop("timing", {}),
    }

    # Keep the federated round history as its own block when E_06 provides it.
    if "rounds" in diagnostics_payload: report["rounds"] = diagnostics_payload.pop("rounds")
    return report

# Write the canonical run report with the script-specific file name.
def write_run_report(output_dir: Path, script_id: str, payload: Mapping[str, Any]) -> Path:
    path = output_dir / f"{script_id}_run_report.json"
    save_json(path, payload)
    return path

# ----------------------------------------------------------------------------------------------------------------------
# 4. ENTRY POINT

# MAIN: record one environment snapshot for the calling workflow invocation.
def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Record one environment snapshot beside a training output root.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--use-dp", action="store_true", help="Also record the Opacus version.")
    args = parser.parse_args(argv)
    print(f"environment snapshot written to {write_environment_snapshot(args.output_root, use_dp=args.use_dp)}")

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────