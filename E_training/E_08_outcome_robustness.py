"""
Step 8: Quantify the outcome final-prefix bias of every finished run without touching training.

Prefix generation runs to the final event of a case, so for every case at or below the prefix cap, the last prefix
already contains the decision event while the outcome label is derived from that same event.
The remaining-time head masks that position, the outcome head deliberately does not.
This stage measures the resulting optimism instead of removing it, so the all-prefixes evaluation protocol stays
intact and the bias is reported as a robustness result.

Pipeline:
- Discovers every centralized, local and federated run below one output root through the E_07 discovery
- Reads the exported test prediction parquet of each run and the processed parquets for the per-case event count
- Marks every prefix whose label position is the last event of its case and reports its share and accuracy
- Recomputes outcome macro-F1, weighted-F1 and balanced accuracy including and excluding those prefixes
- Adds one per-source row for joint runs, so the cross-dataset reversal is visible

Run: python E_training/E_08_outcome_robustness.py --output-root <root> --analysis-root <root>/analysis

REQUIRED FILES:
    <output_root>/**/E_0{5,6}_run_report.json: run identity and artifact manifest of every finished run
    <output_root>/**/predictions/E_0{5,6}_predictions_test.parquet: exported test predictions per prefix
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: approved dataset mapping with the processed split roots
    E_main_BPIC_2017/data/processed/**/*.parquet: BPIC 2017 split parquets for the per-case event count
    E_ablation_BPIC_2012/data/processed/**/*.parquet: BPIC 2012 split parquets for the per-case event count

CREATED FILES:
    <analysis_root>/E_08_outcome_robustness.csv: one flat row per run and scope
    <analysis_root>/E_08_outcome_robustness.json: the same rows plus the run count and the warnings
    <analysis_root>/E_08_outcome_robustness.md: the sectioned Markdown report for the thesis write-up
"""

# IMPORTS
# ----------------------------------------------------------------------------------------------------------------------
from __future__ import annotations
import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Optional
import numpy as np
import pandas as pd

# Make direct script execution imports work.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The stage-numbered module names start with a capital letter, so the lowercase aliases are intentional.
# noinspection PyPep8Naming
from E_training import E_05_central_and_local_baselines_final as baseline
# noinspection PyPep8Naming
from E_training import E_07_generate_training_analysis as analysis
from E_training import training_core_final as core

# CONFIGURATION
# ----------------------------------------------------------------------------------------------------------------------

# Column order of the robustness CSV.
ROBUSTNESS_COLUMNS: list[str] = [
    "script_id", "dataset", "run_name", "regime", "strategy", "bank", "use_dp", "dp_target_epsilon", "scope",
    "n_prefixes", "n_final_prefixes", "final_prefix_share", "final_prefix_outcome_accuracy", "outcome_macro_f1",
    "outcome_weighted_f1", "outcome_balanced_accuracy", "outcome_macro_f1_excluding_final",
    "outcome_weighted_f1_excluding_final", "outcome_balanced_accuracy_excluding_final", "outcome_macro_f1_inflation",
    "outcome_weighted_f1_inflation", "outcome_balanced_accuracy_inflation", "run_dir",
]

# Probability columns the prediction export writes per outcome class.
OUTCOME_PROBABILITY_COLUMNS: list[str] = [f"outcome_prob_{label}" for label in core.OUTCOME_CLASSES]

# Configure the logger.
log = logging.getLogger("E_08")
def _configure_logging() -> None: logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------------
# 1. CLI

# Parse the output root, the analysis root and the local-run switch from the CLI.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report the E_08 outcome final-prefix robustness.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--skip-local", action="store_true", help="Skip the local baselines to shorten the run.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any warning was recorded.")
    return parser.parse_args(argv)

# ----------------------------------------------------------------------------------------------------------------------
# 2. EVENT COUNTS AND FINAL-PREFIX MARKING

# Build an E_05 config that points at the processed splits of one discovered run.
def baseline_config_for_run(config: dict[str, Any]) -> baseline.BaselineRunConfig:
    run_name = str(config.get("run_name", ""))
    heterogeneity, _, clients = run_name.rpartition("_")
    return baseline.BaselineRunConfig(
        dataset=str(config.get("dataset", "")),
        heterogeneity=heterogeneity,
        n_clients=int(str(clients).removesuffix("banks") or 0),
        regime="centralized",
    )

# Count the events of every case in one split, keyed by dataset and case id.
# The union of the bank splits is the pooled split, so one lookup serves centralized, local and federated runs alike.
def event_counts_for_run(config: dict[str, Any], mapping: dict[str, Any],
                         split_name: str = "test") -> dict[tuple[str, str], int]:
    run_config = baseline_config_for_run(config)
    counts: dict[tuple[str, str], int] = {}
    for bank in baseline.bank_names_for_config(run_config):
        dataset_id, path = baseline.bank_split_parquet_path(run_config, mapping, bank, split_name)
        if not path.exists(): raise FileNotFoundError(f"missing split parquet for the event count: {path}")

        # Read only the case-id column because the event count is the number of rows per case.
        case_column = str(mapping["datasets"][dataset_id]["column_mapping"][baseline.encoding.CASE_ID])
        case_ids = pd.read_parquet(path, columns=[case_column])[case_column].astype(str)
        for case_id, n_events in case_ids.value_counts().items():
            counts[(dataset_id, str(case_id))] = int(n_events)
    return counts

# Mark every prefix whose label position is the last event of its own case.
# Cases longer than the prefix cap never reach their last event, so they are correctly left unmarked.
def mark_final_prefixes(predictions: pd.DataFrame, event_counts: dict[tuple[str, str], int]) -> np.ndarray:
    keys = list(zip(predictions["dataset_id"].astype(str), predictions["case_id"].astype(str)))
    missing = [key for key in keys if key not in event_counts]
    if missing: raise ValueError(f"{len(missing)} predicted cases have no event count, first is {missing[0]}")
    n_events = np.array([event_counts[key] for key in keys], dtype=np.int64)
    return np.asarray(predictions["label_pos"].to_numpy(dtype=np.int64) == (n_events - 1))

# ----------------------------------------------------------------------------------------------------------------------
# 3. METRIC BLOCKS

# Score the outcome head on one prefix subset through the shared metric helper.
# The export stores probabilities, so the logits are reconstructed as their log, which leaves the softmax unchanged.
def outcome_metrics_for(predictions: pd.DataFrame) -> dict[str, Any]:
    logits = np.log(np.clip(predictions[OUTCOME_PROBABILITY_COLUMNS].to_numpy(dtype=float), 1e-12, 1.0))
    return core.compute_outcome_metrics(predictions["outcome_label"].to_numpy(dtype=int), logits)

# Build one robustness row from one prediction frame and its final-prefix marking.
def robustness_row(predictions: pd.DataFrame, is_final: np.ndarray) -> dict[str, Any]:
    n_prefixes = int(len(predictions))
    n_final = int(is_final.sum())
    final_accuracy = ""
    if n_final:
        final = predictions.loc[is_final]
        final_accuracy = float((final["outcome_label"] == final["outcome_pred"]).mean())

    # Score the full split and the split without its final prefixes with the same metric implementation.
    full = outcome_metrics_for(predictions)
    row: dict[str, Any] = {
        "n_prefixes": n_prefixes,
        "n_final_prefixes": n_final,
        "final_prefix_share": float(n_final / n_prefixes) if n_prefixes else "",
        "final_prefix_outcome_accuracy": final_accuracy,
        "outcome_macro_f1": float(full["macro_f1"]),
        "outcome_weighted_f1": float(full["weighted_f1"]),
        "outcome_balanced_accuracy": float(full["balanced_accuracy"]),
    }

    # A split without a final prefix has nothing to correct, so the corrected columns stay empty.
    if n_final == 0 or n_final == n_prefixes:
        for name in ("macro_f1", "weighted_f1", "balanced_accuracy"):
            row[f"outcome_{name}_excluding_final"] = ""
            row[f"outcome_{name}_inflation"] = ""
        return row

    corrected = outcome_metrics_for(predictions.loc[~is_final])
    for name in ("macro_f1", "weighted_f1", "balanced_accuracy"):
        row[f"outcome_{name}_excluding_final"] = float(corrected[name])
        row[f"outcome_{name}_inflation"] = float(full[name]) - float(corrected[name])
    return row

# ----------------------------------------------------------------------------------------------------------------------
# 4. RUN PROCESSING

# Resolve the exported test prediction parquet of one discovered run.
def test_predictions_path(script_id: str, run_dir: Path) -> Path:
    return run_dir / baseline.PREDICTIONS_DIR_NAME / f"{script_id}_predictions_test.parquet"

# Build every robustness row of one discovered run, with one extra row per source dataset for joint runs.
def rows_for_run(run: dict[str, Any], mapping: dict[str, Any], output_root: Path,
                 warnings: list[dict[str, str]]) -> list[dict[str, Any]]:
    config = dict(run["config"])
    run_dir = Path(run["run_dir"])
    script_id = str(config.get("script_id", ""))
    path = test_predictions_path(script_id, run_dir)
    if not path.exists():
        warnings.append({"severity": "warning", "message": f"missing test predictions: {run_dir}"})
        return []

    # Mark the final prefixes once for the whole split, then score the split and every source dataset inside it.
    predictions = pd.read_parquet(path)
    is_final = mark_final_prefixes(predictions, event_counts_for_run(config, mapping))
    identity = {
        "script_id": script_id,
        "dataset": config.get("dataset", ""),
        "run_name": config.get("run_name", ""),
        "regime": analysis.resolve_regime(config),
        "strategy": config.get("strategy", ""),
        "bank": config.get("bank", ""),
        "use_dp": analysis.is_truthy(config.get("use_dp", False)),
        "dp_target_epsilon": config.get("dp_target_epsilon", ""),
        "run_dir": str(run_dir.relative_to(output_root)) if run_dir.is_absolute() else str(run_dir),
    }
    rows = [{**identity, "scope": "all", **robustness_row(predictions, is_final)}]

    # A joint run pools two source datasets whose bias differs, so each source gets its own row.
    sources = sorted(set(predictions["dataset_id"].astype(str)))
    if len(sources) > 1:
        for source in sources:
            selector = (predictions["dataset_id"].astype(str) == source).to_numpy()
            rows.append({**identity, "scope": source,
                         **robustness_row(predictions.loc[selector], is_final[selector])})
    return rows

# ----------------------------------------------------------------------------------------------------------------------
# 5. MARKDOWN REPORT

# HELPER: Name one run inside its split, so a flipped comparison can be reported precisely.
def _run_label(row: dict[str, Any]) -> str:
    parts = [str(row.get("regime") or ""), str(row.get("strategy") or ""), str(row.get("bank") or "")]
    if analysis.is_truthy(row.get("use_dp")): parts.append(f"dp_eps_{row.get('dp_target_epsilon', '')}")
    return " ".join(part for part in parts if part) or "run"

# Compare every pair of runs that share a dataset and a split to see how many keep the sign of their gap.
# The bias cancels only approximately, so a comparison already inside the noise band can still flip.
def within_split_sign_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("scope") != "all": continue
        if row.get("outcome_macro_f1_excluding_final") in {"", None}: continue
        grouped.setdefault((str(row.get("dataset", "")), str(row.get("run_name", ""))), []).append(row)

    # Every unordered pair inside one split is one within-dataset comparison the thesis can make.
    total, flipped = 0, []
    for (dataset, run_name), split_rows in sorted(grouped.items()):
        ordered = sorted(split_rows, key=_run_label)
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                left, right = ordered[left_index], ordered[right_index]
                raw = float(left["outcome_macro_f1"]) - float(right["outcome_macro_f1"])
                corrected = (float(left["outcome_macro_f1_excluding_final"])
                             - float(right["outcome_macro_f1_excluding_final"]))
                total += 1
                if (raw > 0.0) != (corrected > 0.0):
                    flipped.append({
                        "dataset": dataset, "run_name": run_name,
                        "comparison": f"{_run_label(left)} minus {_run_label(right)}",
                        "raw_delta": raw, "corrected_delta": corrected,
                    })
    return {"comparisons": total, "sign_kept": total - len(flipped), "flipped": flipped}

# Build the sectioned Markdown report from the robustness rows.
def build_markdown(rows: list[dict[str, Any]], warnings: list[dict[str, str]], output_root: Path) -> str:
    # One append per report line, so each paragraph stays readable next to the section it belongs to.
    # noinspection PyListCreation
    lines = ["# E_08 Outcome Final-Prefix Robustness", ""]
    lines.append("## Scope")
    lines.append("")
    lines.append("Prefix generation runs to the final event of a case, so the last prefix of every case at or below "
                 "the prefix cap already contains the decision event that defines the outcome label.")
    lines.append("The outcome head is deliberately not masked at that position, so the reported outcome metrics carry "
                 "a measurable optimism. This stage measures it and never changes a training result.")
    lines.append("The bias is uniform in magnitude inside one dataset, so it largely cancels in a comparison of two "
                 "runs of that dataset. It does not cancel exactly, so a comparison whose raw gap is already smaller "
                 "than the difference of the two biases can still change sign.")
    lines.append("It is not uniform across datasets, so every cross-dataset outcome ordering must use the corrected "
                 "column.")
    lines.append("")

    # State the sign property as a computed result rather than as an assertion.
    agreement = within_split_sign_agreement(rows)
    lines.append("## Within-split sign agreement")
    lines.append("")
    lines.append(f"Comparisons examined: {agreement['comparisons']}. These are every unordered pair of runs that "
                 "share a dataset and a split, which is the set of within-dataset comparisons the thesis makes.")
    lines.append(f"Comparisons whose outcome macro-F1 gap keeps its sign after excluding the final prefix: "
                 f"{agreement['sign_kept']} of {agreement['comparisons']}.")
    if agreement["flipped"]:
        lines.append(f"Sign changes: {len(agreement['flipped'])}. Each one is listed below with both gaps, so the "
                     "manuscript can state the exception instead of claiming universal stability.")
        lines.append("Two kinds appear. A comparison whose raw gap is already inside the noise band can flip on a "
                     "difference of a few ten-thousandths. A joint comparison between a BPIC 2012 bank and a "
                     "BPIC 2017 bank is a cross-dataset comparison despite sharing the joint split, so the two biases "
                     "differ by about three percentage points and the flip is expected there.")
        lines.append(analysis.markdown_table(
            agreement["flipped"], ["dataset", "run_name", "comparison", "raw_delta", "corrected_delta"]))
    else:
        lines.append("No comparison changed sign in this matrix.")
    lines.append("")

    # Report the centralized runs first, because they carry the headline outcome numbers.
    centralized = [row for row in rows if row.get("regime") == "centralized" and row.get("scope") == "all"]
    lines.append("## Centralized baselines")
    lines.append(analysis.markdown_table(
        sorted(centralized, key=lambda row: (str(row["dataset"]), str(row["run_name"]))),
        ["dataset", "run_name", "n_prefixes", "final_prefix_share", "final_prefix_outcome_accuracy",
         "outcome_macro_f1", "outcome_macro_f1_excluding_final", "outcome_macro_f1_inflation"],
    ))

    federated = [row for row in rows if row.get("script_id") == analysis.SCRIPT_ID_FEDERATED
                 and row.get("scope") == "all"]
    lines.append("## Federated runs")
    lines.append(analysis.markdown_table(
        sorted(federated, key=lambda row: (str(row["dataset"]), str(row["run_name"]), str(row["strategy"]))),
        ["dataset", "run_name", "strategy", "use_dp", "dp_target_epsilon", "final_prefix_share",
         "outcome_macro_f1", "outcome_macro_f1_excluding_final", "outcome_macro_f1_inflation"],
    ))

    per_source = [row for row in rows if row.get("scope") not in {"all", None}]
    lines.append("## Joint runs per source dataset")
    lines.append(analysis.markdown_table(
        sorted(per_source, key=lambda row: (str(row["run_name"]), str(row["regime"]), str(row["scope"]))),
        ["run_name", "regime", "strategy", "scope", "final_prefix_share", "final_prefix_outcome_accuracy",
         "outcome_macro_f1", "outcome_macro_f1_excluding_final", "outcome_macro_f1_inflation"],
    ))

    lines.append("## Warnings")
    if warnings: lines.extend(f"- {warning['severity']}: {warning['message']}" for warning in warnings)
    else: lines.append("No warnings.")
    lines.append("")
    lines.append("## Artifact index")
    lines.append(f"Output root: {output_root}")
    lines.append("Files: E_08_outcome_robustness.md, E_08_outcome_robustness.json, E_08_outcome_robustness.csv")
    return "\n".join(lines)

# ----------------------------------------------------------------------------------------------------------------------
# 6. ANALYSIS ENTRY POINT

# Run the robustness stage over one output root and write the three artifacts.
def run_robustness(output_root: Path, analysis_root: Path,
                   skip_local: bool = False) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    mapping = baseline.load_dataset_mapping(baseline.DATASET_MAPPING_PATH, require_approved=True)
    runs = analysis.discover_runs(output_root, warnings)

    # Score every discovered run, skipping the local baselines only when the caller asked for the short pass.
    rows: list[dict[str, Any]] = []
    for run in runs:
        if skip_local and str(run["config"].get("regime", "")) == "local": continue
        log.info("Scoring %s", run["run_dir"])
        rows.extend(rows_for_run(run, mapping, output_root, warnings))

    # Write the flat CSV, the JSON payload and the Markdown report next to the E_07 outputs.
    analysis_root.mkdir(parents=True, exist_ok=True)
    analysis.write_csv(analysis_root / "E_08_outcome_robustness.csv", rows, ROBUSTNESS_COLUMNS)
    analysis.write_json(
        analysis_root / "E_08_outcome_robustness.json",
        {"schema_version": 1, "row_count": len(rows), "warnings": warnings,
         "within_split_sign_agreement": within_split_sign_agreement(rows), "rows": rows},
    )
    (analysis_root / "E_08_outcome_robustness.md").write_text(
        build_markdown(rows, warnings, output_root), encoding="utf-8")
    return rows, warnings

# MAIN: score one output root per invocation and return a nonzero exit code in strict mode.
def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    args = parse_args(argv)
    _rows, warnings = run_robustness(args.output_root, args.analysis_root, skip_local=args.skip_local)
    return 1 if args.strict and warnings else 0

if __name__ == "__main__":
    sys.exit(main())

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────