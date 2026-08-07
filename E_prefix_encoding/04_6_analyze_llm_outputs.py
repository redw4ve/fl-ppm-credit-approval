"""
Step 4.6: Analyze E_04 LLM mapping drafts.

Run this optional script only for the LLM side experiment.
It compares LLM canonical schema and dataset mapping drafts against the frozen manual reference files after generation.

The script writes review metrics under `mappings/llm_mapping/llm_analysis/`. These reports are not encoder artifacts.
They document how well different prompt strategies match the manually reviewed E_04 ground truth.

OUTPUT:
    04_06_llm_analysis_results.json: all machine-readable analysis results
    04_06_llm_analysis_summary.txt: human-readable analysis summary
    04_06_llm_dataset_mapping_accuracy.png: repeated-run mapping accuracy chart
    04_06_llm_dataset_mapping_errors.png: repeated-run mapping error chart

REQUIRED FILES:
    E_prefix_encoding/mappings/MANUAL_canonical_schemas.json: frozen manual schema reference
    E_prefix_encoding/mappings/MANUAL_dataset_mapping.json: frozen manual dataset mapping reference
    E_prefix_encoding/mappings/llm_mapping/canonical_schemas/*.json: LLM canonical schema drafts
    E_prefix_encoding/mappings/llm_mapping/dataset_mappings/*.json: LLM dataset mapping drafts

CREATED FILES:
    E_prefix_encoding/mappings/llm_mapping/llm_analysis/04_06_llm_analysis_results.json: full analysis results
    E_prefix_encoding/mappings/llm_mapping/llm_analysis/04_06_llm_analysis_summary.txt: readable summary
    E_prefix_encoding/mappings/llm_mapping/llm_analysis/04_06_llm_dataset_mapping_accuracy.png: accuracy plot
    E_prefix_encoding/mappings/llm_mapping/llm_analysis/04_06_llm_dataset_mapping_errors.png: error plot
"""

# IMPORTS
from __future__ import annotations
import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Optional
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Allow direct script execution from the repository root.
if __package__ in {None, ""}: sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import the contract only to identify legal model input fields.
contract = importlib.import_module("E_prefix_encoding.04_1_contract")

# CONFIGURATION
SCRIPT_DIR: Path = Path(__file__).resolve().parent                          # Folder that contains this script
MAPPING_ROOT: Path = SCRIPT_DIR / "mappings"                                # Folder for manual references
LLM_MAPPING_ROOT: Path = MAPPING_ROOT / "llm_mapping"                       # Root folder for LLM side experiment files
LLM_SCHEMA_ROOT: Path = LLM_MAPPING_ROOT / "canonical_schemas"              # Folder with LLM schema drafts
LLM_DATASET_MAPPING_ROOT: Path = LLM_MAPPING_ROOT / "dataset_mappings"      # Folder with LLM mapping drafts
ANALYSIS_ROOT: Path = LLM_MAPPING_ROOT / "llm_analysis"                     # Folder for generated comparison reports
MANUAL_SCHEMA_PATH: Path = MAPPING_ROOT / "MANUAL_canonical_schemas.json"   # Schema reference
MANUAL_MAPPING_PATH: Path = MAPPING_ROOT / "MANUAL_dataset_mapping.json"    # Mapping reference

# Graph style for the thesis figures, consistent with the dataset split graphics.
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#666666"
LEGEND_EDGE = "#bfbfbf"
TEXT_DARK = "#222222"
NEUTRAL_GREY = "#cccccc"
BANK_COLORS = ["#003E7A", "#0065BF", "#5A9DDC", "#99C2E5", "#D6E4F2"]

LLM_SCHEMA_PATTERNS: tuple[str, ...] = ("*.json",)
LLM_MAPPING_PATTERNS: tuple[str, ...] = ("*.json",)
MAPPING_STRATEGY_ORDER: tuple[str, ...] = (
    "semantic_character",
    "semantic_word",
    "llm_strategy_1_character",
    "llm_strategy_1_word",
    "llm_strategy_2_split",
    "llm_strategy_3_classify",
)
MAPPING_STRATEGY_LABELS: dict[str, str] = {
    "semantic_character": "Semantic char",
    "semantic_word": "Semantic word",
    "llm_strategy_1_character": "LLM S1 char",
    "llm_strategy_1_word": "LLM S1 word",
    "llm_strategy_2_split": "LLM S2 split",
    "llm_strategy_3_classify": "LLM S3 classify",
}
SEEDLESS_STRATEGY_FILES: set[str] = {
    "04_03_strategy_2_split_prompt_dataset_mapping.json",
    "04_03_strategy_3_target_recipe_dataset_mapping.json",
}

# Contract fields that are real columns but illegal as a model input feature.
# A schema draft that selects any of them would leak a target or feed bookkeeping instead of a feature.
FORBIDDEN_MODEL_INPUT_FIELDS = {
    contract.CASE_ID, contract.EVENT_INDEX, contract.TIMESTAMP, contract.RAW_ACTIVITY, contract.LIFECYCLE,
    contract.RAW_ACTIVITY_TOKEN, contract.NEXT_ACTIVITY_RAW, contract.OUTCOME, contract.DATASET_ID,
    contract.CLIENT_ID, contract.SPLIT, contract.CANONICAL_ACTIVITY_LABEL, contract.NEXT_ACTIVITY_TARGET,
    contract.NEXT_ACTIVITY_MASK, contract.REMAINING_TIME, contract.REMAINING_TIME_MASK, contract.REQUESTED_AMOUNT_MASK,
    contract.LOAN_GOAL_MASK, contract.APPLICATION_TYPE_MASK, contract.OFFER_PRESENT, contract.OFFER_FEATURE_MASK,
}

# ----------------------------------------------------------------------------------------------------------------------
# 1. IO HELPERS

# HELPER: Parse optional automation arguments while keeping script defaults for WORKFLOW_run_encoding.sh.
def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze E_04 LLM mapping drafts.")
    parser.add_argument("--manual-schema-path", type=Path, default=MANUAL_SCHEMA_PATH)
    parser.add_argument("--manual-mapping-path", type=Path, default=MANUAL_MAPPING_PATH)
    parser.add_argument("--schema-root", type=Path, default=LLM_SCHEMA_ROOT)
    parser.add_argument("--mapping-root", type=Path, default=LLM_DATASET_MAPPING_ROOT)
    parser.add_argument("--analysis-root", type=Path, default=ANALYSIS_ROOT)
    parser.add_argument("--write-plots", action="store_true")
    return parser.parse_args(argv)

# HELPER: Load one JSON artifact.
def load_json(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))

# HELPER: Save a JSON artifact with stable formatting.
def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

# HELPER: Save one text artifact for review.
def save_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")

# HELPER: Find LLM draft files without reading manual references.
def discover_llm_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []

    # Patterns allow the analysis to compare all saved strategy drafts in one run.
    for pattern in patterns: files.extend(sorted(root.rglob(pattern)))
    return sorted(set(files))

# HELPER: Return a stable score key that keeps run folders visible.
def _score_key(path: Path, root: Path) -> str: return path.relative_to(root).as_posix()

# HELPER: Return a stable division result for metric calculations.
def safe_divide(numerator: int, denominator: int) -> float: return float(numerator / denominator) if denominator else 0.0

# ----------------------------------------------------------------------------------------------------------------------
# 2. SCHEMA SCORING

# HELPER: Return all model input fields selected by one profile.
def _profile_input_fields(profile: dict[str, Any]) -> set[str]:
    fields: set[str] = set()
    for key in ["sequence_categorical_columns", "sequence_numerical_columns", "offer_numerical_columns"]:
        fields.update(str(field) for field in profile.get(key, []))
    return fields

# HELPER: Count invalid (absent or reserved for targets, ids, masks, ...) model input fields in one LLM schema profile.
def _illegal_model_fields(profile: dict[str, Any]) -> list[str]:
    selected = _profile_input_fields(profile)
    unknown = selected - set(contract.FIELD_CATALOG)
    forbidden = selected & FORBIDDEN_MODEL_INPUT_FIELDS
    return sorted(unknown.union(forbidden))

# HELPER: Score one canonical schema draft against the manual reference.
def score_canonical_schema(manual_schema: dict[str, Any], llm_schema: dict[str, Any]) -> dict[str, Any]:
    # manual_profiles are the frozen ground truth, llm_profiles are the candidate being evaluated
    manual_profiles = manual_schema.get("schema_profiles", {})
    llm_profiles = llm_schema.get("schema_profiles", {})
    true_positive = 0
    false_positive = 0
    false_negative = 0
    exact_profile_match_count = 0
    cap_match_count = 0
    illegal_field_count = 0

    # Compare selected input fields per profile and count illegal model facing fields.
    for profile_name, manual_profile in manual_profiles.items():
        llm_profile = llm_profiles.get(profile_name, {})
        manual_fields = _profile_input_fields(manual_profile)
        llm_fields = _profile_input_fields(llm_profile)
        true_positive += len(manual_fields & llm_fields)
        false_positive += len(llm_fields - manual_fields)
        false_negative += len(manual_fields - llm_fields)
        illegal_field_count += len(_illegal_model_fields(llm_profile))
        if manual_profile == llm_profile: exact_profile_match_count += 1
        if manual_profile.get("max_prefix_length_for_encoding") == llm_profile.get("max_prefix_length_for_encoding"):
            cap_match_count += 1

    # Field precision and recall describe how close the selected input field sets are.
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    field_f1 = safe_divide(2 * true_positive, 2 * true_positive + false_positive + false_negative)

    return {
        "valid_json": isinstance(llm_schema, dict) and "validation_error" not in llm_schema,
        "profile_count": len(manual_profiles),
        "profile_coverage": safe_divide(len(set(manual_profiles) & set(llm_profiles)), len(manual_profiles)),
        "exact_profile_match_count": exact_profile_match_count,
        "cap_match_count": cap_match_count,
        "field_precision": precision,
        "field_recall": recall,
        "field_f1": field_f1,
        "illegal_field_count": illegal_field_count,
    }

# ----------------------------------------------------------------------------------------------------------------------
# 3. COLUMN MAPPING SCORING

# Score parquet column mappings against the manual mapping.
def score_column_mapping(manual_mapping: dict[str, Any], llm_mapping: dict[str, Any]) -> dict[str, Any]:

    # Exact counts source columns that match the frozen manual mapping
    exact = 0
    wrong = 0
    unresolved = 0
    total = 0
    manual_datasets = manual_mapping.get("datasets", {})
    llm_datasets = llm_mapping.get("datasets", {})

    # Compare every manually mapped canonical field per dataset.
    for dataset_id, manual_dataset in manual_datasets.items():
        manual_columns = manual_dataset.get("column_mapping", {})
        llm_columns = llm_datasets.get(dataset_id, {}).get("column_mapping", {})
        for field, manual_column in manual_columns.items():
            total += 1
            llm_column = str(llm_columns.get(field, ""))
            if not llm_column: unresolved += 1
            elif llm_column == manual_column: exact += 1
            else: wrong += 1

    return {
        "total_fields": total,
        "exact_column_count": exact,
        "wrong_column_count": wrong,
        "unresolved_field_count": unresolved,
        "mapping_accuracy": safe_divide(exact, total),
        "required_field_coverage": safe_divide(total - unresolved, total),
    }

# ----------------------------------------------------------------------------------------------------------------------
# 4. ACTIVITY MAPPING SCORING

# Flatten grouped raw label mappings into one comparable lookup.
def flatten_activity_mapping(mapping_payload: dict[str, Any]) -> dict[tuple[str, str], str]:
    flat: dict[tuple[str, str], str] = {}
    groups = mapping_payload.get("activity_mapping", {}).get("canonical_activities", {})

    # Flatten labels_by_dataset into dataset plus raw label lookup keys.
    for canonical_label, group in groups.items():
        for dataset_id, labels in group.get("labels_by_dataset", {}).items():
            for label in labels: flat[(str(dataset_id), str(label))] = str(canonical_label)
    return flat

# Return the activity origin prefix of one raw or canonical label.
def _activity_prefix(value: str) -> str: return value.split("_", 1)[0] if "_" in value else ""

# Score raw activity mappings against the manual mapping.
def score_activity_mapping(manual_mapping: dict[str, Any], llm_mapping: dict[str, Any]) -> dict[str, Any]:
    # Flatten both mappings so every raw label can be compared directly.
    manual_flat = flatten_activity_mapping(manual_mapping)
    llm_flat = flatten_activity_mapping(llm_mapping)
    correct = 0
    wrong_count = 0
    missing_count = 0
    cross_prefix_count = 0

    # Missing labels are labels present in the manual reference but absent from the LLM draft.
    for key, manual_label in manual_flat.items():
        llm_label = llm_flat.get(key)
        if llm_label is None: missing_count += 1
        elif llm_label == manual_label: correct += 1
        else:
            wrong_count += 1
            if _activity_prefix(key[1]) != _activity_prefix(llm_label): cross_prefix_count += 1

    # Extra labels are labels the LLM introduced beyond the manual reference.
    extra_count = sum(1 for key in llm_flat if key not in manual_flat)
    unresolved = list(llm_mapping.get("activity_mapping", {}).get("unresolved_labels", []))

    return {
        "total_manual_labels": len(manual_flat),
        "correct_label_count": correct,
        "wrong_label_count": wrong_count,
        "missing_manual_label_count": missing_count,
        "extra_label_count": extra_count,
        "unresolved_label_count": len(unresolved),
        "cross_prefix_error_count": cross_prefix_count,
        "activity_accuracy": safe_divide(correct, len(manual_flat)),
    }

# ----------------------------------------------------------------------------------------------------------------------
# 5. STRATEGY SUMMARY

# Build one ranking entry for a schema draft.
def _schema_summary_entry(path: Path, score: dict[str, Any]) -> dict[str, Any]:

    # validity_gate marks whether the draft passed the minimum structural JSON checks needed for scoring.
    return {
        "file": path.name,
        "kind": "canonical_schema",
        "validity_gate": bool(score["valid_json"]),
        "illegal_field_count": score["illegal_field_count"],
        "field_f1": score["field_f1"],
        "profile_coverage": score["profile_coverage"],
    }

# Build one ranking entry for a dataset-mapping draft.
def _mapping_summary_entry(path: Path, column_score: dict[str, Any], activity_score: dict[str, Any],
    marked_invalid: bool = False) -> dict[str, Any]:

    # Cross-prefix activity mappings and drafts rejected by the generator both fail the validity gate.
    validity_gate = activity_score["cross_prefix_error_count"] == 0 and not marked_invalid
    return {
        "file": path.name,
        "kind": "dataset_mapping",
        "validity_gate": validity_gate,
        "cross_prefix_error_count": activity_score["cross_prefix_error_count"],
        "extra_label_count": activity_score["extra_label_count"],
        "missing_manual_label_count": activity_score["missing_manual_label_count"],
        "unresolved_label_count": activity_score["unresolved_label_count"],
        "wrong_column_count": column_score["wrong_column_count"],
        "wrong_label_count": activity_score["wrong_label_count"],
        "activity_accuracy": activity_score["activity_accuracy"],
        "column_mapping_accuracy": column_score["mapping_accuracy"],
    }

# Return the methodological strategy name for one dataset-mapping draft.
def _mapping_strategy_name(file_key: str) -> str:
    path = Path(file_key)
    strategy = path.name.removesuffix(".json")
    parts = path.parts
    if strategy == "04_03_semantic_character_dataset_mapping": return "semantic_character"
    if strategy == "04_03_semantic_word_dataset_mapping": return "semantic_word"
    if strategy == "04_03_strategy_1_baseline_dataset_mapping":
        if "semantic_character" in parts: return "llm_strategy_1_character"
        if "semantic_word" in parts: return "llm_strategy_1_word"
    if strategy == "04_03_strategy_2_split_prompt_dataset_mapping": return "llm_strategy_2_split"
    if strategy == "04_03_strategy_3_target_recipe_dataset_mapping": return "llm_strategy_3_classify"
    return strategy

# Return the arithmetic mean of a score list.
def _mean(values: list[float]) -> float: return float(sum(values) / len(values)) if values else 0.0

# Return a stable display order for known and unknown dataset-mapping strategies.
def _mapping_strategy_sort_key(strategy: str) -> tuple[int, str]:
    if strategy in MAPPING_STRATEGY_ORDER: return MAPPING_STRATEGY_ORDER.index(strategy), strategy
    return len(MAPPING_STRATEGY_ORDER), strategy

# Return a mean count while preserving old result payload compatibility.
def _mean_count(rows: list[dict[str, Any]], key: str) -> float:
    return _mean([float(row.get(key, 0)) for row in rows])

# Keep legacy seed-variant duplicates only until the current seedless files exist.
def _filter_seedless_strategy_duplicates(mapping_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    current_files: set[str] = set()
    for entry in mapping_entries:
        path = Path(str(entry.get("file", "")))
        if len(path.parts) == 2 and path.parts[0].startswith("run_") and path.name in SEEDLESS_STRATEGY_FILES:
            current_files.add(path.name)
    if not current_files: return mapping_entries

    filtered: list[dict[str, Any]] = []
    for entry in mapping_entries:
        path = Path(str(entry.get("file", "")))
        if path.name in current_files and any(part in {"semantic_character", "semantic_word"} for part in path.parts):
            continue
        filtered.append(entry)
    return filtered

# Filter only dataset-mapping rows while preserving schema rows in the human report.
def _filter_summary_seedless_strategy_duplicates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapping_entries = [entry for entry in entries if entry.get("kind") == "dataset_mapping"]
    filtered_mapping_ids = {id(entry) for entry in _filter_seedless_strategy_duplicates(mapping_entries)}
    return [
        entry for entry in entries
        if entry.get("kind") != "dataset_mapping" or id(entry) in filtered_mapping_ids
    ]

# Aggregate repeated dataset-mapping strategy runs.
def build_repeated_mapping_summary(entries: list[dict[str, Any]]) -> dict[str, Any]:
    mapping_entries = [entry for entry in entries if entry.get("kind") == "dataset_mapping"]
    run_entries = [
        entry for entry in mapping_entries
        if Path(str(entry.get("file", ""))).parts and Path(str(entry.get("file", ""))).parts[0].startswith("run_")
    ]
    if run_entries: mapping_entries = run_entries
    mapping_entries = _filter_seedless_strategy_duplicates(mapping_entries)

    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in mapping_entries: groups.setdefault(_mapping_strategy_name(str(entry["file"])), []).append(entry)

    strategies: dict[str, dict[str, Any]] = {}
    for strategy, rows in sorted(groups.items(), key=lambda item: _mapping_strategy_sort_key(item[0])):
        activity = [float(row.get("activity_accuracy", 0.0)) for row in rows]
        columns = [float(row.get("column_mapping_accuracy", 0.0)) for row in rows]
        blocking = [
            int(row.get("cross_prefix_error_count", 0))
            + int(row.get("wrong_column_count", 0))
            + int(row.get("wrong_label_count", 0))
            for row in rows
        ]
        correction_burden = [
            int(row.get("cross_prefix_error_count", 0))
            + int(row.get("wrong_column_count", 0))
            + int(row.get("wrong_label_count", 0))
            + int(row.get("missing_manual_label_count", 0))
            + int(row.get("extra_label_count", 0))
            + int(row.get("unresolved_label_count", 0))
            for row in rows
        ]
        strategies[strategy] = {
            "run_count": len(rows),
            "valid_run_count": sum(1 for row in rows if row.get("validity_gate")),
            "approval_ready_run_count": sum(1 for value in correction_burden if value == 0),
            "activity_accuracy_mean": _mean(activity),
            "column_mapping_accuracy_mean": _mean(columns),
            "blocking_error_count_sum": int(sum(blocking)),
            "blocking_error_count_mean": _mean([float(value) for value in blocking]),
            "correction_burden_sum": int(sum(correction_burden)),
            "correction_burden_mean": _mean([float(value) for value in correction_burden]),
            "missing_manual_label_count_sum": int(sum(int(row.get("missing_manual_label_count", 0)) for row in rows)),
            "missing_manual_label_count_mean": _mean_count(rows, "missing_manual_label_count"),
            "extra_label_count_sum": int(sum(int(row.get("extra_label_count", 0)) for row in rows)),
            "extra_label_count_mean": _mean_count(rows, "extra_label_count"),
            "unresolved_label_count_sum": int(sum(int(row.get("unresolved_label_count", 0)) for row in rows)),
            "unresolved_label_count_mean": _mean_count(rows, "unresolved_label_count"),
            "wrong_column_count_sum": int(sum(int(row.get("wrong_column_count", 0)) for row in rows)),
            "wrong_column_count_mean": _mean_count(rows, "wrong_column_count"),
            "wrong_label_count_sum": int(sum(int(row.get("wrong_label_count", 0)) for row in rows)),
            "wrong_label_count_mean": _mean_count(rows, "wrong_label_count"),
        }
    return {"strategies": strategies}

# Shorten strategy names for compact chart labels.
def _short_strategy_name(value: str) -> str:
    if value in MAPPING_STRATEGY_LABELS: return MAPPING_STRATEGY_LABELS[value]
    return value.replace("04_03_", "").replace("_dataset_mapping", "").replace("_", " ")

# Return strategies in the fixed thesis display order.
def _ordered_repeated_mapping_strategies(summary: dict[str, Any]) -> list[str]:
    rows = summary.get("strategies", {})
    return sorted(rows, key=_mapping_strategy_sort_key)

# Apply the thesis matplotlib style before plotting.
def _configure_plot_style() -> None:
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 11, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "legend.fontsize": 9, "figure.dpi": 150, "savefig.dpi": 300, "axes.spines.top": False,
        "axes.spines.right": False, "axes.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

# Apply the thesis non-pie axis style.
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

# Apply the thesis legend frame style.
def style_reference_legend(legend: Any) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_alpha(0.78)
    frame.set_linewidth(0.8)

# Add compact numeric labels above bars.
def _label_bars(ax: Any, bars: Any, values: list[float], fmt: str) -> None:
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height(), fmt.format(value), ha="center", va="bottom",
            color=TEXT_DARK, fontsize=8,
        )

# Save a compact accuracy plot for thesis reporting (thesis figure style).
def save_repeated_mapping_accuracy_plot(analysis_root: Path, summary: dict[str, Any]) -> None:
    _configure_plot_style()
    rows = summary.get("strategies", {})
    if not rows: return
    strategies = _ordered_repeated_mapping_strategies(summary)
    activity = [rows[name]["activity_accuracy_mean"] for name in strategies]
    columns = [rows[name]["column_mapping_accuracy_mean"] for name in strategies]
    x_values = list(range(len(strategies)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 4.5))
    column_bars = ax.bar(
        [value - width / 2 for value in x_values], columns, width=width, label="Column mapping", color=BANK_COLORS[0],
        edgecolor="white",
    )
    activity_bars = ax.bar(
        [value + width / 2 for value in x_values], activity, width=width, label="Activity mapping",
        color=BANK_COLORS[2], edgecolor="white",
    )
    _label_bars(ax, column_bars, columns, "{:.2f}")
    _label_bars(ax, activity_bars, activity, "{:.2f}")
    ax.set_ylabel("Mean accuracy")
    ax.set_ylim(0.0, 1.4)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticklabels(["0.0", "0.2", "0.4", "0.6", "0.8", "1.0", "", ""])
    ax.set_xticks(x_values)
    ax.set_xticklabels([_short_strategy_name(strategy) for strategy in strategies], rotation=20, ha="right")
    style_non_pie_axes(ax)
    legend = ax.legend(loc="upper right", frameon=True)
    style_reference_legend(legend)
    fig.tight_layout(pad=1.2)
    fig.savefig(analysis_root / "04_06_llm_dataset_mapping_accuracy.png", dpi=300)
    plt.close(fig)

# Return mean per-run error series for repeated dataset-mapping plots.
def _repeated_mapping_error_series(summary: dict[str, Any]) -> tuple[list[str], list[tuple[str, str, list[float]]]]:
    rows = summary.get("strategies", {})
    strategies = _ordered_repeated_mapping_strategies(summary)
    series = [
        ("Missing labels", BANK_COLORS[0], [float(rows[name].get("missing_manual_label_count_mean", 0.0)) for name in strategies]),
        ("Wrong labels", BANK_COLORS[1], [float(rows[name].get("wrong_label_count_mean", 0.0)) for name in strategies]),
        ("Wrong columns", BANK_COLORS[2], [float(rows[name].get("wrong_column_count_mean", 0.0)) for name in strategies]),
        ("Extra labels", BANK_COLORS[3], [float(rows[name].get("extra_label_count_mean", 0.0)) for name in strategies]),
        ("Unresolved labels", NEUTRAL_GREY, [float(rows[name].get("unresolved_label_count_mean", 0.0)) for name in strategies]),
    ]
    return strategies, series

# Save a compact error plot for thesis reporting (thesis figure style).
def save_repeated_mapping_error_plot(analysis_root: Path, summary: dict[str, Any]) -> None:
    _configure_plot_style()
    rows = summary.get("strategies", {})
    if not rows: return
    strategies, series = _repeated_mapping_error_series(summary)
    totals = [sum(values[index] for _, _, values in series) for index in range(len(strategies))]
    x_values = list(range(len(strategies)))
    bottom = [0 for _ in strategies]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for label, color, values in series:
        ax.bar(
            x_values, values, bottom=bottom, label=label, color=color, edgecolor="white",
        )
        bottom = [current + value for current, value in zip(bottom, values)]
    for x_value, total in zip(x_values, totals):
        ax.text(x_value, total, f"{total:.1f}", ha="center", va="bottom", color=TEXT_DARK, fontsize=8)
    ax.set_ylabel("Mean mapping errors per run")
    ax.set_xticks(x_values)
    ax.set_xticklabels([_short_strategy_name(strategy) for strategy in strategies], rotation=20, ha="right")
    style_non_pie_axes(ax)
    legend = ax.legend(loc="upper right", frameon=True)
    style_reference_legend(legend)
    fig.tight_layout(pad=1.2)
    fig.savefig(analysis_root / "04_06_llm_dataset_mapping_errors.png", dpi=300)
    plt.close(fig)

# Return the ranking metrics for one strategy entry.
def _ranking_metrics(entry: dict[str, Any]) -> tuple[float, float]:
    # Schema and mapping drafts use different quality metrics but share one ranking table.
    if entry.get("kind") == "canonical_schema":
        return float(entry.get("field_f1", 0.0)), float(entry.get("profile_coverage", 0.0))
    return float(entry.get("activity_accuracy", 0.0)), float(entry.get("column_mapping_accuracy", 0.0))

# Sort strategies by validity, quality and file name.
def rank_strategy_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        entries,
        key=lambda entry: (
            bool(entry.get("validity_gate")),
            _ranking_metrics(entry)[0],
            _ranking_metrics(entry)[1],
            str(entry.get("file", "")),
        ),
        reverse=True,
    )

# Format one metric value for the review table.
def _format_metric(value: Any) -> str:
    if isinstance(value, float): return f"{value:.4f}"
    return str(value)

# Return the short issue summary for one ranked strategy.
def _format_issue_summary(entry: dict[str, Any]) -> str:
    if entry.get("kind") == "canonical_schema": return f"illegal_fields={entry.get('illegal_field_count', 0)}"
    return (
        f"cross_prefix={entry.get('cross_prefix_error_count', 0)}, "
        f"missing={entry.get('missing_manual_label_count', 0)}, "
        f"extra={entry.get('extra_label_count', 0)}, "
        f"unresolved={entry.get('unresolved_label_count', 0)}, "
        f"wrong_labels={entry.get('wrong_label_count', 0)}, "
        f"wrong_columns={entry.get('wrong_column_count', 0)}"
    )

# Build rows for JSON and text review.
def build_strategy_table(summary_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Convert raw score dictionaries into one row per strategy review file.
    for rank, entry in enumerate(rank_strategy_entries(summary_entries), start=1):
        row: dict[str, Any] = {
            "rank": rank,
            "strategy": str(entry.get("file", "")).removesuffix(".json"),
            "kind": entry.get("kind", ""),
            "validity_gate": bool(entry.get("validity_gate")),
            "field_f1": None,
            "profile_coverage": None,
            "activity_accuracy": None,
            "column_mapping_accuracy": None,
            "issue_summary": _format_issue_summary(entry),
        }

        # Fill only the metrics that apply to the draft kind.
        if entry.get("kind") == "canonical_schema":
            row["field_f1"] = entry.get("field_f1", 0.0)
            row["profile_coverage"] = entry.get("profile_coverage", 0.0)
        else:
            row["activity_accuracy"] = entry.get("activity_accuracy", 0.0)
            row["column_mapping_accuracy"] = entry.get("column_mapping_accuracy", 0.0)
        rows.append(row)
    return rows

# Render a table for quick comparison.
def format_strategy_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Rank", "Kind", "Strategy", "Valid", "Field F1", "Profile coverage", "Activity accuracy", "Column accuracy",
        "Issue summary",
    ]

    # Convert rows into strings before computing fixed width table columns.
    body = [
        [
            str(row["rank"]),
            str(row["kind"]),
            str(row["strategy"]),
            "yes" if row["validity_gate"] else "no",
            _format_metric(row["field_f1"]) if row["field_f1"] is not None else "",
            _format_metric(row["profile_coverage"]) if row["profile_coverage"] is not None else "",
            _format_metric(row["activity_accuracy"]) if row["activity_accuracy"] is not None else "",
            _format_metric(row["column_mapping_accuracy"]) if row["column_mapping_accuracy"] is not None else "",
            str(row["issue_summary"]),
        ]
        for row in rows
    ]
    widths = [len(header) for header in headers]

    # Size each column from the widest header or value.
    for line in body: widths = [max(width, len(value)) for width, value in zip(widths, line)]

    # Write a plain text table that can be reviewed without opening JSON.
    lines = [
        "E_04 LLM strategy comparison",
        "",
        "  ".join(header.ljust(width) for header, width in zip(headers, widths)),
        "  ".join("-" * width for width in widths),
        *["  ".join(value.ljust(width) for value, width in zip(line, widths)) for line in body],
    ]
    return "\n".join(lines) + "\n"

# Return short metric explanations for the consolidated JSON and text summary.
def metric_explanations() -> dict[str, str]:
    return {
        "validity_gate": (
            "Shows whether the LLM output passed the structural validity checks needed for scoring."
        ),
        "field_f1": (
            "Measures whether the canonical schema selected the same active fields as the manual reference."
        ),
        "profile_coverage": (
            "Measures whether all expected schema profiles were created."
        ),
        "activity_accuracy": (
            "Measures how many raw activity labels match the manual canonical activity mapping."
        ),
        "column_mapping_accuracy": (
            "Measures how many source parquet columns match the manual column mapping."
        ),
        "correction_burden_sum": (
            "Counts missing labels, extra labels, unresolved labels and wrong mappings across repeated runs."
        ),
        "correction_burden_mean": (
            "Counts the mean number of missing labels, extra labels, unresolved labels and wrong mappings per run."
        ),
    }

# Render repeated-run metrics as compact text lines.
def format_repeated_mapping_summary(summary: dict[str, Any]) -> str:
    lines = ["Repeated dataset-mapping runs", ""]
    strategies = summary.get("strategies", {})
    for strategy, values in strategies.items():
        lines.extend(
            [
                f"* {_short_strategy_name(strategy)}",
                f"  * runs: {values.get('run_count', 0)}",
                f"  * valid runs: {values.get('valid_run_count', 0)}",
                f"  * activity accuracy mean: {values.get('activity_accuracy_mean', 0.0):.4f}",
                f"  * column accuracy mean: {values.get('column_mapping_accuracy_mean', 0.0):.4f}",
                f"  * correction burden mean: {values.get('correction_burden_mean', 0.0):.2f}",
                f"  * correction burden total: {values.get('correction_burden_sum', 0)}",
            ]
        )
    return "\n".join(lines) + "\n"

# Render one consolidated human-readable LLM analysis report.
def format_analysis_summary(results: dict[str, Any]) -> str:
    metric_lines = ["Metric explanations", ""]
    for metric, explanation in results["metric_explanations"].items(): metric_lines.append(f"* {metric}: {explanation}")
    sections = [
        "E_04 LLM side-experiment analysis",
        "",
        "Reference files:",
        f"* manual schema: {results['manual_schema_path']}",
        f"* manual mapping: {results['manual_mapping_path']}",
        "",
        format_strategy_table(results["strategy_table"]).rstrip(),
        "",
        format_repeated_mapping_summary(results["repeated_mapping_summary"]).rstrip(),
        "",
        "\n".join(metric_lines),
    ]
    return "\n".join(sections) + "\n"

# ----------------------------------------------------------------------------------------------------------------------
# 6. MAIN ANALYSIS

# Compare all available LLM drafts and write experiment reports.
def analyze_llm_outputs(manual_schema_path: Path, manual_mapping_path: Path, schema_root: Path, mapping_root: Path,
    analysis_root: Path, write_plots: bool = False) -> dict[str, Any]:

    # Load manual references only for scoring, never for LLM prompting.
    manual_schema = load_json(manual_schema_path)
    manual_mapping = load_json(manual_mapping_path)
    schema_scores: dict[str, Any] = {}
    column_scores: dict[str, Any] = {}
    activity_scores: dict[str, Any] = {}
    summary_entries: list[dict[str, Any]] = []

    # Score each saved LLM canonical schema draft.
    for path in discover_llm_files(schema_root, LLM_SCHEMA_PATTERNS):
        score = score_canonical_schema(manual_schema, load_json(path))
        score_key = _score_key(path, schema_root)
        schema_scores[score_key] = score
        entry = _schema_summary_entry(path, score)
        entry["file"] = score_key
        summary_entries.append(entry)

    # Score each saved LLM dataset mapping draft across columns and activities.
    for path in discover_llm_files(mapping_root, LLM_MAPPING_PATTERNS):
        payload = load_json(path)
        column_score = score_column_mapping(manual_mapping, payload)
        activity_score = score_activity_mapping(manual_mapping, payload)
        score_key = _score_key(path, mapping_root)
        column_scores[score_key] = column_score
        activity_scores[score_key] = activity_score
        entry = _mapping_summary_entry(path, column_score, activity_score, "validation_error" in payload)
        entry["file"] = score_key
        summary_entries.append(entry)

    # Store detailed scores inside one consolidated result payload.
    report_entries = _filter_summary_seedless_strategy_duplicates(summary_entries)
    strategy_table = build_strategy_table(report_entries)
    repeated_mapping_summary = build_repeated_mapping_summary(report_entries)
    results = {
        "manual_schema_path": str(manual_schema_path),
        "manual_mapping_path": str(manual_mapping_path),
        "schema_scores": schema_scores,
        "column_mapping_scores": column_scores,
        "activity_mapping_scores": activity_scores,
        "strategy_table": strategy_table,
        "strategy_ranking": rank_strategy_entries(summary_entries),
        "repeated_mapping_summary": repeated_mapping_summary,
        "metric_explanations": metric_explanations(),
    }
    save_json(analysis_root / "04_06_llm_analysis_results.json", results)
    save_text(analysis_root / "04_06_llm_analysis_summary.txt", format_analysis_summary(results))
    if write_plots:
        save_repeated_mapping_accuracy_plot(analysis_root, repeated_mapping_summary)
        save_repeated_mapping_error_plot(analysis_root, repeated_mapping_summary)
    return results

# Run the LLM side experiment analysis
def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    analyze_llm_outputs(
        args.manual_schema_path, args.manual_mapping_path, args.schema_root, args.mapping_root, args.analysis_root,
        args.write_plots,
    )

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb · Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────