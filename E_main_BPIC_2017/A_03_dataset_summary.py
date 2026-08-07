"""
Step 3: Dataset analytics for partitioned splits (BPIC 2017 - main).

- Read A_02_partition_stats.csv and A_02_metadata.json from each data/processed/{heterogeneity}_{n_clients}banks/ folder.
- Build a unified comparison table for the thesis.
- Renders figures (PNG only, 300 DPI).

Run: python A_03_dataset_summary.py

Outputs:
  - one row per (config, bank): data/processed/A_03_comparison_table.csv
  - case shares per config: plots/A_03_partition_pie_charts.png
  - grouped bars, all configs: plots/A_03_approval_rate_by_bank.png
  - 100% stacked bars: plots/A_03_client_size_distribution.png
  - line over heterogeneity: plots/A_03_approval_rate_spread.png
  - strong-config undersampling: plots/A_03_cases_dropped.png
  - RequestedAmount quintile shares: plots/A_03_requested_amount_quintiles.png
  - LoanGoal group distribution: plots/A_03_loangoal_group_distribution.png
"""

# IMPORTS
from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Render figures directly to files.
matplotlib.use("Agg")

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data/processed"
PLOTS_DIR = SCRIPT_DIR / "plots"

# Keep a dark-to-light bank palette anchored on TUM blue ("#0065BF").
# This is consistent with BPIC 2012.
BANK_COLORS = ["#003E7A", "#0065BF", "#5A9DDC", "#99C2E5", "#D6E4F2"]
LINE_COLOR = "#0065BF"
WARN_COLOR = "#C8102E"
NEUTRAL_COLOR = "#cccccc"
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#666666"
LEGEND_EDGE = "#bfbfbf"
TEXT_COLOR = "#222222"

# Order configs by increasing heterogeneity, then secondary by client count.
CONFIG_ORDER = ["iid", "weak", "medium", "strong"]
CONFIG_DISPLAY_ORDER = ["iid_3banks", "weak_3banks", "medium_3banks", "strong_3banks", "medium_5banks", "strong_5banks"]
BANK_ORDER = ["A", "B", "C", "D", "E"]
SHORT_LABELS = {
    "iid_3banks":    "iid-3",
    "weak_3banks":   "weak-3",
    "medium_3banks": "medium-3",
    "medium_5banks": "medium-5",
    "strong_3banks": "strong-3",
    "strong_5banks": "strong-5",
}
# Order RequestedAmount bands from zero to high values.
AMOUNT_BAND_ORDER = ["Zero", "Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
AMOUNT_BAND_COLORS = ["#cccccc", "#D6E4F2", "#99C2E5", "#5A9DDC", "#0065BF", "#003E7A"]

# Group LoanGoal values by approval tendency.
LOANGOAL_HIGH_APPROVAL = {"Remaining debt home", "Boat", "Caravan / Camper", "Unknown"}
LOANGOAL_LOW_APPROVAL = {
    "Tax payments", "Not speficied", "Other, see explanation",
    "Extra spending limit", "Motorcycle", "Business goal",
}

# Mark LoanGoal values that define specialist banks.
LOANGOAL_SPECIALIST_D = "Home improvement"
LOANGOAL_SPECIALIST_E = "Existing loan takeover"

# Order LoanGoal groups for stacked bars and legends.
LOANGOAL_GROUP_ORDER = ["High-approval", "Core mixed", "Low-approval", "Specialist D", "Specialist E", "Other"]
LOANGOAL_GROUP_COLORS = ["#003E7A", "#0065BF", "#5A9DDC", "#2E7D32", "#66BB6A", "#cccccc"]

# Store LoanGoal legend labels with grouped raw values.
LOANGOAL_GROUP_LABELS = {
    "High-approval": "High-approval\nRemaining debt home, Boat, Caravan / Camper, Unknown",
    "Core mixed": "Core mixed\nCar, Debt restructuring",
    "Low-approval": "Low-approval\nTax payments, Not specified, Other, Extra spending limit, Motorcycle, Business goal",
    "Specialist D": "Specialist D\nHome improvement",
    "Specialist E": "Specialist E\nExisting loan takeover",
    "Other": "Other\nunmapped or missing",
}

# Apply shared matplotlib defaults once.
plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "standard",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Configure the script logger.
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("analytics")

# ----------------------------------------------------------------------------------------------------------------------
# 1. COLLECT CONFIGS

# Collect valid partition folders.
def collect_configs() -> list[tuple[str, int, Path]]:
    configs = []

    # Iterate process directory.
    for d in sorted(PROCESSED_DIR.iterdir()):
        if not d.is_dir() or d.name == "centralized": continue
        parts = d.name.split("_")
        if len(parts) != 2 or not parts[1].endswith("banks"): continue
        het, suffix = parts[0], parts[1]
        try: n = int(suffix.replace("banks", ""))
        except ValueError: continue
        configs.append((het, n, d))

    # Rank configs by heterogeneity level and client count.
    rank = {h: i for i, h in enumerate(CONFIG_ORDER)}
    configs.sort(key=lambda x: (rank.get(x[0], 99), x[1]))
    return configs

# ----------------------------------------------------------------------------------------------------------------------
# 2. COMPARISON TABLE

# Read metadata for one partition config.
def load_partition_metadata(path: Path) -> dict[str, Any]:
    with (path / "A_02_metadata.json").open() as f:
        return json.load(f)

# Extract partition provenance fields for the comparison table.
def provenance_columns(metadata: dict[str, Any]) -> dict[str, object]:
    provenance = metadata.get("partition_provenance", {})
    strong = provenance.get("strong_approval_enforcement", {}) or {}
    medium = provenance.get("medium_approval_enforcement", {}) or {}
    return {
        "n_cases_available_after_filters": provenance.get("n_cases_available_after_filters"),
        "n_cases_assigned":                provenance.get("n_cases_assigned"),
        "n_cases_unassigned_or_dropped":   provenance.get("n_cases_unassigned_or_dropped"),
        "strong_target_ab_gap":            strong.get("target_ab_gap"),
        "strong_target_bc_gap":            strong.get("target_bc_gap"),
        "strong_final_ab_gap":             strong.get("final_ab_gap"),
        "strong_final_bc_gap":             strong.get("final_bc_gap"),
        "strong_dropped_total":            strong.get("n_cases_dropped_total"),
        "medium_target_ab_gap":            medium.get("target_ab_gap"),
        "medium_target_bc_gap":            medium.get("target_bc_gap"),
        "medium_final_ab_gap":             medium.get("final_ab_gap"),
        "medium_final_bc_gap":             medium.get("final_bc_gap"),
        "medium_dropped_total":            medium.get("n_cases_dropped_total"),
    }

# Build one comparison table across partition configs.
def build_comparison_table(configs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    rows = []

    # Read partition stats for each config.
    for het, n, path in configs:
        stats = pd.read_csv(path / "A_02_partition_stats.csv")
        provenance = provenance_columns(load_partition_metadata(path))

        # Add config identifiers as leading columns and populate provenance fields.
        stats.insert(0, "heterogeneity", het)
        stats.insert(1, "n_clients", n)
        stats.insert(2, "config", f"{het}_{n}banks")
        for col, value in provenance.items():
            stats[col] = value

        # Add metrics for every configuration.
        stats["case_share"] = stats["n_cases"] / stats["n_cases"].sum()
        stats["approval_rate_spread"] = stats["approval_rate"].max() - stats["approval_rate"].min()
        rows.append(stats)

    # Suppress pandas dtype warning for optional provenance columns, NaN values are valid.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="The behavior of DataFrame concatenation with empty or all-NA entries is deprecated",
            category=FutureWarning,
        )
        return pd.concat(rows, ignore_index=True)

# ----------------------------------------------------------------------------------------------------------------------
# 2.5 CASE SPLIT TABLE

# Stream one parquet batch (whole parquet too large) and keep one row per case -> Enough for case attributes.
def _read_case_rows(parquet_path: Path, columns: list[str]) -> pd.DataFrame:
    rows = []
    seen: set[str] = set()
    parquet = pq.ParquetFile(parquet_path)
    for batch in parquet.iter_batches(batch_size=50_000, columns=columns):
        frame = batch.to_pandas()
        frame = frame.drop_duplicates("case:concept:name")
        mask = ~frame["case:concept:name"].astype(str).isin(seen)
        frame = frame.loc[mask].copy()
        seen.update(frame["case:concept:name"].astype(str).tolist())
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=columns)

# Build one case-level table from train, val and test files.
def build_case_level_table(configs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    rows = []
    columns = ["case:concept:name", "case:RequestedAmount", "case:LoanGoal", "outcome"]
    for het, n, path in configs:
        config = f"{het}_{n}banks"
        before = len(rows)

        # Read each bank split and keep only case attributes.
        for parquet_path in sorted(path.glob("A_02_bank_*_*.parquet")):
            parts = parquet_path.stem.split("_")
            if len(parts) < 5: continue
            bank = parts[3]
            split = parts[4]
            assert bank in BANK_ORDER, f"Unexpected bank in {parquet_path.name}: {bank}"
            assert split in {"train", "val", "test"}, f"Unexpected split in {parquet_path.name}: {split}"
            cases = _read_case_rows(parquet_path, columns)

            # Standardize column names for downstream diagnostics.
            cases = cases.rename(columns={
                "case:concept:name": "case_id", "case:RequestedAmount": "requested_amount", "case:LoanGoal": "loan_goal",
            })

            # Attach partition metadata to every case row
            cases.insert(0, "heterogeneity", het)
            cases.insert(1, "n_clients", n)
            cases.insert(2, "config", config)
            cases.insert(3, "bank", bank)
            cases.insert(4, "split", split)
            rows.append(cases)

        # Log the number of case rows loaded for this config.
        n_cases = sum(len(frame) for frame in rows[before:])
        log.info(f"Loaded case-level split data for {config}: {n_cases:,} cases")
    if not rows: raise FileNotFoundError("No BPIC 2017 per-bank parquet files found for split analytics")
    return pd.concat(rows, ignore_index=True)

# Add RequestedAmount bands specific to every config, keep zero as an own group.
def add_requested_amount_bands(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["amount_band"] = "Zero"
    for cfg, idx in out.groupby("config").groups.items():
        amounts = out.loc[idx, "requested_amount"].astype(float)
        nonzero_idx = amounts[amounts > 0].index
        if len(nonzero_idx) == 0: continue

        # Split non-zero amounts into quintiles with pd.qcut().
        q = pd.qcut(out.loc[nonzero_idx, "requested_amount"].astype(float), q=5, labels=False, duplicates="drop")
        out.loc[nonzero_idx, "amount_band"] = q.map(lambda value: AMOUNT_BAND_ORDER[int(value) + 1])
    return out

# Map raw LoanGoal values to partition groups as set in the CONFIG section.
def _loangoal_group(goal: object) -> str:
    if pd.isna(goal): return "Other"
    goal_str = str(goal)
    if goal_str in LOANGOAL_HIGH_APPROVAL: return "High-approval"
    if goal_str in LOANGOAL_LOW_APPROVAL: return "Low-approval"
    if goal_str == LOANGOAL_SPECIALIST_D: return "Specialist D"
    if goal_str == LOANGOAL_SPECIALIST_E: return "Specialist E"
    if goal_str in {"Car", "Debt restructuring"}: return "Core mixed"
    return "Other"

# Add LoanGoal groups used by the partition design.
def add_loangoal_groups(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["loangoal_group"] = out["loan_goal"].map(_loangoal_group)
    return out

# ----------------------------------------------------------------------------------------------------------------------
# 3. PLOTTING UTILITIES

# HELPER: Save a figure as PNG and close it.
def _save(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)
    log.info(f"Saved plot: {stem.with_suffix('.png').name}")

# HELPER: Save a dense non-pie figure with compact whitespace.
def _save_tight(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    log.info(f"Saved plot: {stem.with_suffix('.png').name}")

# HELPER: Compact x-tick label for one config.
def _config_short(cfg: str) -> str:
    return SHORT_LABELS.get(cfg, cfg)

# HELPER: Keep only available configs while preserving the caller's plot order.
def _available_config_order(values: pd.Series | pd.Index, order: list[str]) -> list[str]:
    available = set(values)
    return [cfg for cfg in order if cfg in available]

# HELPER: Apply the common axis style, consistent across graphs (not for pie charts).
def _style_non_pie_axes(ax: plt.Axes, x_margin: float = 0.04) -> None:
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

# HELPER: Style reference-line legends.
def _style_reference_legend(legend: plt.Legend) -> None:
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor(LEGEND_EDGE)
    frame.set_alpha(0.78)
    frame.set_linewidth(0.8)

# HELPER: Finish a figure with shared padding (not for pie charts).
def _finish_non_pie_figure(fig: plt.Figure, bottom: Optional[float] = None) -> None:
    fig.tight_layout(pad=1.2)
    if bottom is not None: fig.subplots_adjust(bottom=bottom)

# HELPER: Return config-bank pairs in the display order defined in CONFIG.
def _ordered_config_bank_index(cases: pd.DataFrame) -> pd.MultiIndex:
    pairs = []
    for cfg in CONFIG_DISPLAY_ORDER:
        cfg_banks = cases.loc[cases["config"] == cfg, "bank"].drop_duplicates().tolist()
        for bank in BANK_ORDER:
            if bank in cfg_banks: pairs.append((cfg, bank))
    return pd.MultiIndex.from_tuples(pairs, names=["config", "bank"])

# HELPER: Convert case counts into category shares within banks.
def _category_share_pivot(cases: pd.DataFrame, category: str, order: list[str]) -> pd.DataFrame:
    counts = (
        cases.groupby(["config", "bank", category], observed=False)["case_id"]
        .nunique()
        .unstack(category, fill_value=0)
    )

    # Keep display order and include missing categories as zero.
    counts = counts.reindex(index=_ordered_config_bank_index(cases), fill_value=0)
    counts = counts.reindex(columns=order, fill_value=0)

    # Normalize each row to percentages.
    totals = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(totals, axis=0).fillna(0.0) * 100

# HELPER: Return x positions with gaps between configurations.
def _spaced_group_positions(index: pd.MultiIndex, gap: float = 0.90) -> np.ndarray:
    configs = index.get_level_values("config").tolist()
    positions = []
    offset = 0.0
    previous = None
    for i, cfg in enumerate(configs):
        if previous is not None and cfg != previous: offset += gap
        positions.append(i + offset)
        previous = cfg
    return np.asarray(positions, dtype=float)

# HELPER: Draw separators between configuration groups (for the stacked bar charts).
def _draw_config_separators(ax: plt.Axes, index: pd.MultiIndex, x: np.ndarray) -> None:
    configs = index.get_level_values("config").tolist()
    for pos in range(1, len(configs)):
        if configs[pos] != configs[pos - 1]:
            boundary = float((x[pos - 1] + x[pos]) / 2)
            ax.axvline(boundary, color=SPINE_COLOR, linewidth=1.2, alpha=0.50)

# HELPER: Place one bold configuration label below each bank group.
def _label_config_groups(ax: plt.Axes, index: pd.MultiIndex, x: np.ndarray, y: float = -0.26,
                         fontsize: float = 9) -> None:

    configs = index.get_level_values("config").tolist()
    start = 0
    for pos in range(1, len(configs) + 1):
        if pos == len(configs) or configs[pos] != configs[start]:
            mid = float((x[start] + x[pos - 1]) / 2)
            ax.text(mid, y, _config_short(configs[start]), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=fontsize, fontweight="bold", color=TEXT_COLOR)
            start = pos

# HELPER: Return RequestedAmount entries for the compact legend.
def _amount_legend_entries(cases: pd.DataFrame) -> list[tuple[str, str]]:
    entries = []
    for band in AMOUNT_BAND_ORDER:
        values = cases.loc[cases["amount_band"] == band, "requested_amount"].astype(float)
        if band == "Zero": entries.append((band, "€0"))
        elif values.empty: entries.append((band, "n/a"))
        else: entries.append((band, f"€{values.min():,.0f} - €{values.max():,.0f}"))
    return entries

# Return LoanGoal entries for the compact legend.
def _loangoal_legend_entries() -> list[tuple[str, str]]:
    entries = []
    for group in LOANGOAL_GROUP_ORDER:
        title, detail = LOANGOAL_GROUP_LABELS[group].split("\n", 1)
        entries.append((title, detail))
    return entries

# Draw 100% stacked config-bank bars.
def _plot_share_stack(ax: plt.Axes, pivot: pd.DataFrame, colors: list[str], y_label: str,
    config_label_y: float = -0.26, show_config_labels: bool = True,) -> None:

    x = _spaced_group_positions(pivot.index)
    bottom = np.zeros(len(pivot.index))

    # Stack one category layer at a time.
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values
        ax.bar(x, vals, bottom=bottom, color=colors[i], edgecolor="white", linewidth=0.5)
        bottom += vals

    # Show bank letters as ticks and config names as group labels.
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.get_level_values("bank"))
    ax.tick_params(axis="both", labelsize=10.5)
    ax.set_ylabel(y_label, fontsize=12.5)
    ax.set_ylim(0, 108)
    _draw_config_separators(ax, pivot.index, x)
    if show_config_labels: _label_config_groups(ax, pivot.index, x, y=config_label_y, fontsize=10.5)

    # Apply shared axis styling after all bars and group labels exist.
    _style_non_pie_axes(ax, x_margin=0.01)

# Draw compact swatch legends with separate label and detail columns.
def _draw_compact_swatch_legend(legend_ax: plt.Axes, entries: list[tuple[str, str]], colors: list[str],
    row_y: list[float], col_specs: list[tuple[float, float, float]], swatch_width: float, fontsize: float = 10.8,
    ) -> None:

    legend_ax.axis("off")

    # Fill rows first within each two-column legend.
    for i, (title, detail) in enumerate(entries):
        col = i % 2
        row = i // 2
        swatch_x, title_x, detail_x = col_specs[col]
        y = row_y[row]
        legend_ax.add_patch(
            mpatches.Rectangle(
                (swatch_x, y - 0.034), swatch_width, 0.068, transform=legend_ax.transAxes,
                facecolor=colors[i], edgecolor="white", linewidth=0.5,
            )
        )
        legend_ax.text(
            title_x, y, f"{title}:", transform=legend_ax.transAxes, ha="left", va="center", fontsize=fontsize,
            fontweight="bold", color=TEXT_COLOR,
        )
        legend_ax.text(
            detail_x, y, detail, transform=legend_ax.transAxes, ha="left", va="center",
            fontsize=fontsize, color=TEXT_COLOR,
        )

# Draw LoanGoal legend in two compact columns.
def _draw_loangoal_legend(legend_ax: plt.Axes) -> None:

    # Use wider columns because LoanGoal details are long text lists.
    _draw_compact_swatch_legend(
        legend_ax, _loangoal_legend_entries(), LOANGOAL_GROUP_COLORS, row_y=[0.55, 0.40, 0.25],
        col_specs=[(0.025, 0.080, 0.170), (0.660, 0.715, 0.800)], swatch_width=0.034,
    )

# Draw the RequestedAmount legend in two compact columns.
def _draw_amount_legend(legend_ax: plt.Axes, cases: pd.DataFrame) -> None:
    # Center the shorter amount ranges under the stacked bars.
    _draw_compact_swatch_legend(
        legend_ax, _amount_legend_entries(cases), AMOUNT_BAND_COLORS, row_y=[0.55, 0.40, 0.25],
        col_specs=[(0.145, 0.235, 0.315), (0.535, 0.625, 0.705)], swatch_width=0.032,
    )

# Draw a stacked category-share figure with the shared compact legend row.
def _plot_category_share_distribution(cases: pd.DataFrame, out_stem: Path, category: str, category_order: list[str],
    colors: list[str], figure_size: tuple[float, float], legend_drawer: Callable[[plt.Axes, pd.DataFrame], None],
    ) -> None:

    pivot = _category_share_pivot(cases, category, category_order)
    fig, (ax, legend_ax) = plt.subplots(2, 1, figsize=figure_size, constrained_layout=False,
                                        gridspec_kw={"height_ratios": [4.0, 1.10], "hspace": 0.10})
    _plot_share_stack(ax, pivot, colors, "Case share (%)", config_label_y=-0.065)
    ax.set_xlabel("")
    legend_drawer(legend_ax, cases)

    # Keep the plot-to-legend gap above the compact legend.
    fig.subplots_adjust(left=0.055, right=0.99, top=0.98, bottom=0.010, hspace=2.00)
    _save_tight(fig, out_stem)

# ----------------------------------------------------------------------------------------------------------------------
# 4. PLOTS

# PLOT: Draw partition pies with inline labels and approval captions.
def plot_partition_pies(configs: list[tuple[str, int, Path]], out_stem: Path) -> None:
    n = len(configs)
    cols = 2
    rows = (n + cols - 1) // cols

    # Size pies for readable captions across all 03 partition layouts.
    fig, axes_arr = plt.subplots(rows, cols, figsize=(17.0, 26.5), constrained_layout=False)
    fig.subplots_adjust(left=0.036, right=0.964, top=0.94, bottom=0.115, wspace=0.02, hspace=0.20)
    flat = np.atleast_1d(axes_arr).ravel().tolist()

    # Place 3-bank configs before 5-bank configs.
    cfg_3 = [c for c in configs if c[1] == 3]
    cfg_5 = [c for c in configs if c[1] == 5]
    ordered_configs = cfg_3 + cfg_5

    # Draw 3-bank pies first, then 5-bank pies.
    for i, (het, nc, path) in enumerate(ordered_configs): _draw_pie(flat[i], path, het, nc)
    for j in range(n, len(flat)): flat[j].set_visible(False)

    # Center the shared bank legend below the pie captions.
    max_banks = max((len(pd.read_csv(path / "A_02_partition_stats.csv")) for _, _, path in configs), default=3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BANK_COLORS[i], edgecolor="white")
        for i in range(max_banks)
    ]
    labels = [f"Bank {chr(ord('A') + i)}" for i in range(max_banks)]
    fig.legend(
        handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.018), fontsize=22, ncol=max_banks, frameon=False,
        handlelength=2.0, handleheight=1.6, columnspacing=1.6, handletextpad=0.8
    )
    _save(fig, out_stem)

# HELPER: Draw a pie with inline slice labels and a separate approval-rate caption.
def _draw_pie(ax: plt.Axes, path: Path, het: str, nc: int) -> None:
    stats = pd.read_csv(path / "A_02_partition_stats.csv")
    n = len(stats)
    sizes = stats["n_cases"].to_numpy()
    rates = stats["approval_rate"].to_numpy()
    banks = stats["bank"].to_numpy()
    total = sizes.sum()

    # Draw large pies with fixed bounds so all partition panels keep the same visual scale.
    wedges, _ = ax.pie(
        sizes, colors=BANK_COLORS[:n], wedgeprops={"edgecolor": "white", "linewidth": 1.5}, startangle=90,
        counterclock=False, radius=1.65,
    )
    ax.set_xlim(-1.95, 1.95)
    ax.set_ylim(-1.95, 1.95)

    # Annotate readable slices inline and route small slices through leader lines.
    for i, (wedge, size, bank) in enumerate(zip(wedges, sizes, banks)):
        share_pct = size / total * 100
        ang = np.deg2rad((wedge.theta1 + wedge.theta2) / 2)
        r = 0.90 if share_pct >= 15 else 1.00
        x, y = r * np.cos(ang), r * np.sin(ang)
        text_color = "white" if i <= 1 else "#222222"

        # Adjust 5-bank pie labels to keep dense slices readable.
        if share_pct >= 10:
            # Nudge Bank B in 5-bank pies to keep the label inside its wedge.
            dx, dy = (0.18, -0.12) if (n == 5 and i == 1) else (0.0, 0.0)
            # Place the bank letter above the count and share.
            ax.text(x + dx, y + dy + 0.22, bank, ha="center", va="center",  fontsize=20, color=text_color, fontweight="bold")
            ax.text(x + dx, y + dy - 0.11, f"{int(size):,}\n({share_pct:.1f}%)",
                    ha="center", va="center", fontsize=20, color=text_color, fontweight="normal", linespacing=1.4)
        else:
            # Place small-slice labels outside the pie with leader lines.
            x_out, y_out = 1.78 * np.cos(ang), 1.78 * np.sin(ang)
            ax.annotate(
                f"{bank}: {int(size):,} ({share_pct:.1f}%)",
                xy=(1.45 * np.cos(ang), 1.45 * np.sin(ang)),
                xytext=(x_out, y_out),
                ha="left" if x_out >= 0 else "right",
                va="center",
                fontsize=15,
                color="#222222",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"),
            )

    # Split 5-bank approval captions across two rate lines.
    appr_segments = []
    for bank, rate in zip(banks, rates):
        rate_str = f"{float(rate) * 100:.1f}%" if pd.notna(rate) else "n/a"
        appr_segments.append(f"{bank} {rate_str}")
    if len(appr_segments) <= 3:
        appr_text = "Approval rate\n" + "   ".join(appr_segments)
    else:
        half = (len(appr_segments) + 1) // 2
        line_a = "   ".join(appr_segments[:half])
        line_b = "   ".join(appr_segments[half:])
        appr_text = "Approval rate\n" + line_a + "\n" + line_b
    ax.text(0.5, 0.054, appr_text, transform=ax.transAxes, ha="center", va="top",
            fontsize=22, color="#222222", linespacing=1.35)
    ax.set_title(f"{het} ({nc} banks)", fontsize=26, fontweight="bold", pad=5)
    # Anchor the pie at the top of the panel.
    ax.set_aspect("equal", adjustable="box", anchor="N")

# PLOT: Draw approval rates by bank with the dataset-wide approval rate as reference.
def plot_approval_rate_by_bank(table: pd.DataFrame, out_stem: Path) -> None:
    pivot = (
        table.pivot_table(index="config", columns="bank", values="approval_rate")
        .reindex(index=_available_config_order(table["config"], CONFIG_DISPLAY_ORDER))
    )
    bank_cols = [b for b in ["A", "B", "C", "D", "E"] if b in pivot.columns]
    pivot = pivot[bank_cols]

    # Use iid as the stable approval-rate reference.
    ref = table.copy()
    ref_cfg = "iid_3banks" if "iid_3banks" in ref["config"].values else ref["config"].iloc[0]
    ref_subset = ref[ref["config"] == ref_cfg]
    # Include canceled cases in the denominator because the outcome is a three-class target.
    overall_rate = ref_subset["n_approved"].sum() / ref_subset["n_cases"].sum()

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    n_banks = len(bank_cols)
    x = np.arange(len(pivot.index))
    width = 0.8 / n_banks
    bank_patches = []
    for i, bank in enumerate(bank_cols):
        offset = (i - (n_banks - 1) / 2) * width
        bars = ax.bar(x + offset, pivot[bank].values, width, color=BANK_COLORS[i], edgecolor="white", linewidth=0.5)
        # Label bars and lift labels near the overall mean
        for bar, val in zip(bars, pivot[bank].values):
            if pd.notna(val):
                _y = bar.get_height() + 0.010
                if overall_rate - 0.04 < bar.get_height() < overall_rate + 0.025: _y = overall_rate + 0.04
                ax.text(bar.get_x() + bar.get_width() / 2, _y,
                        f"{val * 100:.0f}%", ha="center", va="bottom", fontsize=7, color="#222222")
        bank_patches.append(mpatches.Patch(color=BANK_COLORS[i], label=f"Bank {bank}"))

    mean_line = ax.axhline(overall_rate, linestyle="--", linewidth=1.2, color=WARN_COLOR,
                           label=f"overall mean = {overall_rate:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in pivot.index])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Approval rate")

    # Leave headroom for bar labels and the overall mean legend.
    ax.set_ylim(0, 1.12)
    _style_non_pie_axes(ax)

    # Place the mean legend in the upper-right corner.
    mean_legend = ax.legend(handles=[mean_line], loc="upper right", frameon=True, fontsize=10)
    _style_reference_legend(mean_legend)
    ax.add_artist(mean_legend)

    # Place the bank legend below the axis.
    ax.legend(handles=bank_patches, frameon=False, ncol=n_banks, loc="lower center", bbox_to_anchor=(0.5, -0.26))
    _finish_non_pie_figure(fig, bottom=0.23)
    _save(fig, out_stem)

# PLOT: Draw 100% stacked case share bars per config.
def plot_client_size_distribution(table: pd.DataFrame, out_stem: Path) -> None:
    pivot = table.pivot_table(index="config", columns="bank", values="case_share")

    # Keep the display order defined in CONFIG and convert shares to percentages.
    pivot = pivot.reindex(index=_available_config_order(pivot.index, CONFIG_DISPLAY_ORDER))
    bank_cols = [b for b in ["A", "B", "C", "D", "E"] if b in pivot.columns]
    pivot = pivot[bank_cols].fillna(0.0) * 100
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))

    # Stack banks within each configuration to show the size imbalance.
    for i, bank in enumerate(bank_cols):
        vals = pivot[bank].values
        ax.bar(x, vals, bottom=bottom, color=BANK_COLORS[i], edgecolor="white", linewidth=0.6, label=f"Bank {bank}")

        # Label only readable segments to avoid clutter in small slices.
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v >= 6.0:
                ax.text(
                    xi, b0 + v / 2, f"{v:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="white" if i <= 1 else "#222222",
                )
        bottom += vals

    # Match the other config bar charts.
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in pivot.index])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Case share (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=len(bank_cols), loc="lower center", bbox_to_anchor=(0.5, -0.26))
    _style_non_pie_axes(ax)
    _finish_non_pie_figure(fig, bottom=0.23)
    _save(fig, out_stem)

# PLOT: Draw approval-rate spread across increasing heterogeneity.
def plot_approval_rate_spread(table: pd.DataFrame, out_stem: Path) -> None:

    # Keep one spread value per config, preserve display order.
    spread = (
        table.groupby("config", sort=False)["approval_rate_spread"].first()
        .reindex(_available_config_order(table["config"], list(SHORT_LABELS)))
    )
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=False)
    x = np.arange(len(spread))

    # Plot the max-min approval rate gap as the heterogeneity signal.
    ax.plot(x, spread.values, color=LINE_COLOR, linewidth=1.8, marker="o",
            markersize=7, markerfacecolor=LINE_COLOR, markeredgecolor="white", markeredgewidth=1.0)

    # Offset labels so they stay readable above the markers.
    spread_values = [float(value) for value in spread.to_numpy()]
    label_offset = max(max(spread_values) * 0.04, 0.003)
    for xi in range(len(spread_values)):
        value = spread_values[xi]
        ax.text(x=float(xi), y=value + label_offset, s=f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    # Use compact config labels and leave headroom for marker labels.
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in spread.index])
    ax.set_xlabel("Configuration (increasing heterogeneity)")
    ax.set_ylabel("Approval-rate spread (max - min)")
    ax.set_ylim(0, max(0.05, spread.max() * 1.38))
    _style_non_pie_axes(ax)
    _finish_non_pie_figure(fig)
    _save(fig, out_stem)

# PLOT: Draw cases dropped by medium and strong undersampling.
def plot_cases_dropped(table: pd.DataFrame, out_stem: Path) -> None:

    # Read one drop count per config and set missing enforcement values at zero.
    order = _available_config_order(table["config"], list(SHORT_LABELS))
    medium = table.groupby("config", sort=False)["medium_dropped_total"].first()
    medium = pd.to_numeric(medium, errors="coerce").fillna(0).astype(int).reindex(order, fill_value=0)
    strong = table.groupby("config", sort=False)["strong_dropped_total"].first()
    strong = pd.to_numeric(strong, errors="coerce").fillna(0).astype(int).reindex(order, fill_value=0)
    total = medium + strong
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=False)
    x = np.arange(len(order))

    # Stack medium and strong drops to show the total undersampling loss.
    ax.bar(x, medium.values, color="#5A9DDC", edgecolor="white", linewidth=0.6, label="Medium enforcement")
    ax.bar(x, strong.values, bottom=medium.values, color=LINE_COLOR, edgecolor="white", linewidth=0.6,
           label="Strong enforcement")

    # Mark zero-drop configs with a small neutral bar so they stay visible.
    for xi, t in zip(x, total.values):
        if t == 0:
            ax.bar([xi], [max(int(total.max() * 0.012), 1)], color=NEUTRAL_COLOR, edgecolor="white", linewidth=0.6)

    # Place total drop counts above each stacked bar.
    for xi, t in zip(x, total.values):
        ax.text(xi, t + max(total.max() * 0.02, 5),  f"{int(t):,}", ha="center", va="bottom", fontsize=9)

    # Leave headroom for the count labels and keep the legend only when both layers exist.
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in order])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Cases dropped (undersampling)")
    ymax = max(int(total.max() * 1.30), 50)
    ax.set_ylim(0, ymax)
    _style_non_pie_axes(ax)
    if (medium > 0).any() and (strong > 0).any():
        legend = ax.legend(loc="upper left", frameon=True, fontsize=9, borderaxespad=1.5)
        _style_reference_legend(legend)
    _finish_non_pie_figure(fig)
    _save(fig, out_stem)

# HELPER: Draw RequestedAmount band shares by config and bank.
def plot_requested_amount_quintiles(cases: pd.DataFrame, out_stem: Path) -> None:
    _plot_category_share_distribution(cases, out_stem, "amount_band", AMOUNT_BAND_ORDER,
                                      AMOUNT_BAND_COLORS, (15, 7.4), _draw_amount_legend)

# HELPER: Draw LoanGoal group shares by config and bank.
def plot_loangoal_group_distribution(cases: pd.DataFrame, out_stem: Path) -> None:
    _plot_category_share_distribution(cases, out_stem, "loangoal_group", LOANGOAL_GROUP_ORDER,
                                      LOANGOAL_GROUP_COLORS, (17, 8.0),
                                      legend_drawer=lambda legend_ax, _summary: _draw_loangoal_legend(legend_ax))

# ----------------------------------------------------------------------------------------------------------------------
# MAIN FLOW

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = collect_configs()
    if not configs: raise FileNotFoundError(f"No partitioned configs found under {PROCESSED_DIR}")
    log.info(f"Found {len(configs):,} partition configuration(s): " + ", ".join(f"{h}_{n}banks" for h, n, _ in configs))

    # Build the comparison table across all available partition configs.
    table = build_comparison_table(configs)
    out_csv = PROCESSED_DIR / "A_03_comparison_table.csv"
    table.to_csv(out_csv, index=False)
    log.info(f"Saved table: {out_csv} ({len(table):,} rows, {len(table.columns):,} columns)")

    # Build case-split data for RequestedAmount and LoanGoal analysis.
    cases = build_case_level_table(configs)
    cases = add_requested_amount_bands(cases)
    cases = add_loangoal_groups(cases)

    # Render partition figures.
    plot_partition_pies(configs, PLOTS_DIR / "A_03_partition_pie_charts")
    plot_approval_rate_by_bank(table, PLOTS_DIR / "A_03_approval_rate_by_bank")
    plot_client_size_distribution(table, PLOTS_DIR / "A_03_client_size_distribution")
    plot_approval_rate_spread(table, PLOTS_DIR / "A_03_approval_rate_spread")
    plot_cases_dropped(table, PLOTS_DIR / "A_03_cases_dropped")
    plot_requested_amount_quintiles(cases, PLOTS_DIR / "A_03_requested_amount_quintiles")
    plot_loangoal_group_distribution(cases, PLOTS_DIR / "A_03_loangoal_group_distribution")

    log.info("Per-configuration summary:")
    for cfg, grp in table.groupby("config", sort=False):
        spread = grp["approval_rate_spread"].iloc[0]
        log.info(f"  {cfg}: {len(grp)} banks, total {int(grp['n_cases'].sum()):,} cases, "
                 f"approval-rate spread {spread:.3f}")

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb  |  Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────