"""
Step 3: Dataset analytics for partitioned splits (BPIC 2012 - ablation).

- Read B_02_partition_stats.csv and B_02_metadata.json from each data/processed/{heterogeneity}_{n_clients}banks/ folder.
- Build a unified comparison table for the thesis.
- Renders figures (PNG only, 300 DPI) parallel to the BPIC 2017 A_03_dataset_summary.py outputs.

Run: python B_03_dataset_summary.py

Outputs:
  - Comparison, one row per (config, bank): data/processed/B_03_comparison_table.csv
  - Case shares per config: plots/B_03_partition_pie_charts.png
  - Approval rate grouped bars, all configs: plots/B_03_approval_rate_by_bank.png
  - Client size distributions as stacked bars: plots/B_03_client_size_distribution.png
  - Line over heterogeneity: plots/B_03_approval_rate_spread.png
  - Cases dropped (none): plots/B_03_cases_dropped.png
  - RequestedAmount quintile shares: plots/B_03_requested_amount_quintiles.png
"""

# IMPORTS
from __future__ import annotations
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Optional
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Render figures directly to files instead of opening them.
matplotlib.use("Agg")

# CONFIGURATION
SCRIPT_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = SCRIPT_DIR / "data/processed"
PLOTS_DIR = SCRIPT_DIR / "plots"

# Unified color scheme, to keep BPIC 2012 figures visually aligned with BPIC 2017.
BANK_COLORS = ["#003E7A", "#0065BF", "#5A9DDC", "#99C2E5", "#D6E4F2"]
LINE_COLOR = "#0065BF"
WARN_COLOR = "#C8102E"
NEUTRAL_COLOR = "#cccccc"
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#666666"
LEGEND_EDGE = "#bfbfbf"
TEXT_COLOR = "#222222"

# Order BPIC 2012 configs by increasing heterogeneity.
CONFIG_ORDER = ["iid", "weak", "medium"]
CONFIG_DISPLAY_ORDER = ["iid_3banks", "weak_3banks", "medium_3banks"]
BANK_ORDER = ["A", "B", "C", "D", "E"]
SHORT_LABELS = {"iid_3banks": "iid-3", "weak_3banks": "weak-3", "medium_3banks": "medium-3"}

# Order RequestedAmount bands from zero to high values.
AMOUNT_BAND_ORDER = ["Zero", "Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
AMOUNT_BAND_COLORS = ["#cccccc", "#D6E4F2", "#99C2E5", "#5A9DDC", "#0065BF", "#003E7A"]

# Use B_02-prefixed inputs for the BPIC 2012 ablation.
PARTITION_STATS_FILE = "B_02_partition_stats.csv"
METADATA_FILE = "B_02_metadata.json"

# Apply shared matplotlib defaults once, keep explicit pie margins by avoiding tight save bounding boxes.
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
    for d in sorted(PROCESSED_DIR.iterdir()):
        if not d.is_dir() or d.name == "centralized": continue
        parts = d.name.split("_")
        if len(parts) != 2 or not parts[1].endswith("banks"): continue
        het, suffix = parts[0], parts[1]
        try:
            n = int(suffix.replace("banks", ""))
        except ValueError:
            continue
        configs.append((het, n, d))

    # Rank configs by heterogeneity level and client count.
    rank = {h: i for i, h in enumerate(CONFIG_ORDER)}
    configs.sort(key=lambda x: (rank.get(x[0], 99), x[1]))
    return configs


# ----------------------------------------------------------------------------------------------------------------------
# 2. COMPARISON TABLE

# Read metadata for one partition config.
def load_partition_metadata(path: Path) -> dict[str, Any]:
    with (path / METADATA_FILE).open() as f: return json.load(f)


# Extract partition provenance fields for the comparison table.
def provenance_columns(metadata: dict[str, Any]) -> dict[str, object]:
    provenance = metadata.get("partition_provenance", {})
    return {
        "n_cases_available_after_filters": provenance.get("n_cases_available_after_filters"),
        "n_cases_assigned": provenance.get("n_cases_assigned"),
        "n_cases_unassigned_or_dropped": provenance.get("n_cases_unassigned_or_dropped"),
    }


# Build one comparison table across partition configs.
def build_comparison_table(configs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    rows = []
    for het, n, path in configs:
        stats = pd.read_csv(path / PARTITION_STATS_FILE)
        provenance = provenance_columns(load_partition_metadata(path))

        # Add config identifiers as leading columns and populate provenance fields.
        stats.insert(0, "heterogeneity", het)
        stats.insert(1, "n_clients", n)
        stats.insert(2, "config", f"{het}_{n}banks")
        for col, value in provenance.items(): stats[col] = value

        # Add within-config metrics for quick comparisons.
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
# 2.5 CASE-LEVEL SPLIT TABLE

# Stream one parquet batch and keep one row per case for case attributes.
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


# Build a table with one row per case from train, validation and test files.
def build_case_level_table(configs: list[tuple[str, int, Path]]) -> pd.DataFrame:
    rows = []
    columns = ["case:concept:name", "case:AMOUNT_REQ", "outcome"]
    for het, n, path in configs:
        config = f"{het}_{n}banks"
        before = len(rows)

        # Read each bank split and keep only case attributes.
        for parquet_path in sorted(path.glob("B_02_bank_*_*.parquet")):
            parts = parquet_path.stem.split("_")
            if len(parts) < 5: continue
            bank = parts[3]
            split = parts[4]
            cases = _read_case_rows(parquet_path, columns)

            # Standardize column names for downstream analysis.
            cases = cases.rename(columns={"case:concept:name": "case_id", "case:AMOUNT_REQ": "requested_amount"})

            # Attach partition metadata to every case row.
            cases.insert(0, "heterogeneity", het)
            cases.insert(1, "n_clients", n)
            cases.insert(2, "config", config)
            cases.insert(3, "bank", bank)
            cases.insert(4, "split", split)
            rows.append(cases)

        # Log the number of case rows loaded for this config.
        n_cases = sum(len(frame) for frame in rows[before:])
        log.info(f"Loaded case-level split data for {config}: {n_cases:,} cases")
    if not rows: raise FileNotFoundError("No BPIC 2012 per-bank parquet files found for split analytics")
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
        q = pd.qcut(
            out.loc[nonzero_idx, "requested_amount"].astype(float),
            q=5,
            labels=False,
            duplicates="drop",
        )
        out.loc[nonzero_idx, "amount_band"] = q.map(lambda value: AMOUNT_BAND_ORDER[int(value) + 1])
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


# HELPER: Keep only available configs while preserving the plot order.
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


# HELPER: Style reference-line legends
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


# HELPER: Convert case counts into category shares between banks.
def _category_share_pivot(cases: pd.DataFrame, category: str, order: list[str]) -> pd.DataFrame:
    counts = (
        cases.groupby(["config", "bank", category], observed=False)["case_id"]
        .nunique()
        .unstack(category, fill_value=0)
    )
    counts = counts.reindex(index=_ordered_config_bank_index(cases), fill_value=0)
    counts = counts.reindex(columns=order, fill_value=0)
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


# HELPER: Draw separators between configuration groups.
def _draw_config_separators(ax: plt.Axes, index: pd.MultiIndex, x: np.ndarray) -> None:
    configs = index.get_level_values("config").tolist()
    for pos in range(1, len(configs)):
        if configs[pos] != configs[pos - 1]:
            boundary = float((x[pos - 1] + x[pos]) / 2)
            ax.axvline(boundary, color=SPINE_COLOR, linewidth=1.2, alpha=0.50)


# HELPER: Place one bold configuration label below each bank group.
def _label_config_groups(ax: plt.Axes, index: pd.MultiIndex, x: np.ndarray, y: float = -0.26) -> None:
    configs = index.get_level_values("config").tolist()
    start = 0
    for pos in range(1, len(configs) + 1):
        if pos == len(configs) or configs[pos] != configs[start]:
            mid = float((x[start] + x[pos - 1]) / 2)
            ax.text(mid, y, _config_short(configs[start]), transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9, fontweight="bold", color=TEXT_COLOR)
            start = pos


# HELPER: Return RequestedAmount entries for the compact legend.
def _amount_legend_entries(cases: pd.DataFrame) -> list[tuple[str, str]]:
    entries = []
    for band in AMOUNT_BAND_ORDER:
        values = cases.loc[cases["amount_band"] == band, "requested_amount"].astype(float)
        if band == "Zero":
            entries.append((band, "€0"))
        elif values.empty:
            entries.append((band, "n/a"))
        else:
            entries.append((band, f"€{values.min():,.0f} - €{values.max():,.0f}"))
    return entries


# HELPER: Draw the RequestedAmount legend in two compact columns.
def _draw_amount_legend(legend_ax: plt.Axes, cases: pd.DataFrame) -> None:
    legend_ax.axis("off")
    entries = _amount_legend_entries(cases)

    # Keep top and bottom legend margins symmetric.
    row_y = [0.55, 0.40, 0.25]
    col_specs = [(0.125, 0.225, 0.335), (0.545, 0.645, 0.755)]
    for i, (band, detail) in enumerate(entries):
        col = i % 2
        row = i // 2
        swatch_x, title_x, detail_x = col_specs[col]
        y = row_y[row]
        legend_ax.add_patch(
            mpatches.Rectangle(
                (swatch_x, y - 0.032),
                0.030,
                0.064,
                transform=legend_ax.transAxes,
                facecolor=AMOUNT_BAND_COLORS[i],
                edgecolor="white",
                linewidth=0.5,
            )
        )
        legend_ax.text(
            title_x,
            y,
            f"{band}:",
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.6,
            fontweight="bold",
            color=TEXT_COLOR,
        )
        legend_ax.text(
            detail_x,
            y,
            detail,
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=9.6,
            color=TEXT_COLOR,
        )


# HELPER: Draw 100% stacked bank configuration bars.
def _plot_share_stack(ax: plt.Axes, pivot: pd.DataFrame, colors: list[str], y_label: str, config_label_y: float = -0.26,
                      ) -> None:
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
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 108)
    _draw_config_separators(ax, pivot.index, x)
    _label_config_groups(ax, pivot.index, x, y=config_label_y)

    # Apply shared axis styling after all bars and group labels exist.
    _style_non_pie_axes(ax, x_margin=0.01)


# ----------------------------------------------------------------------------------------------------------------------
# 4. PLOTS

# PLOT: Draw partition pies with inline labels and approval captions.
def plot_partition_pies(configs: list[tuple[str, int, Path]], out_stem: Path) -> None:
    n = len(configs)
    cols = 3
    rows = (n + cols - 1) // cols

    # Size pies for readable captions across the BPIC 2012 partition layouts.
    fig, axes_arr = plt.subplots(rows, cols, figsize=(cols * 6.5 + 2.0, rows * 8.5 + 1.13), constrained_layout=False)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.917, bottom=0.218, wspace=0.18, hspace=0.561)
    flat = np.atleast_1d(axes_arr).ravel().tolist()

    for i, (het, nc, path) in enumerate(configs): _draw_pie(flat[i], path, het, nc)
    for j in range(n, len(flat)): flat[j].set_visible(False)

    # Center the shared bank legend below the pie captions.
    max_banks = max((len(pd.read_csv(path / PARTITION_STATS_FILE)) for _, _, path in configs), default=3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=BANK_COLORS[i], edgecolor="white")
        for i in range(max_banks)
    ]
    labels = [f"Bank {chr(ord('A') + i)}" for i in range(max_banks)]
    fig.legend(handles, labels, loc="lower center", ncol=max_banks, frameon=False, bbox_to_anchor=(0.5, 0.075),
               fontsize=22, handlelength=2.2, handleheight=2.0, columnspacing=3.0, handletextpad=1.0)
    _save(fig, out_stem)


# HELPER: Draw a pie with inline slice labels and a separate approval rate caption.
def _draw_pie(ax: plt.Axes, path: Path, het: str, nc: int) -> None:
    stats = pd.read_csv(path / PARTITION_STATS_FILE)
    n = len(stats)
    sizes = stats["n_cases"].to_numpy()
    rates = stats["approval_rate"].to_numpy()
    banks = stats["bank"].to_numpy()
    total = sizes.sum()

    # Draw pies with a fixed scale so all panels remain visually comparable.
    wedges, _ = ax.pie(
        sizes,
        colors=BANK_COLORS[:n],
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        startangle=90,
        counterclock=False,
        radius=1.50,
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
        if share_pct >= 10:
            # Place the bank letter above the count and share.
            ax.text(x, y + 0.22, bank, ha="center", va="center", fontsize=18, color=text_color, fontweight="bold")
            ax.text(x, y - 0.10, f"{int(size):,}\n({share_pct:.1f}%)", ha="center", va="center",
                    fontsize=18, color=text_color, fontweight="normal", linespacing=1.4)
        else:
            # Place small-slice labels outside the pie with leader lines.
            x_out, y_out = 1.78 * np.cos(ang), 1.78 * np.sin(ang)
            ax.annotate(f"{bank}: {int(size):,} ({share_pct:.1f}%)", xy=(1.45 * np.cos(ang), 1.45 * np.sin(ang)),
                        xytext=(x_out, y_out), ha="left" if x_out >= 0 else "right", va="center", fontsize=14,
                        color="#222222", arrowprops=dict(arrowstyle="-", lw=0.6, color="#888888"))

    # Place approval rates below the pie.
    appr_segments = []
    for bank, rate in zip(banks, rates):
        rate_str = f"{float(rate) * 100:.1f}%" if pd.notna(rate) else "n/a"
        appr_segments.append(f"{bank} {rate_str}")
    appr_text = "Approval rate\n" + "   ".join(appr_segments)
    ax.text(0.5, -0.018, appr_text, transform=ax.transAxes, ha="center", va="top",
            fontsize=22, color="#222222", linespacing=1.35)
    ax.set_title(f"{het} ({nc} banks)", fontsize=24, fontweight="bold", pad=4)

    # Anchor the pie at the top of the panel.
    ax.set_aspect("equal", adjustable="box", anchor="N")


# PLOT: Draw approval rates by bank with the dataset-wide approval rate as reference.
def plot_approval_rate_by_bank(table: pd.DataFrame, out_stem: Path) -> None:
    pivot = (
        table.pivot_table(index="config", columns="bank", values="approval_rate")
        .reindex(index=_available_config_order(table["config"], list(SHORT_LABELS)))
    )
    bank_cols = [b for b in ["A", "B", "C", "D", "E"] if b in pivot.columns]
    pivot = pivot[bank_cols]

    # Use iid as the stable approval-rate reference.
    ref = table.copy()
    ref_cfg = "iid_3banks" if "iid_3banks" in ref["config"].values else ref["config"].iloc[0]
    ref_subset = ref[ref["config"] == ref_cfg]
    # Include canceled cases in the denominator because the outcome is now a three-class target.
    overall_rate = ref_subset["n_approved"].sum() / ref_subset["n_cases"].sum()

    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    n_banks = len(bank_cols)
    x = np.arange(len(pivot.index))
    width = 0.8 / n_banks
    bank_patches = []
    for i, bank in enumerate(bank_cols):
        offset = (i - (n_banks - 1) / 2) * width
        bars = ax.bar(x + offset, pivot[bank].values, width, color=BANK_COLORS[i], edgecolor="white", linewidth=0.5)

        # Label bars and lift labels near the overall mean.
        for bar, val in zip(bars, pivot[bank].values):
            if pd.notna(val):
                _y = bar.get_height() + 0.010
                if overall_rate - 0.025 < bar.get_height() < overall_rate + 0.020: _y = overall_rate + 0.030
                ax.text(bar.get_x() + bar.get_width() / 2, _y,
                        f"{val * 100:.0f}%", ha="center", va="bottom", fontsize=7, color="#222222")
        bank_patches.append(mpatches.Patch(color=BANK_COLORS[i], label=f"Bank {bank}"))

    mean_line = ax.axhline(overall_rate, linestyle="--", linewidth=1.2, color=WARN_COLOR,
                           label=f"overall mean = {overall_rate:.2f}")
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in pivot.index])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Approval rate")

    # Auto-scale for the lower BPIC 2012 approval-rate level.
    ax.set_ylim(0, max(0.5, float(pivot.max().max()) * 1.38))
    _style_non_pie_axes(ax)

    # Place the mean legend in the upper right corner.
    mean_legend = ax.legend(handles=[mean_line], loc="upper right", frameon=True, fontsize=10)
    _style_reference_legend(mean_legend)
    ax.add_artist(mean_legend)

    # Place the bank legend below the axis.
    ax.legend(handles=bank_patches, frameon=False, ncol=n_banks, loc="lower center", bbox_to_anchor=(0.5, -0.26))
    _finish_non_pie_figure(fig, bottom=0.23)
    _save(fig, out_stem)


# PLOT: Draw 100% stacked case-share bars per config.
def plot_client_size_distribution(table: pd.DataFrame, out_stem: Path) -> None:
    pivot = table.pivot_table(index="config", columns="bank", values="case_share")

    # Keep the display order and convert shares to percentages.
    pivot = pivot.reindex(index=_available_config_order(pivot.index, list(SHORT_LABELS)))
    bank_cols = [b for b in ["A", "B", "C", "D", "E"] if b in pivot.columns]
    pivot = pivot[bank_cols].fillna(0.0) * 100
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=False)
    bottom = np.zeros(len(pivot.index))
    x = np.arange(len(pivot.index))

    # Stack banks within each configuration to show the client-size imbalance.
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

    # Use compact config labels and leave room for the bank legend.
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in pivot.index])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Case share (%)")
    ax.set_ylim(0, 112)
    ax.legend(frameon=False, ncol=len(bank_cols), loc="lower center", bbox_to_anchor=(0.5, -0.26))
    _style_non_pie_axes(ax)
    _finish_non_pie_figure(fig, bottom=0.23)
    _save(fig, out_stem)


# PLOT: Draw approval rate spread across increasing heterogeneity.
def plot_approval_rate_spread(table: pd.DataFrame, out_stem: Path) -> None:
    # Keep one spread value per config and preserve thesis display order.
    spread = (
        table.groupby("config", sort=False)["approval_rate_spread"]
        .first()
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
    ax.set_ylim(0, max(0.05, float(spread.max()) * 1.38))
    _style_non_pie_axes(ax)
    _finish_non_pie_figure(fig)
    _save(fig, out_stem)


# PLOT: Draw cases dropped by undersampling for BPIC 2017 figure parity.
def plot_cases_dropped(table: pd.DataFrame, out_stem: Path) -> None:
    order = _available_config_order(table["config"], list(SHORT_LABELS))
    dropped = (
        table.groupby("config", sort=False)["n_cases_unassigned_or_dropped"]
        .first()
        .reindex(order)
        .fillna(0)
        .astype(int)
    )
    fig, ax = plt.subplots(figsize=(9, 4.2), constrained_layout=False)
    x = np.arange(len(dropped))

    # Keep zero-drop configs visible with neutral bars.
    colors = [LINE_COLOR if v > 0 else NEUTRAL_COLOR for v in dropped.values]
    bars = ax.bar(x, dropped.values, color=colors, edgecolor="white", linewidth=0.6)

    # Place exact drop counts above each bar.
    label_offset = max(float(dropped.max()) * 0.02, 0.2)
    for bar, v in zip(bars, dropped.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{int(v):,}", ha="center", va="bottom", fontsize=9,
        )

    # Leave headroom for the count labels.
    ax.set_xticks(x)
    ax.set_xticklabels([_config_short(c) for c in dropped.index])
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Cases dropped (undersampling)")
    ymax = max(int(dropped.max() * 1.30), 10)
    ax.set_ylim(0, ymax)
    _style_non_pie_axes(ax)
    _finish_non_pie_figure(fig)
    _save(fig, out_stem)


# PLOT: Draw RequestedAmount band shares by config and bank.
def plot_requested_amount_quintiles(cases: pd.DataFrame, out_stem: Path) -> None:
    pivot = _category_share_pivot(cases, "amount_band", AMOUNT_BAND_ORDER)
    fig, (ax, legend_ax) = plt.subplots(2, 1, figsize=(10, 6.5), constrained_layout=False,
                                        gridspec_kw={"height_ratios": [4.0, 0.95], "hspace": 0.10})
    _plot_share_stack(ax, pivot, AMOUNT_BAND_COLORS, "Case share (%)", config_label_y=-0.065)
    ax.set_xlabel("")
    _draw_amount_legend(legend_ax, cases)

    # Keep the plot-to-legend gap above the compact legend.
    fig.subplots_adjust(left=0.080, right=0.99, top=0.98, bottom=0.010, hspace=2.00)
    _save_tight(fig, out_stem)


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FLOW

def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    configs = collect_configs()
    if not configs: raise FileNotFoundError(f"No partitioned configs found under {PROCESSED_DIR}")
    log.info(f"Found {len(configs):,} partition configuration(s): " + ", ".join(f"{h}_{n}banks" for h, n, _ in configs))

    # Build the comparison table across all available partition configs.
    table = build_comparison_table(configs)
    out_csv = PROCESSED_DIR / "B_03_comparison_table.csv"
    table.to_csv(out_csv, index=False)
    log.info(f"Saved table: {out_csv} ({len(table):,} rows, {len(table.columns):,} columns)")

    # Build case-level split data for RequestedAmount diagnostics.
    cases = build_case_level_table(configs)
    cases = add_requested_amount_bands(cases)

    # Render thesis partition figures (PNG only).
    plot_partition_pies(configs, PLOTS_DIR / "B_03_partition_pie_charts")
    plot_approval_rate_by_bank(table, PLOTS_DIR / "B_03_approval_rate_by_bank")
    plot_client_size_distribution(table, PLOTS_DIR / "B_03_client_size_distribution")
    plot_approval_rate_spread(table, PLOTS_DIR / "B_03_approval_rate_spread")
    plot_cases_dropped(table, PLOTS_DIR / "B_03_cases_dropped")
    plot_requested_amount_quintiles(cases, PLOTS_DIR / "B_03_requested_amount_quintiles")

    log.info("Per-configuration summary:")
    for cfg, grp in table.groupby("config", sort=False):
        spread = grp["approval_rate_spread"].iloc[0]
        log.info(f" {cfg}: {len(grp)} banks, total {int(grp['n_cases'].sum()):,} cases, approval spread {spread:.3f}")


if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb  |  Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────