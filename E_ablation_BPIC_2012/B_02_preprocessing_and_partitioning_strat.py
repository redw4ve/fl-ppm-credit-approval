"""
Step 2 (BPIC 2012 - ablation): Preprocessing and stratified client partitioning on the case level.

Pipeline:
- Loads the unfiltered event log. Filters lifecycle:transition == COMPLETE (Tax et al. 2017).
- Derives multiclass outcome / next-activity / remaining-time labels.
- Cuts traces at the final decision event, so RemainingTime measures time-to-decision.
- Drops truncation cases. Keeps zero-amount cases (iid: seeded shuffle; weak/medium: routed to Bank A).
- Partitions cases across simulated banks under one of {iid, weak, medium}.
- Performs a temporal train/val/test split within each bank.
- Writes per-bank parquets, a per-config centralized baseline and a metadata/validation report.

Heterogeneity design:
- weak / medium use a stratified amount-quintile soft-mix on AMOUNT_REQ (the only static feature).
    - Every bank receives a share of every quintile, weighted by a graded skew and unequal to zero.
    - Bank A drifts toward low amounts and larger size, Bank C drifts toward high amounts and smaller size.
- No strong tier on BPIC 2012: no LoanGoal / offer attributes that BPIC 2017's strong config uses.
- Note: BPIC 2012 has an inverted approval-rate pattern: The largest bank has the lowest approval rate.

Run: python B_02_preprocessing_and_partitioning_strat.py

Configuration: constants block below (N_CLIENTS, HETEROGENEITY).
On the first run, parses BPI_Challenge_2012.xes and writes the parquet cache.

Outputs:
- per-bank splits + metadata: data/processed/{HETEROGENEITY}_{N_CLIENTS}banks/
- per-config centralized baseline: data/processed/centralized/{HETEROGENEITY}_{N_CLIENTS}banks/
"""

# IMPORTS
from __future__ import annotations
import json
import logging
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
import numpy as np
import pandas as pd
from pm4py.objects.log.importer.xes.variants import iterparse
import pm4py

# CONFIGURATION
# WORKFLOW overrides environment variables HETEROGENEITY / N_CLIENTS (direct runs use the defaults).
N_CLIENTS: int = int(os.environ.get("N_CLIENTS", "3"))                  # BPIC 2012 only supports 3 banks
HETEROGENEITY: str = os.environ.get("HETEROGENEITY", "iid")             # iid | weak | medium
SPLIT_MODE: str = os.environ.get("SPLIT_MODE", "stratified_quintile")   # stratified_quintile | linear_quintile
TRAIN_VAL_TEST_RATIO: tuple[float, float, float] = (0.6, 0.2, 0.2)      # train/val/test distribution
MAX_PREFIX_LENGTH_FOR_ENCODING: int = 42                                # p98 after COMPLETE filtering, before decision cut

# Outcome mapping (BPIC 2012 event names and shared class schema).
APPROVED  = "A_APPROVED"
DENIED    = "A_DECLINED"
CANCELLED = "A_CANCELLED"
END_TOKEN = "[END]"
TERMINAL_OUTCOMES = (APPROVED, DENIED, CANCELLED)
OUTCOME_MAPPING = {APPROVED: 2, DENIED: 1, CANCELLED: 0}
BANK_NAMES = ("A", "B", "C", "D", "E")

# Resolve file paths from the script directory.
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH:  Path = SCRIPT_DIR / "data/cache/B_02_bpic2012_events.parquet"
XES_PATH:    Path = SCRIPT_DIR / "BPI Challenge 2012/BPI_Challenge_2012.xes"
OUTPUT_ROOT: Path = SCRIPT_DIR / "data/processed"

# Lifecycle filter: Keep COMPLETE only, drops approx. 37% of the events.
LIFECYCLE_KEEP: str = "COMPLETE"

# Stratified per-quintile bank weights for WEAK / MEDIUM (rows: Q1 low -> Q5 high; columns: A, B, C).
# Rows sum to 1, all cells are > 0, column bias preserves |A|>|B|>|C| with A toward low and C toward high amounts.
WEAK_QUINTILE_WEIGHTS = np.array([
    [0.55, 0.30, 0.15],   # Q1 (lowest amounts) -> heavier A
    [0.48, 0.32, 0.20],
    [0.40, 0.35, 0.25],
    [0.32, 0.39, 0.29],
    [0.26, 0.34, 0.40],   # Q5 (highest amounts) -> heavier C
])
MEDIUM_QUINTILE_WEIGHTS = np.array([
    [0.85, 0.10, 0.05],   # Q1 -> dominant A
    [0.75, 0.15, 0.10],
    [0.45, 0.35, 0.20],
    [0.20, 0.40, 0.40],
    [0.10, 0.35, 0.55],   # Q5 -> highest C share
])

# Initialize logger.
logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S", level=logging.INFO)
log = logging.getLogger("preprocess")

# Set the random seed for reproducibility.
RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Log and record warnings for the final validation report.
_WARNINGS: list[str] = []
_PARTITION_PROVENANCE: dict[str, Any] = {}
_TRACE_CUT_REPORT: dict[str, Any] = {}
def warn(msg: str) -> None:
    log.warning(msg)
    _WARNINGS.append(msg)

# Validate configuration before reading or writing any artifacts.
def validate_configuration() -> None:
    if N_CLIENTS != 3: raise ValueError("BPIC 2012 supports N_CLIENTS=3 only")

# ----------------------------------------------------------------------------------------------------------------------
# 1. LOAD

# Parse BPIC 2012 XES and persist it as parquet.
def _build_cache_from_xes(xes_path: Path, cache_path: Path) -> None:
    log.info(f"Cache not found at {cache_path}; parsing XES from {xes_path} (slow, one-off)")

    # Verify the XES path and parse it with iterparse to avoid optional backend dependencies.
    assert xes_path.exists(), f"XES file not found at {xes_path}"
    event_log = iterparse.apply(str(xes_path))
    df = pm4py.convert_to_dataframe(event_log) if not isinstance(event_log, pd.DataFrame) else event_log

    # Convert time:timestamp to UTC datetime.
    if not pd.api.types.is_datetime64_any_dtype(df["time:timestamp"]):
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

    # Cache the event log.
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log.info(f"Wrote cache: {len(df):,} events to {cache_path}")

# Load the unfiltered event log and build the parquet cache from XES (if missing).
def load_event_log(path: Path) -> pd.DataFrame:

    # Check if the Cache already exists.
    if not path.exists(): _build_cache_from_xes(XES_PATH, path)
    df = pd.read_parquet(path)
    n_events = len(df)
    n_cases = df["case:concept:name"].nunique()
    log.info(f"Loaded {n_events:,} events across {n_cases:,} cases from {path}")

    # Assert the expected case and event count.
    assert n_events == 262_200, f"unexpected event count: {n_events:,} (want 262,200)"
    assert n_cases == 13_087, f"unexpected case count: {n_cases:,} (want 13,087)"
    if not pd.api.types.is_datetime64_any_dtype(df["time:timestamp"]):
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    df["case:AMOUNT_REQ"] = pd.to_numeric(df["case:AMOUNT_REQ"], errors="raise").astype(float)
    return df

# Filter the dataframe to lifecycle:transition == COMPLETE.
def filter_lifecycle(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.loc[df["lifecycle:transition"] == LIFECYCLE_KEEP].copy()
    log.info(f"Lifecycle filter: {n0:,} -> {len(df):,} events ({1 - len(df) / n0:.1%} dropped)")
    return df

# ----------------------------------------------------------------------------------------------------------------------
# 2. CASE FEATURE DERIVATION

# Build lifecycle-aware activity tokens as the target for next activity prediction.
def add_activity_token(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["activity_token"] = df["concept:name"].astype(str) + "+" + df["lifecycle:transition"].astype(str)
    return df

# Build one row per case from the last recognized decision event.
def build_outcome_table(df: pd.DataFrame) -> pd.DataFrame:

    # Preserve chronological order and use event position to break equal timestamps.
    ordered = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    ordered["_event_pos"] = ordered.groupby("case:concept:name", sort=False).cumcount()
    terminals = ordered.loc[
        ordered["concept:name"].isin(TERMINAL_OUTCOMES),
        ["case:concept:name", "concept:name", "time:timestamp", "_event_pos", "lifecycle:transition"],
    ].copy()

    # Return an empty outcome table with the expected schema when no terminal events exist.
    if terminals.empty:
        return pd.DataFrame(columns=[
            "case_id", "outcome_event", "outcome_time", "outcome_pos", "outcome_lifecycle",
            "outcome", "n_outcome_events", "n_distinct_outcome_events",
        ]).set_index("case_id")

    # Count repeated decision events before selecting the final decision.
    counts = terminals.groupby("case:concept:name")["concept:name"].agg(
        n_outcome_events="size", n_distinct_outcome_events="nunique",
    )
    lifecycle_counts = terminals.groupby("concept:name")["lifecycle:transition"].nunique()
    multi_lifecycle = {str(k): int(v) for k, v in lifecycle_counts.items() if int(v) > 1}
    if multi_lifecycle: warn(f"outcome lifecycle: multiple lifecycle values observed for {multi_lifecycle}")

    # Use the last decision event per case as the prediction target.
    last_terminal = (
        terminals.sort_values(["case:concept:name", "time:timestamp", "_event_pos"])
        .groupby("case:concept:name", sort=False)
        .tail(1)
        .rename(columns={
            "case:concept:name": "case_id", "concept:name": "outcome_event", "time:timestamp": "outcome_time",
            "_event_pos": "outcome_pos", "lifecycle:transition": "outcome_lifecycle",
        })
        .set_index("case_id")
    )
    last_terminal["outcome"] = last_terminal["outcome_event"].map(OUTCOME_MAPPING).astype("int64")
    last_terminal = last_terminal.join(counts, how="left")

    # Log multiclass decision label counts and repeated label diagnostics.
    n_app = int((last_terminal["outcome"] == OUTCOME_MAPPING[APPROVED]).sum())
    n_den = int((last_terminal["outcome"] == OUTCOME_MAPPING[DENIED]).sum())
    n_can = int((last_terminal["outcome"] == OUTCOME_MAPPING[CANCELLED]).sum())
    n_multi = int((last_terminal["n_outcome_events"] > 1).sum())
    n_multi_distinct = int((last_terminal["n_distinct_outcome_events"] > 1).sum())
    n_no_terminal = df["case:concept:name"].nunique() - len(last_terminal)
    log.info(f"Outcome counts: A_APPROVED={n_app:,}, A_DECLINED={n_den:,}, "
             f"A_CANCELLED={n_can:,}, no terminal event={n_no_terminal:,}")
    log.info(f"Outcome diagnostics: {n_multi:,} case(s) with repeated decision events, "
             f"{n_multi_distinct:,} with multiple distinct decision labels")
    return last_terminal

# Remove administrative events after the final decision event.
def cut_traces_at_outcome(df: pd.DataFrame, outcome_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:

    # Preserve cases with no outcome event until the truncation filter accounts for them.
    ordered = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    ordered["_event_pos"] = ordered.groupby("case:concept:name", sort=False).cumcount()
    ordered["_outcome_pos"] = ordered["case:concept:name"].map(outcome_table["outcome_pos"])
    outcome_case_mask = ordered["_outcome_pos"].notna()
    keep_mask = ~outcome_case_mask | (ordered["_event_pos"] <= ordered["_outcome_pos"])
    cut = ordered.loc[keep_mask].drop(columns=["_event_pos", "_outcome_pos"]).copy()

    # Record how much activity past the decision event is removed.
    events_in_outcome_cases = int(outcome_case_mask.sum())
    events_removed = int((~keep_mask.astype(bool)).sum())
    report = {
        "cases_with_outcome": int(len(outcome_table)),
        "cases_with_multiple_outcome_events": int((outcome_table["n_outcome_events"] > 1).sum()),
        "cases_with_multiple_distinct_outcomes": int((outcome_table["n_distinct_outcome_events"] > 1).sum()),
        "events_before_cut": int(len(df)),
        "events_after_cut": int(len(cut)),
        "events_removed_after_outcome": events_removed,
        "share_events_removed_after_outcome": 0.0 if events_in_outcome_cases == 0 else events_removed / events_in_outcome_cases,
    }
    _TRACE_CUT_REPORT.clear()
    _TRACE_CUT_REPORT.update(report)
    log.info(f"Trace cutting: removed {events_removed:,} post-decision events "
             f"({report['share_events_removed_after_outcome']:.2%} of events in cases with a decision)")
    return cut, report

# Add [END] activity target, RemainingTime until decision (s), TimeDelta (s).
def derive_event_features(df: pd.DataFrame) -> pd.DataFrame:

    # Sort events chronologically within each case.
    df = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    if "activity_token" not in df.columns: df = add_activity_token(df)
    grp = df.groupby("case:concept:name", sort=False)

    # Add lifecycle-aware next activity targets and emit [END] after the decision event.
    df["NextActivity"] = grp["activity_token"].shift(-1).fillna(END_TOKEN)

    # Use the case end after the cut, which is the final decision timestamp.
    case_max = grp["time:timestamp"].transform("max")
    case_min = grp["time:timestamp"].transform("min")

    # Store raw time-to-decision in seconds. The prefix encoding owns optional log1p targets.
    df["RemainingTime"] = (case_max - df["time:timestamp"]).dt.total_seconds()
    df["TimeDelta"] = (df["time:timestamp"] - case_min).dt.total_seconds()
    return df

# Build one metadata row per case: Timing, size, AMOUNT_REQ, outcome, terminal-event flag.
def build_case_metadata(df: pd.DataFrame, outcome_table: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("case:concept:name", sort=False)

    # Collect metadata for each case (basically only AMOUNT_REQ).
    meta = pd.DataFrame({
        "start_time":       grp["time:timestamp"].min(),
        "end_time":         grp["time:timestamp"].max(),
        "n_events":         grp.size(),
        "requested_amount": grp["case:AMOUNT_REQ"].first().astype(float),
    })

    # Add metadata to each case.
    meta.index.name = "case_id"
    meta = meta.reset_index()
    meta["outcome"] = meta["case_id"].map(outcome_table["outcome"])
    meta["outcome_event"] = meta["case_id"].map(outcome_table["outcome_event"])
    meta["has_terminal_event"] = meta["case_id"].isin(outcome_table.index)
    return meta

# ----------------------------------------------------------------------------------------------------------------------
# 3. DATA QUALITY FILTERING

# Drop truncation cases with no terminal event. Keep RequestedAmount zero and canceled cases.
def filter_cases(meta: pd.DataFrame) -> pd.DataFrame:
    n0 = len(meta)
    log.info(f"Filtering: Starting from {n0:,} cases")

    # Apply truncated mask, flip the boolean (the case has a terminal event (true) -> not truncated (false)).
    truncated_mask = ~meta["has_terminal_event"]
    n_trunc = int(truncated_mask.sum())

    # Keep only cases that do have a terminal event.
    meta = meta.loc[~truncated_mask].copy()
    log.info(f" 3.1 dropped {n_trunc:,} truncation cases")

    # Find cases where requested_amount = 0 (BPIC2012 has only 1 such case).
    n_zero = int((meta["requested_amount"] == 0).sum())
    log.info(f" 3.2 keeping {n_zero:,} zero-amount case(s) (routed per heterogeneity level)")

    # Retain canceled as class 0 in the multiclass outcome target.
    n_can = int((meta["outcome"] == OUTCOME_MAPPING[CANCELLED]).sum())
    log.info(f" 3.3 keeping {n_can:,} cancelled cases as outcome class 0")
    if meta["outcome"].isna().any(): raise ValueError("filter_cases: retained cases contain missing outcome labels")
    meta["outcome"] = meta["outcome"].astype(int)
    log.info(f"Filtered case count: {len(meta):,} (started {n0:,})")
    return meta.drop(columns=["has_terminal_event"])

# ----------------------------------------------------------------------------------------------------------------------
# 4. CLIENT PARTITION (iid / weak / medium, for BPIC 2012 no strong variant)

# Return the first n configured bank names.
def _bank_names(n_clients: int) -> list[str]: return list(BANK_NAMES[:n_clients])

# Append one structured entry to the partition provenance report.
def record_provenance_list(key: str, value: dict[str, Any]) -> None:
    entries = cast(list[dict[str, Any]], _PARTITION_PROVENANCE.setdefault(key, []))
    entries.append(value)

# Record how cases with RequestedAmount zero were routed for the active heterogeneity level.
def record_zero_routing(heterogeneity: str, n: int, counts: dict[str, int]) -> None:
    record_provenance_list("zero_amount_routing", {
        "heterogeneity": heterogeneity,
        "n_zero_amount_cases": int(n),
        "counts": {bank: int(count) for bank, count in counts.items()},
    })

# Convert NumPy scalars to JSON save int / float / NaN before writing metadata.
def _json_scalar(value: object) -> object:
    if pd.isna(value): return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    return value

# Summarize the assigned cases for metadata and validation.
def _partition_summary(meta: pd.DataFrame, assignment: dict[str, str]) -> list[dict[str, object]]:

    # Attach bank labels and ignore unassigned cases (no strong undersampling -> no dropped cases).
    df = meta.assign(_bank=meta["case_id"].map(assignment)).dropna(subset=["_bank"])
    rows: list[dict[str, object]] = []
    for bank, sub in df.groupby("_bank"):
        labelled = sub["outcome"]
        if labelled.isna().any(): raise ValueError(f"partition summary: Bank {bank} contains missing outcomes")
        n_approved = int((labelled == OUTCOME_MAPPING[APPROVED]).sum())
        rows.append({
            "bank": str(bank),
            "n_cases": int(len(sub)),
            "n_labelled": int(labelled.shape[0]),
            "n_cancelled": int((labelled == OUTCOME_MAPPING[CANCELLED]).sum()),
            "n_approved": n_approved,
            "n_denied": int((labelled == OUTCOME_MAPPING[DENIED]).sum()),
            "approval_rate": None if labelled.empty else float(n_approved / labelled.shape[0]),
            "mean_requested_amount": float(sub["requested_amount"].mean()),
            "n_loan_goals": None,
            "top_loan_goal": None,
        })
    return sorted(rows, key=lambda row: str(row["bank"]))

# Index partition summary rows by bank name.
def _summary_by_bank(meta: pd.DataFrame, assignment: dict[str, str]) -> dict[str, dict[str, object]]:
    return {str(row["bank"]): row for row in _partition_summary(meta, assignment)}

# Return approval share per bank across all labeled decision outcomes.
def _per_bank_rates(meta: pd.DataFrame, assignment: dict[str, str]) -> dict[str, float]:
    df = meta.assign(_bank=meta["case_id"].map(assignment))
    df = df.dropna(subset=["_bank"])
    if df["outcome"].isna().any(): raise ValueError("_per_bank_rates: assigned cases contain missing outcomes")
    rates = df.groupby("_bank")["outcome"].apply(
        lambda values: float((values == OUTCOME_MAPPING[APPROVED]).sum() / len(values))
    )
    return rates.to_dict()

# IID random split: Shuffle and split into banks of equal size (zeros included).
def _partition_iid(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    rng = np.random.default_rng(RANDOM_SEED)
    case_ids = meta["case_id"].to_numpy().copy()
    rng.shuffle(case_ids)
    splits = np.array_split(case_ids, n_clients)
    return {cid: bank for bank, ids in zip(_bank_names(n_clients), splits) for cid in ids}

# ZERO AMOUNT: Route zero-amount case(s) to Bank A (n=1, logged for provenance).
def _route_zero_amount_cases(meta: pd.DataFrame, heterogeneity: str) -> dict[str, str]:
    zero_ids = meta.loc[meta["requested_amount"] == 0, "case_id"].tolist()
    n = len(zero_ids)
    if n == 0: return {}
    routing = {cid: "A" for cid in zero_ids}
    record_zero_routing(heterogeneity, n, {"A": n, "B": 0, "C": 0})
    log.info(f" zero-amount routing [{heterogeneity}]: {n:,} case(s) -> Bank A")
    return routing

# Stratified amount-quintile soft-mix: Distributes cases across A/B/C according to one row of `weights`.
# Random assignment is seeded for reproducibility. Every bank receives each quintile, no all-low or all-high clients.
def _quintile_skew_assignment(meta: pd.DataFrame, weights: np.ndarray, seed: int) -> dict[str, str]:

    # Keep RequestedAmount zero cases and assign each to a requested amount quintile.
    nonzero = meta.loc[meta["requested_amount"] > 0].copy()
    nonzero["quintile"] = pd.qcut(nonzero["requested_amount"], q=5, labels=False, duplicates="drop").astype(int)

    # Cut points used in the `pd.qcut` logged (amount range and case count) for each computed quintile.
    quintile_cuts = nonzero.groupby("quintile")["requested_amount"].agg(["min", "max", "count"])
    log.info(" amount-quintile cuts (computed on >0 subset):")
    for q, row in quintile_cuts.iterrows():
        log.info(f"    Q{q + 1}: amount in [{row['min']:,.0f}, {row['max']:,.0f}]  n={int(row['count']):,}")

    # Shuffle cases within each quintile and split them into A/B/C according to the configured weights.
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    bank_letters = ["A", "B", "C"]
    for q in range(weights.shape[0]):
        q_cases = nonzero.loc[nonzero["quintile"] == q, "case_id"].to_numpy().copy()
        rng.shuffle(q_cases)
        n = len(q_cases)

        # Convert cumulative weights into split indices for this quintile.
        cuts = np.clip(np.round(np.cumsum(weights[q]) * n).astype(int), 0, n)
        chunks = {"A": q_cases[:cuts[0]], "B": q_cases[cuts[0]:cuts[1]], "C": q_cases[cuts[1]:],}

        # Store the bank assignment for each case.
        for bank in bank_letters:
            for cid in chunks[bank]:
                assignment[cid] = bank

    # Record the weights and seed used in provenance.
    record_provenance_list("quintile_skew", {
        "weights_per_quintile": weights.tolist(),
        "seed": int(seed),
        "n_nonzero_cases": int(len(nonzero)),
    })
    return assignment

# LINEAR: Assign whole AMOUNT_REQ quintiles to contiguous banks without random sampling.
def _linear_quintile_assignment(meta: pd.DataFrame) -> dict[str, str]:

    # Keep non-zero amounts and assign deterministic amount quintiles.
    nonzero = meta.loc[meta["requested_amount"] > 0].copy()
    nonzero = nonzero.sort_values(["requested_amount", "case_id"]).copy()
    nonzero["quintile"] = pd.qcut(nonzero["requested_amount"], q=5, labels=False, duplicates="drop").astype(int)
    mapping = {0: "A", 1: "A", 2: "B", 3: "C", 4: "C"}

    # Map every case to the bank owning its full amount quintile.
    assignment = {
        cid: mapping[int(q)]
        for cid, q in zip(nonzero["case_id"], nonzero["quintile"])
    }
    record_provenance_list("linear_quintile", {
        "n_nonzero_cases": int(len(nonzero)),
        "quintile_to_bank": {str(k + 1): v for k, v in mapping.items()},
    })
    return assignment

# Dispatch the configured AMOUNT_REQ split mode.
def _amount_quintile_assignment(meta: pd.DataFrame, weights: np.ndarray, seed: int) -> dict[str, str]:
    if SPLIT_MODE == "stratified_quintile": return _quintile_skew_assignment(meta, weights, seed)
    if SPLIT_MODE == "linear_quintile": return _linear_quintile_assignment(meta)
    raise ValueError(f"unknown SPLIT_MODE: '{SPLIT_MODE}'")

# WEAK: Stratified amount-quintile soft-mix (graded skew, all banks see every quintile).
def _partition_weak(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    if n_clients != 3: raise ValueError("weak heterogeneity requires N_CLIENTS=3")
    log.info(" weak: AMOUNT_REQ quintile skew with graded A/B/C size imbalance")
    base = _amount_quintile_assignment(meta, WEAK_QUINTILE_WEIGHTS, RANDOM_SEED)
    zero = _route_zero_amount_cases(meta, "weak")
    final = {**base, **zero}
    counts = Counter(final.values())
    log.info(f" weak: final sizes A={counts['A']:,} B={counts['B']:,} C={counts['C']:,}")
    return final

# MEDIUM: Steeper amount-quintile skew, still mixed at every quintile (no LoanGoal moves for BPIC2012).
def _partition_medium(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    if n_clients != 3: raise ValueError("medium heterogeneity requires N_CLIENTS=3")
    log.info(" medium: steeper AMOUNT_REQ quintile skew with graded A/B/C size imbalance")
    base = _amount_quintile_assignment(meta, MEDIUM_QUINTILE_WEIGHTS, RANDOM_SEED)
    zero = _route_zero_amount_cases(meta, "medium")
    final = {**base, **zero}
    counts = Counter(final.values())
    log.info(f" medium: final sizes A={counts['A']:,} B={counts['B']:,} C={counts['C']:,}")
    return final

# Dispatch to the partitioning strategy decided by HETEROGENEITY.
def partition_cases(meta: pd.DataFrame) -> dict[str, str]:

    # Start a fresh provenance record for this partitioning run.
    log.info(f"Partitioning: strategy='{HETEROGENEITY}', N_CLIENTS={N_CLIENTS}")
    _PARTITION_PROVENANCE.clear()
    _PARTITION_PROVENANCE["strategy"] = HETEROGENEITY
    _PARTITION_PROVENANCE["n_clients"] = int(N_CLIENTS)
    _PARTITION_PROVENANCE["split_mode"] = SPLIT_MODE
    _PARTITION_PROVENANCE["n_cases_available_after_filters"] = int(len(meta))

    # Store a description of the selected partitioning strategy.
    _PARTITION_PROVENANCE["criteria"] = {
        "iid":    "random equal-size split",
        "weak":   f"{SPLIT_MODE} AMOUNT_REQ quintile split with graded A/B/C size imbalance",
        "medium": f"steeper {SPLIT_MODE} AMOUNT_REQ quintile split with graded A/B/C size imbalance",
    }.get(HETEROGENEITY, "unknown")

    # Dispatch to the selected heterogeneity (BPIC 2012 supports iid | weak | medium).
    strategies = {"iid": _partition_iid, "weak": _partition_weak, "medium": _partition_medium,}
    if HETEROGENEITY in strategies: return strategies[HETEROGENEITY](meta, N_CLIENTS)
    raise ValueError(f"unknown HETEROGENEITY: '{HETEROGENEITY}' (BPIC 2012 supports iid | weak | medium)")

# Sanitycheck the partition: Only known case IDs exist and no bank is empty.
def assert_partition_valid(assignment: dict[str, str], meta: pd.DataFrame) -> None:
    case_ids = set(meta["case_id"])
    assigned = set(assignment.keys())

    # No case in the assignment should be unknown to meta.
    assert assigned.issubset(case_ids), f"partition contains {len(assigned - case_ids):,} unknown case IDs"

    # Count cases per bank and assert none is empty.
    bank_counts = dict(Counter(str(bank) for bank in assignment.values()))
    for b, n in bank_counts.items(): assert n > 0, f"Bank {b} has zero cases (R1 violation)"

    # Record case accounting for the validation report.
    n_dropped = len(case_ids) - len(assigned)
    _PARTITION_PROVENANCE["n_cases_assigned"] = int(len(assigned))
    _PARTITION_PROVENANCE["n_cases_unassigned_or_dropped"] = int(n_dropped)
    _PARTITION_PROVENANCE["bank_case_counts"] = {str(b): int(n) for b, n in bank_counts.items()}
    log.info(f" per-bank case counts: {bank_counts}")

# Log per-bank case count, approval rate, mean RequestedAmount.
def log_partition_stats(meta: pd.DataFrame, assignment: dict[str, str]) -> None:

    # Attach the bank label to each case.
    df = meta.assign(_bank=meta["case_id"].map(assignment)).dropna(subset=["_bank"])
    log.info("Per-bank statistics:")
    for bank, sub in df.groupby("_bank"):
        labelled = sub["outcome"]
        if labelled.isna().any(): raise ValueError(f"log_partition_stats: Bank {bank} contains missing outcomes")
        rate = float((labelled == OUTCOME_MAPPING[APPROVED]).sum() / len(labelled))
        mean_amt = sub["requested_amount"].mean()
        log.info(f" Bank {bank}: n_cases={len(sub):,}  approval={rate:.3f}  "
                 f"mean_requested_amount={mean_amt:,.0f}  n_loan_goals=N/A")

# ----------------------------------------------------------------------------------------------------------------------
# 5. TEMPORAL TRAIN / VAL / TEST SPLIT WITHIN EACH BANK

# Slice a chronologically sorted case list 60/20/20 (per TRAIN_VAL_TEST_RATIO).
def temporal_split(case_ids_sorted: list[str]) -> tuple[list[str], list[str], list[str]]:
    n = len(case_ids_sorted)
    n_train = int(round(n * TRAIN_VAL_TEST_RATIO[0]))
    n_val   = int(round(n * TRAIN_VAL_TEST_RATIO[1]))
    return case_ids_sorted[:n_train], case_ids_sorted[n_train:n_train + n_val], case_ids_sorted[n_train + n_val:]

# Sort the cases by start_time and produce train/val/test event frames (per bank).
def split_bank(bank: str, case_ids: list[str], meta: pd.DataFrame, events: pd.DataFrame) -> dict[str, pd.DataFrame]:

    # Sort cases chronologically before slicing.
    bank_meta = meta.set_index("case_id").loc[case_ids].sort_values("start_time")
    train_ids, val_ids, test_ids = temporal_split(bank_meta.index.tolist())

    # Warn if a split becomes too small for stable evaluation.
    for name, ids in (("train", train_ids), ("val", val_ids), ("test", test_ids)):
        if len(ids) < 200: warn(f"Bank {bank} {name} split has only {len(ids):,} cases (<200)")
    log.info(f" Bank {bank}: train={len(train_ids):,}  val={len(val_ids):,}  test={len(test_ids):,} cases")

    # Return event rows for each case split.
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    return {name: events.loc[events["case:concept:name"].isin(set(ids))].copy() for name, ids in split_ids.items()}

# ----------------------------------------------------------------------------------------------------------------------
# 6. SAVE OUTPUTS

# Collect the run configuration for metadata.json.
def config_metadata() -> dict[str, object]:
    return {
        "DATASET": "BPIC 2012",
        "N_CLIENTS": N_CLIENTS,
        "HETEROGENEITY": HETEROGENEITY,
        "SPLIT_MODE": SPLIT_MODE,
        "RANDOM_SEED": RANDOM_SEED,
        "TRAIN_VAL_TEST_RATIO": list(TRAIN_VAL_TEST_RATIO),
        "OUTCOME_MAPPING": OUTCOME_MAPPING,
        "REMAINING_TIME_UNIT": "raw_seconds_to_decision",
        "REMAINING_TIME_LOG_TRANSFORM_STAGE": "prefix_encoding",
        "MAX_PREFIX_LENGTH_FOR_ENCODING": MAX_PREFIX_LENGTH_FOR_ENCODING,
        "LIFECYCLE_KEEP": LIFECYCLE_KEEP,
    }

# Write parquets per bank, B_02_partition_stats.csv, B_02_metadata.json and centralized splits.
def write_partition_outputs(bank_splits: dict[str, dict[str, pd.DataFrame]], central_splits: dict[str, pd.DataFrame],
    meta: pd.DataFrame, assignment: dict[str, str], out_dir: Path, central_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    central_dir.mkdir(parents=True, exist_ok=True)

    # Write train, validation and test parquets for each bank.
    for bank, splits in bank_splits.items():
        for split_name, frame in splits.items():
            frame.to_parquet(out_dir / f"B_02_bank_{bank}_{split_name}.parquet", index=False)

    # Write one human-readable partition summary table.
    summary_by_bank = _summary_by_bank(meta, assignment)
    rows = [
        {**summary_by_bank[str(bank)], "n_events": int(sum(len(split) for split in splits.values()))}
        for bank, splits in bank_splits.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "B_02_partition_stats.csv", index=False)

    # Persist configuration, warnings and partition provenance.
    _PARTITION_PROVENANCE["per_bank_summary"] = list(summary_by_bank.values())
    metadata = {
        "config": config_metadata(),
        "partition_provenance": _PARTITION_PROVENANCE,
        "trace_cut_report": _TRACE_CUT_REPORT,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(_WARNINGS),
    }
    with (out_dir / "B_02_metadata.json").open("w") as f: json.dump(metadata, f, indent=2)

    # Write the centralized splits per configuration.
    for split_name, frame in central_splits.items():
        frame.to_parquet(central_dir / f"B_02_{split_name}.parquet", index=False)

# ----------------------------------------------------------------------------------------------------------------------
# 7. VALIDATION REPORT

# Format one-per-bank line for the validation report.
def _format_bank_summary_line(bank: str, row: dict[str, object], splits: dict[str, pd.DataFrame]) -> str:
    approval_rate = row["approval_rate"]
    approval_text = "nan" if approval_rate is None else f"{float(cast(float, approval_rate)):.3f}"
    n_cases = int(cast(int, row["n_cases"]))
    n_labelled = int(cast(int, row["n_labelled"]))
    n_cancelled = int(cast(int, row["n_cancelled"]))
    return (
        f"  Bank {bank}: cases={n_cases:>6,}  approval={approval_text}  "
        f"labelled={n_labelled:>6,}  cancelled={n_cancelled:>6,}  "
        f"events: train={len(splits['train']):>7,} "
        f"val={len(splits['val']):>7,} test={len(splits['test']):>7,}"
    )

# Print the final summary to stdout and write it to {out_dir}/B_02_run.log.
def write_validation_report(out_dir: Path, meta: pd.DataFrame, assignment: dict[str, str],
    bank_splits: dict[str, dict[str, pd.DataFrame]], n_cases_initial: int) -> None:
    lines: list[str] = [
        "-" * 78,
        f"VALIDATION REPORT  ({datetime.now(timezone.utc).isoformat()})",
        "-" * 78,
        "Configuration:",
        f"  DATASET              = BPIC 2012",
        f"  HETEROGENEITY        = {HETEROGENEITY}",
        f"  N_CLIENTS            = {N_CLIENTS}",
        f"  SPLIT_MODE           = {SPLIT_MODE}",
        f"  RANDOM_SEED          = {RANDOM_SEED}",
        f"  TRAIN_VAL_TEST_RATIO = {TRAIN_VAL_TEST_RATIO}",
        f"  RemainingTime        = raw_seconds_to_decision",
        f"  RT log transform     = prefix_encoding",
        f"  LIFECYCLE_KEEP       = {LIFECYCLE_KEEP}",
        "",
    ]
    # Report case accounting (no strong undersampling).
    n_unassigned = len(meta) - len(assignment)
    lines.append(f"Cases: {n_cases_initial:,} initial -> {len(meta):,} after filters "
                 f"-> {len(assignment):,} assigned to banks")
    if n_unassigned: lines.append(f"Unassigned/dropped after filtering: {n_unassigned:,}")

    # Add one validation line per bank.
    lines.append("")
    lines.append("Per-bank summary:")
    summary_by_bank = _summary_by_bank(meta, assignment)
    for bank in sorted(summary_by_bank):
        lines.append(_format_bank_summary_line(bank, summary_by_bank[bank], bank_splits[bank]))
    lines.append("")

    # Add any accumulated warnings to the validation report.
    if _WARNINGS:
        lines.append(f"Warnings ({len(_WARNINGS)}):")
        lines.extend(f"  - {w}" for w in _WARNINGS)
    else: lines.append("Warnings: none")
    lines.append("-" * 78)

    # Print the report and write the run log for the current configuration.
    text = "\n".join(lines)
    print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "B_02_run.log").open("w") as f: f.write(text + "\n")

# ----------------------------------------------------------------------------------------------------------------------
# MAIN FLOW

def main() -> None:

    # Validate the run configuration before reading data or writing outputs.
    validate_configuration()

    # Load the raw event log and record the original case count.
    df = load_event_log(INPUT_PATH)
    n_cases_initial = df["case:concept:name"].nunique()

    # Apply the BPIC 2012 lifecycle filter.
    df = filter_lifecycle(df)

    # Build lifecycle-aware activity tokens, derive multiclass outcomes and cut traces at the decision.
    df = add_activity_token(df)
    outcome_table = build_outcome_table(df)
    df, _trace_report = cut_traces_at_outcome(df, outcome_table)
    df = derive_event_features(df)
    meta = build_case_metadata(df, outcome_table)

    # Apply case filters and keep only events from retained cases.
    meta_filtered = filter_cases(meta)
    kept_ids = set(meta_filtered["case_id"])
    df_filtered = df.loc[df["case:concept:name"].isin(kept_ids)].copy()

    # Attach the case outcome to every event.
    df_filtered = df_filtered.merge(
        meta_filtered[["case_id", "outcome"]].rename(columns={"case_id": "case:concept:name"}),
        on="case:concept:name", how="left",
    ).sort_values(["case:concept:name", "time:timestamp"])

    # Partition retained cases into banks and validate the resulting assignment.
    assignment = partition_cases(meta_filtered)
    assert_partition_valid(assignment, meta_filtered)
    log_partition_stats(meta_filtered, assignment)

    # Restrict to assigned cases.
    kept_bank_ids = set(assignment.keys())
    meta_for_banks = meta_filtered.loc[meta_filtered["case_id"].isin(kept_bank_ids)].copy()
    df_for_banks = df_filtered.loc[df_filtered["case:concept:name"].isin(kept_bank_ids)].copy()

    # Build temporal splits for each simulated bank.
    log.info("Temporal train/val/test split per bank:")
    bank_splits = {
        bank: split_bank(bank, [c for c, b in assignment.items() if b == bank], meta_for_banks, df_for_banks)
        for bank in sorted(set(assignment.values()))
    }

    # Build the centralized baseline as the union of bank splits already created.
    central_dir = OUTPUT_ROOT / "centralized" / f"{HETEROGENEITY}_{N_CLIENTS}banks"
    log.info("Centralized baseline: union of bank train/val/test splits")
    central_splits = {
        split_name: pd.concat([splits[split_name] for splits in bank_splits.values()], ignore_index=True)
        for split_name in ("train", "val", "test")
    }

    # Write partition outputs and validation report.
    out_dir = OUTPUT_ROOT / f"{HETEROGENEITY}_{N_CLIENTS}banks"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_partition_outputs(bank_splits, central_splits, meta_filtered, assignment, out_dir, central_dir)
    write_validation_report(out_dir, meta_filtered, assignment, bank_splits, n_cases_initial)

if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb  |  Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────