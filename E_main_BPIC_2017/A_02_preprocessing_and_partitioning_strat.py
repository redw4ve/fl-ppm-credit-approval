"""
Step 2 (BPIC2017 - main): Preprocessing and non-IID client partitioning on the case level.

Pipeline:
- Loads the unfiltered event log, derives multiclass outcome / next-activity / remaining-time labels.
- Cuts traces at the final decision event, so RemainingTime measures time-to-decision.
- Drops truncation cases, but keeps RequestedAmount=0 cases (routed per heterogeneity level).
- Partitions cases across simulated banks under one of {iid, weak, medium, strong}.
- Performs a temporal train/val/test split within each bank.
- Writes parquets per bank, a per-config centralized baseline and a metadata/validation report.

Heterogeneity design:
- weak / medium use a stratified amount-quintile soft-mix.
    - Every bank receives cases from each quintile, with proportions set by a graded skew.
    - Bank A tilts toward high amounts, while Bank C tilts toward low amounts.
- Strong = stratified medium base combined with bidirectional approval rate enforcement.
    - Undersample denials from Bank A, approvals from Bank C. Undersample cap = 0.25.

Run: python A_02_preprocessing_and_partitioning_strat.py

Configuration: constants block below (N_CLIENTS, HETEROGENEITY, etc.).
On the first run, parses BPI Challenge 2017.xes and writes the parquet cache.

Outputs:
- per-bank splits + metadata: data/processed/{HETEROGENEITY}_{N_CLIENTS}banks/
- centralized baseline per config: data/processed/centralized/{HETEROGENEITY}_{N_CLIENTS}banks/
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
import pm4py
from pm4py.objects.log.importer.xes.variants import iterparse

# CONFIGURATION
# WORKFLOW overrides environment variables HETEROGENEITY / N_CLIENTS (direct runs use the defaults).
N_CLIENTS: int = int(os.environ.get("N_CLIENTS", "3"))  # 3 or 5
HETEROGENEITY: str = os.environ.get("HETEROGENEITY", "medium")  # iid | weak | medium | strong
SPLIT_MODE: str = os.environ.get("SPLIT_MODE", "stratified_quintile")  # stratified_quintile | linear_quintile
TRAIN_VAL_TEST_RATIO: tuple[float, float, float] = (0.6, 0.2, 0.2)  # train, val, test distribution
MAX_PREFIX_LENGTH_FOR_ENCODING: int = 83  # p98 before outcome event cutting

# Outcome mapping (BPIC 2017).
APPROVED = "O_Accepted"
DENIED = "A_Denied"
CANCELLED = "A_Cancelled"
END_TOKEN = "[END]"
OUTCOME_EVENTS = (APPROVED, DENIED, CANCELLED)
OUTCOME_MAPPING = {APPROVED: 2, DENIED: 1, CANCELLED: 0}
BANK_NAMES = ("A", "B", "C", "D", "E")

# File Paths (resolved from the script directory).
SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_PATH: Path = SCRIPT_DIR / "data/cache/A_02_bpic2017_events.parquet"
XES_PATH: Path = SCRIPT_DIR / "BPI Challenge 2017/BPI Challenge 2017.xes"
OUTPUT_ROOT: Path = SCRIPT_DIR / "data/processed"

# LoanGoal pool membership.
# BANK B LoanGoals (Car, Home improvement, Existing loan takeover) keep tier received from the weak base.
LOANGOAL_HIGH_APPROVAL = ("Remaining debt home", "Boat", "Caravan / Camper", "Unknown")  # BANK A (medium/strong)
LOANGOAL_LOW_APPROVAL = ("Tax payments", "Not speficied", "Other, see explanation",  # BANK C (medium/strong)
                         "Extra spending limit", "Motorcycle", "Business goal")
LOANGOAL_SPECIALIST_D = "Home improvement"  # BANK D (speciality home)
LOANGOAL_SPECIALIST_E = "Existing loan takeover"  # BANK E (speciality takeover)

# Strong-config target approval-rate gaps (B-C is larger than A-B).
STRONG_AB_GAP_TARGET: float = 0.10  # Min. distance A-B
STRONG_BC_GAP_TARGET: float = 0.15  # Min. distance B-C
STRONG_UNDERSAMPLE_CAP: float = 0.25  # Max. cases dropped during rate enforcement
MEDIUM_BC_MARGIN: int = 0  # Min. case-count buffer B-C (0 = full LoanGoal LOW flow to C)
MEDIUM_AB_GAP_TARGET: float = 0.05  # Medium: target gap between A and B
MEDIUM_BC_GAP_TARGET: float = 0.07  # Medium: target gap between B and C
MEDIUM_UNDERSAMPLE_CAP: float = 0.08  # 3-Medium: Max. cases dropped
MEDIUM_UNDERSAMPLE_CAP_5BANK: float = 0.10  # 5-Medium: Higher max. cases dropped (D/E reduce the A/B/C case pools)

# Stratified bank weights for weak / medium per quintile.
# Rows sum to 1, all cells are > 0, column bias preserves |A|>|B|>|C| and aligns A with high and C with low amounts.
WEAK_QUINTILE_WEIGHTS = np.array([
    [0.30, 0.30, 0.40],  # Q1 (lowest amounts) -> heavier C
    [0.35, 0.30, 0.35],
    [0.40, 0.35, 0.25],
    [0.45, 0.40, 0.15],
    [0.50, 0.40, 0.10],  # Q5 (highest amounts) -> heavier A
])
MEDIUM_QUINTILE_WEIGHTS = np.array([
    [0.05, 0.30, 0.65],  # Q1 -> dominant C
    [0.20, 0.30, 0.50],
    [0.40, 0.40, 0.20],
    [0.55, 0.35, 0.10],
    [0.75, 0.20, 0.05],  # Q5 -> dominant A
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


# ----------------------------------------------------------------------------------------------------------------------
# 1. LOAD

# Parse BPIC 2017 XES and persist it as parquet.
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

    # Assert the correct Case and Event count.
    assert n_events == 1_202_267, f"unexpected event count: {n_events:,} (want 1,202,267)"
    assert n_cases == 31_509, f"unexpected case count: {n_cases:,} (want 31,509)"
    if not pd.api.types.is_datetime64_any_dtype(df["time:timestamp"]):
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
    return df


# ----------------------------------------------------------------------------------------------------------------------
# 2. CASE FEATURE DERIVATION

# Build activity tokens for next-activity prediction (activity + lifecycle).
def add_activity_token(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["activity_token"] = df["concept:name"].astype(str) + "+" + df["lifecycle:transition"].astype(str)
    return df


# Build one row per case from its last decision event (by the latest timestamp, event position as tie-breaker).
def build_outcome_table(df: pd.DataFrame) -> pd.DataFrame:
    # Preserve chronological order and use event position to break equal timestamps.
    ordered = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    ordered["_event_pos"] = ordered.groupby("case:concept:name", sort=False).cumcount()
    outcome_events = ordered.loc[
        ordered["concept:name"].isin(OUTCOME_EVENTS),
        ["case:concept:name", "concept:name", "time:timestamp", "_event_pos", "lifecycle:transition"],
    ].copy()

    if outcome_events.empty:
        return pd.DataFrame(columns=[
            "case_id", "outcome_event", "outcome_time", "outcome_pos", "outcome_lifecycle",
            "outcome", "n_outcome_events", "n_distinct_outcome_events",
        ]).set_index("case_id")

    # Count repeated decision events before selecting the final decision.
    counts = (outcome_events.groupby("case:concept:name")["concept:name"]
              .agg(n_outcome_events="size", n_distinct_outcome_events="nunique"))
    lifecycle_counts = outcome_events.groupby("concept:name")["lifecycle:transition"].nunique()
    multi_lifecycle = {str(k): int(v) for k, v in lifecycle_counts.items() if int(v) > 1}
    if multi_lifecycle: warn(f"outcome lifecycle: multiple lifecycle values observed for {multi_lifecycle}")

    # Use the last decision event per case as the prediction target.
    last_outcome = (
        outcome_events.sort_values(["case:concept:name", "time:timestamp", "_event_pos"])
        .groupby("case:concept:name", sort=False)
        .tail(1)
        .rename(columns={
            "case:concept:name": "case_id", "concept:name": "outcome_event", "time:timestamp": "outcome_time",
            "_event_pos": "outcome_pos", "lifecycle:transition": "outcome_lifecycle",
        })
        .set_index("case_id")
    )
    last_outcome["outcome"] = last_outcome["outcome_event"].map(OUTCOME_MAPPING).astype("int64")
    last_outcome = last_outcome.join(counts, how="left")

    # Log multiclass decision-label counts and repeated-label diagnostics.
    n_app = int((last_outcome["outcome"] == OUTCOME_MAPPING[APPROVED]).sum())
    n_den = int((last_outcome["outcome"] == OUTCOME_MAPPING[DENIED]).sum())
    n_can = int((last_outcome["outcome"] == OUTCOME_MAPPING[CANCELLED]).sum())
    n_multi = int((last_outcome["n_outcome_events"] > 1).sum())
    n_multi_distinct = int((last_outcome["n_distinct_outcome_events"] > 1).sum())
    n_no_outcome = df["case:concept:name"].nunique() - len(last_outcome)
    log.info(f"Outcome counts: O_Accepted={n_app:,}, A_Denied={n_den:,}, "
             f"A_Cancelled={n_can:,}, no outcome event={n_no_outcome:,}")
    log.info(f"Outcome diagnostics: {n_multi:,} case(s) with repeated decision events, "
             f"{n_multi_distinct:,} with multiple distinct decision labels")
    return last_outcome


# Remove administrative events after the final decision event.
def cut_traces_at_outcome(df: pd.DataFrame, outcome_table: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    # Drop every event recorded after the case decision. Cases without a decision pass through untouched.
    ordered = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    ordered["_event_pos"] = ordered.groupby("case:concept:name", sort=False).cumcount()
    ordered["_outcome_pos"] = ordered["case:concept:name"].map(outcome_table["outcome_pos"])
    outcome_case_mask = ordered["_outcome_pos"].notna()
    keep_mask = ~outcome_case_mask | (ordered["_event_pos"] <= ordered["_outcome_pos"])
    cut = ordered.loc[keep_mask].drop(columns=["_event_pos", "_outcome_pos"]).copy()

    # Record how many activities are removed post the outcome decision.
    events_in_outcome_cases = int(outcome_case_mask.sum())
    events_removed = int(keep_mask.size - keep_mask.sum())
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


# Add [END] activity target, RemainingTime (s), TimeDelta (s).
def derive_event_features(df: pd.DataFrame) -> pd.DataFrame:
    # Sort events chronologically within each case.
    df = df.sort_values(["case:concept:name", "time:timestamp"]).copy()
    if "activity_token" not in df.columns: df = add_activity_token(df)
    grp = df.groupby("case:concept:name", sort=False)

    # Set NextActivity to the following activity_token. The final event of each cut trace gets [END].
    df["NextActivity"] = grp["activity_token"].shift(-1).fillna(END_TOKEN)

    # Use the case end post-cut, which is the final decision timestamp.
    case_max = grp["time:timestamp"].transform("max")
    case_min = grp["time:timestamp"].transform("min")

    # Store raw time-to-decision in seconds. The prefix encoding owns optional log1p targets.
    df["RemainingTime"] = (case_max - df["time:timestamp"]).dt.total_seconds()
    df["TimeDelta"] = (df["time:timestamp"] - case_min).dt.total_seconds()
    return df


# Build one meta data row per case: Timing, size, static attributes, outcome, terminal event flag.
def build_case_metadata(df: pd.DataFrame, outcome_table: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("case:concept:name", sort=False)

    # Collect metadata for each case.
    meta = pd.DataFrame({
        "start_time": grp["time:timestamp"].min(),
        "end_time": grp["time:timestamp"].max(),
        "n_events": grp.size(),
        "requested_amount": grp["case:RequestedAmount"].first(),
        "loan_goal": grp["case:LoanGoal"].first(),
        "application_type": grp["case:ApplicationType"].first(),
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

# Drop truncation cases with no outcome event. Keep cases with RequestedAmount zero and canceled cases.
def filter_cases(meta: pd.DataFrame) -> pd.DataFrame:
    n0 = len(meta)
    log.info(f"Filtering: starting from {n0:,} cases")

    # Truncation cases are those without a decision event, so flip has_terminal_event with ~ to mark them.
    truncated_mask = ~meta["has_terminal_event"]
    n_trunc = int(truncated_mask.sum())

    # Keep only cases that do have an outcome event.
    meta = meta.loc[~truncated_mask].copy()
    log.info(f" 3.1 dropped {n_trunc:,} truncation cases")

    # Find cases where requested_amount = 0.
    n_zero = int((meta["requested_amount"] == 0).sum())
    log.info(f" 3.2 keeping {n_zero:,} zero-amount cases (routed per heterogeneity level)")

    # Keep canceled as class 0 in the multiclass outcome target.
    n_can = int((meta["outcome"] == OUTCOME_MAPPING[CANCELLED]).sum())
    log.info(f" 3.3 keeping {n_can:,} cancelled cases as outcome class 0")
    if meta["outcome"].isna().any(): raise ValueError("filter_cases: retained cases contain missing outcome labels")
    meta["outcome"] = meta["outcome"].astype(int)
    log.info(f"Filtered case count: {len(meta):,} (started {n0:,})")
    return meta.drop(columns=["has_terminal_event"])


# ----------------------------------------------------------------------------------------------------------------------
# 4. NON-IID CLIENT PARTITION

# Return the first n configured bank names.
def _bank_names(n_clients: int) -> list[str]: return list(BANK_NAMES[:n_clients])


# Append one structured entry to the partition provenance report.
def record_provenance_list(key: str, value: dict[str, Any]) -> None:
    entries = cast(list[dict[str, Any]], _PARTITION_PROVENANCE.setdefault(key, []))
    entries.append(value)


# Record how zero-amount cases were routed for the active heterogeneity level.
def record_zero_routing(heterogeneity: str, n: int, counts: dict[str, int]) -> None:
    record_provenance_list("zero_amount_routing", {
        "heterogeneity": heterogeneity,
        "n_zero_amount_cases": int(n),
        "counts": {bank: int(count) for bank, count in counts.items()},
    })


# Convert NumPy scalars to JSON save values before writing metadata.
def _json_scalar(value: object) -> object:
    if pd.isna(value): return None
    if isinstance(value, np.integer): return int(value)
    if isinstance(value, np.floating): return float(value)
    return value


# Summarize the assigned cases for metadata and validation.
def _partition_summary(meta: pd.DataFrame, assignment: dict[str, str]) -> list[dict[str, object]]:
    # Attach bank labels and ignore cases dropped by strong undersampling.
    df = meta.assign(_bank=meta["case_id"].map(assignment)).dropna(subset=["_bank"])
    rows: list[dict[str, object]] = []
    for bank, sub in df.groupby("_bank"):
        labelled = sub["outcome"]
        if labelled.isna().any(): raise ValueError(f"partition summary: Bank {bank} contains missing outcomes")
        top_goal = sub["loan_goal"].value_counts().idxmax() if len(sub) else None
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
            "n_loan_goals": int(sub["loan_goal"].nunique()),
            "top_loan_goal": _json_scalar(top_goal),
        })
    return sorted(rows, key=lambda row: str(row["bank"]))


# Index partition summary rows by bank name.
def _summary_by_bank(meta: pd.DataFrame, assignment: dict[str, str]) -> dict[str, dict[str, object]]:
    return {str(row["bank"]): row for row in _partition_summary(meta, assignment)}


# Return approval rate share per bank across all labeled decision outcomes.
def _per_bank_rates(meta: pd.DataFrame, assignment: dict[str, str]) -> dict[str, float]:
    df = meta.assign(_bank=meta["case_id"].map(assignment))
    df = df.dropna(subset=["_bank"])
    if df["outcome"].isna().any(): raise ValueError("_per_bank_rates: assigned cases contain missing outcomes")
    rates = df.groupby("_bank")["outcome"].apply(
        lambda values: float((values == OUTCOME_MAPPING[APPROVED]).sum() / len(values))
    )
    return rates.to_dict()


# IID random split: Shuffle and round-robin into N_CLIENTS banks (zeros included).
def _partition_iid(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    rng = np.random.default_rng(RANDOM_SEED)
    case_ids = meta["case_id"].to_numpy().copy()
    rng.shuffle(case_ids)
    splits = np.array_split(case_ids, n_clients)
    return {cid: bank for bank, ids in zip(_bank_names(n_clients), splits) for cid in ids}


# ZERO AMOUNT: Route requested_amount = 0 cases per heterogeneity level.
def _route_zero_amount_cases(meta: pd.DataFrame, heterogeneity: str, seed: int) -> dict[str, str]:
    # Get ID of all zero-requested amount cases and shuffle.
    zero_ids = meta.loc[meta["requested_amount"] == 0, "case_id"].tolist()
    n = len(zero_ids)
    if n == 0: return {}
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(zero_ids))

    #  WEAK: Equal A/B/C.
    if heterogeneity == "weak":
        chunks = np.array_split(shuffled, 3)
        counts = {bank: int(len(chunk)) for bank, chunk in zip(("A", "B", "C"), chunks)}
        record_zero_routing(heterogeneity, n, counts)
        log.info(f" zero-amount routing [weak]: equal split "
                 f"A={len(chunks[0]):,} B={len(chunks[1]):,} C={len(chunks[2]):,}")
        return {cid: bank for bank, chunk in zip(("A", "B", "C"), chunks) for cid in chunk}

    # MEDIUM: Split 50/25/25.
    if heterogeneity == "medium":
        n_a = int(round(0.50 * n))
        n_b = int(round(0.25 * n))
        n_c = n - n_a - n_b
        record_zero_routing(heterogeneity, n, {"A": n_a, "B": n_b, "C": n_c})
        log.info(f" zero-amount routing [medium]: 50/25/25 split A={n_a:,} B={n_b:,} C={n_c:,}")
        return {
            **{cid: "A" for cid in shuffled[:n_a]},
            **{cid: "B" for cid in shuffled[n_a:n_a + n_b]},
            **{cid: "C" for cid in shuffled[n_a + n_b:]},
        }

    # STRONG: Route 100% to A.
    if heterogeneity == "strong":
        record_zero_routing(heterogeneity, n, {"A": n, "B": 0, "C": 0})
        log.info(f" zero-amount routing [strong]: {n:,} cases -> Bank A")
        return {cid: "A" for cid in zero_ids}

    raise ValueError(f"_route_zero_amount_cases: unsupported heterogeneity '{heterogeneity}'")


# WEAK: Stratified amount-quintile soft-mix: Distributes cases across A/B/C according to one row of `weights`.
# Random assignment is seeded for reproducibility. Every bank receives each quintile, no all-low or all-high clients.
def _quintile_skew_assignment(meta: pd.DataFrame, weights: np.ndarray, seed: int) -> dict[str, str]:
    # Keep non-zero amounts and assign each case to a requested-amount quintile.
    nonzero_meta = meta.loc[meta["requested_amount"] > 0].copy()
    nonzero_meta["quintile"] = pd.qcut(
        nonzero_meta["requested_amount"], q=5, labels=False, duplicates="drop",
    ).astype(int)

    # Cut points used in the `pd.qcut` logged (amount range and case count) for each computed quintile.
    quintile_cuts = nonzero_meta.groupby("quintile")["requested_amount"].agg(["min", "max", "count"])
    log.info(" amount-quintile cuts (computed on >0 subset):")
    for q, row in quintile_cuts.iterrows():
        log.info(f"    Q{q + 1}: amount in [{row['min']:,.0f}, {row['max']:,.0f}]  n={int(row['count']):,}")

    # Shuffle cases within each quintile and split them into A/B/C according to the configured weights.
    rng = np.random.default_rng(seed)
    assignment: dict[str, str] = {}
    bank_letters = ["A", "B", "C"]
    for q in range(weights.shape[0]):
        q_cases = nonzero_meta.loc[nonzero_meta["quintile"] == q, "case_id"].to_numpy().copy()
        rng.shuffle(q_cases)
        n = len(q_cases)

        # Convert cumulative weights into split indices for this quintile.
        cuts = np.clip(np.round(np.cumsum(weights[q]) * n).astype(int), 0, n)
        chunks = {"A": q_cases[:cuts[0]], "B": q_cases[cuts[0]:cuts[1]], "C": q_cases[cuts[1]:], }

        # Store the bank assignment for each case.
        for bank in bank_letters:
            for cid in chunks[bank]:
                assignment[cid] = bank

    # Record the weights and seed used.
    record_provenance_list("quintile_skew", {
        "weights_per_quintile": weights.tolist(),
        "seed": int(seed),
        "n_nonzero_cases": int(len(nonzero_meta)),
    })
    return assignment


# LINEAR: Assign whole RequestedAmount quintiles to contiguous banks without random sampling. (not used)
def _linear_quintile_assignment(meta: pd.DataFrame, high_amount_bank: str) -> dict[str, str]:
    # Keep non-zero amounts and assign deterministic amount quintiles.
    nonzero_meta = meta.loc[meta["requested_amount"] > 0].copy()
    nonzero_meta = nonzero_meta.sort_values(["requested_amount", "case_id"]).copy()
    nonzero_meta["quintile"] = pd.qcut(
        nonzero_meta["requested_amount"], q=5, labels=False, duplicates="drop",
    ).astype(int)

    if high_amount_bank == "A":
        mapping = {0: "C", 1: "C", 2: "B", 3: "A", 4: "A"}
    elif high_amount_bank == "C":
        mapping = {0: "A", 1: "A", 2: "B", 3: "C", 4: "C"}
    else:
        raise ValueError(f"_linear_quintile_assignment: unsupported high_amount_bank '{high_amount_bank}'")

    # Map every case to the bank owning its full amount quintile.
    assignment = {
        cid: mapping[int(q)]
        for cid, q in zip(nonzero_meta["case_id"], nonzero_meta["quintile"])
    }
    record_provenance_list("linear_quintile", {
        "high_amount_bank": high_amount_bank,
        "n_nonzero_cases": int(len(nonzero_meta)),
        "quintile_to_bank": {str(k + 1): v for k, v in mapping.items()},
    })
    return assignment


# Dispatch the configured RequestedAmount split mode.
def _amount_quintile_assignment(meta: pd.DataFrame, weights: np.ndarray, seed: int) -> dict[str, str]:
    if SPLIT_MODE == "stratified_quintile": return _quintile_skew_assignment(meta, weights, seed)
    if SPLIT_MODE == "linear_quintile": return _linear_quintile_assignment(meta, high_amount_bank="A")
    raise ValueError(f"unknown SPLIT_MODE: '{SPLIT_MODE}'")


# MEDIUM: Stratified weight matrix base and bidirectional LoanGoal reassignments.
# This layered design produces stronger heterogeneity than weak partitioning alone.
def _amount_plus_loangoal_base(meta: pd.DataFrame) -> dict[str, str]:
    base = _amount_quintile_assignment(meta, MEDIUM_QUINTILE_WEIGHTS, RANDOM_SEED)
    nonzero_meta = meta.loc[meta["requested_amount"] > 0]
    high_set = set(LOANGOAL_HIGH_APPROVAL)
    low_set = set(LOANGOAL_LOW_APPROVAL)

    # Step 1: HIGH-approval LoanGoals reassigned to Bank A (no cap).
    moved_to_a = 0
    for cid, lg in zip(nonzero_meta["case_id"], nonzero_meta["loan_goal"]):
        if lg in high_set and base[cid] != "A":
            base[cid] = "A"
            moved_to_a += 1
    log.info(f"    HIGH-LoanGoal -> A: {moved_to_a:,} cases moved")

    # Step 2: LOW-approval LoanGoals reassigned to Bank C, capped to keep |C| < |B|.
    # Identify candidates.
    counts = Counter(base.values())
    low_candidates = [
        (cid, base[cid])
        for cid, lg in zip(nonzero_meta["case_id"], nonzero_meta["loan_goal"])
        if lg in low_set and base[cid] != "C"
    ]

    # Shuffle candidates so capped reassignment to Bank C is reproducible but random.
    rng = np.random.default_rng(RANDOM_SEED)
    order = rng.permutation(len(low_candidates))
    low_candidates = [low_candidates[i] for i in order]

    # Move low-approval candidates to Bank C while preserving the B/C size constraint.
    moved_to_c = 0
    skipped = 0
    for cid, origin in low_candidates:
        new_b = counts["B"] - (1 if origin == "B" else 0)
        new_c = counts["C"] + 1
        if new_c >= new_b - MEDIUM_BC_MARGIN:
            skipped += 1
            continue

        # Commit the reassignment and update running bank counts.
        base[cid] = "C"
        counts[origin] -= 1
        counts["C"] += 1
        moved_to_c += 1

    # Step 3: Log moves and store the LoanGoal move configuration and realized move counts.
    log.info(f" LOW-LoanGoal -> C: {moved_to_c:,}/{len(low_candidates):,} cases moved "
             f"({skipped:,} skipped to keep |C| < |B| - {MEDIUM_BC_MARGIN})")
    record_provenance_list("loangoal_moves", {
        "high_approval_goals": list(LOANGOAL_HIGH_APPROVAL),
        "low_approval_goals": list(LOANGOAL_LOW_APPROVAL),
        "moved_to_a": int(moved_to_a),
        "low_candidates": int(len(low_candidates)),
        "moved_to_c": int(moved_to_c),
        "skipped_to_preserve_size_margin": int(skipped),
        "medium_bc_margin": int(MEDIUM_BC_MARGIN),
    })
    return base


# WEAK: Stratified amount-quintile soft-mix on non-zero and zero routing per configuration.
def _partition_weak(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    if n_clients != 3: raise ValueError("weak heterogeneity requires N_CLIENTS=3")
    log.info(" weak: amount-quintile soft-mix (non-zero) + zero routing")
    base = _amount_quintile_assignment(meta, WEAK_QUINTILE_WEIGHTS, RANDOM_SEED)
    zero = _route_zero_amount_cases(meta, "weak", RANDOM_SEED)
    return {**base, **zero}


# HELPER: Drop cases with `drop_outcome` from `bank` until rate gap with ref_rate hits the target (rate enforcement).
def _enforce_rate_gap_by_dropping_outcome(meta: pd.DataFrame, assignment: dict[str, str], bank: str,
                                          drop_outcome: int, ref_rate: float, gap_target: float, direction: str,
                                          cap: float,
                                          rng: np.random.Generator, strategy: str = "strong") -> set[str]:
    # Isolate the target bank and calculate the maximum number of cases that may be dropped.
    bank_ids = [cid for cid, b in assignment.items() if b == bank]
    bank_meta = meta.set_index("case_id").loc[bank_ids]
    bank_size = len(bank_meta)
    cap_n = int(cap * bank_size)

    # Initialize approval rate counters for incremental updates during simulated dropping.
    n_labelled = int(bank_meta["outcome"].notna().sum())
    n_approved = int((bank_meta["outcome"] == OUTCOME_MAPPING[APPROVED]).sum())

    # Build a reproducibly shuffled pool of cases with the outcome targeted for dropping.
    candidate_ids = bank_meta.loc[bank_meta["outcome"] == drop_outcome].index.tolist()
    candidate_ids = list(rng.permutation(candidate_ids))
    dropped: set[str] = set()
    cap_hit = False
    target_hit = False

    for cid in candidate_ids:
        # Stop if the configured drop cap is reached.
        if len(dropped) >= cap_n:
            cap_hit = True
            break

        # Simulate dropping this case: Update running counts before committing.
        n_labelled -= 1
        if drop_outcome == OUTCOME_MAPPING[APPROVED]: n_approved -= 1
        if n_labelled <= 0: break
        new_rate = n_approved / n_labelled
        dropped.add(cid)

        # Stop once the approval rate gap reaches the requested direction and size.
        gap = new_rate - ref_rate
        if direction == "up" and gap >= gap_target: target_hit = True; break
        if direction == "down" and gap <= -gap_target: target_hit = True; break

    # Warn if we hit the cap without reaching the target (gap smaller than intended).
    if cap_hit and not target_hit:
        warn(f"{strategy}: {cap:.0%} cap hit on Bank {bank} at {len(dropped):,} cases "
             f"(target gap {gap_target:.3f} not reached)")
    return dropped


# MEDIUM: Amount + LoanGoal moves + light bidirectional rate enforcement.
# Apply weak rate-gap enforcement by dropping denials from A and approvals from C.
def _partition_medium(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    if n_clients != 3: raise ValueError("medium heterogeneity (3-bank entry point) requires N_CLIENTS=3")
    log.info(" medium: amount-quintile soft-mix + LoanGoal moves (non-zero)")

    # Build the medium assignment and capture pre-enforcement approval rates.
    base = _amount_plus_loangoal_base(meta)
    zero = _route_zero_amount_cases(meta, "medium", RANDOM_SEED)
    assignment = {**base, **zero}
    rates = _per_bank_rates(meta, assignment)
    base_rates = {str(b): float(r) for b, r in rates.items()}
    rng = np.random.default_rng(RANDOM_SEED)

    # Use a slightly higher cap in 5-bank runs (D/E reduce the A/B/C pools), see configuration.
    cap = MEDIUM_UNDERSAMPLE_CAP_5BANK if N_CLIENTS == 5 else MEDIUM_UNDERSAMPLE_CAP

    # Raise the approval rate of Bank A by dropping denied cases.
    dropped_a = _enforce_rate_gap_by_dropping_outcome(
        meta, assignment, bank="A", drop_outcome=OUTCOME_MAPPING[DENIED], ref_rate=rates["B"],
        gap_target=MEDIUM_AB_GAP_TARGET, direction="up", cap=cap, rng=rng, strategy="medium",
    )

    # Lower Bank C approval rate by dropping approved cases.
    dropped_c = _enforce_rate_gap_by_dropping_outcome(
        meta, assignment, bank="C", drop_outcome=OUTCOME_MAPPING[APPROVED], ref_rate=rates["B"],
        gap_target=MEDIUM_BC_GAP_TARGET, direction="down", cap=cap, rng=rng, strategy="medium",
    )

    # Remove all selected cases and recompute final approval rates.
    all_dropped = dropped_a | dropped_c
    final = {cid: b for cid, b in assignment.items() if cid not in all_dropped}
    final_rates = _per_bank_rates(meta, final)

    # Store the pre- and post-enforcement rates, targets, caps and drop counts.
    _PARTITION_PROVENANCE["medium_approval_enforcement"] = {
        "base_approval_rates": base_rates,
        "final_approval_rates": {str(b): float(r) for b, r in final_rates.items()},
        "target_ab_gap": float(MEDIUM_AB_GAP_TARGET),
        "target_bc_gap": float(MEDIUM_BC_GAP_TARGET),
        "final_ab_gap": float(final_rates["A"] - final_rates["B"]),
        "final_bc_gap": float(final_rates["B"] - final_rates["C"]),
        "undersample_cap_per_bank": float(cap),
        "dropped_from_a": {"outcome": "denied", "n_cases": int(len(dropped_a))},
        "dropped_from_c": {"outcome": "approved", "n_cases": int(len(dropped_c))},
        "n_cases_dropped_total": int(len(all_dropped)),
    }
    log.info(f" medium: dropped {len(dropped_a):,} denials from A, {len(dropped_c):,} approvals from C; "
             f"final spread {max(final_rates.values()) - min(final_rates.values()):.3f}")
    counts = Counter(final.values())
    log.info(f" medium: final sizes A={counts['A']:,} B={counts['B']:,} C={counts['C']:,}")
    return final


# STRONG: Medium IID partitioning as base + bidirectional approval-rate enforcement.
def _partition_strong(meta: pd.DataFrame, n_clients: int) -> dict[str, str]:
    # Build a strong base from amount-quintile skew and LoanGoal reassignments.
    if n_clients != 3: raise ValueError("strong heterogeneity (3-bank entry point) requires N_CLIENTS=3")
    log.info(" strong step 1: amount-quintile soft-mix + LoanGoal moves (non-zero)")
    assignment = _amount_plus_loangoal_base(meta)

    # Add strong requested_amount = 0 routing and capture pre-enforcement approval rates.
    log.info(" strong step 2: routing zero-amount cases (per strong rule)")
    zero = _route_zero_amount_cases(meta, "strong", RANDOM_SEED)
    assignment = {**assignment, **zero}
    rates = _per_bank_rates(meta, assignment)
    base_rates = {str(b): float(r) for b, r in rates.items()}

    # Log pre-enforcement approval-rate gaps and seed reproducible enforcement drops.
    log.info(f" strong step 3 (base + zero routing) rates: { {b: round(r, 3) for b, r in rates.items()} } "
             f"(A-B={rates['A'] - rates['B']:+.3f}, B-C={rates['B'] - rates['C']:+.3f})")
    rng = np.random.default_rng(RANDOM_SEED)

    # Phase 1: Drop denials from A, capped by STRONG_UNDERSAMPLE_CAP -> raise A above B.
    dropped_a = _enforce_rate_gap_by_dropping_outcome(
        meta, assignment, bank="A", drop_outcome=OUTCOME_MAPPING[DENIED], ref_rate=rates["B"],
        gap_target=STRONG_AB_GAP_TARGET, direction="up", cap=STRONG_UNDERSAMPLE_CAP, rng=rng,
    )
    # Phase 2: Drop approvals from C, capped by STRONG_UNDERSAMPLE_CAP -> lower C below B.
    dropped_c = _enforce_rate_gap_by_dropping_outcome(
        meta, assignment, bank="C", drop_outcome=OUTCOME_MAPPING[APPROVED], ref_rate=rates["B"],
        gap_target=STRONG_BC_GAP_TARGET, direction="down", cap=STRONG_UNDERSAMPLE_CAP, rng=rng,
    )

    # Remove all cases selected by strong enforcement.
    all_dropped = dropped_a | dropped_c
    final_assignment = {cid: b for cid, b in assignment.items() if cid not in all_dropped}
    final_rates = _per_bank_rates(meta, final_assignment)

    # Log and store strong enforcement targets, realized gaps and drop counts.
    _PARTITION_PROVENANCE["strong_approval_enforcement"] = {
        "base_approval_rates": base_rates,
        "final_approval_rates": {str(b): float(r) for b, r in final_rates.items()},
        "target_ab_gap": float(STRONG_AB_GAP_TARGET),
        "target_bc_gap": float(STRONG_BC_GAP_TARGET),
        "target_total_ac_gap": float(STRONG_AB_GAP_TARGET + STRONG_BC_GAP_TARGET),
        "final_ab_gap": float(final_rates["A"] - final_rates["B"]),
        "final_bc_gap": float(final_rates["B"] - final_rates["C"]),
        "final_total_ac_gap": float(final_rates["A"] - final_rates["C"]),
        "undersample_cap_per_bank": float(STRONG_UNDERSAMPLE_CAP),
        "dropped_from_a": {"outcome": "denied", "n_cases": int(len(dropped_a))},
        "dropped_from_c": {"outcome": "approved", "n_cases": int(len(dropped_c))},
        "n_cases_dropped_total": int(len(all_dropped)),
    }

    log.info(f" strong step 4a: dropped {len(dropped_a):,} denials from Bank A")
    log.info(f" strong step 4b: dropped {len(dropped_c):,} approvals from Bank C")
    log.info(f" strong final rates: { {b: round(r, 3) for b, r in final_rates.items()} } "
             f"(A-B={final_rates['A'] - final_rates['B']:+.3f}, "
             f"B-C={final_rates['B'] - final_rates['C']:+.3f})")
    return final_assignment


# Pull Banks D and E specialist LoanGoal cases out of the filtered meta-table for 5-bank configs.
def _carve_specialists(meta: pd.DataFrame) -> tuple[dict[str, str], pd.DataFrame]:
    # Identify cases assigned to the D/E specialist LoanGoals.
    d_mask = meta["loan_goal"] == LOANGOAL_SPECIALIST_D
    e_mask = meta["loan_goal"] == LOANGOAL_SPECIALIST_E

    # Route specialist cases directly to Banks D and E.
    de_assign = {cid: "D" for cid in meta.loc[d_mask, "case_id"]}
    de_assign.update({cid: "E" for cid in meta.loc[e_mask, "case_id"]})

    # Store specialist bank definitions and realized case counts.
    _PARTITION_PROVENANCE["specialist_banks"] = {
        "D": {"loan_goal": LOANGOAL_SPECIALIST_D, "n_cases": int(d_mask.sum())},
        "E": {"loan_goal": LOANGOAL_SPECIALIST_E, "n_cases": int(e_mask.sum())},
    }
    log.info(f" extended: carved {int(d_mask.sum()):,} cases into Bank D and {int(e_mask.sum()):,} into Bank E")
    return de_assign, meta.loc[~(d_mask | e_mask)].copy()


# Dispatch to the partitioning strategy implied by HETEROGENEITY / N_CLIENTS.
def partition_cases(meta: pd.DataFrame) -> dict[str, str]:
    # Start a fresh provenance record for this partitioning run.
    log.info(f"Partitioning: strategy='{HETEROGENEITY}', N_CLIENTS={N_CLIENTS}")
    _PARTITION_PROVENANCE.clear()
    _PARTITION_PROVENANCE["strategy"] = HETEROGENEITY
    _PARTITION_PROVENANCE["n_clients"] = int(N_CLIENTS)
    _PARTITION_PROVENANCE["split_mode"] = SPLIT_MODE
    _PARTITION_PROVENANCE["n_cases_available_after_filters"] = int(len(meta))

    # Store a compact description of the selected partitioning strategy.
    _PARTITION_PROVENANCE["criteria"] = {
        "iid": "random equal-size split",
        "weak": f"{SPLIT_MODE} RequestedAmount-quintile split plus config-dependent zero-amount routing",
        "medium": f"{SPLIT_MODE} RequestedAmount-quintile split, LoanGoal moves and config-dependent zero-amount routing",
        "strong": "medium criteria plus capped approval-rate enforcement",
    }.get(HETEROGENEITY, "unknown")

    # IID: Has no additional heterogeneity logic.
    if HETEROGENEITY == "iid": return _partition_iid(meta, N_CLIENTS)

    # 5-BANK: Carve D/E specialists first, then partition the remaining A/B/C pool. (WEAK skipped, no LoanGoal).
    if N_CLIENTS == 5 and HETEROGENEITY in ("medium", "strong"):
        de_assign, remaining = _carve_specialists(meta)
        if HETEROGENEITY == "medium":
            abc_assign = _partition_medium(remaining, n_clients=3)
        else:
            abc_assign = _partition_strong(remaining, n_clients=3)
        return {**de_assign, **abc_assign}

    # 3-BANK: Standard path, dispatches directly to the selected heterogeneity.
    strategies = {
        "weak": _partition_weak,
        "medium": _partition_medium,
        "strong": _partition_strong,
    }
    if HETEROGENEITY in strategies: return strategies[HETEROGENEITY](meta, N_CLIENTS)
    raise ValueError(f"unknown HETEROGENEITY: '{HETEROGENEITY}'")


# Sanity-check the partition: Known case IDs only, no bank is empty.
def assert_partition_valid(assignment: dict[str, str], meta: pd.DataFrame) -> None:
    case_ids = set(meta["case_id"])
    assigned = set(assignment.keys())

    # No case in the assignment should be unknown to meta.
    assert assigned.issubset(case_ids), f"partition contains {len(assigned - case_ids):,} unknown case IDs"

    # Count cases per bank and assert none is empty.
    bank_counts = dict(Counter(str(bank) for bank in assignment.values()))
    for b, n in bank_counts.items(): assert n > 0, f"Bank {b} has zero cases (R1 violation)"

    # Record intentionally dropped cases under rate-enforcement.
    n_dropped = len(case_ids) - len(assigned)
    if n_dropped > 0:
        log.info(f" partition assertion: {n_dropped:,} case(s) dropped by rate enforcement")
    _PARTITION_PROVENANCE["n_cases_assigned"] = int(len(assigned))
    _PARTITION_PROVENANCE["n_cases_unassigned_or_dropped"] = int(n_dropped)
    _PARTITION_PROVENANCE["bank_case_counts"] = {str(b): int(n) for b, n in bank_counts.items()}
    log.info(f" per-bank case counts: {bank_counts}")


# Log case count per bank, approval rate, mean RequestedAmount, LoanGoal diversity.
def log_partition_stats(meta: pd.DataFrame, assignment: dict[str, str]) -> None:
    # Attach the bank label to each case and drop unassigned rows (rate-enforcement drops).
    df = meta.assign(_bank=meta["case_id"].map(assignment)).dropna(subset=["_bank"])
    log.info("Per-bank statistics:")
    for bank, sub in df.groupby("_bank"):
        labelled = sub["outcome"]
        if labelled.isna().any(): raise ValueError(f"log_partition_stats: Bank {bank} contains missing outcomes")
        rate = float((labelled == OUTCOME_MAPPING[APPROVED]).sum() / len(labelled))
        mean_amt = sub["requested_amount"].mean()
        n_goals = sub["loan_goal"].nunique()
        log.info(f" Bank {bank}: n_cases={len(sub):,} approval={rate:.3f}  "
                 f"mean_requested_amount={mean_amt:,.0f} n_loan_goals={n_goals}")


# ----------------------------------------------------------------------------------------------------------------------
# 5. TEMPORAL TRAIN/VAL/TEST SPLIT WITHIN EACH BANK

# Slice a chronologically sorted case list 60/20/20 (per TRAIN_VAL_TEST_RATIO).
def temporal_split(case_ids_sorted: list[str]) -> tuple[list[str], list[str], list[str]]:
    n = len(case_ids_sorted)
    n_train = int(round(n * TRAIN_VAL_TEST_RATIO[0]))
    n_val = int(round(n * TRAIN_VAL_TEST_RATIO[1]))
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

    # Return event rows for each split on the case level.
    split_ids = {"train": train_ids, "val": val_ids, "test": test_ids}
    return {name: events.loc[events["case:concept:name"].isin(set(ids))].copy() for name, ids in split_ids.items()}


# ----------------------------------------------------------------------------------------------------------------------
# 6. SAVE OUTPUTS

# Collect the run configuration for metadata.json.
def config_metadata() -> dict[str, object]:
    return {
        "DATASET": "BPIC 2017",
        "N_CLIENTS": N_CLIENTS,
        "HETEROGENEITY": HETEROGENEITY,
        "SPLIT_MODE": SPLIT_MODE,
        "RANDOM_SEED": RANDOM_SEED,
        "TRAIN_VAL_TEST_RATIO": list(TRAIN_VAL_TEST_RATIO),
        "OUTCOME_MAPPING": OUTCOME_MAPPING,
        "REMAINING_TIME_UNIT": "raw_seconds_to_decision",
        "REMAINING_TIME_LOG_TRANSFORM_STAGE": "prefix_encoding",
        "MAX_PREFIX_LENGTH_FOR_ENCODING": MAX_PREFIX_LENGTH_FOR_ENCODING,
        "STRONG_AB_GAP_TARGET": STRONG_AB_GAP_TARGET,
        "STRONG_BC_GAP_TARGET": STRONG_BC_GAP_TARGET,
        "STRONG_UNDERSAMPLE_CAP": STRONG_UNDERSAMPLE_CAP,
        "MEDIUM_BC_MARGIN": MEDIUM_BC_MARGIN,
    }


# Write parquets, partition_stats.csv, metadata.json and centralized splits per bank.
def write_partition_outputs(bank_splits: dict[str, dict[str, pd.DataFrame]], central_splits: dict[str, pd.DataFrame],
                            meta: pd.DataFrame, assignment: dict[str, str], out_dir: Path, central_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    central_dir.mkdir(parents=True, exist_ok=True)

    # Write train, validation and test parquets for each bank.
    for bank, splits in bank_splits.items():
        for split_name, frame in splits.items():
            frame.to_parquet(out_dir / f"A_02_bank_{bank}_{split_name}.parquet", index=False)

    # Write one human-readable partition summary table.
    summary_by_bank = _summary_by_bank(meta, assignment)
    rows = [
        {
            **summary_by_bank[str(bank)],
            "n_events": int(sum(len(split) for split in splits.values())),
        }
        for bank, splits in bank_splits.items()
    ]
    pd.DataFrame(rows).to_csv(out_dir / "A_02_partition_stats.csv", index=False)

    # Persist configuration, warnings and partition provenance.
    _PARTITION_PROVENANCE["per_bank_summary"] = list(summary_by_bank.values())
    metadata = {
        "config": config_metadata(),
        "partition_provenance": _PARTITION_PROVENANCE,
        "trace_cut_report": _TRACE_CUT_REPORT,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(_WARNINGS),
    }
    with (out_dir / "A_02_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Write the per-config centralized splits.
    for split_name, frame in central_splits.items():
        frame.to_parquet(central_dir / f"A_02_{split_name}.parquet", index=False)


# ----------------------------------------------------------------------------------------------------------------------
# 7. VALIDATION REPORT

# Format one validation report row with case counts, approval rate and split event counts.
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


# Print the final summary and append it to {out_dir}/run.log.
def write_validation_report(out_dir: Path, meta: pd.DataFrame, assignment: dict[str, str],
                            bank_splits: dict[str, dict[str, pd.DataFrame]], n_cases_initial: int) -> None:
    lines: list[str] = [
        "-" * 78,
        f"VALIDATION REPORT  ({datetime.now(timezone.utc).isoformat()})",
        "-" * 78,
        "Configuration:",
        f"  HETEROGENEITY        = {HETEROGENEITY}",
        f"  N_CLIENTS            = {N_CLIENTS}",
        f"  SPLIT_MODE           = {SPLIT_MODE}",
        f"  RANDOM_SEED          = {RANDOM_SEED}",
        f"  TRAIN_VAL_TEST_RATIO = {TRAIN_VAL_TEST_RATIO}",
        f"  RemainingTime        = raw_seconds_to_decision",
        f"  RT log transform     = prefix_encoding",
        "",
    ]
    # Report case accounting, including rate-enforcement drops.
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
    else:
        lines.append("Warnings: none")
    lines.append("-" * 78)

    # Print the report and write the run log for the current configuration.
    text = "\n".join(lines)
    print(text)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "A_02_run.log").open("w") as f:
        f.write(text + "\n")


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FLOW

def main() -> None:
    # Load the raw event log and record the original case count.
    df = load_event_log(INPUT_PATH)
    n_cases_initial = df["case:concept:name"].nunique()

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

    # Attach the case outcome to every event, so Step 3 reads it from one parquet.
    df_filtered = df_filtered.merge(
        meta_filtered[["case_id", "outcome"]].rename(columns={"case_id": "case:concept:name"}),
        on="case:concept:name", how="left",
    ).sort_values(["case:concept:name", "time:timestamp"])

    # Partition retained cases into banks and validate the resulting assignment.
    assignment = partition_cases(meta_filtered)
    assert_partition_valid(assignment, meta_filtered)
    log_partition_stats(meta_filtered, assignment)

    # Remove cases intentionally dropped by rate enforcement from FL bank splits.
    kept_bank_ids = set(assignment.keys())
    meta_for_banks = meta_filtered.loc[meta_filtered["case_id"].isin(kept_bank_ids)].copy()
    df_for_banks = df_filtered.loc[df_filtered["case:concept:name"].isin(kept_bank_ids)].copy()

    # Build temporal splits for each simulated bank.
    log.info("Temporal train/val/test split per bank:")
    bank_splits = {
        bank: split_bank(bank, [c for c, b in assignment.items() if b == bank], meta_for_banks, df_for_banks)
        for bank in sorted(set(assignment.values()))
    }

    # Build the centralized baseline as the union of already-created bank splits.
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


# RUN
if __name__ == "__main__":
    main()

# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
#                          TUM · zeb  |  Cornelius Weiss · M.Sc. Wirtschaftsinformatik · 2026
# ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────