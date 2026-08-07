from __future__ import annotations
import importlib.util
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# HELPER: Load one frozen preprocessing module from its stage-numbered file path.
def load_module(relative_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module

# Multiclass outcome labels; test-only helper, kept out of the pipeline scripts.
def derive_outcomes(outcome_table: pd.DataFrame) -> pd.Series: return outcome_table["outcome"].astype("int64")

class BPIC2017DesignDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module(
            "E_main_BPIC_2017/A_02_preprocessing_and_partitioning_strat.py",
            "a02_preprocessing",
        )

    # Verify multiclass trace cutting and activity token targets.
    def test_multiclass_trace_cutting_and_activity_token_targets(self) -> None:
        df = pd.DataFrame(
            [
                ("c1", "A_Start", "start", "2020-01-01 00:00:00", 1000.0),
                ("c1", "W_Check", "complete", "2020-01-01 00:01:00", 1000.0),
                ("c1", "O_Accepted", "complete", "2020-01-01 00:02:00", 1000.0),
                ("c1", "A_After", "complete", "2020-01-01 00:03:00", 1000.0),
                ("c2", "A_Start", "start", "2020-01-02 00:00:00", 0.0),
                ("c2", "A_Cancelled", "complete", "2020-01-02 00:05:00", 0.0),
                ("c3", "A_Start", "start", "2020-01-03 00:00:00", 500.0),
                ("c4", "A_Start", "start", "2020-01-04 00:00:00", 700.0),
                ("c4", "A_Cancelled", "complete", "2020-01-04 00:01:00", 700.0),
                ("c4", "A_Denied", "complete", "2020-01-04 00:02:00", 700.0),
            ],
            columns=[
                "case:concept:name",
                "concept:name",
                "lifecycle:transition",
                "time:timestamp",
                "case:RequestedAmount",
            ],
        )
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)
        df["case:LoanGoal"] = "Unknown"
        df["case:ApplicationType"] = "New credit"

        df = self.module.add_activity_token(df)
        outcome_table = self.module.build_outcome_table(df)
        outcomes = derive_outcomes(outcome_table)
        cut_df, report = self.module.cut_traces_at_outcome(df, outcome_table)
        features = self.module.derive_event_features(cut_df)

        self.assertEqual(outcomes.to_dict(), {"c1": 2, "c2": 0, "c4": 1})
        self.assertNotIn("A_After", set(cut_df["concept:name"]))
        self.assertEqual(report["events_removed_after_outcome"], 1)
        self.assertEqual(report["cases_with_multiple_outcome_events"], 1)

        c1 = features.loc[features["case:concept:name"] == "c1"].reset_index(drop=True)
        self.assertEqual(c1.loc[0, "NextActivity"], "W_Check+complete")
        self.assertEqual(c1.loc[2, "NextActivity"], self.module.END_TOKEN)
        self.assertEqual(float(c1.loc[0, "RemainingTime"]), 120.0)
        self.assertEqual(float(c1.loc[2, "RemainingTime"]), 0.0)

class BPIC2012DesignDecisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = load_module(
            "E_ablation_BPIC_2012/B_02_preprocessing_and_partitioning_strat.py",
            "b02_preprocessing",
        )

    # Verify complete filter then multiclass activity token targets.
    def test_complete_filter_then_multiclass_activity_token_targets(self) -> None:
        df = pd.DataFrame(
            [
                ("b1", "A_SUBMITTED", "START", "2020-01-01 00:00:00", 1000.0),
                ("b1", "A_SUBMITTED", "COMPLETE", "2020-01-01 00:00:05", 1000.0),
                ("b1", "A_APPROVED", "COMPLETE", "2020-01-01 00:10:00", 1000.0),
                ("b1", "W_After", "COMPLETE", "2020-01-01 00:11:00", 1000.0),
                ("b2", "A_SUBMITTED", "COMPLETE", "2020-01-02 00:00:00", 0.0),
                ("b2", "A_CANCELLED", "COMPLETE", "2020-01-02 00:05:00", 0.0),
                ("b3", "A_SUBMITTED", "COMPLETE", "2020-01-03 00:00:00", 500.0),
            ],
            columns=[
                "case:concept:name",
                "concept:name",
                "lifecycle:transition",
                "time:timestamp",
                "case:AMOUNT_REQ",
            ],
        )
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True)

        df = self.module.filter_lifecycle(df)
        df = self.module.add_activity_token(df)
        outcome_table = self.module.build_outcome_table(df)
        outcomes = derive_outcomes(outcome_table)
        cut_df, report = self.module.cut_traces_at_outcome(df, outcome_table)
        features = self.module.derive_event_features(cut_df)

        self.assertEqual(outcomes.to_dict(), {"b1": 2, "b2": 0})
        self.assertTrue(features["activity_token"].str.endswith("+COMPLETE").all())
        self.assertNotIn("START", set(features["lifecycle:transition"]))
        self.assertNotIn("W_After", set(features["concept:name"]))
        self.assertEqual(report["events_removed_after_outcome"], 1)

        b1 = features.loc[features["case:concept:name"] == "b1"].reset_index(drop=True)
        self.assertEqual(b1.loc[0, "NextActivity"], "A_APPROVED+COMPLETE")
        self.assertEqual(b1.loc[1, "NextActivity"], self.module.END_TOKEN)
        self.assertEqual(float(b1.loc[0, "RemainingTime"]), 595.0)

    # Verify BPIC 2012 quintile weights match the multiclass spread design.
    def test_bpic2012_quintile_weights_match_multiclass_spread_design(self) -> None:
        expected_weak = np.array([
            [0.55, 0.30, 0.15],
            [0.48, 0.32, 0.20],
            [0.40, 0.35, 0.25],
            [0.32, 0.39, 0.29],
            [0.26, 0.34, 0.40],
        ])
        expected_medium = np.array([
            [0.85, 0.10, 0.05],
            [0.75, 0.15, 0.10],
            [0.45, 0.35, 0.20],
            [0.20, 0.40, 0.40],
            [0.10, 0.35, 0.55],
        ])

        np.testing.assert_allclose(self.module.WEAK_QUINTILE_WEIGHTS, expected_weak)
        np.testing.assert_allclose(self.module.MEDIUM_QUINTILE_WEIGHTS, expected_medium)
        np.testing.assert_allclose(self.module.WEAK_QUINTILE_WEIGHTS.sum(axis=1), np.ones(5))
        np.testing.assert_allclose(self.module.MEDIUM_QUINTILE_WEIGHTS.sum(axis=1), np.ones(5))
        self.assertTrue((self.module.WEAK_QUINTILE_WEIGHTS > 0).all())
        self.assertTrue((self.module.MEDIUM_QUINTILE_WEIGHTS > 0).all())

if __name__ == "__main__":
    unittest.main()