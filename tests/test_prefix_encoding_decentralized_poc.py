from __future__ import annotations
import importlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import pandas as pd

poc = importlib.import_module("E_prefix_encoding.04_7_decentralized_metadata_poc")
encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

class PrefixEncodingDecentralizedPocTests(unittest.TestCase):
    # Verify numeric statistics aggregate to global scaler.
    def test_numeric_statistics_aggregate_to_global_scaler(self) -> None:
        stats = [
            {"count": 2, "sum": 4.0, "sum_squared": 10.0},
            {"count": 1, "sum": 5.0, "sum_squared": 25.0},
        ]

        scaler = poc.aggregate_numeric_statistics(stats)

        self.assertAlmostEqual(scaler["mean"], 3.0)
        self.assertAlmostEqual(scaler["std"], math.sqrt((35.0 / 3.0) - 9.0))

    # Verify the RT zscore target reconstructed from additive stats.
    def test_remaining_time_zscore_target_reconstructs_from_additive_stats(self) -> None:
        local_stats = [
            {
                "remaining_time_target_statistics": {
                    "transform": "raw",
                    "scaling": "zscore",
                    "count": 2,
                    "sum": 4.0,
                    "sum_squared": 10.0,
                    "median": 2.0,
                    "median_seconds": 2.0,
                }
            },
            {
                "remaining_time_target_statistics": {
                    "transform": "raw",
                    "scaling": "zscore",
                    "count": 1,
                    "sum": 5.0,
                    "sum_squared": 25.0,
                    "median": 5.0,
                    "median_seconds": 5.0,
                }
            },
        ]

        target_repr = poc.remaining_time_target_from_local_stats(local_stats)

        self.assertEqual(target_repr["transform"], "raw")
        self.assertEqual(target_repr["scaling"], "zscore")
        self.assertAlmostEqual(target_repr["center"], 3.0)
        self.assertAlmostEqual(target_repr["scale"], math.sqrt((35.0 / 3.0) - 9.0))
        self.assertFalse(target_repr["use_softplus"])

    # Verify remaining time target statistics ignore zero targets for fitting.
    def test_remaining_time_target_statistics_ignore_zero_targets_for_fitting(self) -> None:
        events = pd.DataFrame(
            {
                encoding.DATASET_ID: ["bpic2012", "bpic2012", "bpic2012"],
                encoding.CLIENT_ID: ["A", "A", "A"],
                encoding.SPLIT: ["train", "train", "train"],
                encoding.CASE_ID: ["case_1", "case_1", "case_1"],
                encoding.EVENT_INDEX: [0, 1, 2],
                encoding.REMAINING_TIME: [10.0, 0.0, 0.0],
            }
        )
        prefix_index = [
            encoding.PrefixIndexRow(dataset_id="bpic2012", case_id="case_1", client_id="A", split="train",
                                    prefix_length=1, label_pos=0),
            encoding.PrefixIndexRow(dataset_id="bpic2012", case_id="case_1", client_id="A", split="train",
                                    prefix_length=2, label_pos=1),
            encoding.PrefixIndexRow(dataset_id="bpic2012", case_id="case_1", client_id="A", split="train",
                                    prefix_length=3, label_pos=2),
        ]

        stats = poc.remaining_time_target_statistics_from_events(events, prefix_index, "raw", "zscore")
        self.assertEqual(stats["count"], 1)
        self.assertEqual(stats["sum"], 10.0)

    # Verify vocabulary from counts preserves reserved indices.
    def test_vocabulary_from_counts_preserves_reserved_indices(self) -> None:
        counts = {"z_token": 1, "a_token": 2, encoding.PAD_TOKEN: 10}
        vocab = poc.build_vocabulary_from_counts(counts)

        self.assertEqual(vocab[encoding.PAD_TOKEN], 0)
        self.assertEqual(vocab[encoding.UNK_TOKEN], 1)
        self.assertEqual(vocab[encoding.MISSING_TOKEN], 2)
        self.assertLess(vocab["a_token"], vocab["z_token"])

    # Verify local stats use train categories only and hide rows.
    def test_local_stats_use_train_categories_only_and_hide_rows(self) -> None:
        frame = pd.DataFrame(
            {
                encoding.DATASET_ID: ["bpic2017", "bpic2017", "bpic2017"],
                encoding.CLIENT_ID: ["A", "A", "A"],
                encoding.SPLIT: ["train", "train", "val"],
                encoding.CASE_ID: ["case_secret_1", "case_secret_1", "case_secret_2"],
                encoding.EVENT_INDEX: [0, 1, 0],
                encoding.CANONICAL_ACTIVITY_TOKEN:
                    ["A_create_application+complete", "O_accept_offer+complete", "val_only"],
                encoding.NEXT_ACTIVITY_TARGET: ["O_accept_offer+complete", encoding.END_TOKEN, encoding.END_TOKEN],
                encoding.RESOURCE: ["user_train", "user_train", "user_val_only"],
                encoding.TIME_DELTA: [0.0, 60.0, 0.0],
                encoding.REQUESTED_AMOUNT_VALUE: [1000.0, 1000.0, 2000.0],
                encoding.REQUESTED_AMOUNT_MASK: [1, 1, 1],
                encoding.OFFER_FEATURE_MASK: [0, 0, 0],
                encoding.OUTCOME: [2, 2, 1],
                encoding.REMAINING_TIME: [60.0, 0.0, 0.0],
                encoding.REMAINING_TIME_MASK: [1, 0, 0],
                encoding.NEXT_ACTIVITY_MASK: [1, 1, 1],
            }
        )

        stats = poc.collect_local_stats_from_events(
            frame, bank_id="A", run_name="iid_3banks", dataset_id="bpic2017",
            categorical_columns=[encoding.CANONICAL_ACTIVITY_TOKEN, encoding.RESOURCE],
            numerical_columns=[encoding.TIME_DELTA, encoding.REQUESTED_AMOUNT_VALUE],
            offer_columns=[], max_prefix_length=83,
        )
        serialized = json.dumps(stats, sort_keys=True)

        self.assertIn("A_create_application+complete",
                      stats["categorical_train_counts"][encoding.CANONICAL_ACTIVITY_TOKEN])
        self.assertNotIn("val_only", stats["categorical_train_counts"][encoding.CANONICAL_ACTIVITY_TOKEN])
        self.assertIn("user_train", stats["categorical_train_counts"][encoding.RESOURCE])
        self.assertNotIn("user_val_only", stats["categorical_train_counts"][encoding.RESOURCE])
        self.assertIn("remaining_time_target_statistics", stats)
        self.assertEqual(stats["remaining_time_target_statistics"]["transform"], "raw")
        self.assertEqual(stats["remaining_time_target_statistics"]["scaling"], "zscore")
        self.assertNotIn("case_secret_1", serialized)

    # Verify compare aggregated to central detects matching metadata.
    def test_compare_aggregated_to_central_detects_matching_metadata(self) -> None:
        aggregated = {
            "counts": {"aggregate": {"case_count": 2, "event_count": 3, "prefix_count": 3}},
            "train_outcome_counts": {"0": 1, "2": 1},
            "vocabularies": {"x": {encoding.PAD_TOKEN: 0, encoding.UNK_TOKEN: 1}},
            "scalers": {"n": {"mean": 2.0, "std": 1.5}},
            "target_scalers":
                {"remaining_time": {"transform": "raw", "scaling": "zscore", "center": 2.0, "scale": 1.5}},
            "prefix": {"cap": 83, "static_padding_length": 83},
        }
        central_spec = {
            "counts": {
                "aggregate": {"case_count": 2, "event_count": 3, "prefix_count": 3},
                "train_outcome_counts": {"0": 1, "2": 1},
            },
            "target_scalers":
                {"remaining_time": {"transform": "raw", "scaling": "zscore", "center": 2.0, "scale": 1.5}},
            "prefix": {"cap": 83, "static_padding_length": 83},
        }
        central_vocab = {"x": {encoding.PAD_TOKEN: 0, encoding.UNK_TOKEN: 1}}
        central_scaler = {"n": {"mean": 2.0, "std": 1.5}}

        report = poc.compare_aggregated_to_central(aggregated, central_spec, central_vocab, central_scaler)

        self.assertTrue(report["matches_central_metadata"])
        self.assertTrue(report["checks"]["counts_match"])
        self.assertEqual(report["differences"], [])

    # Verify masked messages recover global additive payload.
    def test_masked_messages_recover_global_additive_payload(self) -> None:
        local_stats = [
            {
                "bank": "A",
                "counts": {"case_count": 2, "event_count": 4, "prefix_count": 4},
                "train_outcome_counts": {"0": 1},
                "categorical_train_counts": {"x": {"a": 2}},
                "numeric_train_statistics": {"n": {"count": 2, "sum": 5.0, "sum_squared": 13.0}},
            },
            {
                "bank": "B",
                "counts": {"case_count": 1, "event_count": 3, "prefix_count": 3},
                "train_outcome_counts": {"1": 1},
                "categorical_train_counts": {"x": {"b": 3}},
                "numeric_train_statistics": {"n": {"count": 1, "sum": 4.0, "sum_squared": 16.0}},
            },
        ]

        messages = poc.build_masked_client_messages(local_stats, seed=7)
        recovered = poc.aggregate_masked_messages(messages)
        expected = poc.aggregate_additive_payloads([
            poc.local_stats_to_additive_payload(stats)
            for stats in local_stats
        ])

        self.assertEqual(recovered, expected)
        self.assertNotEqual(messages[0]["masked_payload"]["counts"]["case_count"], 2)

    # Verify the secure aggregation path matches plain aggregation.
    def test_secure_aggregation_path_matches_plain_aggregation(self) -> None:
        local_stats = [
            {
                "bank": "A",
                "counts": {"case_count": 2, "event_count": 4, "prefix_count": 4, "observed_max_trace_length": 2},
                "train_outcome_counts": {"0": 1},
                "categorical_train_counts": {encoding.CANONICAL_ACTIVITY_TOKEN: {"a": 2},
                                             encoding.NEXT_ACTIVITY_TARGET: {encoding.END_TOKEN: 1}},
                "numeric_train_statistics": {"n": {"count": 2, "sum": 5.0, "sum_squared": 13.0}},
            },
            {
                "bank": "B",
                "counts": {"case_count": 1, "event_count": 3, "prefix_count": 3, "observed_max_trace_length": 3},
                "train_outcome_counts": {"1": 1},
                "categorical_train_counts": {encoding.CANONICAL_ACTIVITY_TOKEN: {"b": 3},
                                             encoding.NEXT_ACTIVITY_TARGET: {"b": 1}},
                "numeric_train_statistics": {"n": {"count": 1, "sum": 4.0, "sum_squared": 16.0}},
            },
        ]

        plain, plain_messages = poc.aggregate_local_stats(
            local_stats, "bpic2017", "iid_3banks", [encoding.CANONICAL_ACTIVITY_TOKEN], 83,
            use_secure_aggregation=False, seed=11,
        )
        secure, messages = poc.aggregate_local_stats(
            local_stats, "bpic2017", "iid_3banks", [encoding.CANONICAL_ACTIVITY_TOKEN], 83,
            use_secure_aggregation=True, seed=11,
        )

        self.assertEqual(plain_messages, [])
        self.assertEqual(secure["counts"], plain["counts"])
        self.assertEqual(secure["vocabularies"], plain["vocabularies"])
        self.assertEqual(secure["scalers"], plain["scalers"])
        self.assertTrue(secure["secure_aggregation"])
        self.assertTrue(messages)

    # Verify write POC outputs creates three report groups.
    def test_write_poc_outputs_creates_three_report_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_stats = [{"bank": "A", "counts": {"case_count": 1}}]
            aggregated = {"counts": {"aggregate": {"case_count": 1}}}
            comparison = {"matches_central_metadata": True}
            paths = poc.write_poc_outputs(root, "bpic2017", "iid_3banks", local_stats, aggregated, comparison)

            self.assertEqual(len(paths), 3)
            for path in paths:
                self.assertTrue(path.exists())

    # Verify write POC outputs can save masked messages.
    def test_write_poc_outputs_can_save_masked_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            local_stats = [{"bank": "A", "counts": {"case_count": 1}}]
            aggregated = {"counts": {"aggregate": {"case_count": 1}}}
            comparison = {"matches_central_metadata": True}
            masked_messages = [{"bank": "A", "masked_payload": {"counts": {"case_count": 9.0}}}]

            paths = poc.write_poc_outputs(root, "bpic2017", "iid_3banks", local_stats, aggregated, comparison,
                                          masked_messages)

            self.assertEqual(len(paths), 4)
            message_path = root / "secure_aggregation_messages" / "bpic2017" / "iid_3banks.json"
            payload = json.loads(message_path.read_text(encoding="utf-8"))
            self.assertTrue(message_path.exists())
            self.assertIn("bank_A", payload)

    # Verify build POC summary collects run and artifact counts.
    def test_build_poc_summary_collects_run_and_artifact_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            records = [
                {
                    "dataset": "bpic2017",
                    "run": "iid_3banks",
                    "matches_central_metadata": True,
                    "checks": {"counts_match": True, "vocabularies_match": True},
                    "differences": [],
                    "secure_enabled": True,
                    "bank_count": 2,
                    "aggregate_counts": {"case_count": 2, "event_count": 5, "prefix_count": 5},
                }
            ]
            summary = poc.build_poc_summary(records)

            self.assertEqual(summary["run_summary"]["total_runs"], 1)
            self.assertEqual(summary["run_summary"]["matched_runs"], 1)
            self.assertEqual(summary["datasets"]["bpic2017"]["local_client_reports"], 2)
            self.assertTrue(summary["secure_aggregation_summary"]["enabled_in_all_runs"])
            self.assertTrue(summary["datasets"]["bpic2017"]["all_runs_match_central_metadata"])

    # Verify write POC summary creates JSON report.
    def test_write_poc_summary_creates_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = [
                {
                    "dataset": "bpic2012",
                    "run": "weak_3banks",
                    "matches_central_metadata": True,
                    "checks": {"counts_match": True},
                    "differences": [],
                    "secure_enabled": True,
                    "bank_count": 1,
                    "aggregate_counts": {"case_count": 1, "event_count": 2, "prefix_count": 2},
                }
            ]

            path = poc.write_poc_summary(root, records)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertEqual(path.name, "04_07_DECENTRALIZED_poc_summary.json")
            self.assertTrue(path.exists())
            self.assertEqual(payload["run_summary"]["matched_runs"], 1)

    # Verify joint POC lists source clients without joint discovery.
    def test_joint_poc_enumerates_source_clients_without_joint_discovery(self) -> None:
        config = SimpleNamespace(
            dataset="joint", heterogeneity="iid", n_clients=6, max_prefix_length_for_encoding=83,
            remaining_time_transform="raw", remaining_time_scaling="zscore",
        )
        schema_profile = {
            "sequence_categorical_columns": [encoding.CANONICAL_ACTIVITY_TOKEN],
            "sequence_numerical_columns": [],
            "offer_numerical_columns": [],
        }
        mapping = {"datasets": {"bpic2017": {}, "bpic2012": {}}}
        discover_calls: list[tuple[str, str, int]] = []
        stats_banks: list[str] = []

        # Stub the split discovery with one synthetic bank path per client.
        def fake_discover(_mapping: dict[str, object], dataset_id: str, heterogeneity: str,
                          n_clients: int) -> dict[str, dict[str, Path]]:
            discover_calls.append((dataset_id, heterogeneity, n_clients))
            if dataset_id == "joint":
                raise AssertionError("joint must not be passed to discover_split_paths")
            banks = ("A", "B", "C")
            return {
                bank: {split: Path(f"/tmp/{dataset_id}_{bank}_{split}.parquet") for split in encoding.SPLITS}
                for bank in banks
            }

        # Stub the local statistics collection and record the visited banks.
        def fake_collect(_events: pd.DataFrame, bank_id: str, *_args: object, **_kwargs: object) -> dict[str, object]:
            stats_banks.append(bank_id)
            return {
                "bank": bank_id,
                "counts": {"case_count": 1, "event_count": 1, "prefix_count": 1, "observed_max_trace_length": 1},
                "train_outcome_counts": {"2": 1},
                "categorical_train_counts": {},
                "numeric_train_statistics": {},
                "remaining_time_target_statistics": {"transform": "raw", "scaling": "zscore"},
            }

        with mock.patch.object(poc.encoding, "discover_split_paths", side_effect=fake_discover):
            with mock.patch.object(pd, "read_parquet", return_value=pd.DataFrame({"x": [1]})):
                with mock.patch.object(poc.encoding, "to_canonical_events", return_value=pd.DataFrame({"x": [1]})):
                    with mock.patch.object(poc.encoding, "apply_activity_mapping",
                                           return_value=(pd.DataFrame({"x": [1]}), {})):
                        with mock.patch.object(poc, "collect_local_stats_from_events", side_effect=fake_collect):
                            with mock.patch.object(poc, "aggregate_local_stats",
                                                   return_value=({"counts": {"aggregate": {}}}, [])):
                                with mock.patch.object(poc, "load_central_artifacts", return_value=({}, {}, {})):
                                    with mock.patch.object(
                                        poc,
                                        "compare_aggregated_to_central",
                                        return_value={"matches_central_metadata": True, "checks": {},
                                                      "differences": []},
                                    ):
                                        with mock.patch.object(poc, "write_poc_outputs"):
                                            record = poc.run_one_poc(
                                                config,
                                                schema_profile,
                                                mapping,
                                                Path("mapping.json"),
                                                Path("central"),
                                                Path("out"),
                                                True,
                                                42,
                                            )

        self.assertIn(("bpic2017", "iid", 3), discover_calls)
        self.assertIn(("bpic2012", "iid", 3), discover_calls)
        self.assertNotIn(("joint", "iid", 6), discover_calls)
        self.assertIn("bpic2017:A", stats_banks)
        self.assertIn("bpic2012:A", stats_banks)
        self.assertEqual(record["bank_count"], 6)

if __name__ == "__main__":
    unittest.main()