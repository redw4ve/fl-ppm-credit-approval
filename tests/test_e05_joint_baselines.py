from __future__ import annotations
import unittest
from pathlib import Path
from unittest import mock
import pandas as pd
from E_training import E_05_central_and_local_baselines_final as baseline

class E05JointBaselineTests(unittest.TestCase):
    # Verify joint artifact prefix and output paths are safe.
    def test_joint_artifact_prefix_and_output_paths_are_safe(self) -> None:
        centralized = baseline.BaselineRunConfig(dataset="joint", heterogeneity="iid", n_clients=6,
                                                 regime="centralized")
        local = baseline.BaselineRunConfig(dataset="joint", heterogeneity="iid", n_clients=6, regime="local",
                                           bank="bpic2017:A")

        self.assertEqual(baseline._artifact_prefix("joint"), "J_04")
        self.assertEqual(
            baseline.output_dir_for_run(Path("outputs"), centralized),
            Path("outputs/baselines/joint_iid_6banks/centralized_seed_42_lr_0p00025"),
        )
        self.assertEqual(
            baseline.output_dir_for_run(Path("outputs"), local),
            Path("outputs/baselines/joint_iid_6banks/local_bank_bpic2017_A_seed_42_lr_0p00025"),
        )
        self.assertEqual(baseline._safe_bank("bpic2017:A"), "bpic2017_A")

    # Verify joint bank names return dataset qualified clients.
    def test_joint_bank_names_return_dataset_qualified_clients(self) -> None:
        iid = baseline.BaselineRunConfig(dataset="joint", heterogeneity="iid", n_clients=6)
        medium = baseline.BaselineRunConfig(dataset="joint", heterogeneity="medium", n_clients=8)

        self.assertEqual(baseline.bank_names_for_config(iid), (
            "bpic2017:A", "bpic2017:B", "bpic2017:C", "bpic2012:A", "bpic2012:B", "bpic2012:C",
        ))
        self.assertEqual(len(baseline.bank_names_for_config(medium)), 8)

    # Verify joint centralized loading unions all source clients.
    def test_joint_centralized_loading_unions_all_source_clients(self) -> None:
        config = baseline.BaselineRunConfig(dataset="joint", heterogeneity="iid", n_clients=6, regime="centralized")
        mapping = {
            "datasets": {
                "bpic2017": {"input_root": "/tmp/bpic2017", "split_prefix": "A_02"},
                "bpic2012": {"input_root": "/tmp/bpic2012", "split_prefix": "B_02"},
            }
        }
        to_canonical_calls: list[tuple[str, str, str]] = []

                # Stub the canonical conversion with one tagged row per client.
        def fake_to_canonical(_raw: pd.DataFrame, _mapping: dict[str, object], dataset_id: str, client_id: str,
                              split_name: str) -> pd.DataFrame:
            to_canonical_calls.append((dataset_id, client_id, split_name))
            return pd.DataFrame({"dataset": [dataset_id], "client": [client_id], "split": [split_name]})

        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(pd, "read_parquet", return_value=pd.DataFrame({"x": [1]})):
                with mock.patch.object(baseline.encoding, "to_canonical_events", side_effect=fake_to_canonical):
                    with mock.patch.object(baseline.encoding, "apply_activity_mapping",
                                           side_effect=lambda frame, _path: (frame, {})):
                        with mock.patch.object(baseline.encoding, "validate_canonical_events"):
                            events = baseline.load_mapped_events(config, mapping)

        self.assertEqual(len(events), 18)
        self.assertIn(("bpic2017", "A", "train"), to_canonical_calls)
        self.assertIn(("bpic2012", "A", "train"), to_canonical_calls)
        self.assertFalse(any(dataset_id == "joint" for dataset_id, _client_id, _split in to_canonical_calls))

    # Verify joint local loading maps qualified the bank to source split.
    def test_joint_local_loading_maps_qualified_bank_to_source_split(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="joint",
            heterogeneity="iid",
            n_clients=6,
            regime="local",
            bank="bpic2012:A",
        )
        mapping = {
            "datasets": {
                "bpic2017": {"input_root": "/tmp/bpic2017", "split_prefix": "A_02"},
                "bpic2012": {"input_root": "/tmp/bpic2012", "split_prefix": "B_02"},
            }
        }
        to_canonical_calls: list[tuple[str, str, str]] = []

                # Stub the canonical conversion with one tagged row per client.
        def fake_to_canonical(_raw: pd.DataFrame, _mapping: dict[str, object], dataset_id: str, client_id: str,
                              split_name: str) -> pd.DataFrame:
            to_canonical_calls.append((dataset_id, client_id, split_name))
            return pd.DataFrame({"dataset": [dataset_id], "client": [client_id], "split": [split_name]})

        with mock.patch.object(Path, "exists", return_value=True):
            with mock.patch.object(pd, "read_parquet", return_value=pd.DataFrame({"x": [1]})):
                with mock.patch.object(baseline.encoding, "to_canonical_events", side_effect=fake_to_canonical):
                    with mock.patch.object(baseline.encoding, "apply_activity_mapping",
                                           side_effect=lambda frame, _path: (frame, {})):
                        with mock.patch.object(baseline.encoding, "validate_canonical_events"):
                            events = baseline.load_mapped_events(config, mapping)

        self.assertEqual(len(events), 3)
        self.assertEqual(to_canonical_calls,
                         [("bpic2012", "A", "train"), ("bpic2012", "A", "val"), ("bpic2012", "A", "test")])

if __name__ == "__main__":
    unittest.main()