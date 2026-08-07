from __future__ import annotations
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_core_final as core
import importlib

encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

# Minimal E_04 metadata that carries the fields the cache identity reads.
SPEC: dict[str, Any] = {
    "prefix": {"cap": 83, "static_padding_length": 8},
    "target_scalers": {"remaining_time": {"transform": "raw", "scaling": "zscore"}},
}
SCHEMA_PROFILE: dict[str, Any] = {
    "sequence_categorical_columns": ["activity"],
    "sequence_numerical_columns": ["time_delta"],
    "offer_numerical_columns": ["offered_amount"],
}

# HELPER: Write one processed split parquet with the requested number of case rows.
def _write_split(path: Path, n_cases: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case": [f"c{index}" for index in range(n_cases)]}).to_parquet(path, index=False)

# HELPER: Build a dataset mapping and the processed split parquets below one root.
def _build_dataset_root(root: Path, n_cases: int = 12) -> dict[str, Any]:
    _write_split(root / "centralized" / "medium_3banks" / "A_02_train.parquet", n_cases)
    _write_split(root / "centralized" / "medium_3banks" / "A_02_val.parquet", n_cases)
    _write_split(root / "centralized" / "medium_3banks" / "A_02_test.parquet", n_cases)
    return {"datasets": {"bpic2017": {"input_root": str(root), "split_prefix": "A_02",
                                      "column_mapping": {"case_id": "case"}}}}

# HELPER: Copy the frozen approved mapping inputs next to a temporary artifact root.
def _config_for(root: Path, artifact_root: Path) -> baseline.BaselineRunConfig:
    spec_path = artifact_root / "bpic2017" / "medium_3banks" / "A_04_encoding_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text('{"prefix": {"cap": 83}}', encoding="utf-8")
    return baseline.BaselineRunConfig(
        dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="centralized",
        artifact_root=artifact_root, cache_root=root / "cache",
    )

class CacheIdentityTests(unittest.TestCase):
    # HELPER: Resolve the cache build hash of the train split below one dataset root.
    @staticmethod
    def _hash_for(root: Path, artifact_root: Path, n_cases: int = 12) -> str:
        mapping = _build_dataset_root(root, n_cases)
        config = _config_for(root, artifact_root)
        return baseline.cache_context(config, "train", SPEC, {}, {}, SCHEMA_PROFILE, mapping)["cache_build_hash"]

    def setUp(self) -> None:
        baseline._PARQUET_FINGERPRINTS.clear()

    # Verify identical inputs hit the cache and changed parquet content misses it.
    def test_identical_inputs_hit_and_changed_content_misses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            artifacts = Path(tmp) / "metadata"
            first = self._hash_for(root, artifacts)
            baseline._PARQUET_FINGERPRINTS.clear()
            second = self._hash_for(root, artifacts)
            self.assertEqual(first, second)

            # A simulated content change in the split parquet must invalidate the cache.
            baseline._PARQUET_FINGERPRINTS.clear()
            changed = self._hash_for(root, artifacts, n_cases=13)
            self.assertNotEqual(first, changed)

    # Verify the fingerprint is independent of the repository path and of the file modification time.
    def test_fingerprint_is_path_and_time_independent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / "clone_one" / "data"
            artifacts = Path(tmp) / "clone_one" / "metadata"
            first = self._hash_for(original, artifacts)

            # A second checkout at another path with a newer modification time must produce the same hash.
            clone_root = Path(tmp) / "clone_two"
            shutil.copytree(Path(tmp) / "clone_one", clone_root)
            for path in sorted(clone_root.rglob("*.parquet")):
                os.utime(path, (time.time() + 5000.0, time.time() + 5000.0))
            baseline._PARQUET_FINGERPRINTS.clear()
            mapping = {"datasets": {"bpic2017": {"input_root": str(clone_root / "data"), "split_prefix": "A_02",
                                                 "column_mapping": {"case_id": "case"}}}}
            config = _config_for(clone_root / "data", clone_root / "metadata")
            second = baseline.cache_context(config, "train", SPEC, {}, {}, SCHEMA_PROFILE,
                                            mapping)["cache_build_hash"]

        self.assertEqual(first, second)

    # Verify the payload records the tensor layout version, the E_04 spec hash and the input fingerprints.
    def test_payload_records_the_provenance_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            artifacts = Path(tmp) / "metadata"
            mapping = _build_dataset_root(root)
            config = _config_for(root, artifacts)
            payload = baseline.cache_build_payload(config, "train", SPEC, {}, {}, SCHEMA_PROFILE, mapping)

        self.assertEqual(payload["cache_schema_version"], core.PREFIX_TENSOR_CACHE_VERSION)
        self.assertEqual(len(payload["encoding_spec_sha256"]), 64)
        self.assertEqual(list(payload["input_parquet_sha256"]), ["medium_3banks/A_02_train.parquet"])
        self.assertEqual(len(next(iter(payload["input_parquet_sha256"].values()))), 64)

    # Verify a bumped tensor layout version invalidates every existing cache by design.
    def test_cache_schema_version_change_invalidates_the_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            artifacts = Path(tmp) / "metadata"
            mapping = _build_dataset_root(root)
            config = _config_for(root, artifacts)
            payload = baseline.cache_build_payload(config, "train", SPEC, {}, {}, SCHEMA_PROFILE, mapping)
            bumped = {**payload, "cache_schema_version": int(payload["cache_schema_version"]) + 1}

        self.assertNotEqual(baseline.stable_json_hash(payload), baseline.stable_json_hash(bumped))

    # Verify a stored absolute processed-split root is resolved against the checkout's repository root.
    # The checkout is a temporary tree, so the guard holds on a clone that ships without the processed data.
    def test_input_root_resolves_against_the_repository_root(self) -> None:
        stored = "/somewhere/else/entirely/E_main_BPIC_2017/data/processed"
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            (checkout / "E_main_BPIC_2017" / "data" / "processed").mkdir(parents=True)
            with mock.patch.object(baseline, "REPO_ROOT", checkout):
                resolved = baseline.resolve_input_root(stored)
                self.assertEqual(resolved, checkout / "E_main_BPIC_2017" / "data" / "processed")
                self.assertTrue(resolved.is_dir())

                # A repository-relative stored value must resolve through the full-path case of the resolver.
                self.assertEqual(baseline.resolve_input_root("E_main_BPIC_2017/data/processed"),
                                 checkout / "E_main_BPIC_2017" / "data" / "processed")

                # A stored path with no resolvable suffix falls back, so the failure names the recorded path.
                self.assertEqual(baseline.resolve_input_root("/no/such/place"), Path("/no/such/place"))

    # Verify the local checkout wins even when the stored absolute path exists on this machine.
    # A clone beside the original checkout must never silently read the original checkout's processed data.
    def test_existing_foreign_absolute_path_does_not_beat_the_local_checkout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            foreign = root / "foreign" / "E_main_BPIC_2017" / "data" / "processed"
            foreign.mkdir(parents=True)
            checkout = root / "checkout"
            local = checkout / "E_main_BPIC_2017" / "data" / "processed"
            local.mkdir(parents=True)

            with mock.patch.object(baseline, "REPO_ROOT", checkout):
                self.assertEqual(baseline.resolve_input_root(str(foreign)), local)
            with mock.patch.object(encoding, "ENCODING_REPO_ROOT", checkout):
                self.assertEqual(encoding.resolve_input_root(str(foreign)), local)

    # Verify the loaded mapping carries resolved roots rather than the recorded absolute ones.
    # The real frozen payload is loaded with only its machine prefix replaced, so the suffix logic under test is real.
    def test_loaded_mapping_rewrites_every_input_root(self) -> None:
        payload = json.loads(baseline.DATASET_MAPPING_PATH.read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            for dataset_mapping in payload["datasets"].values():
                suffix = Path(*Path(dataset_mapping["input_root"]).parts[-3:])
                (checkout / suffix).mkdir(parents=True)
                dataset_mapping["input_root"] = str(Path("/nonexistent-checkout/thesis") / suffix)
            mapping_path = checkout / "mapping.json"
            mapping_path.write_text(json.dumps(payload), encoding="utf-8")

            with mock.patch.object(baseline, "REPO_ROOT", checkout):
                with mock.patch.object(baseline.encoding, "ENCODING_REPO_ROOT", checkout):
                    mapping = baseline.load_dataset_mapping(mapping_path, require_approved=True)

            for dataset_id, dataset_mapping in mapping["datasets"].items():
                resolved = Path(dataset_mapping["input_root"])
                self.assertTrue(resolved.is_dir(), f"{dataset_id} processed root does not exist: {resolved}")
                self.assertTrue(resolved.is_relative_to(checkout), f"{dataset_id} escaped the repository root")

    # Verify a missing split parquet fails loudly instead of producing a cache identity without provenance.
    def test_missing_split_parquet_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "data"
            artifacts = Path(tmp) / "metadata"
            mapping = _build_dataset_root(root)
            config = _config_for(root, artifacts)
            (root / "centralized" / "medium_3banks" / "A_02_train.parquet").unlink()

            with self.assertRaises(FileNotFoundError):
                baseline.cache_build_payload(config, "train", SPEC, {}, {}, SCHEMA_PROFILE, mapping)

if __name__ == "__main__":
    unittest.main()