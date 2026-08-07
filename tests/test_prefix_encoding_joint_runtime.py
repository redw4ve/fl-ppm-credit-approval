from __future__ import annotations
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from tests.test_prefix_encoding_runtime import _approved_mapping, _bpic2012_frame, _bpic2017_frame

encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")
runner = importlib.import_module("E_prefix_encoding.04_4_runner")

# HELPER: Write split triplet.
def _write_split_triplet(root: Path, run_name: str, prefix: str, bank: str, frame: pd.DataFrame) -> None:
    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for split in encoding.SPLITS:
        frame.to_parquet(run_dir / f"{prefix}_bank_{bank}_{split}.parquet", index=False)

class PrefixEncodingJointRuntimeTests(unittest.TestCase):
    # Verify load joint medium 8-banks uses source runs and plain client ids.
    def test_load_joint_medium_8banks_uses_source_runs_and_plain_client_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            schema_path = tmp_root / "canonical_schemas.json"
            mapping_path = tmp_root / "manual_dataset_mapping.json"
            bpic2017_root = tmp_root / "bpic2017"
            bpic2012_root = tmp_root / "bpic2012"
            mapping = _approved_mapping(mapping_path, schema_path, "joint")
            mapping["datasets"]["bpic2017"]["input_root"] = str(bpic2017_root)
            mapping["datasets"]["bpic2017"]["split_prefix"] = "A_02"
            mapping["datasets"]["bpic2012"]["input_root"] = str(bpic2012_root)
            mapping["datasets"]["bpic2012"]["split_prefix"] = "B_02"
            mapping["datasets"]["bpic2012"].setdefault("default_values", {}).update(
                {
                    encoding.LOAN_GOAL: encoding.MISSING_TOKEN,
                    encoding.APPLICATION_TYPE: encoding.MISSING_TOKEN,
                }
            )

            for bank in ("A", "B", "C", "D", "E"):
                _write_split_triplet(bpic2017_root, "medium_5banks", "A_02", bank, _bpic2017_frame())
            for bank in ("A", "B", "C"):
                _write_split_triplet(bpic2012_root, "medium_3banks", "B_02", bank, _bpic2012_frame())

            schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
            config = runner.build_config("joint", "joint", "medium", 8, 83, tmp_root / "artifacts")
            events = encoding.load_joint_run_events(config, mapping, schema_payload["schema_profiles"]["joint"])
            paths = encoding.build_artifact_paths(tmp_root / "artifacts", "joint", "medium", 8, joint=True)

            self.assertEqual(set(events[encoding.DATASET_ID]), {"bpic2017", "bpic2012"})
            self.assertEqual(set(events[encoding.CLIENT_ID]), {"A", "B", "C", "D", "E"})
            self.assertEqual(paths.root, tmp_root / "artifacts" / "joint" / "medium_8banks")
            self.assertEqual(paths.encoding_spec.name, "J_04_encoding_spec.json")

    # Verify runner all matrix contains joint runs with total client counts.
    def test_runner_all_matrix_contains_joint_runs_with_total_client_counts(self) -> None:
        joint_entries = [entry for entry in runner.matrix_for_profile("all") if entry[0] == "joint"]

        self.assertEqual(joint_entries, [
            ("joint", "iid", 6, 83),
            ("joint", "weak", 6, 83),
            ("joint", "medium", 6, 83),
            ("joint", "medium", 8, 83),
        ])

if __name__ == "__main__":
    unittest.main()