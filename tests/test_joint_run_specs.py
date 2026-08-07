from __future__ import annotations
import importlib
import unittest
from pathlib import Path

encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")
REPO_ROOT = Path(__file__).resolve().parents[1]

class JointRunSpecTests(unittest.TestCase):
    # Verify joint runs cover the locked matrix.
    def test_joint_runs_cover_the_locked_matrix(self) -> None:
        self.assertEqual(
            set(encoding.JOINT_RUNS), {"iid_6banks", "weak_6banks", "medium_6banks", "medium_8banks"},
        )

    # Verify production encoding module exposes joint helpers.
    def test_production_encoding_module_exposes_joint_helpers(self) -> None:
        for name in ("JOINT_RUNS", "resolve_joint_run", "safe_client_id", "load_joint_run_events"):
            self.assertTrue(hasattr(encoding, name), name)

    # Verify resolve joint run returns source specs.
    def test_resolve_joint_run_returns_source_specs(self) -> None:
        spec = encoding.resolve_joint_run("medium_8banks")

        self.assertEqual(spec.run_id, "medium_8banks")
        self.assertEqual(spec.heterogeneity, "medium")
        self.assertEqual(spec.total_clients, 8)
        self.assertEqual([(source.dataset_id, source.heterogeneity, source.n_clients) for source in spec.sources], [
            ("bpic2017", "medium", 5),
            ("bpic2012", "medium", 3),
        ])
        self.assertEqual(spec.qualified_client_ids, (
            "bpic2017:A", "bpic2017:B", "bpic2017:C", "bpic2017:D", "bpic2017:E",
            "bpic2012:A", "bpic2012:B", "bpic2012:C",
        ))

    # Verify joint client ids are unique and file safe.
    def test_joint_client_ids_are_unique_and_file_safe(self) -> None:
        for run_name in encoding.JOINT_RUNS:
            spec = encoding.resolve_joint_run(run_name)

            self.assertEqual(len(spec.qualified_client_ids), spec.total_clients)
            self.assertEqual(len(set(spec.qualified_client_ids)), spec.total_clients)

        self.assertEqual(encoding.safe_client_id("bpic2017:A"), "bpic2017_A")
        self.assertEqual(encoding.safe_client_id("A"), "A")

    # Verify no source file references old joint side module.
    def test_no_source_file_references_old_joint_side_module(self) -> None:
        old_token = "04_5B" + chr(95) + "joint_encoding"
        roots = [REPO_ROOT / "E_prefix_encoding", REPO_ROOT / "E_training", REPO_ROOT / "tests"]
        hits: list[str] = []

        for root in roots:
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix not in {".py", ".sh"}: continue
                if "__pycache__" in path.parts or "training_outputs" in path.parts or "_old" in path.parts: continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if old_token in text: hits.append(str(path.relative_to(REPO_ROOT)))

        self.assertEqual(hits, [])

if __name__ == "__main__":
    unittest.main()