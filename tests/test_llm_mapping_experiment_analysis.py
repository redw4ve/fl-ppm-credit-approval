from __future__ import annotations
import importlib
import json
import tempfile
import unittest
from pathlib import Path

analyze = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")
create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")

# HELPER: Write JSON.
def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")

class LlmMappingExperimentAnalysisTests(unittest.TestCase):
    # Verify LLM analysis uses noninteractive plot backend.
    def test_llm_analysis_uses_noninteractive_plot_backend(self) -> None:
        import matplotlib
        self.assertEqual(matplotlib.get_backend().lower(), "agg")

    # Verify word similarity weights tokens instead of characters.
    def test_word_similarity_weights_tokens_instead_of_characters(self) -> None:
        character_score = create_mapping._activity_score(
            "W_Longwordlongword application",
            "W_validate_application",
            similarity_mode="character",
        )
        word_score = create_mapping._activity_score(
            "W_Longwordlongword application",
            "W_validate_application",
            similarity_mode="word",
        )

        self.assertNotEqual(character_score, word_score)
        self.assertEqual(create_mapping._semantic_variant_name("character"), "semantic_character")
        self.assertEqual(create_mapping._semantic_variant_name("word"), "semantic_word")

    # Verify discover LLM files recurses into run folders.
    def test_discover_llm_files_recurses_into_run_folders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_json(root / "run_01" / "draft.json", {"ok": True})
            _write_json(root / "run_02" / "draft.json", {"ok": True})

            files = analyze.discover_llm_files(root, ("*.json",))

            self.assertEqual(len(files), 2)
            self.assertEqual([path.parent.name for path in files], ["run_01", "run_02"])

    # Verify dataset metadata reads activity labels from all splits.
    def test_dataset_metadata_reads_activity_labels_from_all_splits(self) -> None:
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split_root = root / "iid_3banks"
            train_path = split_root / "T_02_bank_A_train.parquet"
            test_path = split_root / "T_02_bank_A_test.parquet"
            train_path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame({"concept:name": ["A_Create Application"], "case:concept:name": ["case_train"]})
            frame.to_parquet(train_path)
            pd.DataFrame({"concept:name": ["O_Refused"], "case:concept:name": ["case_test"]}).to_parquet(test_path)

            metadata = create_mapping.discover_dataset_metadata(
                "toy", {"input_root": root, "split_prefix": "T_02"}, require_files=True,
            )

            self.assertIn("A_Create Application", metadata["raw_activity_labels"])
            self.assertIn("O_Refused", metadata["raw_activity_labels"])

    # Verify repeated mapping summary groups by strategy.
    def test_repeated_mapping_summary_groups_by_strategy(self) -> None:
        entries = [
            {
                "file": "run_01/04_03_strategy_1_baseline_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.5,
                "column_mapping_accuracy": 0.9,
                "missing_manual_label_count": 2,
                "extra_label_count": 0,
                "unresolved_label_count": 1,
                "wrong_column_count": 1,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_02/04_03_strategy_1_baseline_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 1.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 0,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
        ]

        summary = analyze.build_repeated_mapping_summary(entries)

        row = summary["strategies"]["04_03_strategy_1_baseline_dataset_mapping"]
        self.assertEqual(row["run_count"], 2)
        self.assertEqual(row["valid_run_count"], 2)
        self.assertEqual(row["activity_accuracy_mean"], 0.75)
        self.assertEqual(row["column_mapping_accuracy_mean"], 0.95)
        self.assertEqual(row["blocking_error_count_sum"], 1)
        self.assertEqual(row["approval_ready_run_count"], 1)
        self.assertEqual(row["correction_burden_sum"], 4)

    # Verify repeated mapping summary uses methodological strategy groups.
    def test_repeated_mapping_summary_uses_methodological_strategy_groups(self) -> None:
        entries = [
            {
                "file": "run_semantic_character/04_03_semantic_character_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.6,
                "column_mapping_accuracy": 0.9,
                "missing_manual_label_count": 1,
                "extra_label_count": 0,
                "unresolved_label_count": 1,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_character/04_03_strategy_1_baseline_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.8,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 1,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_word/04_03_strategy_1_baseline_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.7,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 2,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_character/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 1.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 2,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_word/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.5,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 4,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/04_03_strategy_3_target_recipe_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.9,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 3,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
        ]

        summary = analyze.build_repeated_mapping_summary(entries)

        self.assertEqual(
            list(summary["strategies"]),
            [
                "semantic_character",
                "llm_strategy_1_character",
                "llm_strategy_1_word",
                "llm_strategy_2_split",
                "llm_strategy_3_classify",
            ],
        )
        self.assertEqual(summary["strategies"]["llm_strategy_2_split"]["run_count"], 2)
        self.assertEqual(summary["strategies"]["llm_strategy_2_split"]["missing_manual_label_count_mean"], 3.0)
        self.assertEqual(summary["strategies"]["llm_strategy_2_split"]["correction_burden_mean"], 3.0)

    # Verify repeated mapping error series uses mean errors per run.
    def test_repeated_mapping_error_series_uses_mean_errors_per_run(self) -> None:
        summary = {
            "strategies": {
                "llm_strategy_2_split": {
                    "missing_manual_label_count_sum": 6,
                    "wrong_label_count_sum": 2,
                    "wrong_column_count_sum": 0,
                    "extra_label_count_sum": 0,
                    "unresolved_label_count_sum": 0,
                    "missing_manual_label_count_mean": 3.0,
                    "wrong_label_count_mean": 1.0,
                    "wrong_column_count_mean": 0.0,
                    "extra_label_count_mean": 0.0,
                    "unresolved_label_count_mean": 0.0,
                }
            }
        }

        strategies, series = analyze._repeated_mapping_error_series(summary)

        self.assertEqual(strategies, ["llm_strategy_2_split"])
        self.assertEqual(series[0][0], "Missing labels")
        self.assertEqual(series[0][2], [3.0])
        self.assertEqual(series[1][2], [1.0])

    # Verify repeated mapping summary prefers current seedless strategy files.
    def test_repeated_mapping_summary_prefers_current_seedless_strategy_files(self) -> None:
        entries = [
            {
                "file": "run_01/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 1.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 1,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_character/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 10,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_word/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 10,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
        ]

        summary = analyze.build_repeated_mapping_summary(entries)

        row = summary["strategies"]["llm_strategy_2_split"]
        self.assertEqual(row["run_count"], 1)
        self.assertEqual(row["activity_accuracy_mean"], 1.0)
        self.assertEqual(row["missing_manual_label_count_sum"], 1)

    # Verify the human report prefers current seedless strategy files.
    def test_human_report_prefers_current_seedless_strategy_files(self) -> None:
        entries = [
            {
                "file": "run_01/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 1.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 0,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
            {
                "file": "run_01/semantic_character/04_03_strategy_2_split_prompt_dataset_mapping.json",
                "kind": "dataset_mapping",
                "validity_gate": True,
                "activity_accuracy": 0.0,
                "column_mapping_accuracy": 1.0,
                "missing_manual_label_count": 10,
                "extra_label_count": 0,
                "unresolved_label_count": 0,
                "wrong_column_count": 0,
                "wrong_label_count": 0,
                "cross_prefix_error_count": 0,
            },
        ]

        report_entries = analyze._filter_summary_seedless_strategy_duplicates(entries)
        table = analyze.build_strategy_table(report_entries)

        self.assertEqual(len(table), 1)
        self.assertEqual(table[0]["strategy"], "run_01/04_03_strategy_2_split_prompt_dataset_mapping")

if __name__ == "__main__":
    unittest.main()