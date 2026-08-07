from __future__ import annotations
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline

class BaselineEntrypointTests(unittest.TestCase):
    # Verify the config name and output path include seed.
    def test_config_name_and_output_path_include_seed(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="local", bank="A", seed=42, max_epochs=1,
        )

        self.assertEqual(config.run_name, "medium_3banks")
        self.assertEqual(
            baseline.output_dir_for_run(Path("outputs"), config),
            Path("outputs/baselines/bpic2017_medium_3banks/local_bank_A_seed_42_lr_0p00025"),
        )

    # Verify the centralized path stays flat.
    def test_centralized_path_stays_flat(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="centralized", seed=42, max_epochs=1,
        )

        self.assertEqual(
            baseline.output_dir_for_run(Path("outputs"), config),
            Path("outputs/baselines/bpic2017_medium_3banks/centralized_seed_42_lr_0p00025"),
        )

    # Verify run dir name reflects learning rate.
    def test_run_dir_name_reflects_learning_rate(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="centralized", seed=42,
            learning_rate=1.5e-4,
        )

        run_dir = baseline.output_dir_for_run(Path("outputs"), config)
        self.assertTrue(run_dir.name.endswith("lr_0p00015"), run_dir.name)

    # Verify output dir override bypasses standard layout.
    def test_output_dir_override_bypasses_standard_layout(self) -> None:
        override = Path("outputs/sandbox/run_1_lr_0p00025")
        config = baseline.BaselineRunConfig(
            dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="centralized", seed=42, max_epochs=1,
            output_dir_override=override,
        )
        self.assertEqual(baseline.output_dir_for_run(Path("outputs"), config), override)

    # Verify artifact paths include script identifier.
    def test_artifact_paths_include_script_identifier(self) -> None:
        run_dir = Path("outputs/baselines/bpic2017_medium_3banks/centralized_seed_42")

        self.assertEqual(
            baseline.run_artifact_path(run_dir, "config.json"),
            run_dir / "E_05_config.json",
        )
        self.assertEqual(
            baseline.run_artifact_path(run_dir, "model_best.pt"),
            run_dir / "E_05_model_best.pt",
        )

    # Verify prediction artifacts live in the prediction subfolder.
    def test_prediction_artifacts_live_in_prediction_subfolder(self) -> None:
        run_dir = Path("outputs/baselines/bpic2017_medium_3banks/centralized_seed_42")

        self.assertEqual(
            baseline.prediction_artifact_path(run_dir, "predictions_test.parquet"),
            run_dir / "predictions" / "E_05_predictions_test.parquet",
        )
        self.assertEqual(
            baseline.prediction_artifact_path(run_dir, "train_log.csv"),
            run_dir / "predictions" / "E_05_train_log.csv",
        )

    # Verify summary artifacts include script identifier.
    def test_summary_artifacts_include_script_identifier(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="bpic2017", heterogeneity="medium", n_clients=3, regime="local", bank="A", seed=42, max_epochs=1,
        )

        csv_path, json_path = baseline.local_summary_paths(config)

        self.assertEqual(
            csv_path,
            baseline.OUTPUT_ROOT / "baselines/bpic2017_medium_3banks/E_05_local_summary_seed_42_lr_0p00025.csv",
        )
        self.assertEqual(
            json_path,
            baseline.OUTPUT_ROOT / "baselines/bpic2017_medium_3banks/E_05_local_summary_seed_42_lr_0p00025.json",
        )

    # Verify progress bars are enabled by default and disableable.
    def test_progress_bars_are_enabled_by_default_and_disableable(self) -> None:
        default_config = baseline.BaselineRunConfig()
        disabled_config = baseline.BaselineRunConfig(progress_bars=False)

        self.assertTrue(default_config.progress_bars)
        self.assertFalse(disabled_config.progress_bars)
        self.assertFalse(baseline.should_disable_progress(default_config, is_tty=True))
        self.assertTrue(baseline.should_disable_progress(default_config, is_tty=False))
        self.assertTrue(baseline.should_disable_progress(disabled_config, is_tty=True))

    # Verify default output root lives inside training folder.
    def test_default_output_root_lives_inside_training_folder(self) -> None:
        self.assertEqual(
            baseline.OUTPUT_ROOT,
            (REPO_ROOT / "E_training/training_outputs").resolve(),
        )

    # Verify default device is auto for E_05 runs.
    def test_default_device_is_auto_for_e05_runs(self) -> None:
        config = baseline.BaselineRunConfig()

        self.assertEqual(baseline.DEVICE, "auto")
        self.assertEqual(config.device, "auto")

    # Verify config payload records the resolved device.
    def test_config_payload_records_resolved_device(self) -> None:
        config = baseline.BaselineRunConfig(device="mps")

        payload = baseline._config_payload(config, resolved_device="mps")

        self.assertEqual(payload["device"], "mps")
        self.assertEqual(payload["resolved_device"], "mps")

    # Verify outcome regularization defaults reproduce production.
    def test_outcome_regularization_defaults_reproduce_production(self) -> None:
        config = baseline.BaselineRunConfig()
        disabled = baseline.config_from_args(baseline.parse_args([]))

        self.assertEqual(baseline.OUTCOME_LABEL_SMOOTHING, 0.10)
        self.assertEqual(baseline.OUTCOME_HEAD_DROPOUT, 0.45)
        self.assertEqual(config.outcome_label_smoothing, 0.10)
        self.assertEqual(config.outcome_head_dropout, 0.45)
        self.assertEqual(disabled.outcome_head_dropout, 0.45)

    # Verify reporting profile defaults to compact and is parseable.
    def test_reporting_profile_defaults_to_compact_and_is_parseable(self) -> None:
        config = baseline.BaselineRunConfig()
        parsed = baseline.config_from_args(baseline.parse_args(["--reporting-profile", "debug"]))

        self.assertEqual(config.reporting_profile, "compact")
        self.assertEqual(parsed.reporting_profile, "debug")
        self.assertFalse(baseline.training_reporting.should_write_legacy_reports(config.reporting_profile))
        self.assertTrue(baseline.training_reporting.should_write_legacy_reports(parsed.reporting_profile))

if __name__ == "__main__":
    unittest.main()