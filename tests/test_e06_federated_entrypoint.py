from __future__ import annotations
import json
import unittest
import tempfile
from pathlib import Path
from typing import Any, Literal
from unittest import mock
import pandas as pd
import numpy as np
import torch
from E_training import training_core_final as core
from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import E_06_federated_training as federated

# The three head-aggregation modes as the literal type the E_06 signatures declare.
HEAD_AGG_MODES: tuple[Literal["sample", "equal", "contribution"], ...] = ("sample", "equal", "contribution")

# HELPER: Stand in for a DataLoader that the orchestration only hands through and never iterates.
def _stub_loader(name: str) -> Any: return [name]

# HELPER: Build a tiny E_05-compatible model for entrypoint orchestration tests.
def _build_tiny_model() -> core.MultitaskLSTM:
    return core.MultitaskLSTM(
        categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2, next_activity_classes=5,
        hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=6,
    )

# HELPER: Build noop mask context.
def _noop_mask_context() -> core.NextActivityMaskContext:
    return core.NextActivityMaskContext(
        dataset_code_by_id={"bpic2017": 0}, mask_by_dataset=torch.ones((1, 5), dtype=torch.bool), is_noop=True,
    )

# HELPER: Build a minimal prediction frame accepted by E_05 summary writers.
def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prefix_length": [1, 3],
            "outcome_label": [2, 0],
            "outcome_pred": [2, 0],
            "outcome_prob_0": [0.1, 0.8],
            "outcome_prob_1": [0.1, 0.1],
            "outcome_prob_2": [0.8, 0.1],
            "remaining_time_label_seconds": [10.0, 20.0],
            "remaining_time_pred_seconds_clamped": [11.0, 18.0],
            "remaining_time_pred_seconds_raw": [11.0, 18.0],
            "remaining_time_mask": [1, 1],
        }
    )

# HELPER: Build a mocked federated context with one simulated bank.
def _mock_context() -> federated.FederatedRunContext:
    client = federated.FederatedClientContext(
        bank="A",
        train_dataset=["train"],
        val_dataset=["val"],
        test_dataset=["test"],
        train_loader=_stub_loader("train_loader"),
        val_loader=_stub_loader("val_loader"),
        test_loader=_stub_loader("test_loader"),
        train_prefix_count=2,
        next_activity_class_counts=np.array([1.0], dtype=np.float32),
        training_next_activity_mask_context=_noop_mask_context(),
    )
    return federated.FederatedRunContext(
        clients=[client],
        pooled_train_dataset=["pooled_train"],
        pooled_val_dataset=["pooled_val"],
        pooled_test_dataset=["pooled_test"],
        pooled_val_loader=_stub_loader("pooled_val_loader"),
        pooled_test_loader=_stub_loader("pooled_test_loader"),
        pooled_train_labels=pd.Series([0, 1, 2]),
        remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
        evaluation_next_activity_mask_context=_noop_mask_context(),
        spec={},
        vocabularies={},
        scalers={},
        schema_profile={},
        mapping={},
    )

# CLASS: Replace real Flower clients with a deterministic one-client fit stub.
class FakeBankClient:
    # Store the client context and the public protocol inputs the real client receives.
    def __init__(self, *_args: object, **kwargs: Any) -> None:
        self.client_context = kwargs["client_context"]
        self.config = kwargs["config"]
        self.client_index = int(kwargs["client_index"])
        self.n_clients = int(kwargs["n_clients"])
        self.parameter_keys = list(kwargs["parameter_keys"])

    # Return unchanged parameters and flat metrics for one synchronous round.
    def fit(self, parameters: list[Any], fit_config: dict[str, Any]) -> tuple[list[Any], int, dict[str, Any]]:
        round_idx = int(fit_config["server_round"])
        return parameters, self.client_context.train_prefix_count, {
            "round": round_idx,
            "bank": self.client_context.bank,
            "train_prefix_count": int(self.client_context.train_prefix_count),
            "train_loss_total": 1.0,
            "train_loss_total_with_fedprox": 1.0,
            "train_fedprox_penalty": 0.0,
            "n_batches": 2,
        }

    # Mask the stub update through the production client-side helper for secure-aggregation runs.
    def fit_secure(self, parameters: list[Any],
                   fit_config: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        params, train_prefix_count, metrics = self.fit(parameters, fit_config)
        message = federated.build_secure_client_message(
            params, train_prefix_count, self.client_index, self.n_clients,
            next_activity_head_agg=self.config.next_activity_head_agg,
            parameter_keys=self.parameter_keys,
            next_activity_class_counts=self.client_context.next_activity_class_counts,
            is_joint_run=self.config.dataset == "joint",
            round_index=int(fit_config["server_round"]),
            secure_aggregation_seed=self.config.secure_aggregation_seed,
        )
        return message, metrics

# CLASS: Check E_06 entrypoint configuration and artifact layout.
class FederatedEntrypointTests(unittest.TestCase):
    # Validate production workflow defaults that differ from raw E_05 module fallbacks.
    def test_config_defaults_match_final_workflow(self) -> None:
        config = federated.FederatedRunConfig()

        self.assertEqual(config.strategy, "fedprox")
        self.assertEqual(config.next_activity_head_agg, "sample")
        self.assertEqual(config.secure_aggregation_simulation, False)
        self.assertEqual(config.secure_aggregation_seed, 42)
        self.assertEqual(config.fedprox_mu, 1e-4)
        self.assertEqual(config.max_rounds, 40)
        self.assertEqual(config.local_epochs, 1)
        self.assertEqual(config.early_stopping_patience, 7)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.dataset, "bpic2017")
        self.assertEqual(config.heterogeneity, "medium")
        self.assertEqual(config.n_clients, 3)
        self.assertEqual(config.device, "auto")
        self.assertEqual(config.progress_bars, True)
        self.assertEqual(config.learning_rate, 2.5e-4)
        self.assertEqual(config.weight_decay, 1e-4)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.lr_scheduler, "cosine")
        self.assertEqual(config.lr_scheduler_min_lr, 1e-6)
        self.assertEqual(config.lr_scheduler_t_max, 15)
        self.assertEqual(config.batch_size, 512)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertEqual(config.dropout, 0.30)
        self.assertEqual(config.head_hidden_size, 64)
        self.assertEqual(config.outcome_label_smoothing, 0.10)
        self.assertEqual(config.outcome_class_weight_power, baseline.OUTCOME_CLASS_WEIGHT_POWER)
        self.assertEqual(config.outcome_lr_scale, 0.3)
        self.assertEqual(config.next_activity_lr_scale, 1.0)
        self.assertEqual(config.remaining_time_lr_scale, 1.0)
        self.assertEqual(config.outcome_loss_weight, 1.0)
        self.assertEqual(config.next_activity_loss_weight, 0.5)
        self.assertEqual(config.remaining_time_loss_weight, 0.5)
        self.assertEqual(config.remaining_time_transform, "raw")
        self.assertEqual(config.remaining_time_scaling, "zscore")
        self.assertEqual(config.remaining_time_huber_beta, 0.1)
        self.assertEqual(config.outcome_head_dropout, 0.45)
        self.assertEqual(config.reporting_profile, "compact")
        self.assertIsNone(config.dp_smoke_max_batches)
        self.assertEqual(config.output_root, federated.SCRIPT_DIR / "training_outputs")
        self.assertEqual(config.run_name, "medium_3banks")

    # Validate the federated run folder contract for FedProx smoke runs.
    def test_output_path_names_strategy_rounds_local_epochs_and_mu(self) -> None:
        config = federated.FederatedRunConfig()

        self.assertEqual(
            federated.output_dir_for_run(Path("outputs"), config),
            Path("outputs/federated/bpic2017_medium_3banks/fedprox_seed_42_lr_0p00025_rounds_40_le_1_mu_0p0001"),
        )

    # Validate secure-aggregation POC runs the route into their own subfolder with the plain federated run name.
    def test_output_path_names_secure_aggregation_subfolder(self) -> None:
        config = federated.FederatedRunConfig(secure_aggregation_simulation=True)

        self.assertEqual(
            federated.output_dir_for_run(Path("outputs"), config),
            Path(
                "outputs/secure_aggregation/bpic2017_medium_3banks/"
                "fedprox_seed_42_lr_0p00025_rounds_40_le_1_mu_0p0001"),
        )

    # Validate joint run folders separate the three head-aggregation modes while single-dataset names stay unchanged.
    def test_output_path_names_head_aggregation_for_joint_runs_only(self) -> None:
        joint_dirs = {
            federated.output_dir_for_run(
                Path("outputs"),
                federated.FederatedRunConfig(dataset="joint", heterogeneity="medium", n_clients=8,
                                             next_activity_head_agg=mode),
            )
            for mode in HEAD_AGG_MODES
        }

        self.assertEqual(len(joint_dirs), 3)
        self.assertEqual(
            federated.output_dir_for_run(
                Path("outputs"),
                federated.FederatedRunConfig(dataset="joint", heterogeneity="medium", n_clients=8,
                                             next_activity_head_agg="equal"),
            ),
            Path("outputs/federated/joint_medium_8banks/"
                 "fedprox_seed_42_lr_0p00025_rounds_40_le_1_mu_0p0001_agg_equal"),
        )

        # A single-dataset run resolves every mode to sample, so its production directory name must not gain a token.
        single_dirs = {
            federated.output_dir_for_run(Path("outputs"), federated.FederatedRunConfig(next_activity_head_agg=mode))
            for mode in HEAD_AGG_MODES
        }
        self.assertEqual(
            single_dirs,
            {Path("outputs/federated/bpic2017_medium_3banks/fedprox_seed_42_lr_0p00025_rounds_40_le_1_mu_0p0001")},
        )

    # Validate DP run folders distinguish capped smoke checks from full DP passes.
    def test_output_path_names_dp_batch_mode(self) -> None:
        capped = federated.FederatedRunConfig(
            strategy="fedavg", max_rounds=2, local_epochs=1, use_dp=True, dp_target_epsilon=50.0,
            dp_smoke_max_batches=4,
        )
        full = federated.FederatedRunConfig(
            strategy="fedavg", max_rounds=2, local_epochs=1, use_dp=True, dp_target_epsilon=50.0,
            dp_smoke_max_batches=None,
        )

        self.assertEqual(
            federated.output_dir_for_run(Path("outputs"), capped),
            Path(
                "outputs/differential_privacy/bpic2017_medium_3banks/"
                "fedavg_seed_42_lr_0p00025_rounds_2_le_1_dp_eps_50_dp_batches_4"
            ),
        )
        self.assertEqual(
            federated.output_dir_for_run(Path("outputs"), full),
            Path(
                "outputs/differential_privacy/bpic2017_medium_3banks/"
                "fedavg_seed_42_lr_0p00025_rounds_2_le_1_dp_eps_50_dp_batches_full"
            ),
        )

    # Validate prefixed artifact paths inside one E_06 run directory.
    def test_artifact_paths_include_script_identifier(self) -> None:
        run_dir = Path("outputs/federated/bpic2017_medium_3banks/fedprox_seed_42")

        self.assertEqual(federated.run_artifact_path(run_dir, "config.json"), run_dir / "E_06_config.json")
        self.assertEqual(
            federated.prediction_artifact_path(run_dir, "predictions_test.parquet"),
            run_dir / "predictions" / "E_06_predictions_test.parquet",
        )

    # Verify the synchronous loop runs all rounds when pooled validation improves.
    def test_run_one_federated_runs_to_max_rounds_when_validation_improves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = federated.FederatedRunConfig(
                max_rounds=3,
                early_stopping_patience=5,
                output_root=Path(tmp),
                device="cpu",
                progress_bars=False,
            )
            result = self._run_with_validation_losses(config, [3.0, 2.0, 1.0])

            self.assertEqual(result["rounds_completed"], 3)
            self.assertEqual(result["best_round"], 3)
            run_dir = federated.output_dir_for_run(Path(tmp), config)
            round_log = pd.read_csv(run_dir / "E_06_round_log.csv")
            report = json.loads((run_dir / "E_06_run_report.json").read_text(encoding="utf-8"))
            self.assertEqual((run_dir / "E_06_config.json").exists(), False)
            self.assertEqual((run_dir / "E_06_round_metrics.json").exists(), False)
            self.assertNotIn("dp_epsilon_spent", round_log.columns)
            self.assertEqual(float(round_log.loc[0, "train_n_batches_weighted"]), 2.0)
            self.assertEqual(float(round_log.loc[0, "train_n_batches_min"]), 2.0)
            self.assertEqual(float(round_log.loc[0, "train_n_batches_max"]), 2.0)
            self.assertEqual(report["rounds"]["history"][0]["clients"][0]["n_batches"], 2)
            self.assertEqual(report["config"]["reporting_profile"], "compact")
            expected_curves = [
                "E_06_loss_curves.png",
                "E_06_task_loss_curves.png",
                "E_06_outcome_macro_f1_curve.png",
                "E_06_next_activity_accuracy_curve.png",
                "E_06_remaining_time_mae_curve.png",
            ]
            self.assertEqual(report["curves"]["files"], expected_curves)
            for filename in expected_curves:
                self.assertTrue((run_dir / filename).exists())
                self.assertIn(filename, report["artifacts"]["files"])
            summary = pd.read_csv(run_dir / "E_06_federated_summary_test.csv")
            summary_payload = json.loads((run_dir / "E_06_federated_summary_test.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["scope"].tolist(), ["global_test", "bank_A"])
            self.assertEqual([row["scope"] for row in summary_payload["rows"]], ["global_test", "bank_A"])
            self.assertIn("E_06_federated_summary_test.csv", report["artifacts"]["files"])
            self.assertIn("E_06_federated_summary_test.json", report["artifacts"]["files"])
            self.assertIn("best_validation_loss", result)
            self.assertIn("test_metrics", result)

    # Verify debug reporting keeps the legacy fragments besides the compact report.
    def test_run_one_federated_debug_reporting_writes_legacy_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = federated.FederatedRunConfig(
                max_rounds=1, output_root=Path(tmp), device="cpu", progress_bars=False, reporting_profile="debug",
            )
            self._run_with_validation_losses(config, [1.0])
            run_dir = federated.output_dir_for_run(Path(tmp), config)

            self.assertTrue((run_dir / "E_06_run_report.json").exists())
            self.assertTrue((run_dir / "E_06_config.json").exists())
            self.assertTrue((run_dir / "E_06_round_metrics.json").exists())

    # Verify pooled validation loss drives the same patience logic used by E_05.
    def test_run_one_federated_early_stops_on_pooled_validation_loss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = federated.FederatedRunConfig(
                max_rounds=20, early_stopping_patience=2, output_root=Path(tmp), device="cpu", progress_bars=False,
            )
            result = self._run_with_validation_losses(config, [1.0, 1.1, 1.2, 1.3])

            self.assertLess(result["rounds_completed"], 20)
            self.assertEqual(result["early_stopped"], True)
            self.assertEqual(result["best_round"], 1)
            self.assertIn("best_validation_loss", result)
            self.assertIn("test_metrics", result)

    # Verify the secure-aggregation run writes the evidence block and an off run does not.
    def test_run_one_federated_writes_secure_aggregation_block_only_when_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            secure_config = federated.FederatedRunConfig(
                max_rounds=2, output_root=Path(tmp) / "secure", device="cpu", progress_bars=False,
                secure_aggregation_simulation=True, secure_aggregation_seed=42,
            )
            self._run_with_validation_losses(secure_config, [2.0, 1.0])
            secure_run_dir = federated.output_dir_for_run(Path(tmp) / "secure", secure_config)
            secure_report = json.loads((secure_run_dir / "E_06_run_report.json").read_text(encoding="utf-8"))
            secure_block = secure_report["diagnostics"]["secure_aggregation"]
            self.assertEqual(secure_block["enabled"], True)
            self.assertEqual(secure_block["orchestrator_receives_masked_messages_only"], True)
            self.assertEqual(secure_block["production_protocol_implemented"], False)
            self.assertEqual(secure_block["reconstruction_matches_plain_within_tolerance"], True)
            self.assertEqual(secure_block["seed"], 42)

            # The block must deny cryptographic input privacy, because the pair seeds come from the recorded seed.
            self.assertEqual(secure_block["provides_cryptographic_input_privacy"], False)
            self.assertNotIn("server_received_individual_updates", secure_block)
            self.assertTrue(any("recover that client's individual update" in entry
                                for entry in secure_block["limitations"]))
            self.assertTrue(any("Diffie-Hellman" in entry for entry in secure_block["limitations"]))

            plain_config = federated.FederatedRunConfig(
                max_rounds=2, output_root=Path(tmp) / "plain", device="cpu", progress_bars=False,
            )
            self._run_with_validation_losses(plain_config, [2.0, 1.0])
            plain_run_dir = federated.output_dir_for_run(Path(tmp) / "plain", plain_config)
            plain_report = json.loads((plain_run_dir / "E_06_run_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("secure_aggregation", plain_report["diagnostics"])

    # Run E_06 with mocked context, client fit and evaluation boundaries.
    def _run_with_validation_losses(self, config: federated.FederatedRunConfig,
                                    losses: list[float]) -> dict[str, object]:
        validation_losses = iter(losses)

        # Return round validation losses for metrics-only calls and prediction frames for final exports.
        # Fill the requested prefix subsets the way the real evaluation does, so the per-bank artifacts are written.
        def evaluate_model(*_args: object, **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame | None]:
            collect_predictions = bool(kwargs.get("collect_predictions", False))
            if not collect_predictions: return self._metrics(next(validation_losses)), None
            subset_metrics = kwargs.get("subset_metrics")
            for name in (kwargs.get("subset_selectors") or {}):
                if subset_metrics is not None: subset_metrics[name] = self._metrics(0.5)
            return self._metrics(0.5), _prediction_frame()

        # Patch data, client and bank-lookup boundaries so no real cache or parquet IO happens.
        with mock.patch.object(federated, "load_federated_context", return_value=_mock_context()):
            with mock.patch.object(federated, "build_model_for_federated", return_value=_build_tiny_model()):
                with mock.patch.object(federated, "FederatedBankClient", FakeBankClient):
                    with mock.patch.object(baseline, "bank_by_case_for_split", return_value={}):
                        with mock.patch.object(baseline, "bank_selectors_for_dataset",
                                               return_value={"A": np.array([True, True])}):
                            with mock.patch.object(baseline, "evaluate_model", side_effect=evaluate_model):
                                return federated.run_one_federated(config)

    # Build a complete scalar metric block with the requested total loss.
    @staticmethod
    def _metrics(loss_total: float) -> dict[str, object]:
        return {
            "loss_total": float(loss_total),
            "loss_outcome": 0.1,
            "loss_next_activity": 0.1,
            "loss_remaining_time": 0.1,
            "outcome": {"macro_f1": 0.5, "weighted_f1": 0.5, "balanced_accuracy": 0.5, "per_class": {}},
            "next_activity": {"top1_accuracy": 0.5, "top3_accuracy": 0.5},
            "remaining_time": {"mae": 1.0, "rmse": 1.0},
        }

if __name__ == "__main__":
    unittest.main()