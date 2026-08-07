from __future__ import annotations
import os
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
E_TRAINING_DIR = REPO_ROOT / "E_training"
BASELINE_WORKFLOW = E_TRAINING_DIR / "WORKFLOW_run_baseline_final.sh"
FEDERATED_WORKFLOW = E_TRAINING_DIR / "WORKFLOW_run_federated_final.sh"
DP_WORKFLOW = E_TRAINING_DIR / "WORKFLOW_run_federated_dp_final.sh"
MASTER_WORKFLOW = REPO_ROOT / "WORKFLOW_run_FULL_PIPELINE.sh"

class JointWorkflowTests(unittest.TestCase):
    # HELPER: Run one workflow dry run with the caller's stage toggles stripped, so defaults stay observable.
    def _dry_run(self, script: Path, extra_env: dict[str, str] | None = None) -> str:
        dropped = {"RESILIENT", "REPORTING_PROFILE", "OUTPUT_ROOT", "CACHE_ROOT", "ARTIFACT_ROOT",
                   "FAILED_RUNS_LOG", "STRATEGY", "DEVICE", "DRY_RUN"}
        env = {key: value for key, value in os.environ.items()
               if not key.startswith(("RUN_", "CLEAN_")) and key not in dropped}
        env.update({"DRY_RUN": "true"})
        if extra_env is not None:
            env.update(extra_env)
        result = subprocess.run(
            ["bash", str(script)], cwd=REPO_ROOT, env=env, check=False, text=True, capture_output=True,
        )

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout + result.stderr

    # Verify baseline workflow dry run covers single and joint runs.
    def test_baseline_workflow_dry_run_covers_single_and_joint_runs(self) -> None:
        output = self._dry_run(BASELINE_WORKFLOW)

        self.assertIn("E_05 bpic2017 iid_3banks centralized", output)
        self.assertIn("E_05 bpic2012 medium_3banks centralized", output)
        self.assertIn("E_05 joint iid_6banks centralized", output)
        self.assertIn("E_05 joint medium_8banks local bpic2017:E", output)
        self.assertIn("E_05 joint medium_8banks local bpic2012:A", output)
        self.assertIn("REMAINING_TIME_TRANSFORM=raw", output)
        self.assertIn("REMAINING_TIME_SCALING=zscore", output)
        self.assertIn("REMAINING_TIME_HUBER_BETA=0.1", output)
        self.assertIn("REPORTING_PROFILE=compact", output)

    # Verify federated workflow dry run covers single and joint both strategies.
    def test_federated_workflow_dry_run_covers_single_and_joint_both_strategies(self) -> None:
        output = self._dry_run(FEDERATED_WORKFLOW, {"STRATEGY": "both"})

        self.assertIn("E_06 bpic2017 iid_3banks fedavg", output)
        self.assertIn("E_06 joint iid_6banks fedavg", output)
        self.assertIn("E_06 joint medium_8banks fedprox", output)
        self.assertIn("FEDPROX_MU=1e-4", output)
        self.assertIn("USE_DP=false", output)
        self.assertIn("REPORTING_PROFILE=compact", output)
        self.assertNotIn("USE_DP=true", output)
        self.assertIn("NEXT_ACTIVITY_HEAD_AGG=equal", output)
        self.assertIn("NEXT_ACTIVITY_HEAD_AGG=sample", output)

    # Verify baseline workflow resolves per family schedule.
    def test_baseline_workflow_resolves_per_family_schedule(self) -> None:
        output = self._dry_run(BASELINE_WORKFLOW)

        self.assertIn("SCHEDULE dataset=bpic2017 MAX_EPOCHS=40 T_MAX=15 MIN_LR=1e-6", output)
        self.assertIn("SCHEDULE dataset=bpic2012 MAX_EPOCHS=40 T_MAX=35 MIN_LR=1e-5", output)
        self.assertIn("SCHEDULE dataset=joint MAX_EPOCHS=40 T_MAX=35 MIN_LR=1e-5", output)

    # Verify federated workflow resolves per family schedule.
    def test_federated_workflow_resolves_per_family_schedule(self) -> None:
        output = self._dry_run(FEDERATED_WORKFLOW, {"STRATEGY": "fedavg"})

        self.assertIn("SCHEDULE dataset=bpic2017 MAX_ROUNDS=40 T_MAX=15 MIN_LR=1e-6", output)
        self.assertIn("SCHEDULE dataset=bpic2012 MAX_ROUNDS=40 T_MAX=35 MIN_LR=1e-5", output)
        self.assertIn("SCHEDULE dataset=joint MAX_ROUNDS=40 T_MAX=35 MIN_LR=1e-5", output)

    # Verify federated workflow resolves head agg per dataset.
    def test_federated_workflow_resolves_head_agg_per_dataset(self) -> None:
        output = self._dry_run(FEDERATED_WORKFLOW, {"STRATEGY": "fedavg"})

        self.assertIn("E_06 bpic2017 iid_3banks fedavg agg=sample", output)
        self.assertIn("E_06 bpic2012 iid_3banks fedavg agg=sample", output)
        self.assertIn("E_06 joint iid_6banks fedavg agg=equal", output)

    # Verify DP workflow uses FedProx long schedule and standard patience.
    def test_dp_workflow_uses_fedprox_long_schedule_and_standard_patience(self) -> None:
        text = DP_WORKFLOW.read_text(encoding="utf-8")

        self.assertIn("NEXT_ACTIVITY_HEAD_AGG:-sample", text)
        self.assertNotIn("NEXT_ACTIVITY_HEAD_AGG:-equal", text)
        self.assertIn("STRATEGY:-fedprox", text)
        self.assertIn("EARLY_STOPPING_PATIENCE:-7", text)
        self.assertIn("LR_SCHEDULER_T_MAX:-35", text)
        self.assertIn("LR_SCHEDULER_MIN_LR:-1e-5", text)
        self.assertIn('EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}"', text)

    # Verify DP workflow dry run plans twelve FedProx runs.
    def test_dp_workflow_dry_run_plans_twelve_fedprox_runs(self) -> None:
        output = self._dry_run(DP_WORKFLOW)

        self.assertEqual(12, output.count("E_06 DP "))
        self.assertNotIn("fedavg", output)
        for split in ("iid_3banks", "medium_3banks", "strong_3banks"):
            self.assertIn(f"E_06 DP bpic2017 {split} fedprox", output)
        for epsilon in ("eps=1", "eps=5", "eps=10", "eps=50"):
            self.assertIn(epsilon, output)
        self.assertIn("T_MAX=35", output)
        self.assertIn("MIN_LR=1e-5", output)
        self.assertIn("PATIENCE=7", output)
        self.assertIn("MAX_ROUNDS=40", output)

    # Verify the federated workflow default strategy is both.
    def test_federated_workflow_default_strategy_is_both(self) -> None:
        output = self._dry_run(FEDERATED_WORKFLOW)

        self.assertIn("STRATEGY=fedavg", output)
        self.assertIn("STRATEGY=fedprox", output)

    # Verify master workflow bare run is v6 matrix with secure POC.
    def test_master_workflow_bare_run_is_v6_matrix_with_secure_poc(self) -> None:
        output = self._dry_run(MASTER_WORKFLOW)

        self.assertIn("RUN_DATA_DOWNLOAD=false", output)
        self.assertIn("RUN_PREPROCESSING=false", output)
        self.assertIn("RUN_FOCUSED_TESTS=false", output)
        self.assertIn("RUN_ENCODING=false", output)
        self.assertIn("RUN_FEDERATED_DP=false", output)
        self.assertIn("RUN_SECURE_AGGREGATION=true", output)
        self.assertIn("RUN_TRAINING_ANALYSIS=true", output)
        self.assertIn("REPORTING_PROFILE=compact", output)
        self.assertIn("RUN_LLM_EXPERIMENT=false", output)
        self.assertIn("RUN_LLM_SCHEMA_EXPERIMENT=false", output)
        self.assertIn("training_outputs", output)
        self.assertNotIn("training_outputs_v", output)
        self.assertIn("WORKFLOW_run_baseline_final.sh", output)
        self.assertIn("WORKFLOW_run_federated_final.sh", output)
        self.assertIn("E_07_generate_training_analysis.py", output)
        self.assertNotIn("WORKFLOW_run_federated_dp_final.sh", output)
        self.assertNotIn("Run E_04 encoding", output)
        self.assertNotIn("OPENAI_API_KEY is NOT set", output)
        self.assertIn("SECURE_AGGREGATION_SIMULATION=true", output)
        self.assertIn("Secure-aggregation POC joint medium_8banks fedprox", output)
        self.assertIn("Secure-aggregation POC bpic2017 medium_3banks fedprox", output)
        self.assertIn("Secure-aggregation POC bpic2012 medium_3banks fedprox", output)
        self.assertNotIn("WORKFLOW_run_joint_baselines_final.sh", output)
        self.assertNotIn("WORKFLOW_run_joint_federated_final.sh", output)

    # Verify master workflow-focused tests are opt in and listed.
    def test_master_workflow_focused_tests_are_opt_in_and_listed(self) -> None:
        output = self._dry_run(MASTER_WORKFLOW, {"RUN_FOCUSED_TESTS": "true"})

        self.assertIn("RUN_FOCUSED_TESTS=true", output)
        self.assertIn("Run focused unit tests", output)
        self.assertIn("tests.test_joint_run_specs", output)
        self.assertIn("tests.test_prefix_encoding_joint_runtime", output)
        self.assertIn("tests.test_e05_joint_baselines", output)
        self.assertIn("tests.test_e06_federated_core", output)
        self.assertIn("tests.test_prefix_encoding_decentralized_poc", output)
        self.assertIn("tests.test_prefix_encoding_runtime", output)
        self.assertIn("tests.test_joint_workflows", output)
        self.assertIn("tests.test_training_reporting", output)
        self.assertIn("tests.test_training_analysis", output)
        self.assertNotIn("tests.test_llm_mapping_experiment_analysis", output)

if __name__ == "__main__":
    unittest.main()