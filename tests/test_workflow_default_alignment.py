from __future__ import annotations
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import E_06_federated_training as federated

# Shell assignments of the form NAME="${NAME:-value}" or NAME="${NAME-value}" carry the workflow default.
SHELL_DEFAULT_PATTERN = re.compile(r'^(?P<name>[A-Z0-9_]+)="\$\{(?P=name):?-(?P<value>[^}]*)}"')

# Environment reads of the form os.environ.get("NAME", ...) carry the module knobs a workflow must declare.
MODULE_ENV_PATTERN = re.compile(r'os\.environ\.get\(\s*"([A-Z0-9_]+)"')

# Path and orchestration variables are resolved by the workflow and are not training hyperparameters.
NON_HYPERPARAMETER_NAMES: frozenset[str] = frozenset(
    {"OUTPUT_ROOT", "CACHE_ROOT", "ARTIFACT_ROOT", "PYTHON_BIN", "RESILIENT", "FAILED_RUNS_LOG", "DRY_RUN",
     "SINGLE_RUN_MODE", "BANK", "REGIME"}
)

# HELPER: Collect every workflow default assignment from one shell script.
def shell_defaults(path: Path) -> dict[str, str]:
    defaults: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = SHELL_DEFAULT_PATTERN.match(line.strip())
        if match is None: continue
        defaults.setdefault(match.group("name"), match.group("value"))
    return defaults

# HELPER: Collect every environment knob one training module reads.
def module_env_knobs(module: Any) -> set[str]:
    return set(MODULE_ENV_PATTERN.findall(Path(module.__file__).read_text(encoding="utf-8")))

# HELPER: Extract one top-level shell function body from a workflow file by name.
def shell_function(path: Path, name: str) -> str:
    match = re.search(rf"^{name}\(\) \{{\n.*?^}}", path.read_text(encoding="utf-8"), flags=re.MULTILINE | re.DOTALL)
    if match is None: raise AssertionError(f"{path.name} does not define {name}")
    return match.group(0)

# HELPER: Reduce module values and shell strings to one comparable representation.
def normalize(value: Any) -> Optional[str]:
    if value is None: return ""
    if isinstance(value, bool): return "true" if value else "false"
    try: return f"{float(value):.10g}"
    except (TypeError, ValueError): return str(value)

class WorkflowDefaultAlignmentTests(unittest.TestCase):
    # Compare every shared knob of one module against one workflow, skipping the documented workflow-owned entries.
    def assert_aligned(self, module: Any, workflow: Path, workflow_owned: dict[str, str],
                       absent_allowed: dict[str, str] | None = None) -> None:
        defaults = shell_defaults(workflow)
        compared = 0
        for name, shell_value in sorted(defaults.items()):
            if name in NON_HYPERPARAMETER_NAMES or name in workflow_owned: continue
            if not hasattr(module, name): continue
            compared += 1
            self.assertEqual(
                normalize(getattr(module, name)), normalize(shell_value),
                f"{workflow.name} sets {name}={shell_value} while the module default differs",
            )

        # Guard the guard: a broken parser would silently compare nothing.
        self.assertGreater(compared, 20)

        # Every documented exception must still exist in the workflow, so the list cannot rot.
        for name in workflow_owned:
            self.assertIn(name, defaults, f"{workflow.name} no longer defines the workflow-owned knob {name}")

        # A module knob the workflow does not declare would float on the caller's environment, so absence fails.
        for name in sorted(module_env_knobs(module)):
            if name in NON_HYPERPARAMETER_NAMES or name in (absent_allowed or {}): continue
            self.assertIn(name, defaults, f"{workflow.name} does not declare the module knob {name}")

    # Verify a bare E_05 module run resolves the same production config as the baseline workflow.
    def test_baseline_module_defaults_match_the_baseline_workflow(self) -> None:
        self.assert_aligned(
            baseline,
            REPO_ROOT / "E_training" / "WORKFLOW_run_baseline_final.sh",
            {
                "LR_SCHEDULER_T_MAX": "resolved per dataset family, 15 for BPIC 2017 and 35 for BPIC 2012 and joint",
                "LR_SCHEDULER_MIN_LR": "resolved per dataset family, 1e-6 for BPIC 2017 and 1e-5 elsewhere",
            },
        )

    # Verify a bare E_06 module run resolves the same production config as the no-DP federated workflow.
    def test_federated_module_defaults_match_the_federated_workflow(self) -> None:
        self.assert_aligned(
            federated,
            REPO_ROOT / "E_training" / "WORKFLOW_run_federated_final.sh",
            {
                "LR_SCHEDULER_T_MAX": "resolved per dataset family, 15 for BPIC 2017 and 35 for BPIC 2012 and joint",
                "LR_SCHEDULER_MIN_LR": "resolved per dataset family, 1e-6 for BPIC 2017 and 1e-5 elsewhere",
                "NEXT_ACTIVITY_HEAD_AGG": "resolved per dataset family, equal for joint and sample elsewhere",
                "STRATEGY": "both is a workflow keyword that expands to one FedAvg and one FedProx run",
            },
        )

    # Verify the DP workflow differs from the module defaults only where the DP design deliberately overrides them.
    def test_dp_workflow_overrides_are_limited_to_the_documented_exceptions(self) -> None:
        self.assert_aligned(
            federated,
            REPO_ROOT / "E_training" / "WORKFLOW_run_federated_dp_final.sh",
            {
                "LR_SCHEDULER_T_MAX": "long schedule, t_max 35 for every DP run because DP noise slows convergence",
                "LR_SCHEDULER_MIN_LR": "long schedule, floor 1e-5 for every DP run",
                "NEXT_ACTIVITY_HEAD_AGG": "the DP grid is single-dataset, so sample is stated explicitly",
                "USE_DP": "this workflow is DP-only and always passes --use-dp",
            },
            absent_allowed={
                "SECURE_AGGREGATION_SIMULATION": "mutually exclusive with DP, the module default off is required",
                "SECURE_AGGREGATION_SEED": "only read when the simulation is on, which the DP workflow never enables",
            },
        )

    # Verify the two named outcome regularization knobs carry their production values in the module itself.
    def test_outcome_regularization_defaults_are_the_production_values(self) -> None:
        config = baseline.BaselineRunConfig()

        self.assertEqual(config.outcome_label_smoothing, 0.10)
        self.assertEqual(config.outcome_head_dropout, 0.45)
        self.assertEqual(config.outcome_label_smoothing, federated.FederatedRunConfig().outcome_label_smoothing)
        self.assertEqual(config.outcome_head_dropout, federated.FederatedRunConfig().outcome_head_dropout)

class WorkflowGuardTests(unittest.TestCase):
    MASTER = REPO_ROOT / "WORKFLOW_run_FULL_PIPELINE.sh"
    FEDERATED = REPO_ROOT / "E_training" / "WORKFLOW_run_federated_final.sh"
    BASELINE = REPO_ROOT / "E_training" / "WORKFLOW_run_baseline_final.sh"

    # HELPER: Run one extracted shell function under bash and return the completed process.
    @staticmethod
    def run_bash(script: str) -> "subprocess.CompletedProcess[str]":
        return subprocess.run(["bash", "-c", script], capture_output=True, text=True)

    # Verify the master refuses the analysis stage when the output root holds no completed run report.
    def test_master_refuses_analysis_on_an_empty_matrix(self) -> None:
        guard = shell_function(self.MASTER, "refuse_empty_matrix")
        with tempfile.TemporaryDirectory() as tmp:
            prelude = f'OUTPUT_ROOT="{tmp}"; FAILED_RUNS_LOG="{tmp}/none.log"; {guard}\n'
            empty = self.run_bash(prelude + "refuse_empty_matrix")
            report = Path(tmp) / "baselines" / "run" / "E_05_run_report.json"
            report.parent.mkdir(parents=True)
            report.write_text("{}", encoding="utf-8")
            filled = self.run_bash(prelude + "refuse_empty_matrix")

        self.assertNotEqual(empty.returncode, 0)
        self.assertIn("no completed training run", empty.stderr)
        self.assertEqual(filled.returncode, 0, filled.stderr)

    # Verify the master runs the full self-contained suite as its first stage, before the download and the matrices.
    def test_master_runs_the_full_suite_before_every_other_stage(self) -> None:
        text = self.MASTER.read_text(encoding="utf-8")
        self.assertIn('RUN_TEST_SUITE="${RUN_TEST_SUITE:-true}"', text)
        suite_stage = text.index("-m pytest tests/")
        self.assertLess(suite_stage, text.index("download_bpic_from_4tu.py"))
        self.assertLess(suite_stage, text.index("WORKFLOW_run_baseline_final.sh"))

    # Verify the preprocessing workflows execute the frozen notebooks into build copies only.
    # A reintroduced --inplace flag would rewrite the frozen A_01 and B_01 notebooks on the next preprocessing run.
    def test_partitioning_workflows_never_execute_notebooks_in_place(self) -> None:
        workflows = (
            REPO_ROOT / "E_main_BPIC_2017" / "A_WORKFLOW_run_partitioning.sh",
            REPO_ROOT / "E_ablation_BPIC_2012" / "B_WORKFLOW_run_partitioning.sh",
        )
        for workflow in workflows:
            text = workflow.read_text(encoding="utf-8")
            active = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))
            self.assertNotIn("--inplace", active, f"{workflow.name} must never execute a frozen notebook in place")
            self.assertIn("-m nbconvert", active, f"{workflow.name} must launch nbconvert as a module")
            self.assertIn("--output-dir build", active, f"{workflow.name} must route the executed copy to build/")

    # Verify a failed run aborts the federated matrix under RESILIENT=false and is logged and skipped under true.
    def test_failed_run_aborts_unless_resilient(self) -> None:
        run_one = shell_function(self.FEDERATED, "run_one")
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "failed.log"
            prelude = (
                'resolve_family_schedule() { :; }\nrun_python() { return 1; }\n'
                f'OUTPUT_ROOT="{tmp}"; FAILED_RUNS_LOG="{log}"; STRATEGY=fedavg; NEXT_ACTIVITY_HEAD_AGG=sample; '
                'MAX_ROUNDS=40; LOCAL_EPOCHS=1;\n'
            )
            strict = self.run_bash(f'RESILIENT=false\n{prelude}{run_one}\nrun_one bpic2017 medium 3')
            resilient = self.run_bash(f'RESILIENT=true\n{prelude}{run_one}\nrun_one bpic2017 medium 3')
            logged = log.read_text(encoding="utf-8") if log.exists() else ""

        self.assertNotEqual(strict.returncode, 0)
        self.assertEqual(resilient.returncode, 0, resilient.stderr)
        self.assertIn("bpic2017 medium_3banks", logged)

    # Verify the strict shell mode the abort chain relies on exists in the master and both training workflows.
    def test_master_and_training_workflows_run_under_strict_mode(self) -> None:
        for path in (self.MASTER, self.FEDERATED, self.BASELINE):
            self.assertIn("set -euo pipefail", path.read_text(encoding="utf-8"), path.name)

if __name__ == "__main__":
    unittest.main()