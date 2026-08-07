from __future__ import annotations
import ast
import importlib
import contextlib
import io
import json
import sys
import tempfile
import tokenize
import unittest
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

# Blank out every comment token, so a source scan sees code only and string literals stay intact.
def _strip_python_comments(source: str) -> list[str]:
    lines = source.splitlines(keepends=True)
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    for token in tokens:
        if token.type != tokenize.COMMENT: continue
        row, start = token.start[0] - 1, token.start[1]
        lines[row] = lines[row][:start] + lines[row][start:].replace(token.string, "", 1)
    return lines

class PrefixEncodingContractTests(unittest.TestCase):
    # Verify prefix encoding uses eight python files.
    def test_prefix_encoding_uses_eight_python_files(self) -> None:
        paths = sorted(path.name for path in (REPO_ROOT / "E_prefix_encoding").glob("04_*.py"))

        self.assertEqual(
            paths,
            [
                "04_0_extract_contract_context.py",
                "04_1_contract.py",
                "04_2_create_canonical_schema.py",
                "04_3_create_dataset_mapping.py",
                "04_4_runner.py",
                "04_5_encoding.py",
                "04_6_analyze_llm_outputs.py",
                "04_7_decentralized_metadata_poc.py",
            ],
        )

    # Verify public prefix modules import cleanly.
    def test_public_prefix_modules_import_cleanly(self) -> None:
        analyze_llm = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")
        contract_context = importlib.import_module("E_prefix_encoding.04_0_extract_contract_context")
        decentralized_poc = importlib.import_module("E_prefix_encoding.04_7_decentralized_metadata_poc")
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
        encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")

        self.assertTrue(hasattr(analyze_llm, "main"))
        self.assertTrue(hasattr(contract, "CONTRACT_FIELD_GROUPS"))
        self.assertTrue(hasattr(contract_context, "summarize_dataset_context"))
        self.assertTrue(hasattr(decentralized_poc, "aggregate_numeric_statistics"))
        self.assertTrue(hasattr(create_schema, "main"))
        self.assertTrue(hasattr(create_mapping, "main"))
        self.assertTrue(hasattr(encoding, "PrefixDataset"))
        self.assertTrue(hasattr(runner, "run_full_matrix"))

    # Verify script docstrings list required and created files.
    def test_script_docstrings_list_required_and_created_files(self) -> None:
        expected = {
            "04_0_extract_contract_context.py": {
                "required": ["A_02 split parquets", "B_02 split parquets"],
                "created": [
                    "decentralized_poc/contract_context",
                    "04_00_bpic2017_contract_context.json",
                    "04_00_federation_contract_context.json",
                ],
            },
            "04_1_contract.py": {
                "required": ["mappings/MANUAL_contract.json"],
                "created": ["none"],
            },
            "04_2_create_canonical_schema.py": {
                "required": ["mappings/MANUAL_contract.json", "LLM_canonical_schema_targets.json"],
                "created": ["LLM_canonical_schema_template.json", "04_02_llm_canonical_schema_candidate.json"],
            },
            "04_3_create_dataset_mapping.py": {
                "required": ["mappings/MANUAL_contract.json", "MANUAL_canonical_schemas.json"],
                "created": ["MANUAL_dataset_mapping.json"],
            },
            "04_4_runner.py": {
                "required": ["MANUAL_canonical_schemas.json", "MANUAL_dataset_mapping.json"],
                "created": ["encoding_spec.json", "vocabulary.json", "scaler.json", "mapping_report.json"],
            },
            "04_5_encoding.py": {
                "required": ["mappings/MANUAL_contract.json", "MANUAL_canonical_schemas.json",
                             "MANUAL_dataset_mapping.json"],
                "created": ["encoding_spec.json", "vocabulary.json", "scaler.json", "mapping_report.json"],
            },
            "04_6_analyze_llm_outputs.py": {
                "required": ["MANUAL_canonical_schemas.json", "MANUAL_dataset_mapping.json", "canonical_schemas",
                             "dataset_mappings"],
                "created": ["04_06_llm_analysis_results.json", "04_06_llm_analysis_summary.txt"],
            },
            "04_7_decentralized_metadata_poc.py": {
                "required": ["MANUAL_canonical_schemas.json", "MANUAL_dataset_mapping.json", "encoded_metadata"],
                "created": [
                    "decentralized_poc/local_stats/<dataset>/<run>.json",
                    "decentralized_poc/secure_aggregation_messages/<dataset>/<run>.json",
                    "decentralized_poc/server_aggregation/<dataset>/<run>.json",
                    "decentralized_poc/comparison_reports/<dataset>/<run>.json",
                    "04_07_DECENTRALIZED_poc_summary.json",
                ],
            },
        }

        for script_name, checks in expected.items():
            path = (REPO_ROOT / "E_prefix_encoding") / script_name
            docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
            self.assertIn("REQUIRED FILES:", docstring)
            self.assertIn("CREATED FILES", docstring)
            for fragment in checks["required"] + checks["created"]:
                self.assertIn(fragment, docstring, f"{fragment} missing from {script_name}")

    # Verify contract exposes profiles and run matrices.
    def test_contract_exposes_profiles_and_run_matrices(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")

        self.assertEqual(contract.CONTRACT_PATH.name, "MANUAL_contract.json")
        self.assertEqual(contract.PAD_TOKEN, "[PAD]")
        self.assertEqual(contract.CANONICAL_ACTIVITY_TOKEN, "canonical_activity_token")
        self.assertIn(contract.REMAINING_TIME_MASK, contract.MAPPED_COLUMNS)

    # Verify contract JSON is the editable feature contract.
    def test_contract_json_is_the_editable_feature_contract(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")
        path = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_contract.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(contract.load_contract_payload(path), payload)
        self.assertEqual(payload["reserved_tokens"]["pad"], "[PAD]")
        self.assertIn("field_catalog", payload)
        self.assertIn("canonical_activity_labels", payload)
        self.assertEqual(payload["canonical_activity_labels"], contract.CANONICAL_ACTIVITY_LABELS)

    # Verify contract JSON validation rejects missing required sections.
    def test_contract_json_validation_rejects_missing_required_sections(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")
        payload = contract.load_contract_payload()
        payload.pop("field_catalog")

        with self.assertRaisesRegex(ValueError, "field_catalog"):
            contract.validate_contract_payload(payload)

    # Verify contract JSON validation rejects invalid activity prefix.
    def test_contract_json_validation_rejects_invalid_activity_prefix(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")
        payload = contract.load_contract_payload()
        payload["canonical_activity_labels"]["X_bad_label"] = {
            "description": "Invalid activity prefix.",
            "terminal_candidate": False,
        }

        with self.assertRaisesRegex(ValueError, "canonical activity"):
            contract.validate_contract_payload(payload)

    # Verify contract exposes activity label universe.
    def test_contract_exposes_activity_label_universe(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")

        self.assertTrue(hasattr(contract, "FIELD_CATALOG"))
        self.assertTrue(hasattr(contract, "CANONICAL_ACTIVITY_LABELS"))
        self.assertTrue(hasattr(contract, "CANONICAL_ACTIVITY_LABEL_NAMES"))
        self.assertIn("A_create_application", contract.CANONICAL_ACTIVITY_LABEL_NAMES)
        self.assertIn("O_accept_offer", contract.CANONICAL_ACTIVITY_LABEL_NAMES)
        self.assertNotIn(contract.OTHER_ACTIVITY_TOKEN, contract.CANONICAL_ACTIVITY_LABEL_NAMES)

    # Verify contract activity labels preserve event origin prefixes.
    def test_contract_activity_labels_preserve_event_origin_prefixes(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")

        for label in contract.CANONICAL_ACTIVITY_LABEL_NAMES: self.assertRegex(label, r"^[AOW]_")

        self.assertIn("O_offer_created", contract.CANONICAL_ACTIVITY_LABEL_NAMES)
        self.assertIn("W_shortened_completion", contract.CANONICAL_ACTIVITY_LABEL_NAMES)
        self.assertNotIn("create_application", contract.CANONICAL_ACTIVITY_LABEL_NAMES)

    # Verify contract contains no dataset mapping defaults.
    def test_contract_contains_no_dataset_mapping_defaults(self) -> None:
        contract = importlib.import_module("E_prefix_encoding.04_1_contract")
        forbidden_exports = [
            "default_activity_mapping",
            "dataset_mapping_template",
            "build_dataset_mapping_payload",
            "build_schema_payload",
            "RUN_MATRICES",
        ]

        for name in forbidden_exports:
            self.assertFalse(hasattr(contract, name), name)

    # Verify contract contains no source dataset names or columns.
    def test_contract_contains_no_source_dataset_names_or_columns(self) -> None:
        text = (REPO_ROOT / "E_prefix_encoding/04_1_contract.py").read_text(encoding="utf-8")
        forbidden_fragments = [
            "A_Create Application", "A_SUBMITTED", "A_accept_application", "case:concept:name", "case:RequestedAmount",
            "case:AMOUNT_REQ", "E_main_BPIC_2017", "E_ablation_BPIC_2012",
        ]

        for fragment in forbidden_fragments: self.assertNotIn(fragment, text)

    # Verify CLI overrides are optional for user scripts.
    def test_cli_overrides_are_optional_for_user_scripts(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")

        self.assertEqual(create_schema.parse_args([]).schema_mode, create_schema.SCHEMA_MODE)
        self.assertEqual(create_mapping.parse_args([]).schema_profile, create_mapping.SCHEMA_PROFILE)
        self.assertEqual(runner.parse_args([]).schema_profile, runner.SCHEMA_PROFILE)

    # Verify mapping review files live in mappings folder.
    def test_mapping_review_files_live_in_mappings_folder(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")

        self.assertEqual(create_schema.CANONICAL_SCHEMA_PATH.parent.name, "llm_mapping")
        self.assertEqual(create_schema.CANONICAL_SCHEMA_PATH.parent.parent.name, "mappings")
        self.assertEqual(create_mapping.CANONICAL_SCHEMA_PATH.parent.name, "mappings")
        self.assertEqual(create_mapping.DATASET_MAPPING_PATH.parent.name, "mappings")
        self.assertEqual(runner.CANONICAL_SCHEMA_PATH.parent.name, "mappings")
        self.assertEqual(runner.DATASET_MAPPING_PATH.parent.name, "mappings")
        self.assertEqual(create_schema.CANONICAL_SCHEMA_PATH.name, "LLM_canonical_schema_template.json")
        self.assertEqual(create_schema.LLM_CANONICAL_SCHEMA_PATH.parent.name, "canonical_schemas")
        self.assertEqual(create_schema.LLM_CANONICAL_SCHEMA_PATH.parent.parent.name, "llm_mapping")
        self.assertEqual(create_schema.CANONICAL_SCHEMA_TARGETS_PATH.parent.name, "llm_mapping")
        self.assertEqual(create_schema.CANONICAL_SCHEMA_TARGETS_PATH.name, "LLM_canonical_schema_targets.json")
        self.assertEqual(create_mapping.DATASET_MAPPING_PATH.name, "MANUAL_dataset_mapping.json")
        self.assertEqual(create_mapping.CANONICAL_SCHEMA_PATH.name, "MANUAL_canonical_schemas.json")
        self.assertEqual(runner.CANONICAL_SCHEMA_PATH.name, "MANUAL_canonical_schemas.json")

    # Verify contract context summary exports metadata, not rows.
    def test_contract_context_summary_exports_metadata_not_rows(self) -> None:
        contract_context = importlib.import_module("E_prefix_encoding.04_0_extract_contract_context")
        with tempfile.TemporaryDirectory() as tmp:
            input_root = Path(tmp) / "processed"
            run_dir = input_root / "iid_3banks"
            run_dir.mkdir(parents=True)
            frame = pd.DataFrame(
                {
                    "case:concept:name": ["case_1", "case_1", "case_2"],
                    "concept:name": ["A_Create Application", "O_Create Offer", "A_Denied"],
                    "lifecycle:transition": ["complete", "complete", "complete"],
                    "org:resource": ["secret_user_1", "secret_user_2", "secret_user_3"],
                    "case:RequestedAmount": [10000.0, 10000.0, 25000.0],
                    "time:timestamp": pd.to_datetime(
                        ["2017-01-01 00:00:00", "2017-01-01 00:10:00", "2017-01-02 00:00:00"],
                        utc=True,
                    ),
                    "outcome": [2, 2, 1],
                }
            )
            frame.to_parquet(run_dir / "A_02_bank_A_train.parquet", index=False)

            summary = contract_context.summarize_dataset_context(
                dataset_id="bpic2017", input_root=input_root, split_prefix="A_02", heterogeneities=("iid_3banks",),
                splits=("train",), include_activity_counts=False, include_missingness=True,
                include_numeric_train_stats=True,
            )

        serialized = json.dumps(summary, sort_keys=True)
        self.assertEqual(summary["dataset_id"], "bpic2017")
        self.assertIn("case:concept:name", summary["columns"]["names"])
        self.assertIn("case:RequestedAmount", summary["numeric_train_statistics"])
        self.assertEqual(summary["trace_length_summary"]["case_count"], 2)
        self.assertEqual(summary["activity_labels"], ["A_Create Application", "A_Denied", "O_Create Offer"])
        self.assertNotIn("case_1", serialized)
        self.assertNotIn("secret_user_1", serialized)
        self.assertNotIn("2017-01-01 00:00:00", serialized)

    # Verify contract context writer creates dataset and federation files.
    def test_contract_context_writer_creates_dataset_and_federation_files(self) -> None:
        contract_context = importlib.import_module("E_prefix_encoding.04_0_extract_contract_context")
        summaries = [
            {"dataset_id": "bpic2017", "columns": {"names": ["a", "b"]}, "activity_labels": ["A"],
             "trace_length_summary": {"case_count": 1, "event_count": 2}},
            {"dataset_id": "bpic2012", "columns": {"names": ["b", "c"]}, "activity_labels": ["B"],
             "trace_length_summary": {"case_count": 1, "event_count": 3}},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            paths = contract_context.write_contract_context_files(output_root, summaries)

            self.assertEqual(sorted(path.name for path in paths), [
                "04_00_bpic2012_contract_context.json",
                "04_00_bpic2017_contract_context.json",
                "04_00_federation_contract_context.json",
            ])
            federation_path = output_root / "04_00_federation_contract_context.json"
            federation = json.loads(federation_path.read_text(encoding="utf-8"))

        self.assertEqual(federation["datasets"], ["bpic2012", "bpic2017"])
        self.assertEqual(federation["column_overlap"]["shared_columns"], ["b"])

    # Verify manual schema mode writes fillable template.
    def test_manual_schema_mode_writes_fillable_template(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        payload = create_schema.build_schema_payload("manual", "test")

        self.assertFalse(payload["approved"])
        self.assertIn("<profile_name>", payload["schema_profiles"])
        placeholder = payload["schema_profiles"]["<profile_name>"]
        self.assertEqual(placeholder["datasets"], [])
        self.assertEqual(placeholder["sequence_categorical_columns"], [])
        self.assertEqual(placeholder["sequence_numerical_columns"], [])
        self.assertEqual(placeholder["offer_numerical_columns"], [])
        self.assertIsNone(placeholder["max_prefix_length_for_encoding"])
        self.assertIn("Replace <profile_name>", placeholder["review_note"])
        self.assertIn("field_name", payload["contract_field_catalog"]["case_id"])
        self.assertIn("A_create_application", payload["canonical_activity_labels"])

    # Verify canonical schema script has no bpic profile builder logic.
    def test_canonical_schema_script_has_no_bpic_profile_builder_logic(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        forbidden_exports = [
            "SCHEMA_PROFILES_TO_CREATE",
            "DATASETS_BY_SCHEMA_PROFILE",
            "PROFILE_SEQUENCE_CATEGORICAL_COLUMNS",
            "PROFILE_SEQUENCE_NUMERICAL_COLUMNS",
            "PROFILE_OFFER_NUMERICAL_COLUMNS",
            "PROFILE_MAX_PREFIX_LENGTHS",
            "build_default_schema_profiles",
        ]

        for name in forbidden_exports:
            self.assertFalse(hasattr(create_schema, name), name)

    # Verify manual reference schema JSON is valid.
    def test_manual_reference_schema_json_is_valid(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        path = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_canonical_schemas.json")
        payload = json.loads(path.read_text(encoding="utf-8"))

        create_schema.validate_canonical_schema_payload(payload, require_profiles=True)

        self.assertEqual(sorted(payload["schema_profiles"]), ["bpic2012", "bpic2017", "joint"])
        self.assertEqual(sorted(payload["canonical_activity_labels"]),
                         sorted(create_schema.contract.CANONICAL_ACTIVITY_LABEL_NAMES))

    # Verify the bad LLM schema shape is rejected.
    def test_bad_llm_schema_shape_is_rejected(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        bad_payload = create_schema.build_schema_payload("manual", "test")
        bad_payload["schema_profiles"] = [{"profile_name": "bpic2017", "issues": []}]

        with self.assertRaisesRegex(ValueError, "schema_profiles"):
            create_schema.validate_canonical_schema_payload(bad_payload, require_profiles=True)

    # Verify LLM mode records malformed schema response for side experiment.
    def test_llm_mode_records_malformed_schema_response_for_side_experiment(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        bad_response = create_schema.build_schema_payload("manual", "test")
        bad_response["schema_profiles"] = [{"profile_name": "bpic2017"}]
        original = create_schema.request_openai_json
        create_schema.request_openai_json = lambda prompt, model: bad_response
        try:
            payload = create_schema.build_schema_payload("llm", "test")
        finally:
            create_schema.request_openai_json = original

        self.assertFalse(payload["approved"])
        self.assertEqual(payload["mode"], "llm")
        self.assertIn("validation_error", payload)
        self.assertIn("schema_profiles", payload["validation_error"])

    # Verify LLM mode accepts filled template response.
    def test_llm_mode_accepts_filled_template_response(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        schema_path = REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_canonical_schemas.json"
        reference = json.loads(schema_path.read_text(encoding="utf-8"))
        reference["approved"] = False
        reference["mode"] = "llm"
        original = create_schema.request_openai_json
        create_schema.request_openai_json = lambda prompt, model: reference
        try:
            payload = create_schema.build_schema_payload("llm", "test")
        finally:
            create_schema.request_openai_json = original

        self.assertEqual(payload["mode"], "llm")
        self.assertEqual(sorted(payload["schema_profiles"]), ["bpic2012", "bpic2017", "joint"])

    # Verify OpenAI schema timeout raises a clear error.
    def test_openai_schema_timeout_raises_clear_error(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        original_urlopen = create_schema.urllib.request.urlopen
        original_retries = create_schema.OPENAI_RETRIES
        create_schema.OPENAI_RETRIES = 1
        create_schema.urllib.request.urlopen = (
            lambda request, timeout: (_ for _ in ()).throw(TimeoutError("read timed out")))
        try:
            with self.assertRaisesRegex(RuntimeError, "OpenAI schema draft timed out"):
                create_schema.request_openai_json({}, "test", api_key="test-key")
        finally:
            create_schema.urllib.request.urlopen = original_urlopen
            create_schema.OPENAI_RETRIES = original_retries

    # Verify OpenAI mapping timeout raises a clear error.
    def test_openai_mapping_timeout_raises_clear_error(self) -> None:
        create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
        original_urlopen = create_mapping.urllib.request.urlopen
        original_retries = create_mapping.OPENAI_RETRIES
        create_mapping.OPENAI_RETRIES = 1
        create_mapping.urllib.request.urlopen = (
            lambda request, timeout: (_ for _ in ()).throw(TimeoutError("read timed out")))
        try:
            with self.assertRaisesRegex(RuntimeError, "OpenAI mapping draft timed out"):
                create_mapping.request_openai_json({}, "test", api_key="test-key")
        finally:
            create_mapping.urllib.request.urlopen = original_urlopen
            create_mapping.OPENAI_RETRIES = original_retries

    # Verify LLM schema prompt names required profiles and fields.
    def test_llm_schema_prompt_names_required_profiles_and_fields(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        template = create_schema.build_schema_template("llm", "test")
        targets_path = (REPO_ROOT / "E_prefix_encoding/mappings/llm_mapping/LLM_canonical_schema_targets.json")
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        prompt = create_schema.build_llm_prompt(template, targets, "strategy_1")
        prompt_text = json.dumps(prompt)

        self.assertIn("profiles_to_create", prompt_text)
        self.assertIn("bpic2017", prompt_text)
        self.assertIn("bpic2012", prompt_text)
        self.assertIn("joint", prompt_text)
        self.assertIn("sequence_categorical_columns", prompt_text)
        self.assertIn("sequence_numerical_columns", prompt_text)
        self.assertIn("offer_numerical_columns", prompt_text)
        self.assertIn("max_prefix_length_for_encoding", prompt_text)
        self.assertIn("schema_profiles must be an object keyed by profile name", prompt_text)

    # Verify LLM schema prompt strategy two forbids non-input fields.
    def test_llm_schema_prompt_strategy_two_forbids_non_input_fields(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
        template = create_schema.build_schema_template("llm", "test")
        targets_path = (REPO_ROOT / "E_prefix_encoding/mappings/llm_mapping/LLM_canonical_schema_targets.json")
        targets = json.loads(targets_path.read_text(encoding="utf-8"))
        prompt = create_schema.build_llm_prompt(template, targets, "strategy_2")
        prompt_text = json.dumps(prompt)

        self.assertIn("allowed_model_input_fields", prompt_text)
        self.assertIn("forbidden_model_input_fields", prompt_text)
        self.assertIn("remaining_time", prompt["forbidden_model_input_fields"])
        self.assertIn("next_activity_target", prompt["forbidden_model_input_fields"])
        self.assertIn("split", prompt["forbidden_model_input_fields"])

    # Verify LLM dataset mapping strategy two uses split prompts.
    def test_llm_dataset_mapping_strategy_two_uses_split_prompts(self) -> None:
        create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
        schema_path = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_canonical_schemas.json")
        schema_payload = json.loads(schema_path.read_text(encoding="utf-8"))
        profile = create_mapping.get_schema_profile(schema_payload, "bpic2017")
        allowed_labels = create_mapping.allowed_canonical_activity_labels(schema_payload)
        metadata = {
            "bpic2017": {
                "available_columns": {
                    "case:concept:name": "str",
                    "concept:name": "str",
                    "time:timestamp": "datetime64",
                },
                "raw_activity_labels": ["A_Create Application", "O_Sent (online only)"],
            }
        }
        datasets = {"bpic2017": {"column_mapping": {"case_id": ""}, "default_values": {}}}

        column_prompt = create_mapping.build_llm_column_prompt(profile, metadata, datasets, "strategy_2")
        activity_prompt = create_mapping.build_llm_activity_prompt(profile, metadata, allowed_labels, "strategy_2")

        self.assertIn("column_mapping", json.dumps(column_prompt))
        self.assertNotIn("manual", json.dumps(column_prompt).lower())
        self.assertIn("raw_activity_labels", json.dumps(activity_prompt))
        self.assertIn("preserve the A, O and W prefix", json.dumps(activity_prompt))
        self.assertIn("labels_by_dataset", json.dumps(activity_prompt))

    # Verify canonical schema script has no semantic mode.
    def test_canonical_schema_script_has_no_semantic_mode(self) -> None:
        create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")

        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            create_schema.parse_args(["--schema-mode", "semantic"])

    # Verify E_04 python sources avoid pipe operator for the static checker.
    def test_e04_python_sources_avoid_pipe_operator_for_static_checker(self) -> None:
        offenders = {}
        pipe_operator = chr(124)
        for path in (REPO_ROOT / "E_prefix_encoding").glob("04_*.py"):
            # The check targets the PEP 604 union operator, so comments are dropped before the scan.
            # Configuration comments spell out their allowed values as "raw | log" and are not code.
            source = "".join(_strip_python_comments(path.read_text(encoding="utf-8")))
            lines = [line for line in source.splitlines() if pipe_operator in line]
            if lines:
                offenders[str(path)] = lines

        self.assertEqual(offenders, {})

    # Verify runner matrix excludes unapproved five bank iid and weak.
    def test_runner_matrix_excludes_unapproved_five_bank_iid_and_weak(self) -> None:
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")

        self.assertEqual(runner.matrix_for_profile("bpic2017")[0], ("bpic2017", "iid", 3, 83))
        self.assertNotIn(("bpic2017", "iid", 5, 83), runner.matrix_for_profile("bpic2017"))
        self.assertNotIn(("bpic2017", "weak", 5, 83), runner.matrix_for_profile("bpic2017"))

    # Verify runner all matrix combines required dataset matrices.
    def test_runner_all_matrix_combines_required_dataset_matrices(self) -> None:
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")

        expected = (
            runner.matrix_for_profile("bpic2017")
            + runner.matrix_for_profile("bpic2012")
            + runner.matrix_for_profile("joint")
        )

        self.assertEqual(runner.matrix_for_profile("all"), expected)

    # Verify runner single profile matrix uses the selected schema profile.
    def test_runner_single_profile_matrix_uses_selected_schema_profile(self) -> None:
        runner = importlib.import_module("E_prefix_encoding.04_4_runner")
        original_load_schema = runner.encoding.load_approved_json
        original_load_mapping = runner.encoding.load_dataset_mapping
        original_hash = runner.encoding.json_sha256
        original_run_one = runner.run_one
        try:
            runner.encoding.load_approved_json = lambda path, label: {
                "schema_profiles": {"bpic2017": {"datasets": ["bpic2017"]}},
            }
            runner.encoding.load_dataset_mapping = lambda path, require_approved=True: {
                "schema_profile": "bpic2017",
                "schema_sha256": "schema-hash",
                "datasets": {"bpic2017": {}},
            }
            runner.encoding.json_sha256 = lambda path: "schema-hash"
            runner.run_one = lambda config, profile, mapping, path: (config.dataset, config.schema_profile)

            paths = runner.run_full_matrix("bpic2017", Path("encoded"), Path("schema.json"), Path("mapping.json"))
        finally:
            runner.encoding.load_approved_json = original_load_schema
            runner.encoding.load_dataset_mapping = original_load_mapping
            runner.encoding.json_sha256 = original_hash
            runner.run_one = original_run_one

        self.assertEqual(len(paths), len(runner.matrix_for_profile("bpic2017")))
        self.assertTrue(all(schema_profile == "bpic2017" for _, schema_profile in paths))

    # Verify schema uses the multiclass outcome without the outcome mask.
    def test_schema_uses_multiclass_outcome_without_outcome_mask(self) -> None:
        encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

        self.assertNotIn("outcome_mask", encoding.REQUIRED_COLUMNS)
        self.assertFalse(hasattr(encoding, "OUTCOME_MASK"))
        self.assertFalse(hasattr(encoding.EncodedBatch, "outcome_mask"))

    # Verify manual dataset mapping is approved and ground truth ready.
    def test_manual_dataset_mapping_is_approved_and_ground_truth_ready(self) -> None:
        path = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_dataset_mapping.json")
        payload = json.loads(path.read_text(encoding="utf-8"))
        activity = payload["activity_mapping"]["canonical_activities"]

        self.assertTrue(payload["approved"])
        self.assertEqual(payload["schema_profile"], "all")
        self.assertIn("bpic2017", payload["datasets"])
        self.assertIn("bpic2012", payload["datasets"])
        self.assertEqual(
            sorted(activity["O_send_offer"]["labels_by_dataset"]["bpic2017"]),
            ["O_Sent (mail and online)", "O_Sent (online only)"],
        )
        self.assertEqual(activity["O_send_offer"]["labels_by_dataset"]["bpic2012"], ["O_SENT"])
        self.assertEqual(activity["O_create_offer"]["labels_by_dataset"]["bpic2017"], ["O_Create Offer"])
        self.assertEqual(activity["O_offer_created"]["labels_by_dataset"]["bpic2017"], ["O_Created"])
        self.assertEqual(activity["W_personal_loan_collection"]["labels_by_dataset"]["bpic2017"],
                         ["W_Personal Loan collection"])
        self.assertEqual(activity["W_shortened_completion"]["labels_by_dataset"]["bpic2017"],
                         ["W_Shortened completion "])

    # Verify LLM analysis logic stays out of encoder files.
    def test_llm_analysis_logic_stays_out_of_encoder_files(self) -> None:
        forbidden = [
            "score_activity_mapping",
            "llm_strategy_summary",
            "ground_truth",
            "llm_analysis",
        ]
        checked_paths = [
            (REPO_ROOT / "E_prefix_encoding/04_1_contract.py"),
            (REPO_ROOT / "E_prefix_encoding/04_2_create_canonical_schema.py"),
            (REPO_ROOT / "E_prefix_encoding/04_3_create_dataset_mapping.py"),
            (REPO_ROOT / "E_prefix_encoding/04_5_encoding.py"),
            (REPO_ROOT / "E_prefix_encoding/04_4_runner.py"),
        ]

        for path in checked_paths:
            text = path.read_text(encoding="utf-8")
            for fragment in forbidden:
                self.assertNotIn(fragment, text, f"{fragment} leaked into {path}")

    # Verify LLM analysis discovers dataset strategy outputs.
    def test_llm_analysis_discovers_dataset_strategy_outputs(self) -> None:
        analyze_llm = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")

        self.assertEqual(analyze_llm.LLM_MAPPING_ROOT.name, "llm_mapping")
        self.assertEqual(analyze_llm.LLM_SCHEMA_ROOT.name, "canonical_schemas")
        self.assertEqual(analyze_llm.LLM_DATASET_MAPPING_ROOT.name, "dataset_mappings")
        self.assertIn("*.json", analyze_llm.LLM_SCHEMA_PATTERNS)
        self.assertIn("*.json", analyze_llm.LLM_MAPPING_PATTERNS)

    # Verify LLM strategy files use grouped meaningful names.
    def test_llm_strategy_files_use_grouped_meaningful_names(self) -> None:
        text = (REPO_ROOT / "E_prefix_encoding/WORKFLOW_run_encoding.sh").read_text(encoding="utf-8")

        self.assertIn("04_02_strategy_1_baseline_canonical_schema.json", text)
        self.assertIn("04_02_strategy_2_field_rules_canonical_schema.json", text)
        self.assertIn("04_02_strategy_3_target_recipe_canonical_schema.json", text)
        self.assertIn("04_03_strategy_1_baseline_dataset_mapping.json", text)
        self.assertIn("04_03_strategy_2_split_prompt_dataset_mapping.json", text)
        self.assertIn("04_03_strategy_3_target_recipe_dataset_mapping.json", text)

    # Verify LLM dataset mapping workflow runs seedless strategies once.
    def test_llm_dataset_mapping_workflow_runs_seedless_strategies_once(self) -> None:
        text = (REPO_ROOT / "E_prefix_encoding/WORKFLOW_run_encoding.sh").read_text(encoding="utf-8")

        self.assertIn('seed_mapping_root="$LLM_MAPPING_ROOT/$run_id/$semantic_variant"', text)
        self.assertIn('run_mapping_root="$LLM_MAPPING_ROOT/$run_id"', text)
        self.assertIn(
            '--dataset-mapping-path "$seed_mapping_root/04_03_strategy_1_baseline_dataset_mapping.json"', text)
        self.assertIn(
            '--dataset-mapping-path "$run_mapping_root/04_03_strategy_2_split_prompt_dataset_mapping.json"', text)
        self.assertIn(
            '--dataset-mapping-path "$run_mapping_root/04_03_strategy_3_target_recipe_dataset_mapping.json"', text)

    # Verify LLM analysis writes the readable strategy table.
    def test_llm_analysis_writes_readable_strategy_table(self) -> None:
        analyze_llm = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            schema_root = root / "canonical_schemas"
            mapping_root = root / "dataset_mappings"
            analysis_root = root / "llm_analysis"
            schema_root.mkdir()
            mapping_root.mkdir()
            manual_schema = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_canonical_schemas.json")
            manual_mapping = (REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_dataset_mapping.json")
            (schema_root / "04_02_strategy_3_target_recipe_canonical_schema.json").write_text(
                manual_schema.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (mapping_root / "04_03_strategy_3_target_recipe_dataset_mapping.json").write_text(
                manual_mapping.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            summary = analyze_llm.analyze_llm_outputs(
                manual_schema, manual_mapping, schema_root, mapping_root, analysis_root,
            )
            summary_path = analysis_root / "04_06_llm_analysis_summary.txt"
            summary_text = summary_path.read_text(encoding="utf-8")
            results_path = analysis_root / "04_06_llm_analysis_results.json"
            results = json.loads(results_path.read_text(encoding="utf-8"))

        self.assertIn("strategy_table", summary)
        self.assertIn("schema_scores", summary)
        self.assertIn("column_mapping_scores", summary)
        self.assertIn("activity_mapping_scores", summary)
        self.assertIn("repeated_mapping_summary", summary)
        self.assertEqual(results["strategy_table"], summary["strategy_table"])
        self.assertIn("field_f1", summary["strategy_table"][0])
        self.assertNotIn("primary_score", summary["strategy_table"][0])
        self.assertIn("04_02_strategy_3_target_recipe_canonical_schema", summary_text)
        self.assertIn("04_03_strategy_3_target_recipe_dataset_mapping", summary_text)
        self.assertIn("Rank", summary_text)
        self.assertIn("Field F1", summary_text)
        self.assertIn("Activity accuracy", summary_text)
        self.assertIn("Issue summary", summary_text)
        self.assertIn("Metric explanations", summary_text)

    # Verify the workflow runs without user CLI arguments.
    def test_workflow_runs_without_user_cli_arguments(self) -> None:
        path = (REPO_ROOT / "E_prefix_encoding/WORKFLOW_run_encoding.sh")
        text = path.read_text(encoding="utf-8")

        self.assertIn('MANUAL_CONTRACT_PATH="mappings/MANUAL_contract.json"', text)
        self.assertIn('MANUAL_SCHEMA_PATH="mappings/MANUAL_canonical_schemas.json"', text)
        self.assertIn('MANUAL_MAPPING_PATH="mappings/MANUAL_dataset_mapping.json"', text)
        self.assertIn('LLM_TARGETS_PATH="mappings/llm_mapping/LLM_canonical_schema_targets.json"', text)
        self.assertIn('ARTIFACT_ROOT="encoded_metadata"', text)
        self.assertIn('RUN_MANUAL_ENCODING="${RUN_MANUAL_ENCODING:-true}"', text)
        self.assertIn('RUN_LLM_EXPERIMENT="${RUN_LLM_EXPERIMENT:-false}"', text)
        self.assertIn('RUN_LLM_SCHEMA_EXPERIMENT="${RUN_LLM_SCHEMA_EXPERIMENT:-false}"', text)
        self.assertIn('OPENAI_MODEL="${OPENAI_MODEL:-gpt-5-nano}"', text)
        self.assertIn('OPENAI_TIMEOUT_SECONDS="${OPENAI_TIMEOUT_SECONDS:-240}"', text)
        self.assertIn('OPENAI_RETRIES="${OPENAI_RETRIES:-2}"', text)
        self.assertIn("04_4_runner.py", text)
        self.assertIn("04_6_analyze_llm_outputs.py", text)
        self.assertNotIn("getopts", text)
        self.assertNotIn("04_2_create_canonical_schema.py --schema-mode manual", text)
        self.assertNotIn("04_3_create_dataset_mapping.py --mapping-mode manual", text)

    # Verify LLM activity scoring detects wrong and unresolved labels.
    def test_llm_activity_scoring_detects_wrong_and_unresolved_labels(self) -> None:
        analyze_llm = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")
        manual = {
            "activity_mapping": {
                "canonical_activities": {
                    "A_create_application": {"labels_by_dataset": {"bpic2017": ["A_Create Application"]}},
                    "O_send_offer": {"labels_by_dataset": {"bpic2017": ["O_Sent"]}},
                }
            }
        }
        llm = {
            "activity_mapping": {
                "canonical_activities": {
                    "A_create_application": {"labels_by_dataset": {"bpic2017": ["A_Create Application"]}},
                    "O_create_offer": {"labels_by_dataset": {"bpic2017": ["O_Sent"]}},
                },
                "unresolved_labels": ["A_Denied"],
            }
        }

        score = analyze_llm.score_activity_mapping(manual, llm)

        self.assertEqual(score["total_manual_labels"], 2)
        self.assertEqual(score["correct_label_count"], 1)
        self.assertEqual(score["wrong_label_count"], 1)
        self.assertEqual(score["unresolved_label_count"], 1)

    # Verify LLM schema scoring detects illegal input fields.
    def test_llm_schema_scoring_detects_illegal_input_fields(self) -> None:
        analyze_llm = importlib.import_module("E_prefix_encoding.04_6_analyze_llm_outputs")
        manual = {
            "schema_profiles": {
                "bpic2017": {
                    "datasets": ["bpic2017"],
                    "sequence_categorical_columns": ["canonical_activity_token", "resource"],
                    "sequence_numerical_columns": ["time_delta"],
                    "offer_numerical_columns": [],
                    "max_prefix_length_for_encoding": 83,
                }
            }
        }
        llm = {
            "schema_profiles": {
                "bpic2017": {
                    "datasets": ["bpic2017"],
                    "sequence_categorical_columns": ["canonical_activity_token", "split"],
                    "sequence_numerical_columns": ["time_delta", "remaining_time"],
                    "offer_numerical_columns": [],
                    "max_prefix_length_for_encoding": 83,
                }
            }
        }

        score = analyze_llm.score_canonical_schema(manual, llm)

        self.assertEqual(score["profile_count"], 1)
        self.assertEqual(score["exact_profile_match_count"], 0)
        self.assertEqual(score["illegal_field_count"], 2)
        self.assertLess(score["field_f1"], 1.0)

if __name__ == "__main__":
    unittest.main()