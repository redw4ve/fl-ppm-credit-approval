from __future__ import annotations
import importlib
import json
import sys
import tempfile
import unittest
from typing import Any
from pathlib import Path
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")
create_schema = importlib.import_module("E_prefix_encoding.04_2_create_canonical_schema")
create_mapping = importlib.import_module("E_prefix_encoding.04_3_create_dataset_mapping")
runner = importlib.import_module("E_prefix_encoding.04_4_runner")

# HELPER: Build BPIC 2017 frame.
def _bpic2017_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept:name": ["A_Create Application", "O_Create Offer", "O_Accepted"],
            "activity_token": [
                "A_Create Application+complete",
                "O_Create Offer+complete",
                "O_Accepted+complete",
            ],
            "lifecycle:transition": ["complete", "complete", "complete"],
            "org:resource": ["User_1", "User_2", "User_3"],
            "time:timestamp": pd.to_datetime(
                ["2017-01-01 00:00:00", "2017-01-01 00:10:00", "2017-01-01 00:20:00"],
                utc=True,
            ),
            "case:concept:name": ["case_1", "case_1", "case_1"],
            "case:RequestedAmount": [10000.0, 10000.0, 10000.0],
            "case:LoanGoal": ["Car", "Car", "Car"],
            "case:ApplicationType": ["New credit", "New credit", "New credit"],
            "CreditScore": [np.nan, 0.0, np.nan],
            "MonthlyCost": [np.nan, 300.0, np.nan],
            "OfferedAmount": [np.nan, 9000.0, np.nan],
            "NumberOfTerms": [np.nan, 24.0, np.nan],
            "NextActivity": ["O_Create Offer+complete", "O_Accepted+complete", "[END]"],
            "RemainingTime": [1200.0, 600.0, 0.0],
            "TimeDelta": [0.0, 600.0, 1200.0],
            "outcome": [2, 2, 2],
        }
    )

# HELPER: Build BPIC 2012 frame.
def _bpic2012_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept:name": ["A_SUBMITTED", "W_Completeren aanvraag", "A_APPROVED"],
            "activity_token": [
                "A_SUBMITTED+COMPLETE",
                "W_Completeren aanvraag+COMPLETE",
                "A_APPROVED+COMPLETE",
            ],
            "lifecycle:transition": ["COMPLETE", "COMPLETE", "COMPLETE"],
            "org:resource": ["112", None, "114"],
            "time:timestamp": pd.to_datetime(
                ["2012-01-01 00:00:00", "2012-01-01 00:05:00", "2012-01-01 00:10:00"],
                utc=True,
            ),
            "case:concept:name": ["173700", "173700", "173700"],
            "case:AMOUNT_REQ": [5000.0, 5000.0, 5000.0],
            "NextActivity": ["W_Completeren aanvraag+COMPLETE", "A_APPROVED+COMPLETE", "[END]"],
            "RemainingTime": [600.0, 300.0, 0.0],
            "TimeDelta": [0.0, 300.0, 600.0],
            "outcome": [2, 2, 2],
        }
    )

# HELPER: Build approved schema.
def _approved_schema(path: Path) -> dict[str, object]:
    manual_schema_path = REPO_ROOT / "E_prefix_encoding/mappings/MANUAL_canonical_schemas.json"
    payload = json.loads(manual_schema_path.read_text(encoding="utf-8"))
    payload["approved"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload

# HELPER: Build approved mapping.
def _approved_mapping(path: Path, schema_path: Path, profile: str) -> dict[str, Any]:
    schema_payload = _approved_schema(schema_path)
    payload = create_mapping.build_dataset_mapping_payload(schema_payload, schema_path, profile, "manual", "test")
    if profile in {"bpic2017", "joint"}:
        payload["datasets"]["bpic2017"]["column_mapping"].update(
            {
                encoding.CASE_ID: "case:concept:name",
                encoding.TIMESTAMP: "time:timestamp",
                encoding.RAW_ACTIVITY: "concept:name",
                encoding.LIFECYCLE: "lifecycle:transition",
                encoding.RAW_ACTIVITY_TOKEN: "activity_token",
                encoding.NEXT_ACTIVITY_RAW: "NextActivity",
                encoding.RESOURCE: "org:resource",
                encoding.TIME_DELTA: "TimeDelta",
                encoding.REMAINING_TIME: "RemainingTime",
                encoding.OUTCOME: "outcome",
                encoding.REQUESTED_AMOUNT_VALUE: "case:RequestedAmount",
                encoding.LOAN_GOAL: "case:LoanGoal",
                encoding.APPLICATION_TYPE: "case:ApplicationType",
                encoding.CREDIT_SCORE_VALUE: "CreditScore",
                encoding.MONTHLY_COST_VALUE: "MonthlyCost",
                encoding.OFFERED_AMOUNT_VALUE: "OfferedAmount",
                encoding.NUMBER_OF_TERMS_VALUE: "NumberOfTerms",
            }
        )
    if profile in {"bpic2012", "joint"}:
        payload["datasets"]["bpic2012"]["column_mapping"].update(
            {
                encoding.CASE_ID: "case:concept:name",
                encoding.TIMESTAMP: "time:timestamp",
                encoding.RAW_ACTIVITY: "concept:name",
                encoding.LIFECYCLE: "lifecycle:transition",
                encoding.RAW_ACTIVITY_TOKEN: "activity_token",
                encoding.NEXT_ACTIVITY_RAW: "NextActivity",
                encoding.RESOURCE: "org:resource",
                encoding.TIME_DELTA: "TimeDelta",
                encoding.REMAINING_TIME: "RemainingTime",
                encoding.OUTCOME: "outcome",
                encoding.REQUESTED_AMOUNT_VALUE: "case:AMOUNT_REQ",
            }
        )
    payload["activity_mapping"] = {
        "canonical_activities": {
            "A_create_application": {
                "labels_by_dataset": {"bpic2017": ["A_Create Application"], "bpic2012": ["A_SUBMITTED"]},
                "token_overrides": {},
                "rationale": "Test mapping.",
            },
            "O_create_offer": {
                "labels_by_dataset": {"bpic2017": ["O_Create Offer"]},
                "token_overrides": {},
                "rationale": "Test mapping.",
            },
            "O_accept_offer": {
                "labels_by_dataset": {"bpic2017": ["O_Accepted"]},
                "token_overrides": {},
                "rationale": "Test mapping.",
            },
            "A_approve_application": {
                "labels_by_dataset": {"bpic2012": ["A_APPROVED"]},
                "token_overrides": {},
                "rationale": "Test mapping.",
            },
            "W_complete_application": {
                "labels_by_dataset": {"bpic2012": ["W_Completeren aanvraag"]},
                "token_overrides": {},
                "rationale": "Test mapping.",
            },
        },
        "unresolved_labels": [],
    }
    payload["approved"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload

# HELPER: Build schema profile.
def _schema_profile(schema_path: Path, profile: str) -> dict[str, Any]:
    payload = json.loads(schema_path.read_text(encoding="utf-8"))
    return payload["schema_profiles"][profile]

class PrefixEncodingRuntimeTests(unittest.TestCase):
    # Verify manual schema mode writes unapproved profiles.
    def test_manual_schema_mode_writes_unapproved_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical_schemas.json"

            create_schema.write_canonical_schemas(path, "manual", "test", force=False)
            payload = json.loads(path.read_text(encoding="utf-8"))

            self.assertFalse(payload["approved"])
            self.assertIn("<profile_name>", payload["schema_profiles"])
            self.assertEqual(payload["schema_profiles"]["<profile_name>"]["datasets"], [])
            self.assertIn("A_create_application", payload["canonical_activity_labels"])

    # Verify the approved schema is not overwritten without force.
    def test_approved_schema_is_not_overwritten_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "canonical_schemas.json"
            payload = create_schema.build_schema_payload("manual", "test")
            payload["approved"] = True
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already approved"):
                create_schema.write_canonical_schemas(path, "manual", "test", force=False)

    # Verify dataset mapping requires approved schema.
    def test_dataset_mapping_requires_approved_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            schema_path.write_text(json.dumps(create_schema.build_schema_payload("manual", "test")), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "not approved"):
                create_mapping.write_dataset_mapping(schema_path, mapping_path, "bpic2017", "manual", "test",
                                                     force=False)

    # Verify semantic dataset mapping records selected profile.
    def test_semantic_dataset_mapping_records_selected_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            input_root = Path(tmp) / "processed"
            run_dir = input_root / "iid_3banks"
            run_dir.mkdir(parents=True)
            _bpic2017_frame().to_parquet(run_dir / "A_02_bank_A_train.parquet", index=False)
            _approved_schema(schema_path)

            original_inputs = create_mapping.DATASET_INPUTS
            create_mapping.DATASET_INPUTS = {
                **original_inputs,
                "bpic2017": {"input_root": input_root, "split_prefix": "A_02"},
            }
            try:
                create_mapping.write_dataset_mapping(schema_path, mapping_path, "bpic2017", "semantic", "test",
                                                     force=False)
            finally:
                create_mapping.DATASET_INPUTS = original_inputs
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))

            self.assertFalse(payload["approved"])
            self.assertEqual(payload["schema_profile"], "bpic2017")
            self.assertEqual(payload["datasets"]["bpic2017"]["column_mapping"][encoding.RAW_ACTIVITY], "concept:name")
            self.assertIn("A_create_application", payload["activity_mapping"]["canonical_activities"])
            self.assertIn("allowed_canonical_activity_labels", payload["activity_mapping"])

    # Verify dataset mapping rejects unknown canonical activity label.
    def test_dataset_mapping_rejects_unknown_canonical_activity_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            mapping["activity_mapping"]["canonical_activities"]["not_in_contract"] = {
                "labels_by_dataset": {"bpic2017": ["X_Custom"]},
                "token_overrides": {},
                "rationale": "Invalid test label.",
            }
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "unknown canonical activity labels"):
                encoding.load_dataset_mapping(mapping_path, require_approved=True)

    # Verify other activity is reserved not defined by mapping.
    def test_other_activity_is_reserved_not_defined_by_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            mapping["activity_mapping"]["canonical_activities"][encoding.OTHER_ACTIVITY_TOKEN] = {
                "labels_by_dataset": {"bpic2017": ["X_Other"]},
                "token_overrides": {},
                "rationale": "Invalid reserved label.",
            }
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "reserved fallback"):
                encoding.load_dataset_mapping(mapping_path, require_approved=True)

    # Verify dataset mapping rejects cross-prefix activity mapping.
    def test_dataset_mapping_rejects_cross_prefix_activity_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            mapping["activity_mapping"]["canonical_activities"]["O_create_offer"][
                "labels_by_dataset"]["bpic2017"].append(
                "A_Create Application"
            )
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "cross-prefix"):
                encoding.load_dataset_mapping(mapping_path, require_approved=True)

    # Verify runner rejects stale schema hash before loading parquets.
    def test_runner_rejects_stale_schema_hash_before_loading_parquets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            mapping["schema_sha256"] = "stale"
            mapping_path.write_text(json.dumps(mapping), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema hash"):
                runner.run_full_matrix("bpic2017", Path(tmp) / "artifacts", schema_path, mapping_path)

    # Verify BPIC 2017 mapping converts columns correctly.
    def test_bpic2017_mapping_converts_columns_correctly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")

            events = encoding.to_canonical_events(_bpic2017_frame(), mapping, "bpic2017", "A", "train")

            self.assertEqual(events[encoding.CASE_ID].tolist(), ["case_1", "case_1", "case_1"])
            self.assertEqual(events[encoding.REQUESTED_AMOUNT_VALUE].tolist(), [10000.0, 10000.0, 10000.0])
            self.assertEqual(events[encoding.LOAN_GOAL].tolist(), ["Car", "Car", "Car"])
            self.assertEqual(events[encoding.APPLICATION_TYPE].tolist(), ["New credit", "New credit", "New credit"])
            self.assertNotIn("EventOrigin", events.columns)
            self.assertEqual(events[encoding.OFFER_FEATURE_MASK].tolist(), [0, 1, 0])

    # Verify temporal features are derived from the timestamp.
    def test_temporal_features_are_derived_from_the_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")

            events = encoding.to_canonical_events(_bpic2017_frame(), mapping, "bpic2017", "A", "train")

            # The fixture events are ten minutes apart on Sunday 2017-01-01 at midnight.
            self.assertEqual(events[encoding.TIME_SINCE_PREVIOUS].tolist(), [0.0, 600.0, 600.0])
            # Weekday six and hour zero map onto their cyclical sin and cos pairs.
            expected_weekday_sin = float(np.sin(2.0 * np.pi * 6.0 / 7.0))
            expected_weekday_cos = float(np.cos(2.0 * np.pi * 6.0 / 7.0))
            for value in events[encoding.WEEKDAY_SIN].tolist():
                self.assertAlmostEqual(value, expected_weekday_sin, places=6)
            for value in events[encoding.WEEKDAY_COS].tolist():
                self.assertAlmostEqual(value, expected_weekday_cos, places=6)
            for value in events[encoding.HOUR_SIN].tolist():
                self.assertAlmostEqual(value, 0.0, places=6)
            for value in events[encoding.HOUR_COS].tolist():
                self.assertAlmostEqual(value, 1.0, places=6)

    # Verify cyclical features use an identity scaler.
    def test_cyclical_features_use_an_identity_scaler(self) -> None:
        train = pd.DataFrame(
            {
                encoding.WEEKDAY_SIN: [0.5, -0.5, 0.9],
                encoding.HOUR_COS: [1.0, 0.0, -1.0],
                encoding.TIME_DELTA: [0.0, 100.0, 200.0],
            }
        )

        scalers = encoding.fit_all_scalers(
            train, numerical_columns=[encoding.WEEKDAY_SIN, encoding.HOUR_COS, encoding.TIME_DELTA]
        )

        # The four cyclical features pass through unchanged while time_delta keeps a real fitted scaler.
        self.assertEqual(scalers[encoding.WEEKDAY_SIN], {"mean": 0.0, "std": 1.0})
        self.assertEqual(scalers[encoding.HOUR_COS], {"mean": 0.0, "std": 1.0})
        self.assertNotEqual(scalers[encoding.TIME_DELTA], {"mean": 0.0, "std": 1.0})
        passthrough = encoding.transform_with_scaler(pd.Series([0.5, -0.5, 0.9]), scalers[encoding.WEEKDAY_SIN])
        self.assertTrue(np.allclose(passthrough.to_numpy(), np.array([0.5, -0.5, 0.9], dtype=np.float32)))

    # Verify BPIC 2012 mapping converts columns without the offer profile.
    def test_bpic2012_mapping_converts_columns_without_offer_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2012")

            events = encoding.to_canonical_events(_bpic2012_frame(), mapping, "bpic2012", "A", "train")

            self.assertEqual(events[encoding.CASE_ID].tolist(), ["173700", "173700", "173700"])
            self.assertEqual(events[encoding.REQUESTED_AMOUNT_VALUE].tolist(), [5000.0, 5000.0, 5000.0])
            self.assertEqual(events[encoding.RESOURCE].tolist(), ["112", "[MISSING]", "114"])
            self.assertNotIn(encoding.LOAN_GOAL,
                             _schema_profile(schema_path, "bpic2012")["sequence_categorical_columns"])
            self.assertEqual(events[encoding.OFFER_FEATURE_MASK].tolist(), [0, 0, 0])

    # Verify outcome labels are multiclass without the outcome mask.
    def test_outcome_labels_are_multiclass_without_outcome_mask(self) -> None:
        events = pd.concat(
            [
                pd.DataFrame({encoding.OUTCOME: [0]}),
                pd.DataFrame({encoding.OUTCOME: [1]}),
                pd.DataFrame({encoding.OUTCOME: [2]}),
            ],
            ignore_index=True,
        )

        encoding.validate_multiclass_outcome_labels(events)

        self.assertEqual(sorted(events[encoding.OUTCOME].unique().tolist()), [0, 1, 2])
        self.assertNotIn("outcome_mask", events.columns)

    # Verify prefix cap limits samples without truncating source events.
    def test_prefix_cap_limits_samples_without_truncating_source_events(self) -> None:
        events = pd.DataFrame(
            {
                encoding.CASE_ID: ["c1", "c1", "c1"],
                encoding.DATASET_ID: ["bpic2017", "bpic2017", "bpic2017"],
                encoding.CLIENT_ID: ["A", "A", "A"],
                encoding.SPLIT: ["train", "train", "train"],
                encoding.EVENT_INDEX: [0, 1, 2],
            }
        )

        prefix_index, observed_max = encoding.build_prefix_index(events, max_prefix_length=2)

        self.assertEqual(len(events), 3)
        self.assertEqual(observed_max, 3)
        self.assertEqual([row.prefix_length for row in prefix_index], [1, 2])

    # Verify vocabularies fit on train only and unknown inputs map to unk.
    def test_vocabularies_fit_on_train_only_and_unknown_inputs_map_to_unk(self) -> None:
        train = pd.DataFrame(
            {
                encoding.CANONICAL_ACTIVITY_TOKEN: ["known+complete"],
                encoding.NEXT_ACTIVITY_TARGET: ["known_next+complete"],
                encoding.RESOURCE: ["User_1"],
            }
        )
        val = pd.DataFrame({encoding.RESOURCE: ["User_9"]})

        vocabs = encoding.build_all_vocabularies(
            train, categorical_columns=[encoding.CANONICAL_ACTIVITY_TOKEN, encoding.RESOURCE])
        encoded = encoding.encode_categorical_series(val[encoding.RESOURCE], vocabs[encoding.RESOURCE])

        self.assertNotIn("User_9", vocabs[encoding.RESOURCE])
        self.assertEqual(encoded.tolist(), [vocabs[encoding.RESOURCE][encoding.UNK_TOKEN]])

    # Verify scalers fit on train only.
    def test_scalers_fit_on_train_only(self) -> None:
        train = pd.DataFrame({encoding.TIME_DELTA: [0.0, 10.0]})
        val = pd.DataFrame({encoding.TIME_DELTA: [1000.0]})

        scalers = encoding.fit_all_scalers(train, numerical_columns=[encoding.TIME_DELTA])
        transformed = encoding.transform_with_scaler(val[encoding.TIME_DELTA], scalers[encoding.TIME_DELTA])

        self.assertEqual(scalers[encoding.TIME_DELTA]["mean"], 5.0)
        self.assertGreater(float(transformed.iloc[0]), 100.0)

    # Verify unknown next activity target is masked and not trained as unk.
    def test_unknown_next_activity_target_is_masked_and_not_trained_as_unk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            events = encoding.to_canonical_events(_bpic2017_frame(), mapping, "bpic2017", "A", "val")
            mapped, _ = encoding.apply_activity_mapping(events, mapping_path)
            vocabs = encoding.build_all_vocabularies(
                mapped.iloc[:1],
                categorical_columns=_schema_profile(schema_path, "bpic2017")["sequence_categorical_columns"],
            )
            numerical_columns = _schema_profile(schema_path, "bpic2017")["sequence_numerical_columns"]
            scalers = encoding.fit_all_scalers(mapped, numerical_columns=numerical_columns)
            prefix_index, _ = encoding.build_prefix_index(mapped, max_prefix_length=2)
            dataset = encoding.PrefixDataset(mapped, prefix_index, vocabs, scalers, static_padding_length=3)

            sample = dataset[1]

            self.assertEqual(int(sample["next_activity_mask"]), 0)
            self.assertEqual(int(sample["next_activity_label"]),
                             vocabs[encoding.NEXT_ACTIVITY_TARGET][encoding.UNK_TOKEN])

    # Verify offer features forward fill without the future leakage and credit score reset.
    def test_offer_features_forward_fill_without_future_leakage_and_credit_score_reset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            raw = _bpic2017_frame()
            raw.loc[2, "concept:name"] = "O_Create Offer"
            raw.loc[2, "CreditScore"] = 0.0
            raw.loc[2, "MonthlyCost"] = 200.0
            raw.loc[2, "OfferedAmount"] = 8000.0
            raw.loc[2, "NumberOfTerms"] = 12.0
            events = encoding.to_canonical_events(raw, mapping, "bpic2017", "A", "train")

            state_1 = encoding.build_offer_state(events.iloc[:1])
            state_2 = encoding.build_offer_state(events.iloc[:2])
            state_3 = encoding.build_offer_state(events.iloc[:3])

            self.assertEqual(state_1["offer_present"], 0)
            self.assertEqual(state_2[encoding.MONTHLY_COST_VALUE], 300.0)
            self.assertEqual(state_3[encoding.CREDIT_SCORE_VALUE], 0.0)
            self.assertEqual(state_3[encoding.MONTHLY_COST_VALUE], 200.0)

    # Verify the RT target representation supports all encoding modes.
    def test_remaining_time_target_repr_supports_all_encoding_modes(self) -> None:
        values = pd.Series([100.0, 400.0, 900.0, 1600.0])
        sample = pd.Series([100.0, 900.0, 1600.0])

        for transform in encoding.REMAINING_TIME_TRANSFORMS:
            for scaling in encoding.REMAINING_TIME_SCALINGS:
                target_repr = encoding.fit_remaining_time_target_repr(values, transform, scaling)
                model_units = encoding.transform_remaining_time_target(sample, target_repr)
                seconds = encoding.inverse_remaining_time_target(model_units, target_repr)

                self.assertTrue(np.allclose(seconds.to_numpy(), sample.to_numpy(), atol=1e-3), f"{transform}+{scaling}")

        with self.assertRaises(ValueError):
            encoding.fit_remaining_time_target_repr(pd.Series([-1.0]), "raw", "zscore")
        with self.assertRaises(ValueError):
            encoding.fit_remaining_time_target_repr(pd.Series([-1.0, 100.0]), "raw", "zscore")

    # Verify prefix dataset encodes remaining time target from metadata.
    def test_prefix_dataset_encodes_remaining_time_target_from_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            events = encoding.to_canonical_events(_bpic2017_frame(), mapping, "bpic2017", "A", "train")
            mapped, _ = encoding.apply_activity_mapping(events, mapping_path)
            profile = _schema_profile(schema_path, "bpic2017")
            vocabs = encoding.build_all_vocabularies(mapped,
                                                     categorical_columns=profile["sequence_categorical_columns"])
            scalers = encoding.fit_all_scalers(mapped, numerical_columns=profile["sequence_numerical_columns"])
            prefix_index, _ = encoding.build_prefix_index(mapped, max_prefix_length=3)
            target_repr = encoding.fit_remaining_time_target_from_prefixes(
                mapped, prefix_index, transform="raw", scaling="zscore"
            )
            dataset = encoding.PrefixDataset(
                mapped,
                prefix_index,
                vocabs,
                scalers,
                static_padding_length=3,
                remaining_time_target_repr=target_repr,
            )

            self.assertAlmostEqual(float(dataset[0]["remaining_time_label"]), 1.0, places=5)
            self.assertAlmostEqual(float(dataset[1]["remaining_time_label"]), -1.0, places=5)
            self.assertEqual(int(dataset[2]["remaining_time_mask"]), 0)

    # Verify static padding and final remaining time mask are correct.
    def test_static_padding_and_final_remaining_time_mask_are_correct(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2017")
            events = encoding.to_canonical_events(_bpic2017_frame(), mapping, "bpic2017", "A", "train")
            mapped, _ = encoding.apply_activity_mapping(events, mapping_path)
            profile = _schema_profile(schema_path, "bpic2017")
            vocabs = encoding.build_all_vocabularies(mapped,
                                                     categorical_columns=profile["sequence_categorical_columns"])
            scalers = encoding.fit_all_scalers(mapped, numerical_columns=profile["sequence_numerical_columns"])
            prefix_index, _ = encoding.build_prefix_index(mapped, max_prefix_length=3)
            dataset = encoding.PrefixDataset(mapped, prefix_index, vocabs, scalers, static_padding_length=3)

            sample = dataset[2]

            self.assertEqual(sample["categorical_ids"].dtype, torch.long)
            self.assertEqual(sample["numerical"].dtype, torch.float32)
            self.assertEqual(sample["outcome_label"].dtype, torch.long)
            self.assertEqual(sample["padding_mask"].dtype, torch.int8)
            self.assertEqual(sample["padding_mask"].tolist(), [1, 1, 1])
            self.assertEqual(int(sample["remaining_time_mask"]), 0)

    # Verify BPIC 2012 profile emits empty offer tensors.
    def test_bpic2012_profile_emits_empty_offer_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            schema_path = Path(tmp) / "canonical_schemas.json"
            mapping_path = Path(tmp) / "manual_dataset_mapping.json"
            mapping = _approved_mapping(mapping_path, schema_path, "bpic2012")
            events = encoding.to_canonical_events(_bpic2012_frame(), mapping, "bpic2012", "A", "train")
            mapped, _ = encoding.apply_activity_mapping(events, mapping_path)
            profile = _schema_profile(schema_path, "bpic2012")
            vocabs = encoding.build_all_vocabularies(mapped,
                                                     categorical_columns=profile["sequence_categorical_columns"])
            scalers = encoding.fit_all_scalers(mapped, numerical_columns=profile["sequence_numerical_columns"])
            prefix_index, _ = encoding.build_prefix_index(mapped, max_prefix_length=3)
            dataset = encoding.PrefixDataset(
                mapped,
                prefix_index,
                vocabs,
                scalers,
                static_padding_length=3,
                offer_numerical_columns=profile["offer_numerical_columns"],
            )

            sample = dataset[1]

            self.assertEqual(sample["offer_numerical"].shape, (3, 0))

    # Verify encoder artifacts are compact JSON without saved prefix tensors.
    def test_encoder_artifacts_are_compact_json_without_saved_prefix_tensors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = {"ok": True}
            paths = encoding.build_artifact_paths(root, "bpic2017", "iid", 3, joint=False)
            encoding.save_encoding_artifacts(paths, payload, payload, payload, payload)
            files = sorted(path.name for path in paths.root.iterdir())

            self.assertEqual(
                files,
                [
                    "A_04_encoding_spec.json",
                    "A_04_mapping_report.json",
                    "A_04_scaler.json",
                    "A_04_vocabulary.json",
                ],
            )
            self.assertFalse(any(path.suffix in {".pt", ".pth", ".parquet", ".pkl"} for path in paths.root.iterdir()))

    # Verify encoding spec uses plain bank keys and per bank prefix counts.
    def test_encoding_spec_uses_plain_bank_keys_and_per_bank_prefix_counts(self) -> None:
        events = pd.DataFrame(
            {
                encoding.DATASET_ID: ["bpic2017", "bpic2017", "bpic2017"],
                encoding.CLIENT_ID: ["A", "A", "B"],
                encoding.SPLIT: ["train", "train", "train"],
                encoding.CASE_ID: ["case_1", "case_1", "case_2"],
                encoding.EVENT_INDEX: [0, 1, 0],
                encoding.OUTCOME: [2, 2, 1],
            }
        )
        prefix_index, observed_max = encoding.build_prefix_index(events, max_prefix_length=2)

        class Config:
            dataset = "bpic2017"
            schema_profile = "bpic2017"
            heterogeneity = "iid"
            n_clients = 2
            random_seed = 42
            max_prefix_length_for_encoding = 2
            outcome_target_mode = "every_prefix"
            remaining_time_transform = "raw"
            remaining_time_scaling = "zscore"

        target_repr = encoding.RemainingTimeTargetRepr("raw", "zscore", 0.0, 1.0, False, 0.0, 0.0, 0)
        spec = encoding.build_encoding_spec(Config, events, prefix_index, observed_max, {}, {}, {}, target_repr)

        self.assertEqual(set(spec["counts"]["per_bank"]), {"A", "B"})
        self.assertEqual(spec["counts"]["per_bank"]["A"]["prefix_count"], 2)
        self.assertEqual(spec["counts"]["per_bank"]["B"]["prefix_count"], 1)
        self.assertEqual(spec["hyperparameters"]["remaining_time_transform"], "raw")
        self.assertEqual(spec["hyperparameters"]["remaining_time_scaling"], "zscore")
        self.assertEqual(spec["target_scalers"]["remaining_time"]["scaling"], "zscore")

    # Verify joint train outcome counts keep dataset case identity.
    def test_joint_train_outcome_counts_keep_dataset_case_identity(self) -> None:
        events = pd.DataFrame(
            {
                encoding.DATASET_ID: ["bpic2017", "bpic2012"],
                encoding.CLIENT_ID: ["A", "A"],
                encoding.SPLIT: ["train", "train"],
                encoding.CASE_ID: ["shared_case", "shared_case"],
                encoding.EVENT_INDEX: [0, 0],
                encoding.OUTCOME: [2, 1],
            }
        )
        prefix_index, observed_max = encoding.build_prefix_index(events, max_prefix_length=1)

        class Config:
            dataset = "joint"
            schema_profile = "joint"
            heterogeneity = "iid"
            n_clients = 6
            random_seed = 42
            max_prefix_length_for_encoding = 1
            outcome_target_mode = "every_prefix"
            remaining_time_transform = "raw"
            remaining_time_scaling = "zscore"

        target_repr = encoding.RemainingTimeTargetRepr("raw", "zscore", 0.0, 1.0, False, 0.0, 0.0, 0)
        spec = encoding.build_encoding_spec(Config, events, prefix_index, observed_max, {}, {}, {}, target_repr)

        self.assertEqual(spec["counts"]["train_outcome_counts"], {"1": 1, "2": 1})

if __name__ == "__main__":
    unittest.main()