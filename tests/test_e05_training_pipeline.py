from __future__ import annotations
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_core_final as core

class TinyPrefixDataset:
    # HELPER: Build init.
    def __init__(self) -> None:
        self.static_padding_length = 2
        self.prefix_index = [
            SimpleNamespace(dataset_id="bpic2012", client_id="A", case_id="c1", split="train",
                            prefix_length=1, label_pos=0),
            SimpleNamespace(dataset_id="bpic2012", client_id="A", case_id="c2", split="train",
                            prefix_length=2, label_pos=1),
        ]
        self.samples = [
            {
                "categorical_ids": torch.tensor([[1, 2], [0, 0]], dtype=torch.long),
                "numerical": torch.tensor([[0.5], [0.0]], dtype=torch.float32),
                "offer_numerical": torch.tensor([[0.0], [0.0]], dtype=torch.float32),
                "offer_feature_mask": torch.tensor([0, 0], dtype=torch.int8),
                "padding_mask": torch.tensor([1, 0], dtype=torch.int8),
                "prefix_length": torch.tensor(1, dtype=torch.long),
                "outcome_label": torch.tensor(2, dtype=torch.long),
                "next_activity_label": torch.tensor(3, dtype=torch.long),
                "next_activity_mask": torch.tensor(1, dtype=torch.int8),
                "remaining_time_label": torch.tensor(10.0, dtype=torch.float32),
                "remaining_time_mask": torch.tensor(1, dtype=torch.int8),
            },
            {
                "categorical_ids": torch.tensor([[2, 1], [3, 4]], dtype=torch.long),
                "numerical": torch.tensor([[1.5], [2.0]], dtype=torch.float32),
                "offer_numerical": torch.tensor([[1.0], [0.0]], dtype=torch.float32),
                "offer_feature_mask": torch.tensor([1, 0], dtype=torch.int8),
                "padding_mask": torch.tensor([1, 1], dtype=torch.int8),
                "prefix_length": torch.tensor(2, dtype=torch.long),
                "outcome_label": torch.tensor(0, dtype=torch.long),
                "next_activity_label": torch.tensor(4, dtype=torch.long),
                "next_activity_mask": torch.tensor(1, dtype=torch.int8),
                "remaining_time_label": torch.tensor(0.0, dtype=torch.float32),
                "remaining_time_mask": torch.tensor(0, dtype=torch.int8),
            },
        ]

    # HELPER: Build len.
    def __len__(self) -> int: return len(self.samples)

    # HELPER: Build getitem.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: return self.samples[index]

class E05TrainingPipelineTests(unittest.TestCase):
    # Verify defaults match selected calibration.
    def test_defaults_match_selected_calibration(self) -> None:
        self.assertEqual(baseline.SCRIPT_ID, "E_05")
        self.assertEqual(baseline.OUTPUT_ROOT, (REPO_ROOT / "E_training/training_outputs").resolve())
        self.assertEqual(baseline.EARLY_STOPPING_PATIENCE, 7)
        self.assertEqual(baseline.LEARNING_RATE, 0.00025)
        self.assertEqual(baseline.WEIGHT_DECAY, 0.0001)
        self.assertEqual(baseline.BATCH_SIZE, 512)
        self.assertEqual(baseline.MAX_EPOCHS, 40)
        self.assertEqual(baseline.LR_SCHEDULER_T_MAX, 15)

    # Verify cache rejects mismatched build hash.
    def test_cache_rejects_mismatched_build_hash(self) -> None:
        dataset = TinyPrefixDataset()
        context = {"cache_build_hash": "abc", "cache_build_payload": {"target_mode": "raw"}}

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp) / "cache"
            cached = core.load_or_build_prefix_tensor_cache(dataset, cache_dir, overwrite=False, cache_context=context)

            self.assertEqual(cached.metadata["cache_build_hash"], "abc")
            self.assertEqual(cached.prefix_index[0].case_id, "c1")
            self.assertIsNotNone(core.try_load_prefix_tensor_cache(cache_dir, expected_cache_hash="abc"))
            self.assertIsNone(core.try_load_prefix_tensor_cache(cache_dir, expected_cache_hash="def"))

    # Verify remaining time loss trains encoded target units.
    def test_remaining_time_loss_trains_encoded_target_units(self) -> None:
        outputs = core.ModelOutput(
            outcome_logits=torch.tensor([[0.0, 0.0, 2.0]], dtype=torch.float32),
            next_activity_logits=torch.tensor([[0.0, 0.0, 0.0, 4.0, 0.0]], dtype=torch.float32),
            remaining_time_scaled=torch.tensor([1.0], dtype=torch.float32),
        )
        batch = SimpleNamespace(
            outcome_label=torch.tensor([2], dtype=torch.long),
            next_activity_label=torch.tensor([3], dtype=torch.long),
            next_activity_mask=torch.tensor([1], dtype=torch.int8),
            remaining_time_label=torch.tensor([1.0], dtype=torch.float32),
            remaining_time_mask=torch.tensor([1], dtype=torch.int8),
        )

        losses = core.compute_multitask_loss(
            outputs,
            batch,
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=core.RemainingTimeRepr("raw", "zscore", 100.0, 10.0, False, 0.0),
        )

        self.assertLess(losses.remaining_time.item(), 0.01)

    # Verify predictions invert through the encoding representation.
    def test_predictions_invert_through_the_encoding_representation(self) -> None:
        repr_ = core.RemainingTimeRepr("log", "raw", 0.0, 1.0, True, 6.0)
        model_units = np.array([float(torch.log1p(torch.tensor(900.0)))], dtype=float)

        seconds = baseline.remaining_time_model_units_to_seconds(model_units, repr_)

        self.assertAlmostEqual(float(seconds[0]), 900.0, places=2)

    # Verify remaining time representation defaults are behavior neutral.
    def test_remaining_time_representation_defaults_are_behavior_neutral(self) -> None:
        config = baseline.BaselineRunConfig()

        self.assertEqual(config.remaining_time_transform, "raw")
        self.assertEqual(config.remaining_time_scaling, "zscore")
        self.assertEqual(config.remaining_time_huber_beta, 0.1)

    # Verify the cache path separates remaining time encoding variants.
    def test_cache_path_separates_remaining_time_encoding_variants(self) -> None:
        # E_04 now writes encoded RT labels, so transform and scaling must separate cache trees.
        spec = {"prefix": {"static_padding_length": 8}}
        base = baseline.BaselineRunConfig(remaining_time_transform="raw", remaining_time_scaling="zscore")
        other = baseline.BaselineRunConfig(
            remaining_time_transform="log", remaining_time_scaling="median", remaining_time_huber_beta=0.1
        )

        self.assertNotEqual(baseline.prefix_cache_dir(base, "train", spec),
                            baseline.prefix_cache_dir(other, "train", spec))
        self.assertIn("rt_raw_zscore", baseline.prefix_cache_dir(base, "train", spec).as_posix())

    # Verify target diagnostics use encoding representation.
    def test_target_diagnostics_use_encoding_representation(self) -> None:
        dataset = TinyPrefixDataset()
        repr_ = core.RemainingTimeRepr("raw", "zscore", 20.0, 10.0, False, -1.0)

        diagnostics = baseline.remaining_time_target_diagnostics(dataset, repr_)
        self.assertEqual(diagnostics["remaining_time_transform"], "raw")
        self.assertEqual(diagnostics["remaining_time_scaling"], "zscore")
        self.assertAlmostEqual(diagnostics["remaining_time_center"], 20.0, places=4)
        self.assertAlmostEqual(diagnostics["remaining_time_scale"], 10.0, places=4)

        seconds = baseline.remaining_time_model_units_to_seconds(np.array([2.0]), repr_)
        self.assertAlmostEqual(float(seconds[0]), 40.0, places=4)

    # Verify prefix bucket metrics are computed from probability columns.
    def test_prefix_bucket_metrics_are_computed_from_probability_columns(self) -> None:
        predictions = pd.DataFrame(
            {
                "prefix_length": [1, 3, 12, 25],
                "outcome_label": [2, 1, 0, 2],
                "outcome_pred": [2, 0, 0, 2],
                "outcome_prob_0": [0.1, 0.8, 0.7, 0.2],
                "outcome_prob_1": [0.1, 0.1, 0.2, 0.1],
                "outcome_prob_2": [0.8, 0.1, 0.1, 0.7],
                "remaining_time_label_seconds": [10.0, 20.0, 30.0, 40.0],
                "remaining_time_pred_seconds_clamped": [11.0, 10.0, 25.0, 0.0],
                "remaining_time_mask": [1, 1, 1, 1],
            }
        )

        buckets = baseline.compute_prefix_bucket_metrics(predictions)

        self.assertEqual(buckets["1"]["n_prefixes"], 1)
        self.assertIn("2-5", buckets)
        self.assertAlmostEqual(buckets["1"]["remaining_time_mae_seconds"], 1.0)

# HELPER: Minimal prefix dataset carrying only the next activity fields the mask builder reads.
class _NextActivityMaskDataset:
    # HELPER: Build init.
    def __init__(self, dataset_ids: list[str], next_labels: list[int]) -> None:
        self.prefix_index = [SimpleNamespace(dataset_id=dataset_id) for dataset_id in dataset_ids]
        self.samples = [
            {
                "next_activity_label": torch.tensor(int(label), dtype=torch.long),
                "next_activity_mask": torch.tensor(1, dtype=torch.int8),
            }
            for label in next_labels
        ]

    # HELPER: Build len.
    def __len__(self) -> int: return len(self.samples)

    # HELPER: Build getitem.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: return self.samples[index]

class E05WorldAEvaluationMaskTests(unittest.TestCase):
    # Verify a joint local baseline builds a distinct central evaluation mask.
    # The mask must stay finite in a class the bank never trained on.
    def test_joint_local_evaluation_mask_is_distinct_and_finite(self) -> None:
        n_classes = 6
        target_key = baseline.encoding.NEXT_ACTIVITY_TARGET
        vocabularies = {target_key: {str(index): index for index in range(n_classes)}}

        # Bank bpic2017:A trained on next-activity class 3 only.
        bank_train = _NextActivityMaskDataset(["bpic2017", "bpic2017"], [3, 3])
        # The pooled joint train also contains bpic2017 class 5 and bpic2012 class 4, which this bank never trained on.
        pooled_train = _NextActivityMaskDataset(["bpic2017", "bpic2017", "bpic2012"], [3, 5, 4])

        config = baseline.BaselineRunConfig(dataset="joint", regime="local", bank="bpic2017:A")
        training_mask = baseline.build_next_activity_mask_context(config, bank_train, n_classes)
        with mock.patch.object(baseline, "load_cached_prefix_dataset", return_value=pooled_train):
            evaluation_mask = baseline.build_evaluation_next_activity_mask_context(
                config, training_mask, {}, vocabularies, {}, {}, {}
            )

        # The evaluation mask is a separate context built from the pooled train, not the bank's own classes.
        self.assertIsNot(evaluation_mask, training_mask)
        train_code = training_mask.dataset_code_by_id["bpic2017"]
        eval_code = evaluation_mask.dataset_code_by_id["bpic2017"]
        self.assertFalse(bool(training_mask.mask_by_dataset[train_code][5]))
        self.assertTrue(bool(evaluation_mask.mask_by_dataset[eval_code][5]))

        # A validation target of class 5 is minus infinity under the bank-local training mask.
        # The same target is finite under the central evaluation mask.
        logits = torch.zeros((1, n_classes), dtype=torch.float32)
        train_masked = core.mask_next_activity_logits(logits, torch.tensor([train_code], dtype=torch.long),
                                                      training_mask)
        eval_masked = core.mask_next_activity_logits(logits, torch.tensor([eval_code], dtype=torch.long),
                                                     evaluation_mask)
        self.assertTrue(torch.isneginf(train_masked[0, 5]))
        self.assertTrue(torch.isfinite(eval_masked[0, 5]))

    # Verify single-dataset and joint centralized runs reuse the training mask as the evaluation mask.
    # Reusing it keeps those regimes byte-identical.
    def test_single_dataset_and_joint_centralized_reuse_training_mask(self) -> None:
        n_classes = 6
        vocabularies = {baseline.encoding.NEXT_ACTIVITY_TARGET: {str(index): index for index in range(n_classes)}}

        single_config = baseline.BaselineRunConfig(dataset="bpic2017", regime="local", bank="A")
        single_train = _NextActivityMaskDataset(["bpic2017"], [3])
        single_training_mask = baseline.build_next_activity_mask_context(single_config, single_train, n_classes)
        single_eval_mask = baseline.build_evaluation_next_activity_mask_context(
            single_config, single_training_mask, {}, vocabularies, {}, {}, {}
        )
        self.assertIs(single_eval_mask, single_training_mask)

        joint_central_config = baseline.BaselineRunConfig(dataset="joint", regime="centralized")
        pooled_train = _NextActivityMaskDataset(["bpic2017", "bpic2012"], [3, 4])
        joint_training_mask = baseline.build_next_activity_mask_context(joint_central_config, pooled_train, n_classes)
        joint_eval_mask = baseline.build_evaluation_next_activity_mask_context(
            joint_central_config, joint_training_mask, {}, vocabularies, {}, {}, {}
        )
        self.assertIs(joint_eval_mask, joint_training_mask)

if __name__ == "__main__":
    unittest.main()