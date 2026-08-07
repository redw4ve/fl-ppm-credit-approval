from __future__ import annotations
import importlib
from dataclasses import dataclass
import unittest
from typing import Any, Optional
from unittest import mock
import numpy as np
import torch
from opacus.layers import DPLSTM

contract = importlib.import_module("E_prefix_encoding.04_1_contract")
training = importlib.import_module("E_training.training_core_final")

@dataclass(frozen=True)
class DatasetAwareTestBatch:
    categorical_ids: torch.Tensor
    numerical: torch.Tensor
    offer_numerical: torch.Tensor
    offer_feature_mask: torch.Tensor
    padding_mask: torch.Tensor
    prefix_length: torch.Tensor
    outcome_label: torch.Tensor
    next_activity_label: torch.Tensor
    next_activity_mask: torch.Tensor
    remaining_time_label: torch.Tensor
    remaining_time_mask: torch.Tensor
    dataset_code: torch.Tensor

# HELPER: Build batch.
def _batch() -> Any:
    return contract.EncodedBatch(
        categorical_ids=torch.tensor(
            [
                [[1, 2], [3, 4], [0, 0]],
                [[2, 1], [0, 0], [0, 0]],
            ],
            dtype=torch.long,
        ),
        numerical=torch.tensor(
            [
                [[0.1, 1.0], [0.2, 1.1], [0.0, 0.0]],
                [[0.3, 1.2], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        offer_numerical=torch.tensor(
            [
                [[0.0, 0.0], [1.0, -1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        offer_feature_mask=torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.int8),
        padding_mask=torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.int8),
        prefix_length=torch.tensor([2, 1], dtype=torch.long),
        outcome_label=torch.tensor([2, 0], dtype=torch.long),
        next_activity_label=torch.tensor([4, 3], dtype=torch.long),
        next_activity_mask=torch.tensor([1, 0], dtype=torch.int8),
        remaining_time_label=torch.tensor([1.0, 0.0], dtype=torch.float32),
        remaining_time_mask=torch.tensor([1, 0], dtype=torch.int8),
        )

# HELPER: Build the dataset-aware batch.
def _dataset_aware_batch(dataset_codes: list[int], next_activity_labels: Optional[list[int]] = None,
                         next_activity_masks: Optional[list[int]] = None) -> DatasetAwareTestBatch:
    batch = _batch()
    return DatasetAwareTestBatch(
        categorical_ids=batch.categorical_ids,
        numerical=batch.numerical,
        offer_numerical=batch.offer_numerical,
        offer_feature_mask=batch.offer_feature_mask,
        padding_mask=batch.padding_mask,
        prefix_length=batch.prefix_length,
        outcome_label=batch.outcome_label,
        next_activity_label=(
            torch.tensor(next_activity_labels, dtype=torch.long)
            if next_activity_labels is not None
            else batch.next_activity_label
        ),
        next_activity_mask=(
            torch.tensor(next_activity_masks, dtype=torch.int8)
            if next_activity_masks is not None
            else batch.next_activity_mask
        ),
        remaining_time_label=batch.remaining_time_label,
        remaining_time_mask=batch.remaining_time_mask,
        dataset_code=torch.tensor(dataset_codes, dtype=torch.long),
    )

class TrainingCoreTests(unittest.TestCase):
    # Verify embedding dim rule has floor and cap.
    def test_embedding_dim_rule_has_floor_and_cap(self) -> None:
        self.assertEqual(training.embedding_dim_for_vocab(1), 8)
        self.assertEqual(training.embedding_dim_for_vocab(4), 8)
        self.assertEqual(training.embedding_dim_for_vocab(132), 32)
        self.assertEqual(training.embedding_dim_for_vocab(1000), 32)

    # Verify model forward shapes match three tasks.
    def test_model_forward_shapes_match_three_tasks(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.1,
        )

        outputs = model(_batch())

        self.assertEqual(tuple(outputs.outcome_logits.shape), (2, 3))
        self.assertEqual(tuple(outputs.next_activity_logits.shape), (2, 5))
        self.assertEqual(tuple(outputs.remaining_time_scaled.shape), (2,))

    # Verify lstm cls hook defaults to torch lstm.
    def test_lstm_cls_hook_defaults_to_torch_lstm(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.1,
        )
        self.assertIsInstance(model.lstm, torch.nn.LSTM)

    # Verify lstm cls hook accepts DP lstm with the same state keys.
    def test_lstm_cls_hook_accepts_dp_lstm_with_same_state_keys(self) -> None:
        plain = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.1,
        )
        private = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.1, lstm_cls=DPLSTM,
        )

        # Keep parameter serialization compatible between no-DP and DP trunks.
        self.assertIsInstance(private.lstm, DPLSTM)
        self.assertEqual(set(plain.state_dict()), set(private.state_dict()))

    # Verify the RT head uses positive activation.
    def test_remaining_time_head_uses_positive_activation(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=1, dropout=0.0,
        )
        final_layer = model.remaining_time_head[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(-10.0)

        outputs = model(_batch())

        self.assertTrue(torch.all(outputs.remaining_time_scaled > 0.0))

    # Verify remaining time bias initializes to training median in scaled units.
    def test_remaining_time_bias_initializes_to_training_median_in_scaled_units(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=1, dropout=0.0,
        )
        final_layer = model.remaining_time_head[-1]
        with torch.no_grad():
            final_layer.weight.zero_()

        # A median of 100 seconds scaled by 50 puts the scaled train median at 2.0 model units.
        repr_ = training.RemainingTimeRepr("raw", "median", 0.0, 50.0, True, 2.0)
        training.initialize_remaining_time_head_bias(model, repr_)

        outputs = model(_batch())
        self.assertTrue(torch.allclose(outputs.remaining_time_scaled, torch.full((2,), 2.0), atol=1e-5))

    # Verify linear head outputs negative remaining time for zscore.
    def test_linear_head_outputs_negative_remaining_time_for_zscore(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=1, dropout=0.0, remaining_time_softplus=False,
        )
        final_layer = model.remaining_time_head[-1]
        with torch.no_grad():
            final_layer.weight.zero_()
            final_layer.bias.fill_(-1.5)

        outputs = model(_batch())

        # The linear head must pass negative model units through, which softplus could never produce.
        self.assertTrue(torch.all(outputs.remaining_time_scaled < 0.0))

    # Verify outcome head dropout defaults to global dropout.
    def test_outcome_head_dropout_defaults_to_global_dropout(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.3,
        )

        # The outcome-head dropout layer sits at index 2 of the head and matches the global dropout by default.
        self.assertEqual(model.outcome_head[2].p, 0.3)
        self.assertEqual(model.next_activity_head[2].p, 0.3)

    # Verify outcome head dropout only changes the outcome head.
    def test_outcome_head_dropout_only_changes_the_outcome_head(self) -> None:
        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=2, dropout=0.3, outcome_head_dropout=0.45,
        )

        # Only the outcome head takes the stronger dropout; the other two heads keep the global value.
        self.assertEqual(model.outcome_head[2].p, 0.45)
        self.assertEqual(model.next_activity_head[2].p, 0.3)
        self.assertEqual(model.remaining_time_head[2].p, 0.3)

    # Verify multitask loss applies task masks.
    def test_multitask_loss_applies_task_masks(self) -> None:
        outputs = training.ModelOutput(
            outcome_logits=torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]], dtype=torch.float32),
            next_activity_logits=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 4.0], [0.0, 0.0, 0.0, 4.0, 0.0]],
                dtype=torch.float32,
            ),
            remaining_time_scaled=torch.tensor([1.0, 999999.0], dtype=torch.float32),
        )

        losses = training.compute_multitask_loss(
            outputs,
            _batch(),
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=training.RemainingTimeRepr("raw", "median", 0.0, 100.0, True, 1.0),
        )

        self.assertLess(losses.next_activity.item(), 0.1)
        self.assertLess(losses.remaining_time.item(), 0.01)
        self.assertGreater(losses.total.item(), losses.outcome.item())

    # Verify next activity dataset mask leaves single dataset logits unchanged.
    def test_next_activity_dataset_mask_leaves_single_dataset_logits_unchanged(self) -> None:
        context = training.build_next_activity_mask_context(
            {"bpic2017": np.zeros(6, dtype=bool)}, single_dataset_noop=True,
        )
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]], dtype=torch.float32)

        masked = training.mask_next_activity_logits(
            logits, torch.tensor([context.dataset_code_by_id["bpic2017"]], dtype=torch.long), context,
        )
        self.assertTrue(torch.equal(masked, logits))

    # Verify next activity single client joint context masks shared head.
    def test_next_activity_single_client_joint_context_masks_shared_head(self) -> None:
        context = training.build_next_activity_mask_context(
            {"bpic2017": np.array([False, False, False, False, False, True, False], dtype=bool)},
        )
        logits = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 50.0]], dtype=torch.float32)

        masked = training.mask_next_activity_logits(
            logits, torch.tensor([context.dataset_code_by_id["bpic2017"]], dtype=torch.long), context,
        )

        self.assertFalse(context.is_noop)
        self.assertEqual(float(masked[0, 5]), 5.0)
        self.assertTrue(torch.isneginf(masked[0, 6]))

    # Verify multitask loss masks next activity logits by dataset code.
    def test_multitask_loss_masks_next_activity_logits_by_dataset_code(self) -> None:
        context = training.build_next_activity_mask_context(
            {
                "bpic2017": np.array([False, False, False, False, True, True, False], dtype=bool),
                "bpic2012": np.array([False, False, False, False, True, False, True], dtype=bool),
            },
        )
        batch = _dataset_aware_batch(
            [context.dataset_code_by_id["bpic2017"], context.dataset_code_by_id["bpic2012"]],
            next_activity_labels=[5, 6],
            next_activity_masks=[1, 1],
        )
        outputs = training.ModelOutput(
            outcome_logits=torch.tensor([[0.0, 0.0, 2.0], [2.0, 0.0, 0.0]], dtype=torch.float32),
            next_activity_logits=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 50.0], [0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 5.0]],
                dtype=torch.float32,
            ),
            remaining_time_scaled=torch.tensor([1.0, 0.0], dtype=torch.float32),
        )

        losses = training.compute_multitask_loss(
            outputs,
            batch,
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=training.RemainingTimeRepr("raw", "median", 0.0, 100.0, True, 1.0),
            next_activity_mask_context=context,
        )

        self.assertLess(losses.next_activity.item(), 0.1)

    # Verify masked next activity metrics use dataset valid logits for top-k.
    def test_masked_next_activity_metrics_use_dataset_valid_logits_for_topk(self) -> None:
        context = training.build_next_activity_mask_context(
            {
                "bpic2017": np.array([False, False, False, False, True, True, False], dtype=bool),
                "bpic2012": np.array([False, False, False, False, True, False, True], dtype=bool),
            },
        )
        logits = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 50.0], [0.0, 0.0, 0.0, 0.0, 4.0, 50.0, 5.0]],
            dtype=torch.float32,
        )
        dataset_codes = torch.tensor(
            [context.dataset_code_by_id["bpic2017"], context.dataset_code_by_id["bpic2012"]],
            dtype=torch.long,
        )
        masked = training.mask_next_activity_logits(logits, dataset_codes, context)

        metrics = training.compute_next_activity_metrics(
            y_true=torch.tensor([5, 6], dtype=torch.long).numpy(),
            logits=masked.numpy(),
        )

        self.assertEqual(metrics["top1_accuracy"], 1.0)
        self.assertEqual(metrics["top3_accuracy"], 1.0)

    # Verify remaining time loss uses encoded batch target and bias repr.
    def test_remaining_time_loss_uses_encoded_batch_target_and_bias_repr(self) -> None:
        batch = _batch()
        outputs = training.ModelOutput(
            outcome_logits=torch.zeros((2, 3), dtype=torch.float32),
            next_activity_logits=torch.zeros((2, 5), dtype=torch.float32),
            remaining_time_scaled=torch.tensor([1.0, 7.0], dtype=torch.float32),
        )

        repr_ = training.RemainingTimeRepr("raw", "zscore", 900.0, 500.0, False, 0.0)
        losses = training.compute_multitask_loss(
            outputs, batch, outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"), remaining_time_repr=repr_, huber_beta=1.0,
        )
        self.assertLess(losses.remaining_time.item(), 0.01)

        model = training.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=16, num_layers=1, dropout=0.0, remaining_time_softplus=False,
        )
        training.initialize_remaining_time_head_bias(model, repr_)
        self.assertAlmostEqual(float(model.remaining_time_head[-1].bias.detach()[0]), 0.0, places=5)

    # Verify remaining time metrics clamp negative predictions.
    def test_remaining_time_metrics_clamp_negative_predictions(self) -> None:
        metrics = training.compute_remaining_time_metrics(
            y_true=torch.tensor([5.0, 10.0]),
            y_pred=torch.tensor([-100.0, 13.0]),
            mask=torch.tensor([1, 1], dtype=torch.int8),
        )

        self.assertEqual(metrics["mae"], 4.0)
        self.assertAlmostEqual(metrics["rmse"], 17.0 ** 0.5, places=6)

    # Verify explicit mps request fails when mps is unavailable.
    def test_explicit_mps_request_fails_when_mps_is_unavailable(self) -> None:
        with mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "MPS device was requested"):
                training.select_device("mps")

    # Verify that the auto device can still fall back to cpu.
    def test_auto_device_can_still_fall_back_to_cpu(self) -> None:
        with mock.patch.object(torch.backends.mps, "is_available", return_value=False):
            with mock.patch.object(torch.cuda, "is_available", return_value=False):
                self.assertEqual(training.select_device("auto"), torch.device("cpu"))

    # Verify auto device prefers cuda over mps.
    def test_auto_device_prefers_cuda_over_mps(self) -> None:
        with mock.patch.object(torch.backends.mps, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "is_available", return_value=True):
                self.assertEqual(training.select_device("auto"), torch.device("cuda"))

    # Verify explicit cuda request fails when cuda is unavailable.
    def test_explicit_cuda_request_fails_when_cuda_is_unavailable(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA device was requested"):
                training.select_device("cuda")

if __name__ == "__main__":
    unittest.main()