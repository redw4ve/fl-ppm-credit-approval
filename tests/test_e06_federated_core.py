from __future__ import annotations
from collections import OrderedDict
from types import SimpleNamespace
from typing import Any, Literal
import unittest
from unittest import mock
import warnings
import pandas as pd
import numpy as np
from opacus import PrivacyEngine
from opacus.layers import DPLSTM
import torch
from E_training import E_06_federated_training as federated
from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_core_final as core

# The three head-aggregation modes as the literal type the E_06 signatures declare.
HEAD_AGG_MODES: tuple[Literal["sample", "equal", "contribution"], ...] = ("sample", "equal", "contribution")

# HELPER: Build a tiny E_05-compatible multitask model for federated helper tests.
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

# HELPER: Build eval mask context.
def _eval_mask_context() -> core.NextActivityMaskContext:
    return core.NextActivityMaskContext(
        dataset_code_by_id={"bpic2017": 0}, mask_by_dataset=torch.ones((1, 5), dtype=torch.bool), is_noop=False,
    )

# CLASS: Provide a tiny prefix dataset compatible with the E_04 collate function.
class TinyPrefixDataset:
    # Store two trainable samples with one valid next-activity and remaining-time label each.
    def __init__(self) -> None:
        self.samples = [
            {
                "categorical_ids": torch.tensor([[1, 2], [0, 0]], dtype=torch.long),
                "numerical": torch.tensor([[0.5, 1.0], [0.0, 0.0]], dtype=torch.float32),
                "offer_numerical": torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32),
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
                "numerical": torch.tensor([[1.5, 2.0], [2.0, 2.5]], dtype=torch.float32),
                "offer_numerical": torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                "offer_feature_mask": torch.tensor([1, 1], dtype=torch.int8),
                "padding_mask": torch.tensor([1, 1], dtype=torch.int8),
                "prefix_length": torch.tensor(2, dtype=torch.long),
                "outcome_label": torch.tensor(0, dtype=torch.long),
                "next_activity_label": torch.tensor(4, dtype=torch.long),
                "next_activity_mask": torch.tensor(1, dtype=torch.int8),
                "remaining_time_label": torch.tensor(20.0, dtype=torch.float32),
                "remaining_time_mask": torch.tensor(1, dtype=torch.int8),
            },
        ]

        self.prefix_index = [
            SimpleNamespace(dataset_id="bpic2017", client_id="A", case_id="1", split="train",
                            prefix_length=1, label_pos=0),
            SimpleNamespace(dataset_id="bpic2017", client_id="A", case_id="2", split="train",
                            prefix_length=2, label_pos=1),
        ]

    # Return the number of available prefix samples.
    def __len__(self) -> int: return len(self.samples)

    # Return one prefix sample in the shape emitted by E_04.
    def __getitem__(self, index: int) -> dict[str, torch.Tensor]: return self.samples[index]

# CLASS: Provide local target arrays and prefix rows for client-mask construction tests.
class TinyMaskPrefixDataset:
    # Store only the fields needed to derive local next-activity support.
    def __init__(self) -> None:
        self.arrays = {
            "next_activity_label": np.array([5, 6], dtype=np.int64),
            "next_activity_mask": np.array([1, 1], dtype=np.int8),
        }
        self.prefix_index = [
            SimpleNamespace(dataset_id="bpic2017", client_id="A", case_id="1", split="train",
                            prefix_length=1, label_pos=0),
            SimpleNamespace(dataset_id="bpic2017", client_id="A", case_id="2", split="train",
                            prefix_length=2, label_pos=1),
        ]

    # Return the number of available prefix samples.
    def __len__(self) -> int:
        return 2

# CLASS: Expose a reversed state-dict order while keeping the same parameter key set.
class ReversedStateDictModel(core.MultitaskLSTM):
    # Return a reversed state dict so sorted-key serialization proves order independence.
    def state_dict(self, *args: object, **kwargs: object) -> OrderedDict[str, torch.Tensor]:
        state = super().state_dict(*args, **kwargs)
        return OrderedDict(reversed(list(state.items())))

# CLASS: Check federated parameter serialization, aggregation and FedProx helpers.
class FederatedCoreHelperTests(unittest.TestCase):
    # Verify parameters round-trip through numpy arrays without changing tensor values.
    def test_parameter_round_trip_preserves_state_dict_tensors(self) -> None:
        source = _build_tiny_model()
        target = _build_tiny_model()

        params = federated.model_parameters_to_numpy(source)
        federated.load_numpy_parameters(target, params)

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    # Verify sorted-key serialization is independent of state-dict insertion order.
    def test_parameter_round_trip_uses_sorted_keys_not_state_dict_order(self) -> None:
        source = _build_tiny_model()
        target = ReversedStateDictModel(
            categorical_vocab_sizes={"activity": 7, "resource": 6}, numerical_dim=2, offer_dim=2,
            next_activity_classes=5, hidden_size=8, num_layers=1, dropout=0.0, head_hidden_size=6,
        )

        params = federated.model_parameters_to_numpy(source)
        federated.load_numpy_parameters(target, params)

        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(value, target.state_dict()[key]), key)

    # Verify FedAvg aggregation weights each tensor by the client prefix count.
    @staticmethod
    def test_aggregate_parameters_weights_by_prefix_count() -> None:
        params_a = [np.array([1.0, 3.0], dtype=np.float32)]
        params_b = [np.array([5.0, 7.0], dtype=np.float32)]

        result = federated.aggregate_parameters([(params_a, 1), (params_b, 3)])
        np.testing.assert_allclose(result[0], np.array([4.0, 6.0], dtype=np.float32))

    # Verify equal aggregation changes only next-activity head tensors.
    @staticmethod
    def test_aggregate_parameters_equal_averages_next_activity_head_only() -> None:
        params_a = [
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
        ]
        params_b = [
            np.array([5.0, 7.0], dtype=np.float32),
            np.array([5.0, 7.0], dtype=np.float32),
        ]

        result = federated.aggregate_parameters(
            [(params_a, 1), (params_b, 3)],
            next_activity_head_agg="equal",
            parameter_keys=["lstm.weight_ih_l0", "next_activity_head.0.weight"],
        )

        np.testing.assert_allclose(result[0], np.array([4.0, 6.0], dtype=np.float32))
        np.testing.assert_allclose(result[1], np.array([3.0, 5.0], dtype=np.float32))

    # Verify contribution aggregation applies client-local class counts to final output slots only.
    @staticmethod
    def test_aggregate_parameters_contribution_weights_output_slots_by_class_counts() -> None:
        params_a = [
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([[1.0, 1.0], [10.0, 10.0], [100.0, 100.0], [1000.0, 1000.0]], dtype=np.float32),
            np.array([1.0, 10.0, 100.0, 1000.0], dtype=np.float32),
        ]
        params_b = [
            np.array([5.0, 7.0], dtype=np.float32),
            np.array([5.0, 7.0], dtype=np.float32),
            np.array([[2.0, 2.0], [20.0, 20.0], [200.0, 200.0], [2000.0, 2000.0]], dtype=np.float32),
            np.array([2.0, 20.0, 200.0, 2000.0], dtype=np.float32),
        ]
        params_c = [
            np.array([9.0, 11.0], dtype=np.float32),
            np.array([9.0, 11.0], dtype=np.float32),
            np.array([[3.0, 3.0], [30.0, 30.0], [300.0, 300.0], [3000.0, 3000.0]], dtype=np.float32),
            np.array([3.0, 30.0, 300.0, 3000.0], dtype=np.float32),
        ]
        class_counts = [
            np.array([10.0, 5.0, 0.0, 0.0], dtype=np.float32),
            np.array([0.0, 15.0, 3.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0, 7.0, 0.0], dtype=np.float32),
        ]

        parameter_keys = [
            "lstm.weight_ih_l0",
            "next_activity_head.0.weight",
            "next_activity_head.3.weight",
            "next_activity_head.3.bias",
        ]
        contribution_payloads = [
            federated._client_next_activity_contribution_payload(params, parameter_keys, counts)
            for params, counts in zip([params_a, params_b, params_c], class_counts)
        ]

        result = federated.aggregate_parameters(
            [(params_a, 10), (params_b, 20), (params_c, 30)],
            next_activity_head_agg="contribution",
            parameter_keys=parameter_keys,
            next_activity_class_counts=class_counts,
            next_activity_contribution_payloads=contribution_payloads,
        )

        np.testing.assert_allclose(result[0], np.array([6.3333335, 8.333333], dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(result[1], np.array([6.3333335, 8.333333], dtype=np.float32), rtol=1e-5)
        np.testing.assert_allclose(
            result[2],
            np.array(
                [[1.0, 1.0], [17.5, 17.5], [270.0, 270.0], [2333.3333, 2333.3333]],
                dtype=np.float32,
            ),
            rtol=1e-5,
        )
        np.testing.assert_allclose(
            result[3],
            np.array([1.0, 17.5, 270.0, 2333.3333], dtype=np.float32),
            rtol=1e-5,
        )

    # Verify contribution aggregation falls back to sample weighting when a slot has no target counts.
    @staticmethod
    def test_aggregate_parameters_contribution_falls_back_to_sample_for_zero_denominator() -> None:
        params_a = [np.array([10.0, 100.0], dtype=np.float32)]
        params_b = [np.array([20.0, 200.0], dtype=np.float32)]
        parameter_keys = ["next_activity_head.3.bias"]
        class_counts = [
            np.array([0.0, 0.0], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
        ]
        contribution_payloads = [
            federated._client_next_activity_contribution_payload(params, parameter_keys, counts)
            for params, counts in zip([params_a, params_b], class_counts)
        ]

        result = federated.aggregate_parameters(
            [(params_a, 1), (params_b, 3)],
            next_activity_head_agg="contribution",
            parameter_keys=parameter_keys,
            next_activity_class_counts=class_counts,
            next_activity_contribution_payloads=contribution_payloads,
        )

        np.testing.assert_allclose(result[0], np.array([17.5, 175.0], dtype=np.float32))

    # Verify contribution aggregation requires client-side numerator payloads.
    def test_aggregate_parameters_contribution_requires_client_side_payloads(self) -> None:
        params_a = [np.array([10.0, 100.0], dtype=np.float32)]
        params_b = [np.array([20.0, 200.0], dtype=np.float32)]

        with self.assertRaisesRegex(ValueError, "client-side contribution payloads"):
            federated.aggregate_parameters(
                [(params_a, 1), (params_b, 3)],
                next_activity_head_agg="contribution",
                parameter_keys=["next_activity_head.3.bias"],
                next_activity_class_counts=[
                    np.array([1.0, 0.0], dtype=np.float32),
                    np.array([0.0, 1.0], dtype=np.float32),
                ],
            )

    # Verify non-sample modes are strict no-ops for single-dataset runs.
    def test_aggregate_parameters_single_dataset_modes_reduce_to_sample(self) -> None:
        params_a = [
            np.array([1.0, 3.0], dtype=np.float32),
            np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32),
            np.array([1.0, 10.0], dtype=np.float32),
        ]
        params_b = [
            np.array([5.0, 7.0], dtype=np.float32),
            np.array([[2.0, 2.0], [20.0, 20.0]], dtype=np.float32),
            np.array([2.0, 20.0], dtype=np.float32),
        ]
        parameter_keys = ["lstm.weight_ih_l0", "next_activity_head.3.weight", "next_activity_head.3.bias"]
        results = [(params_a, 1), (params_b, 3)]
        sample = federated.aggregate_parameters(results, next_activity_head_agg="sample", parameter_keys=parameter_keys)

        equal = federated.aggregate_parameters(
            results,
            next_activity_head_agg="equal",
            parameter_keys=parameter_keys,
            is_joint_run=False,
        )
        contribution = federated.aggregate_parameters(
            results,
            next_activity_head_agg="contribution",
            parameter_keys=parameter_keys,
            next_activity_class_counts=[
                np.array([100.0, 0.0], dtype=np.float32),
                np.array([0.0, 100.0], dtype=np.float32),
            ],
            is_joint_run=False,
        )

        for actual in equal + contribution:
            self.assertEqual(actual.dtype, np.float32)
        for expected, actual in zip(sample, equal):
            np.testing.assert_array_equal(actual, expected)
        for expected, actual in zip(sample, contribution):
            np.testing.assert_array_equal(actual, expected)

    # HELPER: Build one masked message per client and combine them through the server helper.
    @staticmethod
    def _secure_aggregate(results: list, round_index: int = 1, secure_aggregation_seed: int = 42,
                          next_activity_class_counts: object = None, **kwargs: object) -> list:
        counts = next_activity_class_counts or [None] * len(results)
        client_messages = [
            federated.build_secure_client_message(
                params, count, client_index, len(results), round_index=round_index,
                secure_aggregation_seed=secure_aggregation_seed, next_activity_class_counts=class_counts, **kwargs
            )
            for client_index, ((params, count), class_counts) in enumerate(zip(results, counts))
        ]
        return federated.secure_aggregate_from_masked_messages(client_messages)

    # HELPER: Build one deterministic synthetic federation of the requested size for the cancellation checks.
    @staticmethod
    def _synthetic_results(n_clients: int, n_classes: int = 4) -> tuple[list, list[str], list[np.ndarray]]:
        rng = np.random.default_rng(7)
        keys = ["lstm.weight_ih_l0", "next_activity_head.0.weight", "next_activity_head.3.weight",
                "next_activity_head.3.bias"]
        results = [
            (
                [
                    rng.normal(0.0, 1.0, size=(3, 2)).astype(np.float32),
                    rng.normal(0.0, 1.0, size=(n_classes, 2)).astype(np.float32),
                    rng.normal(0.0, 1.0, size=(n_classes, 2)).astype(np.float32),
                    rng.normal(0.0, 1.0, size=(n_classes,)).astype(np.float32),
                ],
                int(1000 * (index + 1)),
            )
            for index in range(n_clients)
        ]
        class_counts = [
            rng.integers(0, 60, size=n_classes).astype(np.float32) for _ in range(n_clients)
        ]
        return results, keys, class_counts

    # Verify the pairwise masks of all participants cancel exactly, for federations of two to eight clients.
    def test_pairwise_masks_cancel_across_participants(self) -> None:
        for n_clients in (2, 3, 6, 8):
            total = sum(
                federated.secure_client_mask((4, 3), client_index, n_clients, 42, 1, 0, 0)
                for client_index in range(n_clients)
            )
            np.testing.assert_allclose(total, np.zeros((4, 3)), atol=1e-9)

            # A client's own mask must be far from zero, otherwise it would not hide the contribution.
            own_mask = federated.secure_client_mask((4, 3), 0, n_clients, 42, 1, 0, 0)
            self.assertGreater(float(np.max(np.abs(own_mask))), 1.0)

    # Verify the masked sum equals the plain aggregate within 1e-6 for every federation size and head-aggregation mode.
    def test_secure_aggregate_matches_plain_for_every_size_and_mode(self) -> None:
        for n_clients in (2, 3, 6, 8):
            results, keys, class_counts = self._synthetic_results(n_clients)
            payloads = [
                federated._client_next_activity_contribution_payload(params, keys, counts)
                for (params, _), counts in zip(results, class_counts)
            ]
            for mode in HEAD_AGG_MODES:
                plain = federated.aggregate_parameters(
                    results, next_activity_head_agg=mode, parameter_keys=keys,
                    next_activity_class_counts=class_counts, next_activity_contribution_payloads=payloads,
                )
                secure = self._secure_aggregate(
                    results, next_activity_head_agg=mode, parameter_keys=keys,
                    next_activity_class_counts=class_counts,
                )
                for expected, actual in zip(plain, secure):
                    self.assertEqual(actual.dtype, np.float32)
                    self.assertLess(
                        float(np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))), 1e-6,
                        f"{mode} aggregation drifted for {n_clients} clients",
                    )

    # Verify every masked message hides the client contribution and carries no raw update or raw count.
    def test_masked_messages_hide_the_client_contribution(self) -> None:
        results, keys, class_counts = self._synthetic_results(3)
        messages = federated.build_secure_client_message(
            results[0][0], results[0][1], 0, len(results), next_activity_head_agg="contribution",
            parameter_keys=keys, next_activity_class_counts=class_counts[0],
        )

        self.assertEqual(messages[0].kind, "weighted")
        self.assertEqual(set(messages[0].channels), {"contribution", "weight"})
        self.assertEqual(messages[2].kind, "contribution")
        self.assertEqual(set(messages[2].channels),
                         {"numerator", "denominator", "sample_contribution", "sample_weight"})

        # The masked contribution and the masked weight must both differ from the raw client values.
        raw_contribution = results[0][0][0].astype(np.float64) * float(results[0][1])
        self.assertFalse(np.allclose(messages[0].channels["contribution"], raw_contribution))
        self.assertNotAlmostEqual(float(messages[0].channels["weight"]), float(results[0][1]))

    # Verify the server helper refuses raw client updates and malformed masked messages.
    def test_server_helper_rejects_unmasked_input(self) -> None:
        results, keys, _ = self._synthetic_results(2)
        raw_client_updates = [list(params) for params, _ in results]

        with self.assertRaises(TypeError):
            federated.secure_aggregate_from_masked_messages(raw_client_updates)
        with self.assertRaises(ValueError):
            federated.secure_aggregate_from_masked_messages([])

        # A message with the wrong channel set is rejected, so a partially masked payload cannot pass.
        valid = federated.build_secure_client_message(results[0][0], results[0][1], 0, 2, parameter_keys=keys)
        broken = [federated.SecureTensorMessage(
            kind=valid[0].kind, dtype=valid[0].dtype,
            channels={"contribution": valid[0].channels["contribution"]},
        )] + valid[1:]
        with self.assertRaises(ValueError):
            federated.secure_aggregate_from_masked_messages([broken, broken])

    # Verify the recorded public seed lets anyone regenerate a client mask and recover that client's own update.
    # This is the documented limitation of the simulation. The report block must state it.
    def test_recorded_public_seed_does_not_hide_a_client_update(self) -> None:
        seed, round_index, n_clients = 42, 1, 3
        own_update = [np.array([2.0, 4.0], dtype=np.float32)]
        own_weight = 3
        message = federated.build_secure_client_message(
            own_update, own_weight, 0, n_clients, round_index=round_index, secure_aggregation_seed=seed)[0]

        # The masked channels differ from the raw values, so the message is genuinely masked.
        self.assertFalse(np.allclose(message.channels["contribution"], own_update[0] * own_weight))

        # Regenerating the same masks from the public seed recovers the update and the weight exactly.
        contribution_mask = federated.secure_client_mask(
            np.shape(message.channels["contribution"]), 0, n_clients, seed, round_index, 0,
            federated.SECURE_AGGREGATION_CHANNELS["contribution"])
        weight_mask = federated.secure_client_mask(
            (), 0, n_clients, seed, round_index, 0, federated.SECURE_AGGREGATION_CHANNELS["weight"])
        recovered_weight = float(np.asarray(message.channels["weight"]) - weight_mask)
        recovered_update = (np.asarray(message.channels["contribution"]) - contribution_mask) / recovered_weight

        np.testing.assert_allclose(recovered_update, own_update[0], atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(recovered_weight, float(own_weight), places=9)

    # Verify the plain aggregation path stays byte-identical for a fixed input when the knob is off.
    @staticmethod
    def test_plain_aggregation_is_byte_identical_when_secure_is_off() -> None:
        params_a = [np.array([2.0, 4.0], dtype=np.float32)]
        params_b = [np.array([6.0, 8.0], dtype=np.float32)]

        result = federated.aggregate_parameters([(params_a, 1), (params_b, 3)])

        np.testing.assert_array_equal(result[0], np.array([5.0, 7.0], dtype=np.float32))

    # Verify the same seed reproduces the masked messages exactly and another seed still recovers the same aggregate.
    def test_secure_aggregation_is_seed_reproducible(self) -> None:
        results = [
            ([np.array([2.0, 4.0], dtype=np.float32)], 1),
            ([np.array([6.0, 8.0], dtype=np.float32)], 3),
        ]
        first = self._secure_aggregate(results, round_index=2, secure_aggregation_seed=42)
        second = self._secure_aggregate(results, round_index=2, secure_aggregation_seed=42)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)

        other_seed = self._secure_aggregate(results, round_index=2, secure_aggregation_seed=7)
        for expected, actual in zip(first, other_seed):
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)

        masked_42 = federated.build_secure_client_message(results[0][0], results[0][1], 0, 2, round_index=1,
                                                          secure_aggregation_seed=42)
        masked_7 = federated.build_secure_client_message(results[0][0], results[0][1], 0, 2, round_index=1,
                                                         secure_aggregation_seed=7)
        self.assertFalse(np.allclose(masked_42[0].channels["contribution"], masked_7[0].channels["contribution"]))

    # Verify the contribution mode stays equivalent when a next-activity class has no support anywhere.
    # An exact-zero support test would divide one mask residue by another and leave the 1e-6 requirement.
    def test_secure_contribution_matches_plain_on_unsupported_classes(self) -> None:
        keys = ["next_activity_head.3.weight", "next_activity_head.3.bias"]
        worst = 0.0
        for n_clients in (2, 3, 6, 8):
            for n_classes in (3, 4, 5):
                for round_index in (1, 2, 3, 4):
                    rng = np.random.default_rng(n_clients * 100 + n_classes * 10 + round_index)
                    params = [[rng.normal(0.0, 1.0, size=(n_classes, 2)).astype(np.float32),
                               rng.normal(0.0, 1.0, size=(n_classes,)).astype(np.float32)]
                              for _ in range(n_clients)]
                    counts = [int(1000 * (index + 1)) for index in range(n_clients)]

                    # Class zero is observed by no client, so its federation-wide denominator is exactly zero.
                    class_counts = []
                    for _ in range(n_clients):
                        vector = rng.integers(1, 60, size=n_classes).astype(np.float32)
                        vector[0] = 0.0
                        class_counts.append(vector)
                    payloads = [federated._client_next_activity_contribution_payload(param, keys, vector)
                                for param, vector in zip(params, class_counts)]

                    plain = federated.aggregate_parameters(
                        list(zip(params, counts)), next_activity_head_agg="contribution", parameter_keys=keys,
                        next_activity_class_counts=class_counts, next_activity_contribution_payloads=payloads)
                    secure = federated.secure_aggregate_from_masked_messages([
                        federated.build_secure_client_message(
                            param, weight, index, n_clients, next_activity_head_agg="contribution",
                            parameter_keys=keys, next_activity_class_counts=vector, round_index=round_index)
                        for index, (param, weight, vector) in enumerate(zip(params, counts, class_counts))])
                    for expected, actual in zip(plain, secure):
                        deviation = float(np.max(np.abs(actual.astype(np.float64)
                                                        - expected.astype(np.float64))))
                        # A nan deviation would slip through max(), so non-finite values must fail on their own.
                        self.assertTrue(np.isfinite(deviation),
                                        f"unsupported-class contribution deviation is not finite: {deviation}")
                        worst = max(worst, deviation)
        self.assertLess(worst, 1e-6, f"unsupported-class contribution drifted by {worst}")

    # Verify the construction check reports a deviation inside the reported tolerance for every mode and size.
    def test_construction_deviation_stays_inside_the_reported_tolerance(self) -> None:
        keys = ["lstm.weight_ih_l0", "next_activity_head.3.weight", "next_activity_head.3.bias"]
        shapes = [(3, 2), (4, 2), (4,)]
        for n_clients in (2, 3, 6, 8):
            for mode in HEAD_AGG_MODES:
                deviation = federated.secure_construction_deviation(shapes, keys, n_clients, mode, True, 42)
                self.assertLess(deviation, federated.SECURE_AGGREGATION_TOLERANCE)

    # Verify FedProx penalty is zero at the global state and positive after a local update.
    def test_fedprox_penalty_detects_drift_from_reference_parameters(self) -> None:
        model = _build_tiny_model()
        reference = [param.detach().clone() for param in model.parameters()]

        self.assertEqual(float(federated.fedprox_penalty(model, reference, mu=1e-3).item()), 0.0)
        with torch.no_grad():
            next(model.parameters()).add_(1.0)
        self.assertGreater(float(federated.fedprox_penalty(model, reference, mu=1e-3).item()), 0.0)

# CLASS: Check that E_06 reuses E_05 data helpers with local and pooled configs.
class FederatedContextLoadingTests(unittest.TestCase):
    # Verify a joint client mask to exactly the next-activity classes it observed in its own local data.
    def test_build_client_next_activity_mask_context_is_bank_local(self) -> None:
        config = baseline.BaselineRunConfig(
            dataset="joint", heterogeneity="iid", n_clients=6, regime="local", bank="bpic2017:A",
        )

        context = federated.build_client_next_activity_mask_context(config, TinyMaskPrefixDataset(), n_classes=7,)

        # TinyMaskPrefixDataset carries dataset_id bpic2017 and observed next-activity classes 5 and 6.
        mask = context.mask_by_dataset[context.dataset_code_by_id["bpic2017"]]
        self.assertFalse(context.is_noop)
        self.assertTrue(bool(mask[5]))
        self.assertTrue(bool(mask[6]))
        self.assertFalse(bool(mask[0]))

    # Verify client-side target counts use only valid next-activity labels from that client's train dataset.
    def test_next_activity_target_counts_from_dataset_counts_valid_local_targets(self) -> None:
        helper = getattr(baseline, "next_activity_target_counts_from_dataset", None)
        self.assertTrue(callable(helper))

        dataset = TinyMaskPrefixDataset()
        dataset.arrays["next_activity_label"] = np.array([1, 1, 2, 5, -1, 6], dtype=np.int64)
        dataset.arrays["next_activity_mask"] = np.array([1, 1, 0, 1, 1, 1], dtype=np.int8)

        counts = helper(dataset, n_classes=6)

        np.testing.assert_array_equal(counts, np.array([0, 2, 0, 0, 0, 1], dtype=np.float32))

    # Verify local client datasets use local configs and global statistics use the centralized config.
    def test_load_federated_context_uses_local_clients_and_pooled_training_statistics(self) -> None:
        config = federated.FederatedRunConfig(n_clients=3, batch_size=16, num_workers=0)
        spec = {
            "prefix": {"cap": 8, "static_padding_length": 8},
            "target_scalers": {
                "remaining_time": {
                    "transform": "raw",
                    "scaling": "zscore",
                    "center": 0.0,
                    "scale": 10.0,
                    "use_softplus": False,
                    "median_model_units": 0.0,
                    "train_median_seconds": 0.0,
                    "train_value_count": 2,
                }
            },
        }
        vocabularies = {"activity": {"[PAD]": 0}, "next_activity": {"[END]": 0}}
        scalers = {"time_delta": {"mean": 0.0, "std": 1.0}}
        schema_profile = {"sequence_categorical_columns": ["activity"], "sequence_numerical_columns": [],
                          "offer_numerical_columns": []}
        mapping = {"datasets": {"bpic2017": {}}}
        train_labels = pd.Series([0, 1, 2])
        remaining_time_repr = core.RemainingTimeRepr("raw", "zscore", 0.0, 10.0, False, 0.0)

        # Build deterministic fake datasets by regime, bank and split.
        datasets: dict[tuple[str, str, str], list[int]] = {}

        # Stub the cached split loader with a fixed sample list.
        def cached_dataset(local_config: object, split_name: str, *_args: object) -> list[int]:
            bank = getattr(local_config, "bank", None) or "pooled"
            key = (getattr(local_config, "regime"), bank, split_name)
            datasets[key] = [len(datasets) + 1]
            return datasets[key]

        # Stub the loader factory and record the received seed.
        def loader_payload(dataset: object, batch_size: int, shuffle: bool, seed: int, num_workers: int,
            next_activity_mask_context: object = None,) -> dict[str, object]:
            return {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "seed": seed,
                "num_workers": num_workers,
                "mask_context": next_activity_mask_context,
            }

        # Patch every E_05 data boundary so no parquet or cache IO is used.
        with mock.patch.object(
            federated.baseline,
            "load_training_metadata",
            return_value=(spec, vocabularies, scalers, schema_profile, mapping),
        ) as load_metadata:
            with mock.patch.object(federated.baseline, "build_next_activity_mask_context",
                                   return_value=_noop_mask_context()):
                with mock.patch.object(federated.baseline, "load_cached_prefix_dataset",
                                       side_effect=cached_dataset) as load_cached:
                    with mock.patch.object(federated.baseline, "load_mapped_events") as load_mapped:
                        with mock.patch.object(federated.baseline, "build_prefix_dataset") as build_prefix:
                            with mock.patch.object(federated.baseline, "cache_prefix_dataset") as cache_prefix:
                                with mock.patch.object(federated.baseline, "make_loader", side_effect=loader_payload):
                                    with mock.patch.object(
                                        federated.baseline,
                                        "training_outcome_labels_from_dataset",
                                        return_value=train_labels,
                                    ) as labels_from_dataset:
                                        with mock.patch.object(
                                            baseline,
                                            "next_activity_class_presence_from_dataset",
                                            return_value=np.array([True], dtype=bool),
                                        ):
                                            with mock.patch.object(
                                                baseline,
                                                "next_activity_target_counts_from_dataset",
                                                side_effect=lambda dataset, _n: np.array([float(dataset[0])],
                                                                                         dtype=np.float32),
                                                create=True,
                                            ) as counts_from_dataset:
                                                context = federated.load_federated_context(config)

        # Confirm metadata is loaded once through the pooled centralized config.
        self.assertEqual(load_metadata.call_args.args[0].regime, "centralized")
        self.assertEqual(load_metadata.call_args.args[0].bank, None)

        # Confirm no cache misses touched the slower parquet rebuild path.
        load_mapped.assert_not_called()
        build_prefix.assert_not_called()
        cache_prefix.assert_not_called()

        # Confirm every simulated bank received a local E_05 config for train, val and test.
        local_calls = [call for call in load_cached.call_args_list if call.args[0].regime == "local"]
        self.assertEqual({call.args[0].bank for call in local_calls}, {"A", "B", "C"})
        self.assertEqual({call.args[1] for call in local_calls}, {"train", "val", "test"})

        # Confirm pooled labels are read from the centralized train dataset and the RT representation comes from E_04.
        pooled_train = datasets[("centralized", "pooled", "train")]
        labels_from_dataset.assert_called_once_with(pooled_train)
        self.assertIs(context.pooled_train_labels, train_labels)
        self.assertEqual(context.remaining_time_repr, remaining_time_repr)

        # Confirm the context exposes clients and pooled evaluation loaders.
        self.assertEqual([client.bank for client in context.clients], ["A", "B", "C"])
        self.assertIs(context.pooled_val_dataset, datasets[("centralized", "pooled", "val")])
        self.assertIs(context.pooled_test_dataset, datasets[("centralized", "pooled", "test")])
        # The loader factory is patched to record its arguments as a dict, so both stubs are read untyped.
        pooled_val_loader: Any = context.pooled_val_loader
        client_train_loader: Any = context.clients[0].train_loader
        self.assertEqual(pooled_val_loader["shuffle"], False)
        self.assertEqual(client_train_loader["shuffle"], True)
        self.assertIs(context.clients[0].training_next_activity_mask_context, client_train_loader["mask_context"])
        self.assertIs(context.evaluation_next_activity_mask_context, pooled_val_loader["mask_context"])
        counts_from_dataset.assert_has_calls(
            [mock.call(datasets[("local", bank, "train")], len(vocabularies["next_activity"]))
             for bank in ["A", "B", "C"]]
        )
        self.assertEqual([client.next_activity_class_counts.tolist() for client in context.clients],
                         [[4.0], [7.0], [10.0]])

    # Verify joint runs build one local context per dataset-qualified client.
    def test_load_federated_context_builds_joint_clients_from_total_client_count(self) -> None:
        config = federated.FederatedRunConfig(dataset="joint", heterogeneity="medium", n_clients=8, batch_size=16,
                                             num_workers=0)
        spec = {
            "prefix": {"cap": 83, "static_padding_length": 83},
            "target_scalers": {
                "remaining_time": {
                    "transform": "raw",
                    "scaling": "zscore",
                    "center": 0.0,
                    "scale": 10.0,
                    "use_softplus": False,
                    "median_model_units": 0.0,
                    "train_median_seconds": 0.0,
                    "train_value_count": 2,
                }
            },
        }
        vocabularies = {"activity": {"[PAD]": 0}, "next_activity": {"[END]": 0}}
        scalers = {"time_delta": {"mean": 0.0, "std": 1.0}}
        schema_profile = {"sequence_categorical_columns": ["activity"], "sequence_numerical_columns": [],
                          "offer_numerical_columns": []}
        mapping = {"datasets": {"bpic2017": {}, "bpic2012": {}}}
        train_labels = pd.Series([0, 1, 2])
        datasets: dict[tuple[str, str, str], list[int]] = {}

        # Stub the cached split loader with a fixed sample list.
        def cached_dataset(local_config: object, split_name: str, *_args: object) -> list[int]:
            bank = getattr(local_config, "bank", None) or "pooled"
            key = (getattr(local_config, "regime"), bank, split_name)
            datasets[key] = [len(datasets) + 1]
            return datasets[key]

        # Stub the loader factory and record the received seed.
        def loader_payload(
            dataset: object,
            batch_size: int,
            shuffle: bool,
            seed: int,
            num_workers: int,
            next_activity_mask_context: object = None,
        ) -> dict[str, object]:
            return {
                "dataset": dataset,
                "batch_size": batch_size,
                "shuffle": shuffle,
                "seed": seed,
                "num_workers": num_workers,
                "mask_context": next_activity_mask_context,
            }

        with mock.patch.object(
            federated.baseline,
            "load_training_metadata",
            return_value=(spec, vocabularies, scalers, schema_profile, mapping),
        ):
            with mock.patch.object(federated.baseline, "build_next_activity_mask_context",
                                   return_value=_noop_mask_context()):
                with mock.patch.object(federated.baseline, "load_cached_prefix_dataset", side_effect=cached_dataset):
                    with mock.patch.object(federated.baseline, "make_loader", side_effect=loader_payload):
                        with mock.patch.object(
                            federated.baseline,
                            "training_outcome_labels_from_dataset",
                            return_value=train_labels,
                        ):
                            with mock.patch.object(
                                baseline,
                                "next_activity_class_presence_from_dataset",
                                return_value=np.array([True], dtype=bool),
                            ):
                                with mock.patch.object(
                                    baseline,
                                    "next_activity_target_counts_from_dataset",
                                    return_value=np.array([1.0], dtype=np.float32),
                                ):
                                    with mock.patch.object(
                                        federated,
                                        "build_client_next_activity_mask_context",
                                        return_value=_noop_mask_context(),
                                    ):
                                        context = federated.load_federated_context(config)

        self.assertEqual([client.bank for client in context.clients], [
            "bpic2017:A",
            "bpic2017:B",
            "bpic2017:C",
            "bpic2017:D",
            "bpic2017:E",
            "bpic2012:A",
            "bpic2012:B",
            "bpic2012:C",
        ])
        self.assertIn(("local", "bpic2017:A", "train"), datasets)
        self.assertIn(("local", "bpic2012:A", "train"), datasets)

    # Verify E_06 file names can safely contain dataset-qualified joint clients.
    def test_safe_bank_for_artifacts_reuses_joint_safe_form(self) -> None:
        self.assertEqual(federated._safe_bank("bpic2017:A"), "bpic2017_A")
        self.assertEqual(federated._safe_bank("A"), "A")

# CLASS: Check local federated training and round learning-rate helpers.
class FederatedLocalTrainingTests(unittest.TestCase):
    # Verify one local federated epoch updates at least one trainable parameter.
    def test_train_federated_epoch_updates_model_parameters(self) -> None:
        config = federated.FederatedRunConfig(batch_size=2, progress_bars=False, device="cpu")
        model = _build_tiny_model()
        loader = baseline.make_loader(TinyPrefixDataset(), batch_size=2, shuffle=False, seed=42, num_workers=0)
        optimizer = core.build_multitask_optimizer(model, 1e-2, 0.0, 1.0, 1.0, 1.0)
        reference = [param.detach().clone() for param in model.parameters()]
        before = [param.detach().clone() for param in model.parameters()]

        metrics = federated.train_federated_epoch(
            model=model,
            loader=loader,
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
            huber_beta=1.0,
            optimizer=optimizer,
            reference_params=reference,
            fedprox_mu=0.0,
            device=torch.device("cpu"),
            config=config,
            use_dp=False,
        )

        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        self.assertIn("loss_total", metrics)

    # Verify FedProx adds and reports a proximal term in local training.
    def test_train_federated_epoch_reports_fedprox_penalty(self) -> None:
        config = federated.FederatedRunConfig(batch_size=2, progress_bars=False, device="cpu")
        model = _build_tiny_model()
        loader = baseline.make_loader(TinyPrefixDataset(), batch_size=2, shuffle=False, seed=42, num_workers=0)
        optimizer = core.build_multitask_optimizer(model, 1e-2, 0.0, 1.0, 1.0, 1.0)
        reference = [param.detach().clone() for param in model.parameters()]

        metrics = federated.train_federated_epoch(
            model=model,
            loader=loader,
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
            huber_beta=1.0,
            optimizer=optimizer,
            reference_params=reference,
            fedprox_mu=1e-3,
            device=torch.device("cpu"),
            config=config,
            use_dp=False,
        )

        self.assertIn("train_fedprox_penalty", metrics)
        self.assertIn("loss_total_with_fedprox", metrics)

    # Verify the explicit DP smoke cap stops local training after the requested batch count.
    def test_train_federated_epoch_respects_dp_smoke_batch_cap(self) -> None:
        config = federated.FederatedRunConfig(
            batch_size=1,
            progress_bars=False,
            device="cpu",
            use_dp=True,
            dp_smoke_max_batches=1,
        )
        model = _build_tiny_model()
        loader = baseline.make_loader(TinyPrefixDataset(), batch_size=1, shuffle=False, seed=42, num_workers=0)
        optimizer = core.build_multitask_optimizer(model, 1e-2, 0.0, 1.0, 1.0, 1.0)
        reference = [param.detach().clone() for param in model.parameters()]

        metrics = federated.train_federated_epoch(
            model=model,
            loader=loader,
            outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
            remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
            huber_beta=1.0,
            optimizer=optimizer,
            reference_params=reference,
            fedprox_mu=0.0,
            device=torch.device("cpu"),
            config=config,
            use_dp=True,
        )

        self.assertEqual(metrics["n_batches"], 1)

    # Verify cosine scheduling is applied per optimizer group from each group's base LR.
    def test_optimizer_lrs_for_round_uses_group_base_lrs(self) -> None:
        config = federated.FederatedRunConfig(
            learning_rate=1e-3,
            outcome_lr_scale=0.3,
            next_activity_lr_scale=1.0,
            remaining_time_lr_scale=1.5,
            lr_scheduler_t_max=5,
            lr_scheduler_min_lr=1e-6,
        )

        lrs = federated.optimizer_lrs_for_round(config, round_idx=1)

        self.assertAlmostEqual(lrs["trunk"], 1e-3)
        self.assertAlmostEqual(lrs["outcome"], 3e-4)
        self.assertAlmostEqual(lrs["next_activity"], 1e-3)
        self.assertAlmostEqual(lrs["remaining_time"], 1.5e-3)

# CLASS: Check Flower client and strategy contracts for the synchronous E_06 loop.
class FederatedFlowerContractTests(unittest.TestCase):
    # Verify a bank client fit returns parameters, prefix count and train metrics.
    def test_bank_client_fit_returns_flower_contract_payload(self) -> None:
        config = federated.FederatedRunConfig(batch_size=2, local_epochs=1, progress_bars=False, device="cpu")
        dataset = TinyPrefixDataset()
        loader = baseline.make_loader(dataset, batch_size=2, shuffle=False, seed=42, num_workers=0)
        client_context = federated.FederatedClientContext(
            bank="A",
            train_dataset=dataset,
            val_dataset=dataset,
            test_dataset=dataset,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            train_prefix_count=len(dataset),
            next_activity_class_counts=np.array([1.0], dtype=np.float32),
            training_next_activity_mask_context=_noop_mask_context(),
        )
        run_context = federated.FederatedRunContext(
            clients=[client_context],
            pooled_train_dataset=dataset,
            pooled_val_dataset=dataset,
            pooled_test_dataset=dataset,
            pooled_val_loader=loader,
            pooled_test_loader=loader,
            pooled_train_labels=pd.Series([0, 2]),
            remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
            evaluation_next_activity_mask_context=_noop_mask_context(),
            spec={},
            vocabularies={},
            scalers={},
            schema_profile={},
            mapping={},
        )
        initial_model = _build_tiny_model()

        # Patch the model factory so the test focuses on the NumPyClient fit contract.
        with mock.patch.object(federated, "build_model_for_federated", return_value=_build_tiny_model()):
            client = federated.FederatedBankClient(
                config=config,
                client_context=client_context,
                run_context=run_context,
                device=torch.device("cpu"),
                outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
                next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
                client_index=0,
                n_clients=1,
                parameter_keys=federated.parameter_key_order(_build_tiny_model()),
            )
            parameters, num_examples, metrics = client.fit(
                federated.model_parameters_to_numpy(initial_model),
                {"server_round": 1, "proximal_mu": 1e-3},
            )

        self.assertTrue(all(array.dtype == np.float32 for array in parameters))
        self.assertEqual(num_examples, len(dataset))
        self.assertEqual(metrics["round"], 1)
        self.assertEqual(metrics["bank"], "A")
        self.assertIn("train_loss_total", metrics)
        self.assertIn("train_loss_total_with_fedprox", metrics)
        self.assertIn("train_fedprox_penalty", metrics)
        self.assertEqual(metrics["train_prefix_count"], len(dataset))
        self.assertNotIn("dp_epsilon_spent", metrics)

        # The secure client step returns masked messages only, so no raw update crosses the orchestration boundary.
        with mock.patch.object(federated, "build_model_for_federated", return_value=_build_tiny_model()):
            messages, secure_metrics = client.fit_secure(
                federated.model_parameters_to_numpy(initial_model),
                {"server_round": 1, "proximal_mu": 1e-3},
            )

        self.assertTrue(all(isinstance(message, federated.SecureTensorMessage) for message in messages))
        self.assertEqual(len(messages), len(parameters))
        self.assertEqual(secure_metrics["bank"], "A")

    # Verify client fit uses the client-local training mask and the run-level evaluation mask.
    def test_bank_client_fit_separates_training_and_evaluation_masks(self) -> None:
        config = federated.FederatedRunConfig(batch_size=2, local_epochs=1, progress_bars=False, device="cpu")
        dataset = TinyPrefixDataset()
        loader = baseline.make_loader(dataset, batch_size=2, shuffle=False, seed=42, num_workers=0)
        training_context = _noop_mask_context()
        evaluation_context = _eval_mask_context()
        client_context = federated.FederatedClientContext(
            bank="A",
            train_dataset=dataset,
            val_dataset=dataset,
            test_dataset=dataset,
            train_loader=loader,
            val_loader=loader,
            test_loader=loader,
            train_prefix_count=len(dataset),
            next_activity_class_counts=np.array([1.0], dtype=np.float32),
            training_next_activity_mask_context=training_context,
        )
        run_context = federated.FederatedRunContext(
            clients=[client_context],
            pooled_train_dataset=dataset,
            pooled_val_dataset=dataset,
            pooled_test_dataset=dataset,
            pooled_val_loader=loader,
            pooled_test_loader=loader,
            pooled_train_labels=pd.Series([0, 2]),
            remaining_time_repr=core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0),
            evaluation_next_activity_mask_context=evaluation_context,
            spec={},
            vocabularies={},
            scalers={},
            schema_profile={},
            mapping={},
        )
        val_metrics = {
            "loss_total": 1.0,
            "outcome": {"macro_f1": 0.5},
            "next_activity": {"top1_accuracy": 0.5},
            "remaining_time": {"mae": 10.0},
        }

        with mock.patch.object(federated, "build_model_for_federated", return_value=_build_tiny_model()):
            with mock.patch.object(
                federated,
                "train_federated_epoch",
                return_value={
                    "loss_total": 1.0,
                    "loss_total_with_fedprox": 1.0,
                    "train_fedprox_penalty": 0.0,
                    "n_batches": 1,
                },
            ) as train_epoch:
                with mock.patch.object(
                    federated,
                    "evaluate_model_for_federated",
                    return_value=(val_metrics, None),
                ) as evaluate_model:
                    client = federated.FederatedBankClient(
                        config=config,
                        client_context=client_context,
                        run_context=run_context,
                        device=torch.device("cpu"),
                        outcome_loss=torch.nn.CrossEntropyLoss(reduction="none"),
                        next_activity_loss=torch.nn.CrossEntropyLoss(reduction="none"),
                        client_index=0,
                        n_clients=1,
                        parameter_keys=federated.parameter_key_order(_build_tiny_model()),
                    )
                    client.fit(federated.model_parameters_to_numpy(_build_tiny_model()), {"server_round": 1})

        self.assertIs(train_epoch.call_args.kwargs["next_activity_mask_context"], training_context)
        self.assertIs(evaluate_model.call_args.kwargs["next_activity_mask_context"], evaluation_context)

    # Verify strategy construction chooses the requested Flower strategy family.
    def test_build_strategy_selects_fedavg_or_fedprox(self) -> None:
        initial_params = [np.array([1.0], dtype=np.float32)]
        evaluate_fn = lambda _round, _params, _config: (0.0, {})

        fedavg = federated.build_strategy(
            federated.FederatedRunConfig(strategy="fedavg"), initial_params, evaluate_fn
        )
        fedprox = federated.build_strategy(
            federated.FederatedRunConfig(strategy="fedprox"), initial_params, evaluate_fn
        )

        self.assertIsInstance(fedavg, federated.SavingFedAvgStrategy)
        self.assertIsInstance(fedprox, federated.SavingFedProxStrategy)

# CLASS: Check optional DP-SGD construction while keeping no-DP behavior untouched.
class FederatedDpPathTests(unittest.TestCase):
    # Verify no-DP model construction keeps the normal torch LSTM and no epsilon metric appears.
    def test_no_dp_path_keeps_plain_lstm_and_no_epsilon_metric(self) -> None:
        config = federated.FederatedRunConfig(use_dp=False)
        vocabularies = {
            "activity": {"[PAD]": 0, "A": 1},
            "resource": {"[PAD]": 0},
            baseline.encoding.NEXT_ACTIVITY_TARGET: {"[END]": 0, "A": 1},
        }
        schema_profile = {
            "sequence_categorical_columns": ["activity", "resource"],
            "sequence_numerical_columns": ["time_delta"],
            "offer_numerical_columns": ["offered_amount"],
        }

        model = federated.build_model_for_federated(config, vocabularies, schema_profile, use_dp=False)

        self.assertIsInstance(model.lstm, torch.nn.LSTM)

    # Verify DP model construction swaps only the LSTM trunk and preserves sorted-key parameter compatibility.
    def test_dp_model_uses_dplstm_and_round_trips_to_plain_model(self) -> None:
        config = federated.FederatedRunConfig(use_dp=True)
        vocabularies = {
            "activity": {"[PAD]": 0, "A": 1},
            "resource": {"[PAD]": 0},
            baseline.encoding.NEXT_ACTIVITY_TARGET: {"[END]": 0, "A": 1},
        }
        schema_profile = {
            "sequence_categorical_columns": ["activity", "resource"],
            "sequence_numerical_columns": ["time_delta"],
            "offer_numerical_columns": ["offered_amount"],
        }
        plain = federated.build_model_for_federated(config, vocabularies, schema_profile, use_dp=False)
        private = federated.build_model_for_federated(config, vocabularies, schema_profile, use_dp=True)

        self.assertIsInstance(private.lstm, DPLSTM)
        self.assertEqual(set(plain.state_dict()), set(private.state_dict()))
        federated.load_numpy_parameters(plain, federated.model_parameters_to_numpy(private))
        for key, value in private.state_dict().items():
            self.assertTrue(torch.equal(value, plain.state_dict()[key]), key)

    # Verify DP calibration passes the expanded RDP alpha grid into Opacus.
    def test_dp_noise_multiplier_uses_expanded_alpha_grid(self) -> None:
        captured: dict[str, object] = {}

        # Capture the Opacus calibration call without doing a real accountant search.
        def fake_get_noise_multiplier(**kwargs: object) -> float:
            captured.update(kwargs)
            return 1.25

        with mock.patch.object(federated, "load_opacus_symbol", return_value=fake_get_noise_multiplier):
            result = federated.resolve_dp_noise_multiplier(
                sample_rate=0.25,
                target_epsilon=50.0,
                target_delta=1e-5,
                epochs=2,
            )

        self.assertEqual(result, 1.25)
        self.assertEqual(captured["accountant"], "rdp")
        self.assertEqual(captured["alphas"], list(federated.DP_RDP_ALPHAS))
        self.assertLess(min(federated.DP_RDP_ALPHAS), 1.1)
        self.assertGreater(max(federated.DP_RDP_ALPHAS), 500.0)

    # Verify known DP warnings are logged once and then suppressed by exact message filters.
    def test_known_dp_warnings_are_suppressed_after_one_notice(self) -> None:
        with mock.patch.object(federated, "_DP_WARNING_FILTERS_CONFIGURED", False, create=True):
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                federated.configure_dp_warning_filters()
                warnings.warn(
                    "Secure RNG turned off. This is perfectly fine for experimentation as it allows for much "
                    "faster training performance.",
                    UserWarning,
                )
                warnings.warn(
                    "Optimal order is the largest alpha. Please consider expanding the range of alphas to get a "
                    "tighter privacy bound.",
                    UserWarning,
                )
                warnings.warn(
                    "Full backward hook is firing when gradients are computed with respect to module outputs "
                    "since no inputs require gradients.",
                    UserWarning,
                )
                warnings.warn("Unrelated warning still visible.", UserWarning)

        messages = [str(record.message) for record in records]
        self.assertTrue(any("Known DP warnings are suppressed" in message for message in messages))
        self.assertEqual(messages.count("Unrelated warning still visible."), 1)
        self.assertFalse(any(message.startswith("Secure RNG turned off") for message in messages))
        self.assertFalse(any(message.startswith("Optimal order is the") for message in messages))
        self.assertFalse(any(message.startswith("Full backward hook is firing") for message in messages))

    # Verify tighter epsilon targets require larger noise multipliers.
    def test_dp_noise_multiplier_increases_when_epsilon_decreases(self) -> None:
        loose = federated.resolve_dp_noise_multiplier(
            sample_rate=0.25,
            target_epsilon=50.0,
            target_delta=1e-5,
            epochs=1,
        )
        strict = federated.resolve_dp_noise_multiplier(
            sample_rate=0.25,
            target_epsilon=5.0,
            target_delta=1e-5,
            epochs=1,
        )

        self.assertGreater(loose, 0.0)
        self.assertGreater(strict, loose)

    # Verify full DP training without the smoke cap decreases train and validation losses on a learnable toy split.
    def test_full_dp_training_decreases_train_and_validation_loss_without_smoke_cap(self) -> None:
        # Suppress known direct-Opacus warnings because this toy test bypasses E_06 workflow filters.
        warnings.filterwarnings("ignore", message=r"Secure RNG turned off\..*", category=UserWarning)
        warnings.filterwarnings(
            "ignore",
            message=r"Full backward hook is firing when gradients are computed.*",
            category=UserWarning,
        )
        core.set_global_seed(42)
        config = federated.FederatedRunConfig(
            batch_size=2,
            progress_bars=False,
            device="cpu",
            use_dp=True,
            dp_smoke_max_batches=None,
            gradient_clip_norm=1.0,
        )
        dataset = TinyPrefixDataset()
        loader = baseline.make_loader(dataset, batch_size=2, shuffle=False, seed=42, num_workers=0)
        model = core.MultitaskLSTM(
            categorical_vocab_sizes={"activity": 7, "resource": 6},
            numerical_dim=2,
            offer_dim=2,
            next_activity_classes=5,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
            head_hidden_size=6,
            lstm_cls=DPLSTM,
        )
        optimizer = core.build_multitask_optimizer(model, 5e-2, 0.0, 1.0, 1.0, 1.0)

        # Wrap the toy client once with Opacus so every epoch uses DPOptimizer and DPLSTM.
        privacy_engine = PrivacyEngine(secure_mode=False)
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.0,
            max_grad_norm=10.0,
            poisson_sampling=False,
        )
        reference = [param.detach().clone() for param in model.parameters()]
        remaining_time_repr = core.RemainingTimeRepr("raw", "median", 0.0, 10.0, True, 1.0)
        outcome_loss = torch.nn.CrossEntropyLoss(reduction="none")
        next_activity_loss = torch.nn.CrossEntropyLoss(reduction="none")
        train_losses: list[float] = []
        val_losses: list[float] = []
        train_metrics: dict[str, Any] = {}

        # The federated and baseline configs share the fields this path reads, so the evaluation takes it untyped.
        eval_config: Any = config

        # Fit several complete DP epochs and evaluate through the shared E_05 metrics path after each epoch.
        for _epoch in range(8):
            train_metrics = federated.train_federated_epoch(
                model=model,
                loader=loader,
                outcome_loss=outcome_loss,
                next_activity_loss=next_activity_loss,
                remaining_time_repr=remaining_time_repr,
                huber_beta=1.0,
                optimizer=optimizer,
                reference_params=reference,
                fedprox_mu=0.0,
                device=torch.device("cpu"),
                config=config,
                use_dp=True,
            )
            val_metrics, _ = baseline.evaluate_model(
                model,
                loader,
                outcome_loss,
                next_activity_loss,
                remaining_time_repr,
                1.0,
                torch.device("cpu"),
                eval_config,
                dataset,
                progress_label="toy dp val",
                collect_predictions=False,
            )
            train_losses.append(float(train_metrics["loss_total"]))
            val_losses.append(float(val_metrics["loss_total"]))

        self.assertIsNone(config.dp_smoke_max_batches)
        self.assertEqual(train_metrics["n_batches"], len(loader))
        self.assertLess(train_losses[-1], train_losses[0])
        self.assertLess(val_losses[-1], val_losses[0])

if __name__ == "__main__":
    unittest.main()
