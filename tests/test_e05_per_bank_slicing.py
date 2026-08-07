from __future__ import annotations
import sys
import unittest
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_core_final as core

# HELPER: Build one deterministic evaluated split with three outcome classes and mixed head masks.
def _split_arrays(n_prefixes: int = 240) -> baseline.SplitEvaluationArrays:
    rng = np.random.default_rng(11)
    outcome_labels = rng.integers(0, 3, size=n_prefixes)
    next_labels = rng.integers(0, 5, size=n_prefixes)
    next_masks = rng.random(n_prefixes) > 0.2
    remaining_masks = rng.random(n_prefixes) > 0.3
    return baseline.SplitEvaluationArrays(
        outcome_labels=outcome_labels,
        outcome_logits=rng.normal(0.0, 1.0, size=(n_prefixes, 3)),
        next_activity_labels=next_labels,
        next_activity_logits=rng.normal(0.0, 1.0, size=(n_prefixes, 5)),
        next_activity_masks=next_masks,
        remaining_time_true_seconds=rng.uniform(0.0, 1e6, size=n_prefixes),
        remaining_time_pred_seconds=rng.uniform(0.0, 1e6, size=n_prefixes),
        remaining_time_masks=remaining_masks,
        outcome_loss_values=rng.uniform(0.0, 2.0, size=n_prefixes),
        next_activity_loss_values=rng.uniform(0.0, 2.0, size=n_prefixes),
        remaining_time_loss_values=rng.uniform(0.0, 2.0, size=n_prefixes),
    )

# HELPER: Partition the prefixes into one boolean selector per simulated bank.
def _bank_selectors(n_prefixes: int, banks: tuple[str, ...]) -> dict[str, np.ndarray]:
    owners = np.array([banks[index % len(banks)] for index in range(n_prefixes)])
    return {bank: np.asarray(owners == bank) for bank in banks}

class PerBankSlicingTests(unittest.TestCase):
    # Verify the per-bank confusion matrices sum exactly to the global confusion matrix of the same forward pass.
    def test_per_bank_confusion_matrices_sum_to_the_global_matrix(self) -> None:
        arrays = _split_arrays()
        config = baseline.BaselineRunConfig()
        selectors = _bank_selectors(len(arrays.outcome_labels), ("A", "B", "C"))

        pooled = core.compute_outcome_metrics(arrays.outcome_labels, arrays.outcome_logits)
        per_bank = {bank: baseline.subset_evaluation_metrics(arrays, selector, config)
                    for bank, selector in selectors.items()}

        summed = np.zeros((3, 3), dtype=int)
        for metrics in per_bank.values():
            summed += np.array(metrics["outcome"]["confusion_matrix"], dtype=int)

        np.testing.assert_array_equal(summed, np.array(pooled["confusion_matrix"], dtype=int))
        self.assertEqual(sum(int(metrics["n_prefixes"]) for metrics in per_bank.values()), len(arrays.outcome_labels))

    # Verify the per-bank per-class supports and the remaining-time valid counts also decompose the global totals.
    def test_per_bank_supports_and_masked_counts_decompose_the_global_totals(self) -> None:
        arrays = _split_arrays()
        config = baseline.BaselineRunConfig()
        selectors = _bank_selectors(len(arrays.outcome_labels), ("A", "B"))

        pooled = core.compute_outcome_metrics(arrays.outcome_labels, arrays.outcome_logits)
        per_bank = [baseline.subset_evaluation_metrics(arrays, selector, config) for selector in selectors.values()]

        for label in ("0", "1", "2"):
            summed_support = sum(int(metrics["outcome"]["per_class"][label]["support"]) for metrics in per_bank)
            self.assertEqual(summed_support, int(pooled["per_class"][label]["support"]))

        # The pooled remaining-time MAE is the prefix-count weighted mean of the per-bank MAE values.
        pooled_rt = core.compute_remaining_time_metrics(
            torch.tensor(arrays.remaining_time_true_seconds),
            torch.tensor(arrays.remaining_time_pred_seconds),
            torch.tensor(arrays.remaining_time_masks.astype(np.int8)),
        )
        weights = [float((selector & arrays.remaining_time_masks).sum()) for selector in selectors.values()]
        weighted = sum(
            float(metrics["remaining_time"]["mae"]) * weight for metrics, weight in zip(per_bank, weights)
        ) / sum(weights)

        # The metric helper reports seconds in float32, so the recombination matches to float32 resolution.
        self.assertLess(abs(weighted - float(pooled_rt["mae"])) / float(pooled_rt["mae"]), 1e-6)

    # Verify the per-bank losses are the subset means of the same per-sample loss values the pooled block averages.
    def test_per_bank_losses_are_subset_means_of_the_pooled_loss_values(self) -> None:
        arrays = _split_arrays()
        config = baseline.BaselineRunConfig()
        selectors = _bank_selectors(len(arrays.outcome_labels), ("A", "B", "C"))

        for selector in selectors.values():
            metrics = baseline.subset_evaluation_metrics(arrays, selector, config)
            self.assertAlmostEqual(
                float(metrics["loss_outcome"]), float(arrays.outcome_loss_values[selector].mean()), places=10)
            valid_next = selector & arrays.next_activity_masks
            self.assertAlmostEqual(
                float(metrics["loss_next_activity"]),
                float(arrays.next_activity_loss_values[valid_next].mean()), places=10)
            self.assertAlmostEqual(
                float(metrics["loss_total"]),
                core.weighted_total_loss(
                    metrics["loss_outcome"], metrics["loss_next_activity"], metrics["loss_remaining_time"],
                    config.outcome_loss_weight, config.next_activity_loss_weight, config.remaining_time_loss_weight,
                ),
                places=10,
            )

    # Verify a prefix that belongs to no bank split is rejected instead of silently dropping out of the fairness table.
    def test_unmapped_prefixes_are_rejected(self) -> None:
        class _Dataset:
            prefix_index = [
                type("Row", (), {"dataset_id": "bpic2017", "case_id": "c1"})(),
                type("Row", (), {"dataset_id": "bpic2017", "case_id": "c2"})(),
            ]

        with self.assertRaises(ValueError):
            baseline.bank_selectors_for_dataset(_Dataset(), ("A",), {("bpic2017", "c1"): "A"})

        selectors = baseline.bank_selectors_for_dataset(
            _Dataset(), ("A", "B"), {("bpic2017", "c1"): "A", ("bpic2017", "c2"): "B"}
        )
        np.testing.assert_array_equal(selectors["A"], np.array([True, False]))
        np.testing.assert_array_equal(selectors["B"], np.array([False, True]))

if __name__ == "__main__":
    unittest.main()