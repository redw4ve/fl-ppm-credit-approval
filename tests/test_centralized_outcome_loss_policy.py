from __future__ import annotations
import unittest
import pandas as pd
from E_training import E_05_central_and_local_baselines_final as baseline

class BaselineOutcomeLossPolicyTests(unittest.TestCase):
    # Verify class weights use training labels only.
    def test_class_weights_use_training_labels_only(self) -> None:
        labels = pd.Series([0, 1, 1, 2, 2, 2])
        weights = baseline.compute_outcome_class_weights(labels)
        self.assertEqual(weights, {0: 2.0, 1: 1.0, 2: 2 / 3})

    # Verify power zero reproduces unweighted CE.
    def test_power_zero_reproduces_unweighted_ce(self) -> None:
        labels = pd.Series([0, 1, 1, 2, 2, 2])
        weights = baseline.compute_outcome_class_weights(labels, power=0.0)
        self.assertEqual(weights, {0: 1.0, 1: 1.0, 2: 1.0})

    # Verify power half tempers between unweighted and full.
    def test_power_half_tempers_between_unweighted_and_full(self) -> None:
        labels = pd.Series([0, 1, 1, 2, 2, 2])
        weights = baseline.compute_outcome_class_weights(labels, power=0.5)

        self.assertAlmostEqual(weights[0], 2.0 ** 0.5)
        self.assertAlmostEqual(weights[1], 1.0)
        self.assertAlmostEqual(weights[2], (2 / 3) ** 0.5)

    # Verify missing class fails loudly.
    def test_missing_class_fails_loudly(self) -> None:
        labels = pd.Series([0, 0, 2, 2])

        with self.assertRaises(ValueError):
            baseline.compute_outcome_class_weights(labels)

if __name__ == "__main__":
    unittest.main()