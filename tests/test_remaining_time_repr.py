from __future__ import annotations
import importlib
import unittest
import numpy as np
import pandas as pd

encoding = importlib.import_module("E_prefix_encoding.04_5_encoding")

# HELPER: Build labels.
def _labels() -> tuple[pd.Series, pd.Series]:
    # Positive seconds plus masked-out and non-positive entries that must be ignored.
    labels = pd.Series([100.0, 400.0, 900.0, 1600.0, 0.0, -5.0, 2500.0])
    mask = pd.Series([1, 1, 1, 1, 1, 0, 0])
    return labels, mask

class RemainingTimeReprTests(unittest.TestCase):
    # Verify all six combinations round trip to seconds.
    def test_all_six_combinations_round_trip_to_seconds(self) -> None:
        labels, mask = _labels()
        sample = pd.Series([100.0, 900.0, 1600.0])
        for transform in encoding.REMAINING_TIME_TRANSFORMS:
            for scaling in encoding.REMAINING_TIME_SCALINGS:
                repr_ = encoding.fit_remaining_time_target_repr(labels, transform, scaling, mask)
                model_units = encoding.transform_remaining_time_target(sample, repr_)
                seconds = encoding.inverse_remaining_time_target(model_units, repr_)
                self.assertTrue(np.allclose(seconds.to_numpy(), sample.to_numpy(), atol=1e-3), f"{transform}+{scaling}")

    # Verify raw median reproduces legacy scale.
    def test_raw_median_reproduces_legacy_scale(self) -> None:
        labels, mask = _labels()
        repr_ = encoding.fit_remaining_time_target_repr(labels, "raw", "median", mask)
        self.assertEqual(repr_.center, 0.0)
        self.assertAlmostEqual(repr_.scale, 400.0, places=4)
        self.assertTrue(repr_.use_softplus)
        # Median scaling anchors the scaled median at 1.0, the current bias-init target.
        self.assertAlmostEqual(repr_.median_model_units, 1.0, places=4)

    # Verify zscore centers and uses linear head.
    def test_zscore_centers_and_uses_linear_head(self) -> None:
        labels, mask = _labels()
        repr_ = encoding.fit_remaining_time_target_repr(labels, "raw", "zscore", mask)
        self.assertFalse(repr_.use_softplus)
        positive = np.array([100.0, 400.0, 900.0, 1600.0], dtype=float)
        self.assertAlmostEqual(repr_.center, float(positive.mean()), places=3)
        self.assertAlmostEqual(repr_.scale, float(positive.std()), places=3)
        # A centered representation produces negative model units for small targets.
        model_units = encoding.transform_remaining_time_target(pd.Series([100.0]), repr_)
        self.assertLess(float(model_units.iloc[0]), 0.0)

    # Verify raw scaling is identity.
    def test_raw_scaling_is_identity(self) -> None:
        labels, mask = _labels()
        repr_ = encoding.fit_remaining_time_target_repr(labels, "raw", "raw", mask)
        self.assertEqual((repr_.center, repr_.scale), (0.0, 1.0))
        self.assertTrue(repr_.use_softplus)

    # Verify log transform uses log space.
    def test_log_transform_uses_log_space(self) -> None:
        labels, mask = _labels()
        repr_ = encoding.fit_remaining_time_target_repr(labels, "log", "median", mask)
        model_units = encoding.transform_remaining_time_target(pd.Series([900.0]), repr_)
        expected = np.log1p(900.0) / repr_.scale
        self.assertAlmostEqual(float(model_units.iloc[0]), float(expected), places=4)

    # Verify unknown modes fail loudly.
    def test_unknown_modes_fail_loudly(self) -> None:
        labels, mask = _labels()
        with self.assertRaises(ValueError):
            encoding.fit_remaining_time_target_repr(labels, "sqrt", "median", mask)
        with self.assertRaises(ValueError):
            encoding.fit_remaining_time_target_repr(labels, "raw", "minmax", mask)

if __name__ == "__main__":
    unittest.main()