from __future__ import annotations

import unittest

import numpy as np

from tlpp_embedding.core import (
    interpolate_signal,
    make_TLPP,
    make_occupancy,
    make_occupancy_counts,
    make_tlpp_counts,
)


class CoreRegressionTests(unittest.TestCase):
    def test_linear_interpolation_matches_numpy(self):
        rng = np.random.default_rng(14)
        signal = rng.normal(size=700)
        got, fs = interpolate_signal(signal, 1_000_000.0, 16, "linear")
        old_index = np.arange(len(signal), dtype=np.float64)
        new_index = np.linspace(0.0, len(signal) - 1, (len(signal) - 1) * 16 + 1)
        expected = np.interp(new_index, old_index, signal)
        self.assertTrue(np.array_equal(got, expected))
        self.assertEqual(fs, 16_000_000.0)

    def test_integer_histogram_matches_numpy_including_edges(self):
        rng = np.random.default_rng(3)
        lo, hi, bins = -5.0, 105.0, 128
        points = rng.uniform(lo - 20.0, hi + 20.0, size=(20000, 2))
        edges = np.linspace(lo, hi, bins + 1)
        points = np.vstack((points, np.column_stack((edges, edges[::-1])), [[lo, lo], [hi, hi]]))
        expected, _, _ = np.histogram2d(
            np.clip(points[:, 0], lo, hi),
            np.clip(points[:, 1], lo, hi),
            bins=(edges, edges),
        )
        got = make_occupancy_counts(points, bins, (lo, hi), dtype=np.uint16)
        self.assertTrue(np.array_equal(got, expected.astype(np.uint16)))

    def test_counts_reproduce_probability(self):
        rng = np.random.default_rng(7)
        signal = rng.normal(50.0, 20.0, 700)
        points = make_TLPP(
            signal,
            1_000_000.0,
            lag_us=10.0,
            interpolation_factor=16,
            interpolation_method="linear",
        )
        probability = make_occupancy(points, 128, (-5.0, 105.0), 0.0)
        counts = make_tlpp_counts(
            signal,
            1_000_000.0,
            lag_us=10.0,
            bins=128,
            value_range=(-5.0, 105.0),
            interpolation_factor=16,
            interpolation_method="linear",
            dtype=np.uint16,
        )
        rebuilt = (counts.astype(np.float64) / counts.sum()).astype(np.float32)
        self.assertTrue(np.array_equal(probability, rebuilt))
        self.assertEqual(int(counts.sum()), len(points))


if __name__ == "__main__":
    unittest.main()
