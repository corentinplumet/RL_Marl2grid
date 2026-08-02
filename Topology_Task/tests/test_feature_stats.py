import unittest

import numpy as np

from common.feature_stats import (
    compare_samples,
    describe,
    ks_statistic,
    markdown_table,
    wasserstein1,
)


class DescribeTest(unittest.TestCase):
    def test_basic_summary(self):
        stats = describe(np.arange(101, dtype=np.float64))
        self.assertEqual(stats["n"], 101)
        self.assertAlmostEqual(stats["mean"], 50.0)
        self.assertAlmostEqual(stats["min"], 0.0)
        self.assertAlmostEqual(stats["max"], 100.0)
        self.assertAlmostEqual(stats["p50"], 50.0)
        self.assertAlmostEqual(stats["p01"], 1.0)
        self.assertAlmostEqual(stats["p99"], 99.0)

    def test_empty_is_nan_not_a_crash(self):
        stats = describe(np.empty(0))
        self.assertEqual(stats["n"], 0)
        self.assertTrue(np.isnan(stats["mean"]))
        self.assertTrue(np.isnan(stats["p50"]))


class KsTest(unittest.TestCase):
    def test_identical_samples_are_zero(self):
        values = np.linspace(0.0, 1.0, 500)
        self.assertAlmostEqual(ks_statistic(values, values), 0.0)

    def test_disjoint_supports_are_one(self):
        self.assertAlmostEqual(
            ks_statistic(np.linspace(0, 1, 200), np.linspace(10, 11, 200)), 1.0
        )

    def test_half_overlap(self):
        # b is a's support shifted by half its width, so the CDFs separate by 0.5.
        a = np.linspace(0.0, 1.0, 20001)
        b = a + 0.5
        self.assertAlmostEqual(ks_statistic(a, b), 0.5, places=2)

    def test_empty_is_nan(self):
        self.assertTrue(np.isnan(ks_statistic(np.empty(0), np.arange(5))))


class WassersteinTest(unittest.TestCase):
    def test_pure_shift_equals_shift(self):
        a = np.linspace(0.0, 1.0, 5000)
        self.assertAlmostEqual(wasserstein1(a, a + 0.5), 0.5, places=3)

    def test_identical_is_zero(self):
        a = np.linspace(0.0, 1.0, 5000)
        self.assertAlmostEqual(wasserstein1(a, a), 0.0, places=6)

    def test_scale_change_is_detected(self):
        a = np.linspace(-1.0, 1.0, 5000)
        self.assertGreater(wasserstein1(a, 3.0 * a), 0.5)


class CompareSamplesTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = {
            "node/node/p": rng.normal(0.0, 1.0, 4000),
            "node/node/rho": rng.normal(5.0, 2.0, 4000),
        }
        self.b = {
            "node/node/p": rng.normal(0.0, 1.0, 4000),
            "node/node/rho": rng.normal(15.0, 2.0, 4000),
        }

    def test_matching_feature_shows_no_shift(self):
        rows = {r["feature"]: r for r in compare_samples(self.a, self.b, "x", "y")}
        row = rows["node/node/p"]
        self.assertLess(abs(row["std_mean_shift"]), 0.15)
        self.assertLess(row["ks"], 0.1)
        self.assertTrue(row["present_in_both"])

    def test_shifted_feature_is_flagged(self):
        rows = {r["feature"]: r for r in compare_samples(self.a, self.b, "x", "y")}
        row = rows["node/node/rho"]
        # +10 on a spread of 2 is a five-sigma move, so KS should saturate.
        self.assertGreater(row["std_mean_shift"], 4.0)
        self.assertGreater(row["ks"], 0.95)
        self.assertAlmostEqual(row["std_ratio"], 1.0, places=1)

    def test_feature_missing_on_one_side(self):
        rows = {
            r["feature"]: r
            for r in compare_samples(self.a, {"node/node/p": self.b["node/node/p"]},
                                     "x", "y")
        }
        self.assertFalse(rows["node/node/rho"]["present_in_both"])
        self.assertEqual(rows["node/node/rho"]["n_y"], 0)

    def test_constant_feature_does_not_divide_by_zero(self):
        constant = {"node/node/flag": np.ones(100)}
        rows = compare_samples(constant, constant, "x", "y")
        self.assertTrue(np.isnan(rows[0]["std_mean_shift"]))
        self.assertAlmostEqual(rows[0]["ks"], 0.0)


class MarkdownTableTest(unittest.TestCase):
    def test_worst_shift_first(self):
        rng = np.random.default_rng(1)
        a = {"small": rng.normal(0, 1, 500), "big": rng.normal(0, 1, 500)}
        b = {"small": rng.normal(0.1, 1, 500), "big": rng.normal(20, 1, 500)}
        table = markdown_table(compare_samples(a, b, "x", "y"), "x", "y")
        lines = table.strip().split("\n")
        self.assertEqual(len(lines), 4)  # header, separator, two features
        self.assertIn("big", lines[2])
        self.assertIn("small", lines[3])



class PlotTest(unittest.TestCase):
    def _samples(self, n_features):
        rng = np.random.default_rng(2)
        a = {f"node/node/f{i}": rng.normal(0, 1, 300) for i in range(n_features)}
        b = {f"node/node/f{i}": rng.normal(1, 2, 300) for i in range(n_features)}
        return a, b

    def test_renders_for_various_feature_counts(self):
        import os
        import tempfile

        from common.feature_stats import plot_distributions

        with tempfile.TemporaryDirectory() as tmp:
            for n_features in (1, 3, 4, 7):
                out = os.path.join(tmp, f"plot_{n_features}.png")
                a, b = self._samples(n_features)
                plot_distributions(a, b, "bus14", "wcci", out)
                self.assertTrue(os.path.getsize(out) > 1000)

    def test_no_shared_features_is_a_noop(self):
        import os
        import tempfile

        from common.feature_stats import plot_distributions

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "empty.png")
            plot_distributions({"a": np.arange(3.0)}, {"b": np.arange(3.0)},
                               "x", "y", out)
            self.assertFalse(os.path.exists(out))

    def test_constant_feature_does_not_break_binning(self):
        import os
        import tempfile

        from common.feature_stats import plot_distributions

        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "const.png")
            plot_distributions({"c": np.ones(50)}, {"c": np.ones(50)}, "x", "y", out)
            self.assertTrue(os.path.getsize(out) > 1000)


if __name__ == "__main__":
    unittest.main()
