from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd


HELPERS_DIR = (
    Path(__file__).resolve().parents[1] / "analysis" / "metrics" / "helpers"
)
sys.path.insert(0, str(HELPERS_DIR))

from survival_comparison import (  # noqa: E402
    extract_survival_curves,
    plot_survival_comparison,
    prepare_survival_comparison,
)


def synthetic_curves() -> pd.DataFrame:
    rows = []
    values = {
        ("a_s0", "a", 0): [10.0, 20.0, 30.0],
        ("a_s1", "a", 1): [20.0, 30.0, 40.0],
        ("b_s0", "b", 0): [15.0, 35.0, 55.0],
        ("b_s1", "b", 1): [25.0, 45.0, 65.0],
    }
    for (run_name, config, seed), survival in values.items():
        for step, value in zip([0, 10, 20], survival):
            rows.append(
                {
                    "run_name": run_name,
                    "step": step,
                    "survival_pct": value,
                    "config": config,
                    "seed": seed,
                    "stage": "screen",
                }
            )
    return pd.DataFrame(rows)


class SurvivalComparisonTests(unittest.TestCase):
    def test_extracts_long_history_and_merges_catalog(self):
        history = pd.DataFrame(
            {
                "run_name": ["run_a", "run_a", "run_a"],
                "step": [0, 10, 10],
                "metric": [
                    "test/charts/episodic_survival",
                    "test/charts/episodic_survival",
                    "train/loss",
                ],
                "value": [0.1, 0.2, 99.0],
            }
        )
        catalog = pd.DataFrame(
            {"run_name": ["run_a"], "graph_type": ["bus"], "seed": [0]}
        )
        curves = extract_survival_curves(history, catalog=catalog)
        self.assertEqual(curves["survival_pct"].tolist(), [10.0, 20.0])
        self.assertEqual(curves["graph_type"].unique().tolist(), ["bus"])

    def test_aggregates_seeds_with_std(self):
        summary, members = prepare_survival_comparison(
            synthetic_curves(),
            filters={"stage": "screen"},
            group_by="config",
            smooth=1,
            uncertainty="std",
            min_members="all",
        )
        config_a = summary[summary["config"].eq("a")]
        self.assertEqual(config_a["center"].tolist(), [15.0, 25.0, 35.0])
        self.assertTrue((config_a["n_members"] == 2).all())
        self.assertEqual(members["run_name"].nunique(), 4)

    def test_trailing_smoothing_is_per_run_before_aggregation(self):
        summary, _ = prepare_survival_comparison(
            synthetic_curves(),
            filters={"config": "a"},
            group_by="config",
            smooth=2,
            uncertainty=None,
        )
        self.assertEqual(summary["center"].tolist(), [15.0, 20.0, 30.0])

    def test_difference_from_baseline(self):
        summary, _ = prepare_survival_comparison(
            synthetic_curves(),
            group_by="config",
            smooth=1,
            uncertainty="sem",
            comparison="difference",
            baseline={"config": "a"},
        )
        baseline = summary[summary["config"].eq("a")]
        alternative = summary[summary["config"].eq("b")]
        self.assertEqual(baseline["center"].tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(alternative["center"].tolist(), [5.0, 15.0, 25.0])

    def test_plot_returns_figure_and_summary(self):
        figure, summary = plot_survival_comparison(
            synthetic_curves(),
            group_by="config",
            show_members=True,
            uncertainty="ci95",
            smooth=1,
            return_data=True,
        )
        self.assertGreater(len(figure.data), 2)
        self.assertEqual(summary["curve_label"].nunique(), 2)


if __name__ == "__main__":
    unittest.main()
