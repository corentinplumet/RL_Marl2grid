from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from merge_downloaded_continuations import (  # noqa: E402
    MERGE_PROFILES,
    _identity,
    _merge_one,
    _pair_continuations,
)


def _write_run(
    root: Path,
    *,
    name: str,
    run_id: str,
    steps: list[int],
    values: list[float],
    training_steps: list[int] | None = None,
    config: dict | None = None,
) -> dict:
    run_dir = root / f"{name}__{run_id}"
    run_dir.mkdir()
    (run_dir / "metadata.json").write_text(
        json.dumps({"name": name, "id": run_id, "group": "gs_s3dw"}),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config or {}),
        encoding="utf-8",
    )
    history = {"_step": steps, "survival": values}
    if training_steps is not None:
        history["charts/global_step"] = training_steps
    pd.DataFrame(history).to_csv(
        run_dir / "history.csv.gz",
        index=False,
    )
    return _identity(run_dir)


class ContinuationMergeTests(unittest.TestCase):
    def test_s3dw_profile_recognizes_original_cloud_names(self):
        run_re = MERGE_PROFILES["nl_s3dw"]["base_run_re"]
        self.assertIsNotNone(
            run_re.fullmatch("gs_s3dw_bus_n0_none_e0n0v0_mp3_h32_s0")
        )
        self.assertIsNone(
            run_re.fullmatch("nl_s3dw_bus_n0_none_e0n0v0_mp3_h32_s0")
        )
        self.assertIsNone(
            run_re.fullmatch("gs_s3dw_bus_n0_none_e0n0v0_mp3_h32_s1")
        )

    def test_continuation_replaces_overlap_and_keeps_canonical_identity(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base_name = "gs_s3dw_bus_n0_none_e0n0v0_mp3_h32_s0"
            base = _write_run(
                root,
                name=base_name,
                run_id="base-id",
                steps=[0, 1, 2, 3],
                training_steps=[0, 10, 20, 30],
                values=[0.0, 1.0, 2.0, 3.0],
            )
            continuation = _write_run(
                root,
                name=f"{base_name} [continuation 123]",
                run_id="continuation-id",
                # W&B's internal counter restarted in the resumed run.
                steps=[0, 1, 2],
                training_steps=[20, 30, 40],
                values=[20.0, 30.0, 40.0],
                config={
                    "is_continuation": True,
                    "continuation_of_run_id": "base-id",
                    "continuation_of_run_name": base_name,
                },
            )

            pairs = _pair_continuations([base], [continuation])
            merged, boundaries = _merge_one(base, pairs["base-id"])

            self.assertEqual(merged["_step"].tolist(), [0, 10, 20, 30, 40])
            self.assertEqual(merged["survival"].tolist(), [0.0, 1.0, 20.0, 30.0, 40.0])
            self.assertEqual(merged["run_name"].unique().tolist(), [base_name])
            self.assertEqual(merged["run_id"].unique().tolist(), ["base-id"])
            self.assertEqual(
                merged["wandb_internal_step"].tolist(), [0, 1, 0, 1, 2]
            )
            self.assertEqual(boundaries[0]["start_step"], 20)
            self.assertEqual(boundaries[0]["previous_retained_step"], 10)

    def test_declared_end_recovers_reset_counter_without_global_step_column(self):
        with TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base_name = "gs_s3dw_bus_n0_none_e0n0v0_mp1_h16_s0"
            base = _write_run(
                root,
                name=base_name,
                run_id="base-id",
                steps=[0, 10, 20, 30],
                values=[0.0, 1.0, 2.0, 3.0],
            )
            continuation = _write_run(
                root,
                name=f"{base_name} [continuation 456]",
                run_id="continuation-id",
                steps=[10, 20, 30],
                values=[20.0, 30.0, 40.0],
                config={
                    "is_continuation": True,
                    "continuation_of_run_id": "base-id",
                    "continuation_of_run_name": base_name,
                    "continuation_end_step": 50,
                },
            )

            pairs = _pair_continuations([base], [continuation])
            merged, boundaries = _merge_one(base, pairs["base-id"])

            self.assertEqual(merged["_step"].tolist(), [0, 10, 20, 30, 40, 50])
            self.assertEqual(boundaries[0]["start_step"], 30)
            self.assertEqual(boundaries[0]["end_step"], 50)


if __name__ == "__main__":
    unittest.main()
