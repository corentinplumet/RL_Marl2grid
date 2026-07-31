import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

import wandb

from common.logger import Logger


def _logger_args():
    return Namespace(
        wandb_mode="online",
        wandb_project="Grid2Op",
        wandb_entity="test",
        resume_run_name="checkpoint",
        resume_wandb_run_id="original-id",
    )


class LoggerTests(unittest.TestCase):
    def test_online_timeout_falls_back_to_offline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            offline_dir = Path(tmp_dir) / "offline-run"
            files_dir = offline_dir / "files"
            files_dir.mkdir(parents=True)
            offline_run = Mock(dir=str(files_dir))

            with (
                patch.dict(
                    os.environ,
                    {
                        "WANDB_INIT_TIMEOUT": "7",
                        "WANDB_OFFLINE_FALLBACK": "true",
                    },
                    clear=False,
                ),
                patch(
                    "common.logger.wb.init",
                    side_effect=[
                        wandb.errors.CommError("network unavailable"),
                        offline_run,
                    ],
                ) as init,
                patch("common.logger.wb.teardown") as teardown,
            ):
                logger = Logger("fallback-test", _logger_args())

            self.assertEqual(logger.wb_mode, "offline")
            self.assertEqual(init.call_count, 2)
            self.assertEqual(init.call_args_list[0].kwargs["mode"], "online")
            self.assertEqual(init.call_args_list[0].kwargs["resume"], True)
            self.assertEqual(
                init.call_args_list[0].kwargs["settings"].init_timeout,
                7.0,
            )
            self.assertEqual(init.call_args_list[1].kwargs["mode"], "offline")
            self.assertIsNone(init.call_args_list[1].kwargs["resume"])
            teardown.assert_called_once_with()

    def test_failed_offline_sync_keeps_local_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            offline_dir = Path(tmp_dir) / "offline-run"
            offline_dir.mkdir()
            logger = Logger.__new__(Logger)
            logger.wb_mode = "offline"
            logger.wb_path = str(offline_dir)
            logger.run_name = "fallback-test"
            logger.wb_project = "Grid2Op"
            logger.wb_entity = "test"
            logger.is_resume = True
            logger.wb_target_id = "original-id"

            with (
                patch("common.logger.wb.finish"),
                patch(
                    "common.logger.subprocess.run",
                    return_value=Mock(returncode=1, stdout=""),
                ),
            ):
                logger.close()

            self.assertTrue(offline_dir.exists())

    def test_successful_offline_sync_also_keeps_local_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            offline_dir = Path(tmp_dir) / "offline-run"
            offline_dir.mkdir()
            logger = Logger.__new__(Logger)
            logger.wb_mode = "offline"
            logger.wb_path = str(offline_dir)
            logger.run_name = "fallback-test"
            logger.wb_project = "Grid2Op"
            logger.wb_entity = "test"
            logger.is_resume = True
            logger.wb_target_id = "original-id"

            with (
                patch("common.logger.wb.finish"),
                patch(
                    "common.logger.subprocess.run",
                    return_value=Mock(returncode=0, stdout="Syncing ... done.\n"),
                ) as sync,
            ):
                logger.close()

            self.assertTrue(offline_dir.exists())
            command = sync.call_args.args[0]
            self.assertIn("--no-mark-synced", command)
            self.assertEqual(command[command.index("--id") + 1], "original-id")

    def test_semantic_sync_error_keeps_local_run(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            offline_dir = Path(tmp_dir) / "offline-run"
            offline_dir.mkdir()
            logger = Logger.__new__(Logger)
            logger.wb_mode = "offline"
            logger.wb_path = str(offline_dir)
            logger.run_name = "fallback-test"
            logger.wb_project = "Grid2Op"
            logger.wb_entity = "test"
            logger.is_resume = True
            logger.wb_target_id = "deleted-id"

            with (
                patch("common.logger.wb.finish"),
                patch(
                    "common.logger.subprocess.run",
                    return_value=Mock(
                        returncode=0,
                        stdout=(
                            "wandb: ERROR run was previously created and deleted\n"
                        ),
                    ),
                ),
            ):
                logger.close()

            self.assertTrue(offline_dir.exists())

    def test_online_resume_without_internal_id_uses_offline_mode(self):
        args = _logger_args()
        args.resume_wandb_run_id = ""
        offline_run = Mock(dir="/tmp/offline-run/files")

        with patch("common.logger.wb.init", return_value=offline_run) as init:
            logger = Logger("visible-name", args)

        self.assertEqual(logger.wb_mode, "offline")
        self.assertEqual(init.call_args.kwargs["mode"], "offline")


if __name__ == "__main__":
    unittest.main()
