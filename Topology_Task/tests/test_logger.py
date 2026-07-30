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

            with (
                patch("common.logger.wb.finish"),
                patch(
                    "common.logger.subprocess.run",
                    return_value=Mock(returncode=1),
                ),
            ):
                logger.close()

            self.assertTrue(offline_dir.exists())


if __name__ == "__main__":
    unittest.main()
