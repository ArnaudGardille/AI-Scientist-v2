from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.constrained.campaign import CampaignRunLock


class CampaignRunLockTest(unittest.TestCase):
    def test_rejects_a_second_writer_and_releases_on_exit(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            with CampaignRunLock(output_dir):
                with self.assertRaisesRegex(RuntimeError, "already locked"):
                    with CampaignRunLock(output_dir):
                        pass

            with CampaignRunLock(output_dir):
                self.assertIn("pid=", (output_dir / ".campaign.lock").read_text())


if __name__ == "__main__":
    unittest.main()
