from __future__ import annotations

import json
import subprocess
import unittest
from unittest.mock import patch

from ai_scientist.constrained.structured_model import ClaudeCodeStructuredModel


class ClaudeCodeStructuredModelTest(unittest.TestCase):
    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_surfaces_json_error_from_stdout(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=["claude"],
            returncode=1,
            stdout=json.dumps(
                {
                    "is_error": True,
                    "api_error_status": 401,
                    "result": "Failed to authenticate: revoked OAuth token.",
                }
            ),
            stderr="",
        )
        model = ClaudeCodeStructuredModel()

        with self.assertRaisesRegex(RuntimeError, "revoked OAuth token"):
            model.complete(
                role="test",
                system="test",
                prompt="test",
                schema={"type": "object"},
            )


if __name__ == "__main__":
    unittest.main()
