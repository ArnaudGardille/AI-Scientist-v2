from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
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
        model = ClaudeCodeStructuredModel(retry_base_seconds=0)

        with self.assertRaisesRegex(RuntimeError, "revoked OAuth token"):
            model.complete(
                role="test",
                system="test",
                prompt="test",
                schema={"type": "object"},
            )
        self.assertEqual(run.call_count, 1)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_retries_transient_error_and_persists_redacted_telemetry(self, run, _which):
        failure = subprocess.CompletedProcess(
            args=["claude"],
            returncode=1,
            stdout=json.dumps({
                "is_error": True,
                "api_error_status": 529,
                "result": "Service overloaded.",
            }),
            stderr="",
        )
        success = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps({
                "structured_output": {"ok": True},
                "duration_ms": 12,
                "usage": {"input_tokens": 3, "output_tokens": 2},
            }),
            stderr="",
        )
        run.side_effect = [failure, success]

        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "telemetry.json"
            model = ClaudeCodeStructuredModel(
                max_retries=1,
                retry_base_seconds=0,
                telemetry_path=telemetry,
            )
            payload = model.complete(
                role="test",
                system="private system",
                prompt="private prompt",
                schema={"type": "object"},
            )
            records = json.loads(telemetry.read_text(encoding="utf-8"))

        self.assertEqual(payload, {"ok": True})
        self.assertEqual(run.call_count, 2)
        self.assertEqual(records[0]["attempts"], 2)
        self.assertEqual(records[0]["status"], "ok")
        self.assertNotIn("private prompt", json.dumps(records))
        self.assertNotIn("private system", json.dumps(records))

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_retries_connection_refused_transport_error(self, run, _which):
        run.side_effect = [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout="",
                stderr="API Error: Unable to connect to API (ConnectionRefused)",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps({"structured_output": {"ok": True}}),
                stderr="",
            ),
        ]
        model = ClaudeCodeStructuredModel(max_retries=1, retry_base_seconds=0)
        payload = model.complete(
            role="test",
            system="test",
            prompt="test",
            schema={"type": "object"},
        )
        self.assertEqual(payload, {"ok": True})
        self.assertEqual(run.call_count, 2)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_does_not_retry_subscription_session_limit(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=["claude"],
            returncode=1,
            stdout=json.dumps(
                {
                    "is_error": True,
                    "api_error_status": 429,
                    "result": (
                        "You've hit your session limit - resets 11:40pm "
                        "(Europe/Paris)"
                    ),
                }
            ),
            stderr="",
        )
        model = ClaudeCodeStructuredModel(max_retries=2, retry_base_seconds=0)

        with self.assertRaisesRegex(RuntimeError, "session limit"):
            model.complete(
                role="test",
                system="test",
                prompt="test",
                schema={"type": "object"},
            )

        self.assertEqual(run.call_count, 1)
        self.assertEqual(model.call_count, 1)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_enforces_global_model_call_budget(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps({"structured_output": {"ok": True}}),
            stderr="",
        )
        model = ClaudeCodeStructuredModel(max_calls=1)
        request = {
            "role": "test",
            "system": "test",
            "prompt": "test",
            "schema": {"type": "object"},
        }
        self.assertEqual(model.complete(**request), {"ok": True})
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            model.complete(**request)
        self.assertEqual(run.call_count, 1)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_writes_pending_event_before_invoking_claude(self, run, _which):
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "telemetry.json"

            def observe_pending(*args, **kwargs):
                del args, kwargs
                records = json.loads(telemetry.read_text(encoding="utf-8"))
                self.assertEqual(records[-1]["status"], "pending")
                self.assertEqual(records[-1]["attempts"], 1)
                return subprocess.CompletedProcess(
                    args=["claude"],
                    returncode=0,
                    stdout=json.dumps({"structured_output": {"ok": True}}),
                    stderr="",
                )

            run.side_effect = observe_pending
            model = ClaudeCodeStructuredModel(telemetry_path=telemetry)
            model.complete(
                role="test",
                system="test",
                prompt="test",
                schema={"type": "object"},
            )

            records = json.loads(telemetry.read_text(encoding="utf-8"))
            self.assertEqual(records[-1]["status"], "ok")
            self.assertEqual(records[-1]["attempts"], 1)


if __name__ == "__main__":
    unittest.main()
