from __future__ import annotations

import json
import subprocess
import tempfile
import time
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

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_resume_preserves_telemetry_and_cumulative_call_budget(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps({"structured_output": {"ok": True}}),
            stderr="",
        )
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "telemetry.json"
            telemetry.write_text(
                json.dumps(
                    [
                        {
                            "role": "prior",
                            "status": "ok",
                            "attempts": 2,
                            "elapsed_seconds": 12.0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            model = ClaudeCodeStructuredModel(
                max_calls=3,
                max_wallclock_seconds=100,
                telemetry_path=telemetry,
            )
            request = {
                "role": "resumed",
                "system": "test",
                "prompt": "test",
                "schema": {"type": "object"},
            }

            self.assertEqual(model.complete(**request), {"ok": True})
            with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                model.complete(**request)
            records = json.loads(telemetry.read_text(encoding="utf-8"))

        self.assertEqual(run.call_count, 1)
        self.assertEqual(records[0]["role"], "prior")
        self.assertEqual(records[1]["role"], "resumed")
        self.assertEqual(records[1]["cumulative_calls"], 3)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_durable_request_replays_response_without_a_second_model_call(self, run, _which):
        run.return_value = subprocess.CompletedProcess(
            args=["claude"],
            returncode=0,
            stdout=json.dumps({"structured_output": {"claim": "condition then marginalize"}}),
            stderr="",
        )
        request = {
            "role": "theorist",
            "system": "derive",
            "prompt": "review hypothesis",
            "schema": {"type": "object"},
            "request_id": "concept:abc:theorist",
        }
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "model_telemetry.json"
            first = ClaudeCodeStructuredModel(
                telemetry_path=telemetry,
                max_wallclock_seconds=100,
            )
            expected = first.complete(**request)

            resumed = ClaudeCodeStructuredModel(
                telemetry_path=telemetry,
                max_wallclock_seconds=100,
            )
            replayed = resumed.complete(**request)
            records = json.loads(telemetry.read_text(encoding="utf-8"))
            responses = list((Path(tmp) / "model_responses").glob("*.json"))

        self.assertEqual(replayed, expected)
        self.assertEqual(run.call_count, 1)
        self.assertEqual(len(responses), 1)
        self.assertEqual(records[-1]["status"], "replayed")
        self.assertEqual(records[-1]["request_id"], "concept:abc:theorist")
        self.assertEqual(resumed.call_count, 1)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_resume_enforces_original_wallclock_origin(self, run, _which):
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "model_telemetry.json"
            telemetry.write_text("[]", encoding="utf-8")
            telemetry.with_name("model_budget.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": "sonnet",
                        "max_calls": 128,
                        "max_wallclock_seconds": 100,
                        "campaign_started_at_epoch": time.time() - 101,
                    }
                ),
                encoding="utf-8",
            )
            model = ClaudeCodeStructuredModel(
                telemetry_path=telemetry,
                max_wallclock_seconds=100,
            )

            with self.assertRaisesRegex(RuntimeError, "wall-clock budget exhausted"):
                model.complete(
                    role="test",
                    system="test",
                    prompt="test",
                    schema={"type": "object"},
                )

        self.assertEqual(run.call_count, 0)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_semantically_rejected_response_does_not_poison_request_id(self, run, _which):
        run.side_effect = [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps({"structured_output": {"value": "invalid"}}),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps({"structured_output": {"value": "valid"}}),
                stderr="",
            ),
        ]
        request = {
            "role": "test",
            "system": "test",
            "prompt": "test",
            "schema": {"type": "object"},
            "request_id": "concept:semantic-validation",
        }
        with tempfile.TemporaryDirectory() as tmp:
            telemetry = Path(tmp) / "model_telemetry.json"
            model = ClaudeCodeStructuredModel(telemetry_path=telemetry)

            invalid = model.complete(**request)
            self.assertEqual(invalid, {"value": "invalid"})
            model.reject_response(request["request_id"], "ValueError")
            valid = model.complete(**request)
            model.accept_response(request["request_id"])

            rejected = list((Path(tmp) / "model_responses" / "rejected").glob("*.json"))
            active = list((Path(tmp) / "model_responses").glob("*.json"))
            active_payload = json.loads(active[0].read_text())

        self.assertEqual(valid, {"value": "valid"})
        self.assertEqual(run.call_count, 2)
        self.assertEqual(len(rejected), 1)
        self.assertEqual(len(active), 1)
        self.assertEqual(
            active_payload["semantic_status"],
            "accepted",
        )

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_discards_response_that_finishes_after_campaign_deadline(self, run, _which):
        def slow_response(*args, **kwargs):
            del args, kwargs
            time.sleep(0.03)
            return subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps({"structured_output": {"ok": True}}),
                stderr="",
            )

        run.side_effect = slow_response
        with tempfile.TemporaryDirectory() as tmp:
            model = ClaudeCodeStructuredModel(
                telemetry_path=Path(tmp) / "model_telemetry.json",
                max_wallclock_seconds=0.01,
                timeout_seconds=1,
                max_retries=0,
            )

            with self.assertRaisesRegex(RuntimeError, "wall-clock budget exhausted"):
                model.complete(
                    role="test",
                    system="test",
                    prompt="test",
                    schema={"type": "object"},
                )

        self.assertEqual(run.call_count, 1)

    @patch("ai_scientist.constrained.structured_model.shutil.which", return_value="/bin/claude")
    @patch("ai_scientist.constrained.structured_model.subprocess.run")
    def test_pins_resolved_model_set_after_first_success(self, run, _which):
        run.side_effect = [
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "structured_output": {"ok": True},
                        "modelUsage": {"claude-sonnet-4-6": {}},
                    }
                ),
                stderr="",
            ),
            subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout=json.dumps(
                    {
                        "structured_output": {"ok": True},
                        "modelUsage": {"claude-sonnet-4-7": {}},
                    }
                ),
                stderr="",
            ),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            model = ClaudeCodeStructuredModel(
                telemetry_path=Path(tmp) / "model_telemetry.json",
                max_retries=0,
            )
            request = {
                "role": "test",
                "system": "test",
                "prompt": "test",
                "schema": {"type": "object"},
            }
            self.assertEqual(model.complete(**request), {"ok": True})
            with self.assertRaisesRegex(RuntimeError, "different model set"):
                model.complete(**request)

        self.assertEqual(run.call_count, 2)


if __name__ == "__main__":
    unittest.main()
