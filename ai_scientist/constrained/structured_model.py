"""Structured model interface, including a tool-free Claude Code subscription backend."""

from __future__ import annotations

import json
import hashlib
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol


class StructuredModel(Protocol):
    def complete(
        self, *, role: str, system: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...


class ClaudeCodeStructuredModel:
    def __init__(
        self,
        model: str = "sonnet",
        timeout_seconds: int = 900,
        *,
        max_calls: int = 128,
        max_wallclock_seconds: int = 21_600,
        max_retries: int = 2,
        retry_base_seconds: float = 1.0,
        telemetry_path: Path | None = None,
    ) -> None:
        if shutil.which("claude") is None:
            raise ValueError("Claude Code CLI is not installed or not on PATH")
        if timeout_seconds <= 0 or max_calls < 1 or max_wallclock_seconds <= 0:
            raise ValueError("Claude Code budgets must be positive")
        if max_retries < 0 or retry_base_seconds < 0:
            raise ValueError("Claude Code retry settings must be non-negative")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_calls = max_calls
        self.max_wallclock_seconds = max_wallclock_seconds
        self.max_retries = max_retries
        self.retry_base_seconds = retry_base_seconds
        self.telemetry_path = telemetry_path
        self.call_count = 0
        self.started_at = time.monotonic()
        self.telemetry: list[dict[str, Any]] = []

    @staticmethod
    def _digest(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_telemetry(self) -> None:
        if self.telemetry_path is None:
            return
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.telemetry_path.with_suffix(self.telemetry_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.telemetry, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(self.telemetry_path)

    def _reserve_call(self) -> None:
        elapsed = time.monotonic() - self.started_at
        if self.call_count >= self.max_calls:
            raise RuntimeError(
                f"Claude Code model-call budget exhausted ({self.max_calls} calls)"
            )
        if elapsed >= self.max_wallclock_seconds:
            raise RuntimeError(
                "Claude Code wall-clock budget exhausted "
                f"({self.max_wallclock_seconds}s)"
            )
        self.call_count += 1

    @staticmethod
    def _error_detail(result: subprocess.CompletedProcess[str]) -> tuple[str, int | None]:
        detail = result.stderr.strip()
        status = None
        if result.stdout.strip():
            try:
                envelope = json.loads(result.stdout)
                status = envelope.get("api_error_status")
                detail = str(
                    envelope.get("result")
                    or envelope.get("error")
                    or envelope.get("subtype")
                    or result.stdout
                )
            except json.JSONDecodeError:
                detail = result.stdout.strip()
        return detail[-4000:] or "no diagnostic output", status

    @staticmethod
    def _is_transient(status: int | None, detail: str) -> bool:
        lowered = detail.lower()
        if any(
            marker in lowered
            for marker in (
                "session limit",
                "usage limit",
                "resets at",
                "resets ",
            )
        ):
            return False
        if status in {408, 409, 425, 429, 500, 502, 503, 504, 529}:
            return True
        return any(
            marker in lowered
            for marker in (
                "rate limit",
                "overloaded",
                "temporarily unavailable",
                "timeout",
                "connectionrefused",
                "connection refused",
                "unable to connect",
                "network error",
                "econnreset",
            )
        )

    def complete(
        self, *, role: str, system: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
        request_started = time.monotonic()
        schema_payload = json.dumps(schema, sort_keys=True)
        record: dict[str, Any] = {
            "role": role,
            "model": self.model,
            "prompt_sha256": self._digest(prompt),
            "system_sha256": self._digest(system),
            "schema_sha256": self._digest(schema_payload),
            "status": "pending",
            "attempts": 0,
        }
        self.telemetry.append(record)
        self._write_telemetry()
        last_error: Exception | None = None
        try:
            for attempt in range(self.max_retries + 1):
                self._reserve_call()
                record["attempts"] = attempt + 1
                self._write_telemetry()
                try:
                    result = subprocess.run(
                        [
                            "claude", "--print", prompt,
                            "--model", self.model,
                            "--system-prompt", f"Role: {role}. {system}",
                            "--tools", "",
                            "--permission-mode", "dontAsk",
                            "--safe-mode",
                            "--no-session-persistence",
                            "--output-format", "json",
                            "--json-schema", schema_payload,
                        ],
                        text=True,
                        capture_output=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                    if result.returncode != 0:
                        detail, status = self._error_detail(result)
                        error = RuntimeError(
                            f"Claude Code failed with exit code {result.returncode}: {detail}"
                        )
                        if attempt < self.max_retries and self._is_transient(status, detail):
                            last_error = error
                            time.sleep(self.retry_base_seconds * (2**attempt))
                            continue
                        raise error

                    envelope = json.loads(result.stdout)
                    payload = envelope.get("structured_output")
                    if payload is None:
                        payload = json.loads(envelope["result"])
                    if not isinstance(payload, dict):
                        raise ValueError("structured model output must be a JSON object")
                    record.update({
                        "status": "ok",
                        "duration_api_ms": envelope.get("duration_api_ms"),
                        "duration_ms": envelope.get("duration_ms"),
                        "total_cost_usd": envelope.get("total_cost_usd"),
                        "usage": envelope.get("usage"),
                        "model_usage": envelope.get("modelUsage"),
                    })
                    return payload
                except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
                    last_error = exc
                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"Claude Code did not return valid structured output: {exc}"
                        ) from exc
                    time.sleep(self.retry_base_seconds * (2**attempt))

            raise RuntimeError(f"Claude Code request failed: {last_error}")
        except Exception as exc:
            record["status"] = "error"
            record["error_class"] = type(exc).__name__
            record["error"] = str(exc)[-1000:]
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - request_started
            record["cumulative_calls"] = self.call_count
            self._write_telemetry()
