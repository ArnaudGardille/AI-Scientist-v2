"""Structured model interface, including a tool-free Claude Code subscription backend."""

from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol


class StructuredModel(Protocol):
    def complete(
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        request_id: str | None = None,
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
        self.telemetry = self._load_telemetry()
        self.call_count = sum(
            int(record.get("attempts", 0)) for record in self.telemetry
        )
        self.budget_state_path = (
            self.telemetry_path.with_name("model_budget.json")
            if self.telemetry_path is not None
            else None
        )
        self.budget_state: dict[str, Any] = {}
        self.campaign_started_at: float | None = None
        self.last_request_id: str | None = None

    def begin_locked_run(self) -> None:
        """Refresh persisted counters after the campaign has acquired its writer lock."""
        self.telemetry = self._load_telemetry()
        self.call_count = sum(
            int(record.get("attempts", 0)) for record in self.telemetry
        )
        self.campaign_started_at = self._load_or_create_budget_state()

    def _ensure_budget_state(self) -> None:
        if self.campaign_started_at is None:
            self.begin_locked_run()

    def _load_telemetry(self) -> list[dict[str, Any]]:
        if self.telemetry_path is None or not self.telemetry_path.exists():
            return []
        payload = json.loads(self.telemetry_path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or any(
            not isinstance(record, dict) for record in payload
        ):
            raise ValueError("existing Claude telemetry must be a JSON list of records")
        return payload

    @staticmethod
    def _atomic_write_json(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    def _load_or_create_budget_state(self) -> float:
        if self.telemetry_path is None:
            return time.time()
        if self.budget_state_path is None:
            raise AssertionError("budget state path must exist with telemetry")
        path = self.budget_state_path
        expected_policy = {
            "schema_version": 1,
            "model": self.model,
            "max_calls": self.max_calls,
            "max_wallclock_seconds": self.max_wallclock_seconds,
        }
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            observed_policy = {
                key: payload.get(key) for key in expected_policy
            }
            if observed_policy != expected_policy:
                raise ValueError("existing Claude budget policy does not match this run")
            self.budget_state = payload
            return float(payload["campaign_started_at_epoch"])

        # Legacy telemetry did not carry a campaign start. Preserve at least the model time
        # already consumed; all newly created campaigns use an immutable wall-clock origin.
        legacy_elapsed = sum(
            float(record.get("elapsed_seconds", 0.0)) for record in self.telemetry
        )
        started_at = time.time() - legacy_elapsed
        self.budget_state = {
            **expected_policy,
            "campaign_started_at_epoch": started_at,
        }
        self._atomic_write_json(path, self.budget_state)
        return started_at

    @staticmethod
    def _digest(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _write_telemetry(self) -> None:
        if self.telemetry_path is None:
            return
        self._atomic_write_json(self.telemetry_path, self.telemetry)

    def _remaining_wallclock(self) -> float:
        self._ensure_budget_state()
        if self.campaign_started_at is None:
            raise AssertionError("campaign wall-clock origin was not initialized")
        return self.max_wallclock_seconds - (time.time() - self.campaign_started_at)

    def _reserve_call(self) -> float:
        remaining = self._remaining_wallclock()
        if self.call_count >= self.max_calls:
            raise RuntimeError(
                f"Claude Code model-call budget exhausted ({self.max_calls} calls)"
            )
        if remaining <= 0:
            raise RuntimeError(
                "Claude Code wall-clock budget exhausted "
                f"({self.max_wallclock_seconds}s)"
            )
        self.call_count += 1
        return remaining

    def _sleep_for_retry(self, delay: float) -> None:
        remaining = self._remaining_wallclock()
        if remaining <= 0 or delay >= remaining:
            raise RuntimeError(
                "Claude Code wall-clock budget exhausted "
                f"({self.max_wallclock_seconds}s)"
            )
        time.sleep(delay)

    def _record_resolved_models(self, envelope: dict[str, Any]) -> None:
        resolved = sorted((envelope.get("modelUsage") or {}).keys())
        if not resolved or self.budget_state_path is None:
            return
        expected = self.budget_state.get("resolved_models")
        if expected is not None and expected != resolved:
            raise RuntimeError(
                "Claude model alias resolved to a different model set during resume"
            )
        if expected is None:
            self.budget_state["resolved_models"] = resolved
            self._atomic_write_json(self.budget_state_path, self.budget_state)

    def _response_path(self, request_id: str) -> Path:
        if self.telemetry_path is None:
            raise ValueError("durable request IDs require a telemetry path")
        return (
            self.telemetry_path.parent
            / "model_responses"
            / f"{self._digest(request_id)}.json"
        )

    def accept_response(self, request_id: str) -> None:
        path = self._response_path(request_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_id") != request_id:
            raise ValueError(f"durable model response mismatch for request {request_id}")
        payload["semantic_status"] = "accepted"
        self._atomic_write_json(path, payload)

    def _archive_rejected_response(self, path: Path) -> None:
        rejected = (
            path.parent
            / "rejected"
            / f"{path.stem}_call_{self.call_count:04d}_{time.time_ns()}.json"
        )
        rejected.parent.mkdir(parents=True, exist_ok=True)
        path.replace(rejected)
        for directory in (path.parent, rejected.parent):
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)

    def reject_response(self, request_id: str, reason: str) -> None:
        path = self._response_path(request_id)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("request_id") != request_id:
            raise ValueError(f"durable model response mismatch for request {request_id}")
        payload["semantic_status"] = "rejected"
        payload["rejection"] = {
            "reason_class": reason[:200],
        }
        self._atomic_write_json(path, payload)
        self._archive_rejected_response(path)

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
        self,
        *,
        role: str,
        system: str,
        prompt: str,
        schema: dict[str, Any],
        request_id: str | None = None,
    ) -> dict[str, Any]:
        request_started = time.monotonic()
        self._ensure_budget_state()
        schema_payload = json.dumps(schema, sort_keys=True)
        hashes = {
            "prompt_sha256": self._digest(prompt),
            "system_sha256": self._digest(system),
            "schema_sha256": self._digest(schema_payload),
        }
        response_path = None
        if request_id is not None:
            response_path = self._response_path(request_id)
            if response_path.exists():
                persisted = json.loads(response_path.read_text(encoding="utf-8"))
                if persisted.get("semantic_status") == "rejected":
                    self._archive_rejected_response(response_path)
                    persisted = None
                if persisted is None:
                    pass
                elif (
                    persisted.get("request_id") != request_id
                    or persisted.get("role") != role
                    or persisted.get("hashes") != hashes
                    or persisted.get("semantic_status")
                    not in {"pending", "accepted"}
                ):
                    raise ValueError(
                        f"durable model response mismatch for request {request_id}"
                    )
                else:
                    payload = persisted.get("payload")
                    if not isinstance(payload, dict):
                        raise ValueError("durable model response payload must be an object")
                    self.last_request_id = request_id
                    self.telemetry.append(
                        {
                            "role": role,
                            "model": self.model,
                            "request_id": request_id,
                            **hashes,
                            "status": "replayed",
                            "attempts": 0,
                            "elapsed_seconds": time.monotonic() - request_started,
                            "cumulative_calls": self.call_count,
                        }
                    )
                    self._write_telemetry()
                    return payload

        record: dict[str, Any] = {
            "role": role,
            "model": self.model,
            "request_id": request_id,
            **hashes,
            "status": "pending",
            "attempts": 0,
        }
        self.telemetry.append(record)
        self._write_telemetry()
        last_error: Exception | None = None
        try:
            for attempt in range(self.max_retries + 1):
                remaining = self._reserve_call()
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
                        timeout=min(self.timeout_seconds, max(remaining, 0.001)),
                        check=False,
                    )
                    if self._remaining_wallclock() <= 0:
                        raise RuntimeError(
                            "Claude Code wall-clock budget exhausted "
                            f"({self.max_wallclock_seconds}s)"
                        )
                    if result.returncode != 0:
                        detail, status = self._error_detail(result)
                        error = RuntimeError(
                            f"Claude Code failed with exit code {result.returncode}: {detail}"
                        )
                        if attempt < self.max_retries and self._is_transient(status, detail):
                            last_error = error
                            self._sleep_for_retry(
                                self.retry_base_seconds * (2**attempt)
                            )
                            continue
                        raise error

                    envelope = json.loads(result.stdout)
                    self._record_resolved_models(envelope)
                    payload = envelope.get("structured_output")
                    if payload is None:
                        payload = json.loads(envelope["result"])
                    if not isinstance(payload, dict):
                        raise ValueError("structured model output must be a JSON object")
                    if response_path is not None:
                        self._atomic_write_json(
                            response_path,
                            {
                                "schema_version": 1,
                                "request_id": request_id,
                                "role": role,
                                "hashes": hashes,
                                "semantic_status": "pending",
                                "payload": payload,
                            },
                        )
                    self.last_request_id = request_id
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
                    self._sleep_for_retry(self.retry_base_seconds * (2**attempt))

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
