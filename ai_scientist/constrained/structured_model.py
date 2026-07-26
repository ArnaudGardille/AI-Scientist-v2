"""Structured model interface, including a tool-free Claude Code subscription backend."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any, Protocol


class StructuredModel(Protocol):
    def complete(
        self, *, role: str, system: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]: ...


class ClaudeCodeStructuredModel:
    def __init__(self, model: str = "sonnet", timeout_seconds: int = 900) -> None:
        if shutil.which("claude") is None:
            raise ValueError("Claude Code CLI is not installed or not on PATH")
        self.model = model
        self.timeout_seconds = timeout_seconds

    def complete(
        self, *, role: str, system: str, prompt: str, schema: dict[str, Any]
    ) -> dict[str, Any]:
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
                "--json-schema", json.dumps(schema),
            ],
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            detail = result.stderr.strip()
            if result.stdout.strip():
                try:
                    envelope = json.loads(result.stdout)
                    detail = str(
                        envelope.get("result")
                        or envelope.get("error")
                        or envelope.get("subtype")
                        or result.stdout
                    )
                except json.JSONDecodeError:
                    detail = result.stdout.strip()
            raise RuntimeError(
                f"Claude Code failed with exit code {result.returncode}: "
                f"{detail[-4000:] or 'no diagnostic output'}"
            )
        envelope = json.loads(result.stdout)
        payload = envelope.get("structured_output")
        if payload is None:
            payload = json.loads(envelope["result"])
        if not isinstance(payload, dict):
            raise ValueError("structured model output must be a JSON object")
        return payload
