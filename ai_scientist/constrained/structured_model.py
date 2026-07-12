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
            raise RuntimeError(result.stderr[-4000:])
        envelope = json.loads(result.stdout)
        payload = envelope.get("structured_output")
        if payload is None:
            payload = json.loads(envelope["result"])
        if not isinstance(payload, dict):
            raise ValueError("structured model output must be a JSON object")
        return payload
