"""Frozen staged evaluator for validated candidate bundles."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .bundle import CandidateBundle
from .search import EvaluationStage


def _manifest(root: Path, relative_paths: tuple[Path, ...]) -> dict[str, str]:
    manifest = {}
    for relative in relative_paths:
        target = root / relative
        files = [target] if target.is_file() else sorted(
            path for path in target.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
        digest = hashlib.sha256()
        for path in files:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        manifest[str(relative)] = digest.hexdigest()
    return manifest


@dataclass(frozen=True)
class BundleEvaluatorConfig:
    repository: Path
    base_ref: str
    candidate_path: Path
    working_subdir: Path
    frozen_paths: tuple[Path, ...]
    stages: tuple[EvaluationStage, ...]
    timeout_seconds: int = 3600
    shared_python: Path | None = None


@dataclass
class BundleEvaluation:
    status: str
    stage_results: dict = field(default_factory=dict)
    passed_stages: int = 0
    complete: bool = False
    feedback: str = ""
    ranking_score: float | None = None

    def priority(self) -> float:
        score = self.ranking_score
        if score is None:
            score = sum(
                float(value["primary_score"])
                for value in self.stage_results.values()
            )
        return 1000.0 * self.passed_stages + score

    def designer_feedback(self) -> dict:
        """Return adaptive-search diagnostics without private benchmark answers."""
        stage_keys = {
            "status",
            "metric_direction",
            "primary_score",
            "aggregate_mse",
            "robust_risk",
            "worst_mse",
            "buckets",
            "raw_score",
            "support_gate_failures",
        }
        scenario_keys = {
            "support_gate",
            "mse",
            "variance",
            "sample_size",
            "trials",
            "mean_utility_delta",
            "utility_sem",
            "lower_confidence_bound",
            "lower_screening_bound",
        }
        stages = {}
        for name, payload in self.stage_results.items():
            summary = {key: payload[key] for key in stage_keys if key in payload}
            scenarios = payload.get("scenarios")
            if isinstance(scenarios, dict):
                summary["scenarios"] = {
                    scenario: {
                        key: value[key]
                        for key in scenario_keys
                        if key in value
                    }
                    for scenario, value in scenarios.items()
                }
            stages[name] = summary
        return {
            "status": self.status,
            "passed_stages": self.passed_stages,
            "complete": self.complete,
            "stages": stages,
        }


class WorktreeBundleEvaluator:
    def __init__(self, config: BundleEvaluatorConfig) -> None:
        self.config = config

    def _run(self, args: list[str], *, cwd: Path, worktree: Path) -> subprocess.CompletedProcess:
        safe_keys = {"PATH", "LANG", "LC_ALL", "TZ"}
        safe_prefixes = ("JAX_", "XLA_", "OMP_", "TF_CPP_")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in safe_keys or key.startswith(safe_prefixes)
        }
        sandbox_home = worktree / ".sandbox_home"
        sandbox_tmp = worktree / ".sandbox_tmp"
        sandbox_home.mkdir(exist_ok=True)
        sandbox_tmp.mkdir(exist_ok=True)
        environment["HOME"] = str(sandbox_home)
        environment["TMPDIR"] = str(sandbox_tmp)
        environment["PYTHONNOUSERSITE"] = "1"
        source = str(worktree / self.config.working_subdir)
        environment["PYTHONPATH"] = source
        return subprocess.run(
            args,
            cwd=cwd,
            env=environment,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
            start_new_session=True,
        )

    def evaluate(
        self,
        bundle: CandidateBundle,
        stage_inputs: dict[str, dict] | None = None,
    ) -> BundleEvaluation:
        bundle.validate()
        cfg = self.config
        worktree = Path(tempfile.mkdtemp(prefix="aisci-bundle-"))
        shutil.rmtree(worktree)
        evaluation = BundleEvaluation(status="pending", ranking_score=0.0)
        try:
            add = subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), cfg.base_ref],
                cwd=cfg.repository,
                text=True,
                capture_output=True,
                check=False,
            )
            if add.returncode != 0:
                raise RuntimeError(add.stderr)
            before = _manifest(worktree, cfg.frozen_paths)
            bundle.materialize(worktree / cfg.candidate_path)

            for stage in cfg.stages:
                output = worktree / f".bundle_{stage.name}.json"
                stage_config = worktree / f".bundle_{stage.name}_input.json"
                if stage_inputs and stage.name in stage_inputs:
                    stage_config.write_text(
                        json.dumps(stage_inputs[stage.name], indent=2, sort_keys=True),
                        encoding="utf-8",
                    )
                substitutions = {
                    "output": str(output),
                    "python": str(cfg.shared_python) if cfg.shared_python else "python",
                    "stage_config": str(stage_config),
                }
                command = [part.format(**substitutions) for part in stage.command]
                result = self._run(command, cwd=worktree / cfg.working_subdir, worktree=worktree)
                if result.returncode != 0 or not output.exists():
                    evaluation.status = f"failed:{stage.name}"
                    evaluation.feedback = (result.stdout + "\n" + result.stderr)[-8000:]
                    break
                payload = json.loads(output.read_text(encoding="utf-8"))
                score = float(payload["primary_score"])
                if not math.isfinite(score):
                    raise ValueError(f"non-finite primary score in {stage.name}")
                evaluation.stage_results[stage.name] = payload
                if stage.contributes_to_priority:
                    evaluation.ranking_score += score
                if payload.get("status") != stage.required_status:
                    evaluation.status = f"rejected:{stage.name}"
                    evaluation.feedback = json.dumps(payload, sort_keys=True)[-8000:]
                    break
                if stage.minimum_score is not None and score < stage.minimum_score:
                    evaluation.status = f"not_promoted:{stage.name}"
                    break
                evaluation.passed_stages += 1
            else:
                evaluation.status = "ok"
                evaluation.complete = True

            if before != _manifest(worktree, cfg.frozen_paths):
                return BundleEvaluation(
                    status="integrity_failure",
                    feedback="A frozen benchmark path changed during evaluation.",
                )
            return evaluation
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            return BundleEvaluation(status="failed:evaluator", feedback=str(exc)[-8000:])
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=cfg.repository,
                capture_output=True,
                check=False,
            )
            shutil.rmtree(worktree, ignore_errors=True)
