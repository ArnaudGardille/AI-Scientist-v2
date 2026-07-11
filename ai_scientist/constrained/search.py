"""Best-first tree search where an LLM may edit one file and nothing else."""

from __future__ import annotations

import hashlib
import heapq
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class EvaluationStage:
    name: str
    command: tuple[str, ...]
    minimum_score: float | None = None
    required_status: str = "ok"


@dataclass(frozen=True)
class SearchConfig:
    repository: Path
    base_ref: str
    candidate_path: Path
    working_subdir: Path
    frozen_paths: tuple[Path, ...]
    stages: tuple[EvaluationStage, ...]
    output_dir: Path
    branching_factor: int = 3
    max_nodes: int = 12
    timeout_seconds: int = 3600


@dataclass(frozen=True)
class Proposal:
    hypothesis: str
    code: str


@dataclass
class CandidateNode:
    node_id: int
    parent_id: int | None
    depth: int
    hypothesis: str
    code: str
    score: float = float("-inf")
    status: str = "pending"
    stage_results: dict = field(default_factory=dict)
    feedback: str = ""


class Proposer(Protocol):
    def propose(
        self, parent: CandidateNode, *, count: int, task_context: str
    ) -> list[Proposal]: ...


def _hash_path(root: Path, relative: Path) -> str:
    target = root / relative
    digest = hashlib.sha256()
    if target.is_dir():
        paths = sorted(
            path
            for path in target.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix not in {".pyc", ".pyo"}
        )
    else:
        paths = [target]
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class ConstrainedBFTS:
    """Evaluate tree nodes in disposable Git worktrees with frozen-file checks."""

    def __init__(self, config: SearchConfig, proposer: Proposer, task_context: str) -> None:
        self.config = config
        self.proposer = proposer
        self.task_context = task_context
        self.nodes: list[CandidateNode] = []
        self._seen_code: set[str] = set()

    def _run(self, args: list[str], *, cwd: Path) -> subprocess.CompletedProcess:
        return subprocess.run(
            args,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )

    def _evaluate(self, node: CandidateNode) -> None:
        cfg = self.config
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        worktree = Path(tempfile.mkdtemp(prefix=f"aisci-node-{node.node_id}-"))
        shutil.rmtree(worktree)
        try:
            add = self._run(
                ["git", "worktree", "add", "--detach", str(worktree), cfg.base_ref],
                cwd=cfg.repository,
            )
            if add.returncode != 0:
                raise RuntimeError(add.stderr)

            candidate = worktree / cfg.candidate_path
            candidate.write_text(node.code, encoding="utf-8")
            frozen_before = {
                str(path): _hash_path(worktree, path) for path in cfg.frozen_paths
            }
            stage_results = {}
            completed_all = True
            for stage in cfg.stages:
                output = worktree / f".constrained_{stage.name}.json"
                command = [part.format(output=str(output)) for part in stage.command]
                result = self._run(command, cwd=worktree / cfg.working_subdir)
                if result.returncode != 0 or not output.exists():
                    node.feedback = (result.stdout + "\n" + result.stderr)[-8000:]
                    node.status = f"failed:{stage.name}"
                    completed_all = False
                    break
                payload = json.loads(output.read_text(encoding="utf-8"))
                stage_results[stage.name] = payload
                score = float(payload["primary_score"])
                if payload.get("status") != stage.required_status:
                    node.feedback = json.dumps(payload, sort_keys=True)[-8000:]
                    node.status = f"rejected:{stage.name}"
                    completed_all = False
                    break
                if stage.minimum_score is not None and score < stage.minimum_score:
                    node.status = f"not_promoted:{stage.name}"
                    completed_all = False
                    break

            frozen_after = {
                str(path): _hash_path(worktree, path) for path in cfg.frozen_paths
            }
            if frozen_before != frozen_after:
                node.status = "integrity_failure"
                node.score = float("-inf")
                node.feedback = "A frozen benchmark path changed during evaluation."
                return

            node.stage_results = stage_results
            stage_scores = [float(value["primary_score"]) for value in stage_results.values()]
            node.score = sum(stage_scores) + (1000.0 if completed_all else 0.0)
            if completed_all:
                node.status = "ok"
            if not node.feedback:
                node.feedback = json.dumps(stage_results, sort_keys=True)[-8000:]
        except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
            node.status = "failed:evaluator"
            node.feedback = str(exc)[-8000:]
        finally:
            self._run(["git", "worktree", "remove", "--force", str(worktree)], cwd=cfg.repository)
            shutil.rmtree(worktree, ignore_errors=True)

    def _save(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "best_node_id": self.best_node().node_id,
            "nodes": [asdict(node) for node in self.nodes],
        }
        (self.config.output_dir / "journal.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        (self.config.output_dir / "best_candidate.py").write_text(
            self.best_node().code, encoding="utf-8"
        )

    def best_node(self) -> CandidateNode:
        return max(self.nodes, key=lambda node: node.score)

    def search(self) -> CandidateNode:
        root_code = (self.config.repository / self.config.candidate_path).read_text(
            encoding="utf-8"
        )
        root = CandidateNode(0, None, 0, "existing baseline", root_code)
        self._evaluate(root)
        self.nodes.append(root)
        self._seen_code.add(hashlib.sha256(root_code.encode()).hexdigest())
        frontier = [(-root.score, root.node_id, root)]

        while frontier and len(self.nodes) < self.config.max_nodes:
            _, _, parent = heapq.heappop(frontier)
            proposals = self.proposer.propose(
                parent, count=self.config.branching_factor, task_context=self.task_context
            )
            for proposal in proposals:
                if len(self.nodes) >= self.config.max_nodes:
                    break
                code_hash = hashlib.sha256(proposal.code.encode()).hexdigest()
                if code_hash in self._seen_code:
                    continue
                self._seen_code.add(code_hash)
                node = CandidateNode(
                    len(self.nodes), parent.node_id, parent.depth + 1,
                    proposal.hypothesis, proposal.code,
                )
                self._evaluate(node)
                self.nodes.append(node)
                if node.score != float("-inf"):
                    heapq.heappush(frontier, (-node.score, node.node_id, node))
                self._save()

        self._save()
        return self.best_node()
