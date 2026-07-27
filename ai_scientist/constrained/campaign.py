"""Hierarchical theory-to-code campaign encapsulated around diverse BFTS."""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

from .bundle import ALLOWED_FILES, CandidateBundle
from .bundle_evaluator import BundleEvaluation
from .council import DEFAULT_LENSES, ResearchCouncil, ResearchLens
from .research_schema import ResearchPhase, ResearchRecord
from .selection import ArchiveEntry, DiverseParetoArchive, ResearchObjectives
from .structured_model import StructuredModel


IMPLEMENTATION_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis_summary": {"type": "string"},
        "files": {
            "type": "object",
            "properties": {name: {"type": "string"} for name in sorted(ALLOWED_FILES)},
            "required": ["__init__.py", "estimator.py", "sampler.py"],
            "additionalProperties": False,
        },
    },
    "required": ["hypothesis_summary", "files"],
    "additionalProperties": False,
}


class BundleEvaluator(Protocol):
    def evaluate(
        self, bundle: CandidateBundle, stage_inputs: dict[str, dict] | None = None
    ) -> BundleEvaluation: ...


@dataclass(frozen=True)
class CampaignConfig:
    output_dir: Path
    initial_capacity: int = 5
    max_nodes: int = 15
    revisions_per_expansion: int = 2
    max_per_family: int = 2
    min_conceptual_distance: float = 0.2
    finalist_count: int = 1
    experiment_stage_name: str = "mechanism"
    resume_protocol: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.initial_capacity < 1 or self.max_nodes < self.initial_capacity:
            raise ValueError("invalid campaign node budget")
        if self.revisions_per_expansion < 1 or self.max_per_family < 1:
            raise ValueError("invalid campaign branching parameters")
        if self.finalist_count != 1:
            raise ValueError(
                "finalist_count must be 1: the held-out evaluator is confirmatory, "
                "not a model-selection stage"
            )
        if not self.experiment_stage_name.strip():
            raise ValueError("experiment_stage_name must be non-empty")
        try:
            json.dumps(self.resume_protocol, sort_keys=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("resume_protocol must be JSON serializable") from exc


@dataclass
class CampaignNode:
    node_id: int
    parent_id: int | None
    record: ResearchRecord
    bundle: CandidateBundle
    evaluation: BundleEvaluation
    revision_rationale: str = ""
    final_evaluation: BundleEvaluation | None = None


@dataclass(frozen=True)
class ReviewedConcept:
    record: ResearchRecord
    gate: dict
    objectives: ResearchObjectives


class CampaignRunLock:
    """OS-released single-writer lock for one campaign output directory."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.handle = None

    def __enter__(self) -> "CampaignRunLock":
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / ".campaign.lock"
        self.handle = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError(
                f"campaign output is already locked by another process: {self.output_dir}"
            ) from exc
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f"pid={os.getpid()}\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        if self.handle is None:
            return
        fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
        self.handle.close()
        self.handle = None


class HierarchicalCampaign:
    def __init__(
        self,
        *,
        council: ResearchCouncil,
        implementation_model: StructuredModel,
        evaluator: BundleEvaluator,
        final_evaluator: BundleEvaluator | None,
        config: CampaignConfig,
    ) -> None:
        config.validate()
        self.council = council
        self.model = implementation_model
        self.evaluator = evaluator
        self.final_evaluator = final_evaluator
        self.config = config
        self.archive = DiverseParetoArchive(
            min_distance=config.min_conceptual_distance,
            max_per_family=config.max_per_family,
        )
        self.nodes: list[CampaignNode] = []
        self.concepts: list[ReviewedConcept] = []

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
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

    @staticmethod
    def _objectives(record: ResearchRecord) -> ResearchObjectives:
        complexity = {"low": 0.9, "medium": 0.65, "high": 0.35}[record.hypothesis.complexity]
        signatures = min(1.0, len(record.experiment.expected_signatures) / 4.0)
        controls = min(1.0, len(record.experiment.controls) / 4.0)
        return ResearchObjectives(
            soundness=record.theory.soundness_score,
            robustness=record.falsification.robustness_score,
            novelty=record.novelty.novelty_score,
            information_gain=0.5 * signatures + 0.5 * controls,
            feasibility=complexity,
        )

    def _implement(
        self,
        record: ResearchRecord,
        *,
        parent: CampaignNode | None = None,
        request_id: str,
    ) -> tuple[CandidateBundle, str]:
        parent_context = ""
        if parent is not None:
            parent_context = (
                "\nParent implementation:\n"
                + json.dumps(parent.bundle.files, sort_keys=True)
                + "\nRedacted adaptive-search diagnostics:\n"
                + json.dumps(parent.evaluation.designer_feedback(), sort_keys=True)
            )
        payload = self.model.complete(
            role="method-designer",
            system=(
                "Implement the reviewed mechanism, not a hyperparameter-only variant. "
                "Only return the candidate package. Preserve support and obey the import sandbox."
            ),
            prompt=(
                "Reviewed research record:\n"
                + json.dumps(record.to_dict(), sort_keys=True)
                + parent_context
            ),
            schema=IMPLEMENTATION_SCHEMA,
            request_id=request_id,
        )
        bundle = CandidateBundle(
            files=dict(payload["files"]),
            hypothesis_summary=payload["hypothesis_summary"],
        )
        try:
            bundle.validate()
        except Exception as exc:
            reject = getattr(self.model, "reject_response", None)
            if reject is not None:
                reject(request_id, type(exc).__name__)
            raise
        accept = getattr(self.model, "accept_response", None)
        if accept is not None:
            accept(request_id)
        return bundle, payload["hypothesis_summary"]

    def _evaluate_node(
        self, record: ResearchRecord, bundle: CandidateBundle, parent_id: int | None, rationale: str
    ) -> CampaignNode:
        evaluation = self.evaluator.evaluate(
            bundle,
            stage_inputs={
                self.config.experiment_stage_name: record.experiment.mechanism_config
            },
        )
        # Archive records describe conceptual hypotheses. Each implementation node owns an
        # immutable-at-creation snapshot so later revisions cannot rewrite earlier provenance.
        node_record = copy.deepcopy(record)
        node_record.phase = ResearchPhase.SCREENING
        node_record.empirical_results = copy.deepcopy(evaluation.stage_results)
        node = CampaignNode(
            node_id=len(self.nodes),
            parent_id=parent_id,
            record=node_record,
            bundle=bundle,
            evaluation=evaluation,
            revision_rationale=rationale,
        )
        self.nodes.append(node)
        self._save_node(node)
        self._save_journal()
        return node

    def _save_node(self, node: CampaignNode) -> None:
        directory = self.config.output_dir / f"node_{node.node_id:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self._write_json(directory / "hypothesis.json", node.record.to_dict())
        self._write_json(directory / "evaluation.json", asdict(node.evaluation))
        if node.final_evaluation is not None:
            self._write_json(
                directory / "final_evaluation.json",
                asdict(node.final_evaluation),
            )
        node.bundle.materialize(directory / "candidate")

    def _save_concept(self, concept: ReviewedConcept, index: int) -> None:
        hypothesis_id = concept.record.hypothesis.hypothesis_id
        path = self.config.output_dir / "concepts" / f"{index:04d}_{hypothesis_id}.json"
        self._write_json(
            path,
            {
                "concept_index": index,
                "record": concept.record.to_dict(),
                "gate": concept.gate,
                "objectives": asdict(concept.objectives),
            },
        )

    def _checkpoint_path(self, index: int) -> Path:
        return self.config.output_dir / "checkpoints" / f"concept_{index:04d}.json"

    def _save_checkpoint(
        self,
        index: int,
        lens: ResearchLens,
        record: ResearchRecord,
    ) -> None:
        self._write_json(
            self._checkpoint_path(index),
            {
                "concept_index": index,
                "lens": asdict(lens),
                "record": record.to_dict(),
            },
        )

    def _prepare_resume(
        self,
        problem: str,
        lenses: tuple[ResearchLens, ...],
    ) -> None:
        if len({lens.name for lens in lenses}) != len(lenses):
            raise ValueError("research lens names must be unique for durable request IDs")
        lens_payload = [asdict(lens) for lens in lenses]
        fingerprint = hashlib.sha256(
            json.dumps(
                {"problem": problem, "lenses": lens_payload},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        manifest_path = self.config.output_dir / "checkpoints" / "manifest.json"
        expected = {
            "schema_version": 2,
            "campaign_fingerprint": fingerprint,
            "problem_sha256": hashlib.sha256(problem.encode("utf-8")).hexdigest(),
            "lenses": lens_payload,
            "resume_protocol": self.config.resume_protocol,
            "implementation_schema_sha256": hashlib.sha256(
                json.dumps(IMPLEMENTATION_SCHEMA, sort_keys=True).encode("utf-8")
            ).hexdigest(),
            "source_sha256": {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path(__file__).parent.glob("*.py"))
            },
        }
        if manifest_path.exists():
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
            if observed != expected:
                raise ValueError(
                    "campaign checkpoint does not match the current problem and lenses"
                )
            return
        self._write_json(manifest_path, expected)

    def _generate_or_resume(
        self,
        problem: str,
        index: int,
        lens: ResearchLens,
    ) -> ResearchRecord:
        checkpoint_path = self._checkpoint_path(index)
        if checkpoint_path.exists():
            payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if payload.get("concept_index") != index or payload.get("lens") != asdict(lens):
                raise ValueError(f"invalid conceptual checkpoint at {checkpoint_path}")
            return ResearchRecord.from_dict(payload["record"])

        generated = self.council.generate(problem, lenses=(lens,))
        if len(generated) != 1:
            raise ValueError("each research lens must generate exactly one hypothesis")
        record = generated[0]
        self._save_checkpoint(index, lens, record)
        return record

    def _save_journal(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        best = self.best_node()
        payload = {
            "best_node_id": best.node_id if best else None,
            "accepted": bool(
                best
                and best.final_evaluation is not None
                and best.final_evaluation.complete
            ),
            "concepts": [
                {
                    "concept_index": index,
                    "hypothesis_id": concept.record.hypothesis.hypothesis_id,
                    "title": concept.record.hypothesis.title,
                    "family": concept.record.hypothesis.family,
                    "promotable": concept.gate["promotable"],
                    "failed_checks": concept.gate["failed_checks"],
                    "objectives": asdict(concept.objectives),
                }
                for index, concept in enumerate(self.concepts)
            ],
            "nodes": [
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent_id,
                    "hypothesis_id": node.record.hypothesis.hypothesis_id,
                    "family": node.record.hypothesis.family,
                    "status": node.evaluation.status,
                    "priority": node.evaluation.priority(),
                    "final_priority": (
                        node.final_evaluation.priority() if node.final_evaluation else None
                    ),
                    "final_status": (
                        node.final_evaluation.status if node.final_evaluation else None
                    ),
                }
                for node in self.nodes
            ],
        }
        self._write_json(self.config.output_dir / "journal.json", payload)

    def best_node(self) -> CampaignNode | None:
        # Candidate selection is frozen before the held-out run. The final result can confirm
        # or reject that choice, but must never select a different implementation.
        return max(self.nodes, key=lambda node: node.evaluation.priority(), default=None)

    def _validate_finalists(self) -> None:
        if self.final_evaluator is None:
            return
        selected = self.best_node()
        if selected is None or not selected.evaluation.complete:
            return
        # Held-out validation deliberately ignores agent-designed experiment inputs.
        selected.final_evaluation = self.final_evaluator.evaluate(selected.bundle)
        self._save_node(selected)
        self._save_journal()

    def run(
        self,
        problem: str,
        lenses: tuple[ResearchLens, ...] = DEFAULT_LENSES,
    ) -> CampaignNode | None:
        with CampaignRunLock(self.config.output_dir):
            return self._run_locked(problem, lenses)

    def _run_locked(
        self,
        problem: str,
        lenses: tuple[ResearchLens, ...],
    ) -> CampaignNode | None:
        begin_locked_run = getattr(self.model, "begin_locked_run", None)
        if begin_locked_run is not None:
            begin_locked_run()
        self._prepare_resume(problem, lenses)
        for index, lens in enumerate(lenses):
            record = self._generate_or_resume(problem, index, lens)
            reviewed = self.council.review(
                record,
                problem,
                on_progress=lambda current, i=index, item=lens: self._save_checkpoint(
                    i, item, current
                ),
            )
            self._save_checkpoint(index, lens, reviewed)
            gate = reviewed.conceptual_gate()
            objectives = self._objectives(reviewed)
            concept = ReviewedConcept(
                record=copy.deepcopy(reviewed),
                gate=copy.deepcopy(gate),
                objectives=objectives,
            )
            self.concepts.append(concept)
            self._save_concept(concept, len(self.concepts) - 1)
            self._save_journal()
            if gate["promotable"]:
                self.archive.add(ArchiveEntry(reviewed, objectives))

        entries = self.archive.frontier(self.config.initial_capacity)
        for entry in entries:
            bundle, rationale = self._implement(
                entry.record,
                request_id=f"implementation:{entry.entry_id}:root",
            )
            node = self._evaluate_node(entry.record, bundle, None, rationale)
            normalized = 0.5 + 0.5 * math.tanh(node.evaluation.priority() / 1000.0)
            entry.empirical_score = normalized

        while self.nodes and len(self.nodes) < self.config.max_nodes:
            entry = self.archive.select_for_expansion(
                capacity=self.config.initial_capacity
            )
            candidates = [
                node for node in self.nodes
                if node.record.hypothesis.hypothesis_id == entry.entry_id
            ]
            parent = max(candidates, key=lambda node: node.evaluation.priority())
            for _ in range(self.config.revisions_per_expansion):
                if len(self.nodes) >= self.config.max_nodes:
                    break
                bundle, rationale = self._implement(
                    entry.record,
                    parent=parent,
                    request_id=(
                        f"implementation:{entry.entry_id}:"
                        f"parent:{parent.node_id}:node:{len(self.nodes)}"
                    ),
                )
                child = self._evaluate_node(
                    entry.record, bundle, parent.node_id, rationale
                )
                if child.evaluation.priority() > parent.evaluation.priority():
                    parent = child
                entry.empirical_score = 0.5 + 0.5 * math.tanh(
                    parent.evaluation.priority() / 1000.0
                )
            self._save_journal()
        self._validate_finalists()
        self._save_journal()
        return self.best_node()
