"""Hierarchical theory-to-code campaign encapsulated around diverse BFTS."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
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
    def evaluate(self, bundle: CandidateBundle) -> BundleEvaluation: ...


@dataclass(frozen=True)
class CampaignConfig:
    output_dir: Path
    initial_capacity: int = 5
    max_nodes: int = 15
    revisions_per_expansion: int = 2
    max_per_family: int = 2
    min_conceptual_distance: float = 0.2
    finalist_count: int = 2

    def validate(self) -> None:
        if self.initial_capacity < 1 or self.max_nodes < self.initial_capacity:
            raise ValueError("invalid campaign node budget")
        if self.revisions_per_expansion < 1 or self.max_per_family < 1 or self.finalist_count < 1:
            raise ValueError("invalid campaign branching parameters")


@dataclass
class CampaignNode:
    node_id: int
    parent_id: int | None
    record: ResearchRecord
    bundle: CandidateBundle
    evaluation: BundleEvaluation
    revision_rationale: str = ""
    final_evaluation: BundleEvaluation | None = None


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
    ) -> tuple[CandidateBundle, str]:
        parent_context = ""
        if parent is not None:
            parent_context = (
                "\nParent implementation:\n"
                + json.dumps(parent.bundle.files, sort_keys=True)
                + "\nFrozen numeric results:\n"
                + json.dumps(parent.evaluation.stage_results, sort_keys=True)
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
        )
        bundle = CandidateBundle(
            files=dict(payload["files"]),
            hypothesis_summary=payload["hypothesis_summary"],
        )
        bundle.validate()
        return bundle, payload["hypothesis_summary"]

    def _evaluate_node(
        self, record: ResearchRecord, bundle: CandidateBundle, parent_id: int | None, rationale: str
    ) -> CampaignNode:
        evaluation = self.evaluator.evaluate(bundle)
        record.phase = ResearchPhase.VALIDATION if evaluation.complete else ResearchPhase.SCREENING
        record.empirical_results = evaluation.stage_results
        node = CampaignNode(
            node_id=len(self.nodes),
            parent_id=parent_id,
            record=record,
            bundle=bundle,
            evaluation=evaluation,
            revision_rationale=rationale,
        )
        self.nodes.append(node)
        self._save_node(node)
        return node

    def _save_node(self, node: CampaignNode) -> None:
        directory = self.config.output_dir / f"node_{node.node_id:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "hypothesis.json").write_text(
            json.dumps(node.record.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        (directory / "evaluation.json").write_text(
            json.dumps(asdict(node.evaluation), indent=2, sort_keys=True), encoding="utf-8"
        )
        if node.final_evaluation is not None:
            (directory / "final_evaluation.json").write_text(
                json.dumps(asdict(node.final_evaluation), indent=2, sort_keys=True),
                encoding="utf-8",
            )
        node.bundle.materialize(directory / "candidate")

    def _save_journal(self) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        best = self.best_node()
        payload = {
            "best_node_id": best.node_id if best else None,
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
                }
                for node in self.nodes
            ],
        }
        (self.config.output_dir / "journal.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )

    def best_node(self) -> CampaignNode | None:
        validated = [node for node in self.nodes if node.final_evaluation is not None]
        if validated:
            return max(validated, key=lambda node: node.final_evaluation.priority())
        return max(self.nodes, key=lambda node: node.evaluation.priority(), default=None)

    def _validate_finalists(self) -> None:
        if self.final_evaluator is None:
            return
        eligible = sorted(
            (node for node in self.nodes if node.evaluation.complete),
            key=lambda node: node.evaluation.priority(),
            reverse=True,
        )
        seen_bundles = set()
        finalists = []
        for node in eligible:
            fingerprint = json.dumps(node.bundle.files, sort_keys=True)
            if fingerprint in seen_bundles:
                continue
            seen_bundles.add(fingerprint)
            finalists.append(node)
            if len(finalists) == self.config.finalist_count:
                break
        for node in finalists:
            node.final_evaluation = self.final_evaluator.evaluate(node.bundle)
            self._save_node(node)

    def run(
        self,
        problem: str,
        lenses: tuple[ResearchLens, ...] = DEFAULT_LENSES,
    ) -> CampaignNode | None:
        generated = self.council.generate(problem, lenses=lenses)
        for record in generated:
            reviewed = self.council.review(record, problem)
            if reviewed.conceptually_promotable():
                self.archive.add(ArchiveEntry(reviewed, self._objectives(reviewed)))

        entries = self.archive.frontier(self.config.initial_capacity)
        for entry in entries:
            bundle, rationale = self._implement(entry.record)
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
                bundle, rationale = self._implement(entry.record, parent=parent)
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
