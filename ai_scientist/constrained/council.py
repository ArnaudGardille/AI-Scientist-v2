"""Role-separated research council for hypothesis generation and adversarial review."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Callable

from .research_schema import (
    ExperimentDesign,
    FalsificationReview,
    NoveltyReview,
    ResearchHypothesis,
    ResearchPhase,
    ResearchRecord,
    RevisionAudit,
    TheoryReview,
)
from .structured_model import StructuredModel


HYPOTHESIS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "family": {"type": "string"},
        "mechanism": {"type": "string"},
        "formal_claim": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "predictions": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "falsifiers": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "novelty_claim": {"type": "string"},
        "complexity": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "title", "family", "mechanism", "formal_claim", "assumptions",
        "predictions", "falsifiers", "novelty_claim", "complexity",
    ],
    "additionalProperties": False,
}

THEORY_SCHEMA = {
    "type": "object",
    "properties": {
        "sound": {"type": "boolean"},
        "derivation": {"type": "string"},
        "hidden_assumptions": {"type": "array", "items": {"type": "string"}},
        "bias_analysis": {"type": "string"},
        "variance_analysis": {"type": "string"},
        "support_analysis": {"type": "string"},
        "soundness_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "sound", "derivation", "hidden_assumptions", "bias_analysis",
        "variance_analysis", "support_analysis", "soundness_score",
    ],
    "additionalProperties": False,
}

FALSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "fatal": {"type": "boolean"},
        "counterexamples": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "decisive_test": {"type": "string"},
        "failure_signature": {"type": "string"},
        "robustness_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "fatal", "counterexamples", "decisive_test", "failure_signature", "robustness_score",
    ],
    "additionalProperties": False,
}

EXPERIMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "question": {"type": "string"},
        "manipulated_variables": {"type": "array", "items": {"type": "string"}},
        "controls": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "metrics": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "expected_signatures": {"type": "object", "additionalProperties": {"type": "string"}},
        "rejection_rule": {"type": "string"},
        "budget_class": {
            "type": "string", "enum": ["operator", "mechanism", "learning", "validation"]
        },
        "confounds": {"type": "array", "items": {"type": "string"}, "minItems": 2},
        "mechanism_config": {
            "type": "object",
            "properties": {
                "action_counts": {
                    "type": "array", "items": {"type": "integer", "minimum": 2, "maximum": 32}
                },
                "sample_sizes": {
                    "type": "array", "items": {"type": "integer", "minimum": 32, "maximum": 8192}
                },
                "shift_strengths": {
                    "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "noise_levels": {
                    "type": "array", "items": {"type": "number", "minimum": 0, "maximum": 5}
                },
                "coverage_floor": {"type": "number", "minimum": 0.001, "maximum": 0.25},
                "scenario_seed": {"type": "integer"},
            },
            "required": [
                "action_counts", "sample_sizes", "shift_strengths", "noise_levels",
                "coverage_floor", "scenario_seed",
            ],
            "additionalProperties": False,
        },
    },
    "required": [
        "question", "manipulated_variables", "controls", "metrics",
        "expected_signatures", "rejection_rule", "budget_class", "confounds",
        "mechanism_config",
    ],
    "additionalProperties": False,
}

NOVELTY_SCHEMA = {
    "type": "object",
    "properties": {
        "nearest_methods": {"type": "array", "items": {"type": "string"}},
        "material_difference": {"type": "string"},
        "likely_incremental": {"type": "boolean"},
        "novelty_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": ["nearest_methods", "material_difference", "likely_incremental", "novelty_score"],
    "additionalProperties": False,
}

REVISION_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "preserves_objective": {"type": "boolean"},
        "substantive_mechanism_change": {"type": "boolean"},
        "addresses_reviewed_failure": {"type": "boolean"},
        "objective_analysis": {"type": "string"},
        "mechanism_analysis": {"type": "string"},
        "failure_resolution_analysis": {"type": "string"},
    },
    "required": [
        "preserves_objective",
        "substantive_mechanism_change",
        "addresses_reviewed_failure",
        "objective_analysis",
        "mechanism_analysis",
        "failure_resolution_analysis",
    ],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class ResearchLens:
    name: str
    instruction: str


DEFAULT_LENSES = (
    ResearchLens("statistical-estimation", "Derive a new estimator from bias/variance structure."),
    ResearchLens("counterfactual-causal", "Exploit stationary conditional joint-action dynamics."),
    ResearchLens("game-theoretic", "Address equilibrium selection and teammate adaptation."),
    ResearchLens("coverage-exploration", "Use uncertainty or coverage without starving support."),
    ResearchLens("robust-control", "Optimize against bounded policy drift or misspecification."),
)

QUALITATIVE_GATE_FEEDBACK = {
    "theory_present": "The proposal lacks a complete mathematical analysis.",
    "theory_sound": "The theoretical derivation was judged unsound.",
    "soundness_score": "The theoretical support was judged insufficient.",
    "falsification_present": "The proposal lacks an adversarial falsification analysis.",
    "no_fatal_counterexample": "The falsifier identified a fatal counterexample.",
    "robustness_score": "The mechanism was not robust to the reviewed counterexamples.",
    "experiment_present": "The proposal lacks a decisive controlled experiment.",
    "novelty_present": "The proposal lacks a comparison with its nearest methods.",
    "novelty_score": "The material distinction from prior methods was insufficient.",
    "material_novelty": "The proposal was judged incremental rather than materially new.",
}


class ResearchCouncil:
    def __init__(self, model: StructuredModel) -> None:
        self.model = model

    def _accept(self, request_id: str) -> None:
        accept = getattr(self.model, "accept_response", None)
        if accept is not None:
            accept(request_id)

    def _reject(self, request_id: str, error: Exception) -> None:
        reject = getattr(self.model, "reject_response", None)
        if reject is not None:
            reject(request_id, type(error).__name__)

    @staticmethod
    def _hypothesis(payload: dict[str, Any]) -> ResearchHypothesis:
        for key in ("assumptions", "predictions", "falsifiers"):
            payload[key] = tuple(payload[key])
        hypothesis = ResearchHypothesis(**payload).with_stable_id()
        hypothesis.validate()
        return hypothesis

    def generate(self, problem: str, lenses: tuple[ResearchLens, ...] = DEFAULT_LENSES) -> list[ResearchRecord]:
        records = []
        for lens in lenses:
            request_id = f"hypothesis:{lens.name}"
            payload = self.model.complete(
                role=f"hypothesis-generator:{lens.name}",
                system=(
                    "Generate one falsifiable mechanism, not a hyperparameter change. "
                    "Do not write code and do not merely rename a known baseline."
                ),
                prompt=f"Problem:\n{problem}\n\nResearch lens:\n{lens.instruction}",
                schema=HYPOTHESIS_SCHEMA,
                request_id=request_id,
            )
            try:
                record = ResearchRecord(hypothesis=self._hypothesis(payload))
            except Exception as exc:
                self._reject(request_id, exc)
                raise
            self._accept(request_id)
            record.provenance.append(
                {
                    "role": lens.name,
                    "action": "generated",
                    "request_id": request_id,
                }
            )
            records.append(record)
        return records

    def revise(
        self,
        parent: ResearchRecord,
        problem: str,
        *,
        failed_checks: tuple[str, ...],
        revision_index: int,
    ) -> ResearchRecord:
        parent_id = parent.hypothesis.hypothesis_id
        request_id = f"concept-revision:{parent_id}:{revision_index}"
        qualitative_failures = tuple(
            QUALITATIVE_GATE_FEEDBACK.get(
                check,
                "A reviewer identified an unresolved conceptual weakness.",
            )
            for check in failed_checks
        )
        qualitative_parent = {
            "hypothesis": asdict(parent.hypothesis),
            "theory": {
                key: value
                for key, value in asdict(parent.theory).items()
                if key != "soundness_score"
            },
            "falsification": {
                key: value
                for key, value in asdict(parent.falsification).items()
                if key != "robustness_score"
            },
            "experiment": asdict(parent.experiment),
            "novelty": {
                key: value
                for key, value in asdict(parent.novelty).items()
                if key != "novelty_score"
            },
        }
        payload = self.model.complete(
            role="concept-reviser",
            system=(
                "Revise the scientific mechanism, not its scores or wording. Address the "
                "reviewed technical failures with a materially changed, falsifiable claim. "
                "Do not write code, lower standards, hide assumptions, or propose a "
                "hyperparameter-only variant."
            ),
            prompt=(
                f"Problem:\n{problem}\n\nFailed gate checks:\n"
                f"{json.dumps(qualitative_failures)}\n\nReviewed parent record:\n"
                f"{json.dumps(qualitative_parent, sort_keys=True)}"
            ),
            schema=HYPOTHESIS_SCHEMA,
            request_id=request_id,
        )
        try:
            hypothesis = self._hypothesis(payload)
            if hypothesis.hypothesis_id == parent_id:
                raise ValueError("conceptual revision must materially change the hypothesis")
            if hypothesis.family != parent.hypothesis.family:
                raise ValueError("conceptual revision must preserve the parent family")
        except Exception as exc:
            self._reject(request_id, exc)
            raise
        self._accept(request_id)
        revised = ResearchRecord(
            hypothesis=hypothesis,
            provenance=[
                {
                    "role": "concept-reviser",
                    "action": "revised",
                    "request_id": request_id,
                    "parent_hypothesis_id": parent_id,
                }
            ],
        )
        audit_request_id = (
            f"concept-revision-audit:{parent_id}:{hypothesis.hypothesis_id}"
        )
        audit_payload = self.model.complete(
            role="concept-revision-auditor",
            system=(
                "Audit lineage independently. Approve only if the child preserves the "
                "parent research objective, changes the mechanism substantively rather than "
                "cosmetically, and directly addresses the qualitative reviewed failure. "
                "Do not infer quality from numeric scores."
            ),
            prompt=(
                f"Problem:\n{problem}\n\nQualitative failed checks:\n"
                f"{json.dumps(qualitative_failures)}\n\nParent hypothesis:\n"
                f"{json.dumps(asdict(parent.hypothesis), sort_keys=True)}\n\n"
                f"Child hypothesis:\n"
                f"{json.dumps(asdict(hypothesis), sort_keys=True)}"
            ),
            schema=REVISION_AUDIT_SCHEMA,
            request_id=audit_request_id,
        )
        try:
            revised.revision_audit = RevisionAudit(**audit_payload)
            revised.revision_audit.validate()
        except Exception as exc:
            revised.revision_audit = None
            self._reject(audit_request_id, exc)
            raise
        self._accept(audit_request_id)
        revised.provenance.append(
            {
                "role": "concept-revision-auditor",
                "action": "audited",
                "request_id": audit_request_id,
                "parent_hypothesis_id": parent_id,
            }
        )
        return revised

    def review(
        self,
        record: ResearchRecord,
        problem: str,
        *,
        on_progress: Callable[[ResearchRecord], None] | None = None,
    ) -> ResearchRecord:
        hypothesis = record.hypothesis
        context = f"Problem:\n{problem}\n\nHypothesis:\n{hypothesis}"

        if record.theory is None:
            request_id = f"concept:{hypothesis.hypothesis_id}:theorist"
            theory = self.model.complete(
                role="theorist",
                system=(
                    "Derive the claim and audit bias, variance, support, "
                    "and hidden assumptions."
                ),
                prompt=context,
                schema=THEORY_SCHEMA,
                request_id=request_id,
            )
            try:
                theory["hidden_assumptions"] = tuple(theory["hidden_assumptions"])
                record.theory = TheoryReview(**theory)
                record.theory.validate()
            except Exception as exc:
                record.theory = None
                self._reject(request_id, exc)
                raise
            self._accept(request_id)
            record.phase = ResearchPhase.THEORY
            record.provenance.append(
                {
                    "role": "theorist",
                    "action": "reviewed",
                    "request_id": request_id,
                }
            )
            if on_progress is not None:
                on_progress(record)

        if record.falsification is None:
            request_id = f"concept:{hypothesis.hypothesis_id}:falsifier"
            falsification = self.model.complete(
                role="falsifier",
                system=(
                    "Try to refute the mechanism with concrete counterexamples "
                    "and one decisive test."
                ),
                prompt=context + f"\n\nTheory review:\n{record.theory}",
                schema=FALSIFICATION_SCHEMA,
                request_id=request_id,
            )
            try:
                falsification["counterexamples"] = tuple(
                    falsification["counterexamples"]
                )
                record.falsification = FalsificationReview(**falsification)
                record.falsification.validate()
            except Exception as exc:
                record.falsification = None
                self._reject(request_id, exc)
                raise
            self._accept(request_id)
            record.phase = ResearchPhase.FALSIFICATION
            record.provenance.append(
                {
                    "role": "falsifier",
                    "action": "reviewed",
                    "request_id": request_id,
                }
            )
            if on_progress is not None:
                on_progress(record)

        if record.experiment is None:
            request_id = f"concept:{hypothesis.hypothesis_id}:experimentalist"
            experiment = self.model.complete(
                role="experimentalist",
                system=(
                    "Design the cheapest experiment that distinguishes the proposed mechanism "
                    "from plausible alternatives. Metrics and rejection rules must be numeric."
                ),
                prompt=context + f"\n\nFalsification review:\n{record.falsification}",
                schema=EXPERIMENT_SCHEMA,
                request_id=request_id,
            )
            try:
                for key in ("manipulated_variables", "controls", "metrics", "confounds"):
                    experiment[key] = tuple(experiment[key])
                record.experiment = ExperimentDesign(**experiment)
                record.experiment.validate()
            except Exception as exc:
                record.experiment = None
                self._reject(request_id, exc)
                raise
            self._accept(request_id)
            record.phase = ResearchPhase.EXPERIMENT
            record.provenance.append(
                {
                    "role": "experimentalist",
                    "action": "reviewed",
                    "request_id": request_id,
                }
            )
            if on_progress is not None:
                on_progress(record)

        if record.novelty is None:
            request_id = f"concept:{hypothesis.hypothesis_id}:novelty"
            novelty = self.model.complete(
                role="novelty-reviewer",
                system=(
                    "Identify nearest prior methods and whether the claimed distinction "
                    "is material. Be conservative; this review cannot override "
                    "empirical evidence."
                ),
                prompt=context,
                schema=NOVELTY_SCHEMA,
                request_id=request_id,
            )
            try:
                novelty["nearest_methods"] = tuple(novelty["nearest_methods"])
                record.novelty = NoveltyReview(**novelty)
                record.novelty.validate()
            except Exception as exc:
                record.novelty = None
                self._reject(request_id, exc)
                raise
            self._accept(request_id)
            record.provenance.append(
                {
                    "role": "novelty-reviewer",
                    "action": "reviewed",
                    "request_id": request_id,
                }
            )
            if on_progress is not None:
                on_progress(record)

        record.validate()
        return record
