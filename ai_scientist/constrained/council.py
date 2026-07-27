"""Role-separated research council for hypothesis generation and adversarial review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .research_schema import (
    ExperimentDesign,
    FalsificationReview,
    NoveltyReview,
    ResearchHypothesis,
    ResearchPhase,
    ResearchRecord,
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
