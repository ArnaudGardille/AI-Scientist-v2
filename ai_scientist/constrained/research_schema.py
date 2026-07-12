"""Validated scientific artifacts carried by hierarchical BFTS nodes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class ResearchPhase(StrEnum):
    HYPOTHESIS = "hypothesis"
    THEORY = "theory"
    FALSIFICATION = "falsification"
    EXPERIMENT = "experiment"
    IMPLEMENTATION = "implementation"
    SCREENING = "screening"
    VALIDATION = "validation"


def _require_text(value: str, name: str, minimum: int = 12) -> None:
    if not isinstance(value, str) or len(value.strip()) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} characters")


def _require_items(values: tuple[str, ...], name: str, minimum: int) -> None:
    if len(values) < minimum or any(not str(value).strip() for value in values):
        raise ValueError(f"{name} requires at least {minimum} non-empty items")


def _score(value: float, name: str) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class ResearchHypothesis:
    title: str
    family: str
    mechanism: str
    formal_claim: str
    assumptions: tuple[str, ...]
    predictions: tuple[str, ...]
    falsifiers: tuple[str, ...]
    novelty_claim: str
    complexity: str
    hypothesis_id: str = ""

    def validate(self) -> None:
        _require_text(self.family, "family", 3)
        for name in ("title", "mechanism", "formal_claim", "novelty_claim"):
            _require_text(getattr(self, name), name)
        _require_items(self.assumptions, "assumptions", 2)
        _require_items(self.predictions, "predictions", 2)
        _require_items(self.falsifiers, "falsifiers", 2)
        if self.complexity not in {"low", "medium", "high"}:
            raise ValueError("complexity must be low, medium, or high")

    def with_stable_id(self) -> "ResearchHypothesis":
        payload = f"{self.family}\n{self.title}\n{self.mechanism}".encode()
        stable_id = hashlib.sha256(payload).hexdigest()[:12]
        values = asdict(self)
        values["hypothesis_id"] = stable_id
        return ResearchHypothesis(**values)


@dataclass(frozen=True)
class TheoryReview:
    sound: bool
    derivation: str
    hidden_assumptions: tuple[str, ...]
    bias_analysis: str
    variance_analysis: str
    support_analysis: str
    soundness_score: float

    def validate(self) -> None:
        for name in ("derivation", "bias_analysis", "variance_analysis", "support_analysis"):
            _require_text(getattr(self, name), name, 20)
        _require_items(self.hidden_assumptions, "hidden_assumptions", 1)
        _score(self.soundness_score, "soundness_score")


@dataclass(frozen=True)
class FalsificationReview:
    fatal: bool
    counterexamples: tuple[str, ...]
    decisive_test: str
    failure_signature: str
    robustness_score: float

    def validate(self) -> None:
        _require_items(self.counterexamples, "counterexamples", 2)
        _require_text(self.decisive_test, "decisive_test", 20)
        _require_text(self.failure_signature, "failure_signature", 20)
        _score(self.robustness_score, "robustness_score")


@dataclass(frozen=True)
class ExperimentDesign:
    question: str
    manipulated_variables: tuple[str, ...]
    controls: tuple[str, ...]
    metrics: tuple[str, ...]
    expected_signatures: dict[str, str]
    rejection_rule: str
    budget_class: str
    confounds: tuple[str, ...]
    mechanism_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        _require_text(self.question, "question", 20)
        _require_items(self.manipulated_variables, "manipulated_variables", 1)
        _require_items(self.controls, "controls", 2)
        _require_items(self.metrics, "metrics", 2)
        _require_items(self.confounds, "confounds", 2)
        if len(self.expected_signatures) < 2:
            raise ValueError("expected_signatures must distinguish at least two outcomes")
        _require_text(self.rejection_rule, "rejection_rule", 20)
        if self.budget_class not in {"operator", "mechanism", "learning", "validation"}:
            raise ValueError("invalid experiment budget_class")
        allowed = {
            "action_counts", "sample_sizes", "shift_strengths", "noise_levels",
            "coverage_floor", "scenario_seed",
        }
        unknown = set(self.mechanism_config) - allowed
        if unknown:
            raise ValueError(f"unknown mechanism_config fields: {sorted(unknown)}")
        sequence_bounds = {
            "action_counts": (2, 32),
            "sample_sizes": (32, 8192),
            "shift_strengths": (0.0, 1.0),
            "noise_levels": (0.0, 5.0),
        }
        cells = 1
        for key, (minimum, maximum) in sequence_bounds.items():
            values = self.mechanism_config.get(key, [])
            if not isinstance(values, list) or not values:
                raise ValueError(f"mechanism_config.{key} must be a non-empty list")
            if any(float(value) < minimum or float(value) > maximum for value in values):
                raise ValueError(f"mechanism_config.{key} is outside [{minimum}, {maximum}]")
            cells *= len(values)
        coverage = float(self.mechanism_config.get("coverage_floor", 0.0))
        if not 0.001 <= coverage <= 0.25:
            raise ValueError("mechanism_config.coverage_floor must be in [0.001, 0.25]")
        if cells > 128:
            raise ValueError("mechanism_config exceeds the 128-cell budget")


@dataclass(frozen=True)
class NoveltyReview:
    nearest_methods: tuple[str, ...]
    material_difference: str
    likely_incremental: bool
    novelty_score: float

    def validate(self) -> None:
        _require_items(self.nearest_methods, "nearest_methods", 1)
        _require_text(self.material_difference, "material_difference", 20)
        _score(self.novelty_score, "novelty_score")


@dataclass
class ResearchRecord:
    hypothesis: ResearchHypothesis
    phase: ResearchPhase = ResearchPhase.HYPOTHESIS
    theory: TheoryReview | None = None
    falsification: FalsificationReview | None = None
    experiment: ExperimentDesign | None = None
    novelty: NoveltyReview | None = None
    empirical_results: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, str]] = field(default_factory=list)

    def validate(self) -> None:
        self.hypothesis.validate()
        for artifact in (self.theory, self.falsification, self.experiment, self.novelty):
            if artifact is not None:
                artifact.validate()

    def conceptually_promotable(self) -> bool:
        """Hard pre-code gate; never substitutes for numeric empirical evaluation."""
        self.validate()
        return bool(
            self.theory
            and self.theory.sound
            and self.theory.soundness_score >= 0.6
            and self.falsification
            and not self.falsification.fatal
            and self.falsification.robustness_score >= 0.4
            and self.experiment
            and self.novelty
            and self.novelty.novelty_score >= 0.35
            and not self.novelty.likely_incremental
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        return payload
