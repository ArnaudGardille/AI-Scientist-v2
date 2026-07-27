"""Validated scientific artifacts carried by hierarchical BFTS nodes."""

from __future__ import annotations

import hashlib
import json
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
        values = asdict(self)
        values["hypothesis_id"] = ""
        payload = json.dumps(values, sort_keys=True).encode("utf-8")
        stable_id = hashlib.sha256(payload).hexdigest()[:24]
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


@dataclass(frozen=True)
class RevisionAudit:
    preserves_objective: bool
    substantive_mechanism_change: bool
    addresses_reviewed_failure: bool
    objective_analysis: str
    mechanism_analysis: str
    failure_resolution_analysis: str

    def validate(self) -> None:
        for name in (
            "objective_analysis",
            "mechanism_analysis",
            "failure_resolution_analysis",
        ):
            _require_text(getattr(self, name), name, 20)


@dataclass
class ResearchRecord:
    hypothesis: ResearchHypothesis
    phase: ResearchPhase = ResearchPhase.HYPOTHESIS
    theory: TheoryReview | None = None
    falsification: FalsificationReview | None = None
    experiment: ExperimentDesign | None = None
    novelty: NoveltyReview | None = None
    revision_audit: RevisionAudit | None = None
    empirical_results: dict[str, Any] = field(default_factory=dict)
    provenance: list[dict[str, str]] = field(default_factory=list)

    def validate(self) -> None:
        self.hypothesis.validate()
        for artifact in (
            self.theory,
            self.falsification,
            self.experiment,
            self.novelty,
            self.revision_audit,
        ):
            if artifact is not None:
                artifact.validate()

    def conceptually_promotable(self) -> bool:
        """Hard pre-code gate; never substitutes for numeric empirical evaluation."""
        return bool(self.conceptual_gate()["promotable"])

    def conceptual_gate(self) -> dict[str, Any]:
        """Return the complete, machine-readable pre-code gate decision."""
        self.validate()
        checks = (
            {
                "name": "theory_present",
                "passed": self.theory is not None,
                "observed": self.theory is not None,
                "required": True,
            },
            {
                "name": "theory_sound",
                "passed": bool(self.theory and self.theory.sound),
                "observed": self.theory.sound if self.theory else None,
                "required": True,
            },
            {
                "name": "soundness_score",
                "passed": bool(self.theory and self.theory.soundness_score >= 0.6),
                "observed": self.theory.soundness_score if self.theory else None,
                "required": ">= 0.6",
            },
            {
                "name": "falsification_present",
                "passed": self.falsification is not None,
                "observed": self.falsification is not None,
                "required": True,
            },
            {
                "name": "no_fatal_counterexample",
                "passed": bool(self.falsification and not self.falsification.fatal),
                "observed": self.falsification.fatal if self.falsification else None,
                "required": "fatal is false",
            },
            {
                "name": "robustness_score",
                "passed": bool(
                    self.falsification and self.falsification.robustness_score >= 0.4
                ),
                "observed": (
                    self.falsification.robustness_score if self.falsification else None
                ),
                "required": ">= 0.4",
            },
            {
                "name": "experiment_present",
                "passed": self.experiment is not None,
                "observed": self.experiment is not None,
                "required": True,
            },
            {
                "name": "novelty_present",
                "passed": self.novelty is not None,
                "observed": self.novelty is not None,
                "required": True,
            },
            {
                "name": "novelty_score",
                "passed": bool(self.novelty and self.novelty.novelty_score >= 0.35),
                "observed": self.novelty.novelty_score if self.novelty else None,
                "required": ">= 0.35",
            },
            {
                "name": "material_novelty",
                "passed": bool(self.novelty and not self.novelty.likely_incremental),
                "observed": self.novelty.likely_incremental if self.novelty else None,
                "required": "likely_incremental is false",
            },
        )
        checks = list(checks)
        is_revision = any(
            "parent_hypothesis_id" in item for item in self.provenance
        )
        if is_revision:
            checks.extend(
                (
                    {
                        "name": "revision_audit_present",
                        "passed": self.revision_audit is not None,
                        "observed": self.revision_audit is not None,
                        "required": True,
                    },
                    {
                        "name": "revision_preserves_objective",
                        "passed": bool(
                            self.revision_audit
                            and self.revision_audit.preserves_objective
                        ),
                        "observed": (
                            self.revision_audit.preserves_objective
                            if self.revision_audit
                            else None
                        ),
                        "required": True,
                    },
                    {
                        "name": "revision_changes_mechanism",
                        "passed": bool(
                            self.revision_audit
                            and self.revision_audit.substantive_mechanism_change
                        ),
                        "observed": (
                            self.revision_audit.substantive_mechanism_change
                            if self.revision_audit
                            else None
                        ),
                        "required": True,
                    },
                    {
                        "name": "revision_addresses_failure",
                        "passed": bool(
                            self.revision_audit
                            and self.revision_audit.addresses_reviewed_failure
                        ),
                        "observed": (
                            self.revision_audit.addresses_reviewed_failure
                            if self.revision_audit
                            else None
                        ),
                        "required": True,
                    },
                )
            )
        failed_checks = [check["name"] for check in checks if not check["passed"]]
        return {
            "promotable": not failed_checks,
            "failed_checks": failed_checks,
            "checks": checks,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["phase"] = self.phase.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchRecord":
        """Restore a validated record from an atomic campaign checkpoint."""
        hypothesis_payload = dict(payload["hypothesis"])
        for key in ("assumptions", "predictions", "falsifiers"):
            hypothesis_payload[key] = tuple(hypothesis_payload[key])

        theory_payload = payload.get("theory")
        if theory_payload is not None:
            theory_payload = dict(theory_payload)
            theory_payload["hidden_assumptions"] = tuple(
                theory_payload["hidden_assumptions"]
            )

        falsification_payload = payload.get("falsification")
        if falsification_payload is not None:
            falsification_payload = dict(falsification_payload)
            falsification_payload["counterexamples"] = tuple(
                falsification_payload["counterexamples"]
            )

        experiment_payload = payload.get("experiment")
        if experiment_payload is not None:
            experiment_payload = dict(experiment_payload)
            for key in ("manipulated_variables", "controls", "metrics", "confounds"):
                experiment_payload[key] = tuple(experiment_payload[key])

        novelty_payload = payload.get("novelty")
        if novelty_payload is not None:
            novelty_payload = dict(novelty_payload)
            novelty_payload["nearest_methods"] = tuple(
                novelty_payload["nearest_methods"]
            )
        revision_audit_payload = payload.get("revision_audit")

        hypothesis = ResearchHypothesis(**hypothesis_payload)
        if hypothesis.with_stable_id().hypothesis_id != hypothesis.hypothesis_id:
            raise ValueError("checkpoint hypothesis_id does not match canonical content")

        record = cls(
            hypothesis=hypothesis,
            phase=ResearchPhase(payload.get("phase", ResearchPhase.HYPOTHESIS)),
            theory=TheoryReview(**theory_payload) if theory_payload is not None else None,
            falsification=(
                FalsificationReview(**falsification_payload)
                if falsification_payload is not None
                else None
            ),
            experiment=(
                ExperimentDesign(**experiment_payload)
                if experiment_payload is not None
                else None
            ),
            novelty=NoveltyReview(**novelty_payload) if novelty_payload is not None else None,
            revision_audit=(
                RevisionAudit(**revision_audit_payload)
                if revision_audit_payload is not None
                else None
            ),
            empirical_results=dict(payload.get("empirical_results", {})),
            provenance=[dict(item) for item in payload.get("provenance", [])],
        )
        record.validate()
        return record
