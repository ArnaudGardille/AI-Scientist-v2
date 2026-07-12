"""Pareto- and diversity-aware selection for hierarchical research BFTS."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from .research_schema import ResearchRecord


@dataclass(frozen=True)
class ResearchObjectives:
    soundness: float
    robustness: float
    novelty: float
    information_gain: float
    feasibility: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def values(self) -> tuple[float, ...]:
        self.validate()
        return tuple(vars(self).values())


@dataclass
class ArchiveEntry:
    record: ResearchRecord
    objectives: ResearchObjectives
    visits: int = 0
    empirical_score: float | None = None
    entry_id: str = field(init=False)

    def __post_init__(self) -> None:
        self.record.validate()
        self.objectives.validate()
        self.entry_id = self.record.hypothesis.hypothesis_id


def dominates(left: ResearchObjectives, right: ResearchObjectives) -> bool:
    """True when ``left`` is no worse on every objective and better on at least one."""
    lhs, rhs = left.values(), right.values()
    return all(a >= b for a, b in zip(lhs, rhs, strict=True)) and any(
        a > b for a, b in zip(lhs, rhs, strict=True)
    )


def _tokens(record: ResearchRecord) -> set[str]:
    hypothesis = record.hypothesis
    text = " ".join(
        (hypothesis.title, hypothesis.family, hypothesis.mechanism, hypothesis.formal_claim)
    ).lower()
    return set(re.findall(r"[a-z][a-z0-9_-]{2,}", text))


def conceptual_distance(left: ResearchRecord, right: ResearchRecord) -> float:
    """Token-Jaccard distance used only for deduplication, never scientific scoring."""
    lhs, rhs = _tokens(left), _tokens(right)
    union = lhs | rhs
    return 1.0 if not union else 1.0 - len(lhs & rhs) / len(union)


class DiverseParetoArchive:
    """Maintain non-dominated ideas while preserving families and exploration budget."""

    def __init__(self, *, min_distance: float = 0.2, max_per_family: int = 2) -> None:
        if not 0.0 <= min_distance <= 1.0 or max_per_family < 1:
            raise ValueError("invalid diversity archive parameters")
        self.min_distance = min_distance
        self.max_per_family = max_per_family
        self.entries: list[ArchiveEntry] = []
        self.total_selections = 0

    @staticmethod
    def _mean_objective(entry: ArchiveEntry) -> float:
        return sum(entry.objectives.values()) / len(entry.objectives.values())

    def add(self, entry: ArchiveEntry) -> bool:
        """Add a sufficiently distinct entry, replacing a near-duplicate only if better."""
        if not entry.record.conceptually_promotable():
            return False
        near = [
            existing
            for existing in self.entries
            if conceptual_distance(existing.record, entry.record) < self.min_distance
        ]
        if near:
            incumbent = max(near, key=self._mean_objective)
            if not dominates(entry.objectives, incumbent.objectives):
                return False
            self.entries.remove(incumbent)
        self.entries.append(entry)
        return True

    def pareto_front(self) -> list[ArchiveEntry]:
        return [
            entry
            for entry in self.entries
            if not any(
                other is not entry and dominates(other.objectives, entry.objectives)
                for other in self.entries
            )
        ]

    def frontier(self, capacity: int) -> list[ArchiveEntry]:
        """Return a family-capped Pareto frontier, backfilled by strong diverse entries."""
        if capacity < 1:
            raise ValueError("capacity must be positive")
        selected: list[ArchiveEntry] = []
        family_counts: dict[str, int] = {}
        candidates = sorted(
            self.pareto_front(), key=self._mean_objective, reverse=True
        ) + sorted(self.entries, key=self._mean_objective, reverse=True)
        for entry in candidates:
            if entry in selected:
                continue
            family = entry.record.hypothesis.family.strip().lower()
            if family_counts.get(family, 0) >= self.max_per_family:
                continue
            selected.append(entry)
            family_counts[family] = family_counts.get(family, 0) + 1
            if len(selected) == capacity:
                break
        return selected

    def select_for_expansion(
        self, *, capacity: int, exploration: float = 0.35
    ) -> ArchiveEntry:
        """Select from the diverse frontier using a bounded UCB exploration bonus."""
        candidates = self.frontier(capacity)
        if not candidates:
            raise ValueError("cannot select from an empty archive")
        self.total_selections += 1

        def priority(entry: ArchiveEntry) -> float:
            exploitation = self._mean_objective(entry)
            if entry.empirical_score is not None:
                exploitation = 0.5 * exploitation + 0.5 * entry.empirical_score
            bonus = exploration * math.sqrt(
                math.log(self.total_selections + 1.0) / (entry.visits + 1.0)
            )
            return exploitation + bonus

        selected = max(candidates, key=priority)
        selected.visits += 1
        return selected
