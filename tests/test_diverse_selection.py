from __future__ import annotations

import unittest

from ai_scientist.constrained.research_schema import (
    ExperimentDesign,
    FalsificationReview,
    NoveltyReview,
    ResearchHypothesis,
    ResearchRecord,
    TheoryReview,
)
from ai_scientist.constrained.selection import (
    ArchiveEntry,
    DiverseParetoArchive,
    ResearchObjectives,
    conceptual_distance,
    dominates,
)


def record(title: str, family: str, mechanism: str) -> ResearchRecord:
    hypothesis = ResearchHypothesis(
        title=title,
        family=family,
        mechanism=mechanism,
        formal_claim="The proposed mechanism improves a measurable target under stated assumptions.",
        assumptions=("Finite actions are available", "Conditional dynamics remain stationary"),
        predictions=("Operator error decreases", "Support remains nonzero during learning"),
        falsifiers=("Error does not decrease", "Rare optimal actions disappear from replay"),
        novelty_claim="The mechanism differs materially from plain replay prioritization.",
        complexity="medium",
    ).with_stable_id()
    return ResearchRecord(
        hypothesis=hypothesis,
        theory=TheoryReview(
            True,
            "A complete derivation conditions on teammate actions before target marginalization.",
            ("Coverage is sufficient",),
            "Bias vanishes only when every target-relevant conditional component is identified.",
            "Variance depends on conditional noise and allocation across teammate actions.",
            "A positive support floor prevents deterministic removal of rare useful actions.",
            0.8,
        ),
        falsification=FalsificationReview(
            False,
            ("Coverage collapses", "The target policy changes faster than estimation"),
            "Sweep policy divergence and coverage while comparing paired operator errors.",
            "The proposed advantage disappears specifically in low-coverage target strata.",
            0.7,
        ),
        experiment=ExperimentDesign(
            "Does the mechanism improve estimation without destroying rare-action support?",
            ("policy divergence",),
            ("uniform replay", "exact importance sampling"),
            ("operator MSE", "support mass"),
            {"mechanism": "lower MSE", "null": "no paired improvement"},
            "Reject when the paired confidence interval fails to show positive improvement.",
            "operator",
            ("unequal sample count", "different random seeds"),
            {
                "action_counts": [2, 4],
                "sample_sizes": [128],
                "shift_strengths": [0.0, 0.9],
                "noise_levels": [0.1],
                "coverage_floor": 0.02,
                "scenario_seed": 7,
            },
        ),
        novelty=NoveltyReview(
            ("importance sampling",),
            "It reconstructs a distinct conditional object before target marginalization.",
            False,
            0.6,
        ),
    )


class DiverseSelectionTest(unittest.TestCase):
    def test_pareto_dominance_requires_no_worse_on_every_axis(self):
        strong = ResearchObjectives(0.8, 0.8, 0.7, 0.8, 0.6)
        weak = ResearchObjectives(0.6, 0.7, 0.5, 0.7, 0.6)
        tradeoff = ResearchObjectives(0.9, 0.6, 0.9, 0.5, 0.8)
        self.assertTrue(dominates(strong, weak))
        self.assertFalse(dominates(strong, tradeoff))

    def test_near_duplicate_is_rejected_but_distinct_family_survives(self):
        archive = DiverseParetoArchive(min_distance=0.3, max_per_family=1)
        first = ArchiveEntry(
            record("Action stratified target", "estimation", "Condition by teammate action."),
            ResearchObjectives(0.8, 0.7, 0.6, 0.8, 0.8),
        )
        duplicate = ArchiveEntry(
            record("Action stratified targets", "estimation", "Condition on teammate actions."),
            ResearchObjectives(0.7, 0.7, 0.6, 0.8, 0.8),
        )
        exploration = ArchiveEntry(
            record("Coverage optimism", "exploration", "Explore uncertain joint actions safely."),
            ResearchObjectives(0.7, 0.7, 0.7, 0.8, 0.6),
        )
        self.assertLess(conceptual_distance(first.record, duplicate.record), 0.3)
        self.assertTrue(archive.add(first))
        self.assertFalse(archive.add(duplicate))
        self.assertTrue(archive.add(exploration))
        self.assertEqual({x.record.hypothesis.family for x in archive.frontier(2)}, {
            "estimation", "exploration"
        })

    def test_ucb_eventually_expands_less_visited_frontier_entry(self):
        archive = DiverseParetoArchive(min_distance=0.1, max_per_family=1)
        high = ArchiveEntry(
            record("High confidence estimator", "estimation", "Estimate conditional means robustly."),
            ResearchObjectives(0.9, 0.9, 0.7, 0.8, 0.8),
        )
        diverse = ArchiveEntry(
            record("Partner uncertainty", "modeling", "Model uncertainty in teammate intentions."),
            ResearchObjectives(0.7, 0.7, 0.9, 0.9, 0.7),
        )
        archive.add(high)
        archive.add(diverse)
        selections = [archive.select_for_expansion(capacity=2, exploration=1.0) for _ in range(8)]
        self.assertIn(high, selections)
        self.assertIn(diverse, selections)


if __name__ == "__main__":
    unittest.main()
