from __future__ import annotations

import unittest

from ai_scientist.constrained.council import ResearchCouncil, ResearchLens


class FakeStructuredModel:
    def __init__(self):
        self.roles = []
        self.accepted = []
        self.rejected = []

    def accept_response(self, request_id):
        self.accepted.append(request_id)

    def reject_response(self, request_id, reason):
        self.rejected.append((request_id, reason))

    def complete(self, *, role, system, prompt, schema, request_id=None):
        del system, prompt, schema, request_id
        self.roles.append(role)
        if role.startswith("hypothesis-generator"):
            return {
                "title": "Conditional control variates for teammate drift",
                "family": role.split(":", 1)[1],
                "mechanism": "Estimate stationary action-conditioned residuals and remove drift variance.",
                "formal_claim": "A conditionally centered control variate lowers target-backup variance.",
                "assumptions": ["Finite teammate actions", "Stationary conditional dynamics"],
                "predictions": ["Lower MSE under drift", "No loss when behavior equals target"],
                "falsifiers": ["Residual bias remains", "Variance exceeds exact importance sampling"],
                "novelty_claim": "Combines target reconstruction with learned conditional residual control.",
                "complexity": "medium",
            }
        if role == "theorist":
            return {
                "sound": True,
                "derivation": "Condition on teammate action, center each residual, and marginalize under target.",
                "hidden_assumptions": ["Every relevant stratum has coverage"],
                "bias_analysis": "Unbiased with exact conditional means and complete target support coverage.",
                "variance_analysis": "Control variates reduce within-stratum noise when correlated with outcomes.",
                "support_analysis": "Missing target strata require an explicit conservative fallback estimator.",
                "soundness_score": 0.8,
            }
        if role == "falsifier":
            return {
                "fatal": False,
                "counterexamples": ["A target-only action is absent", "Control is uncorrelated with outcome"],
                "decisive_test": "Sweep minimum stratum coverage and compare paired MSE against SNIS.",
                "failure_signature": "MSE rises sharply as target mass moves to an unseen teammate action.",
                "robustness_score": 0.65,
            }
        if role == "experimentalist":
            return {
                "question": "Does conditional residual centering improve MSE beyond action stratification?",
                "manipulated_variables": ["policy divergence", "stratum coverage"],
                "controls": ["SNIS", "plain stratification"],
                "metrics": ["MSE", "bias"],
                "expected_signatures": {"mechanism": "lower within-stratum variance", "null": "equal MSE"},
                "rejection_rule": "Reject if the paired 95 percent upper bound on MSE improvement is non-positive.",
                "budget_class": "operator",
                "confounds": ["unequal sample count", "target support mismatch"],
                "mechanism_config": {
                    "action_counts": [2, 4, 8],
                    "sample_sizes": [128, 512],
                    "shift_strengths": [0.0, 0.5, 0.9],
                    "noise_levels": [0.1, 0.75],
                    "coverage_floor": 0.02,
                    "scenario_seed": 7,
                },
            }
        if role == "novelty-reviewer":
            return {
                "nearest_methods": ["stratified sampling", "doubly robust off-policy evaluation"],
                "material_difference": "The proposed control is conditioned on teammate action drift in MARL.",
                "likely_incremental": False,
                "novelty_score": 0.55,
            }
        raise AssertionError(role)


class ResearchCouncilTest(unittest.TestCase):
    def test_full_council_produces_promotable_non_code_record(self):
        council = ResearchCouncil(FakeStructuredModel())
        records = council.generate(
            "Off-policy teammate drift",
            lenses=(ResearchLens("statistics", "derive estimator"),),
        )
        self.assertEqual(len(records), 1)
        reviewed = council.review(records[0], "Off-policy teammate drift")
        self.assertTrue(reviewed.conceptually_promotable())
        self.assertIsNone(reviewed.empirical_results.get("code"))
        self.assertEqual(len(reviewed.provenance), 5)

    def test_invalid_hypothesis_is_rejected_before_review(self):
        model = FakeStructuredModel()
        original = model.complete

        def invalid(**kwargs):
            payload = original(**kwargs)
            if kwargs["role"].startswith("hypothesis-generator"):
                payload["predictions"] = []
            return payload

        model.complete = invalid
        council = ResearchCouncil(model)
        with self.assertRaisesRegex(ValueError, "predictions"):
            council.generate(
                "Off-policy teammate drift",
                lenses=(ResearchLens("statistics", "derive estimator"),),
            )
        self.assertEqual(
            model.rejected,
            [("hypothesis:statistics", "ValueError")],
        )
        self.assertEqual(model.accepted, [])

    def test_review_resumes_after_a_completed_role_without_repeating_it(self):
        model = FakeStructuredModel()
        council = ResearchCouncil(model)
        record = council.generate(
            "Off-policy teammate drift",
            lenses=(ResearchLens("statistics", "derive estimator"),),
        )[0]

        def interrupt_after_theory(current):
            if current.theory is not None and current.falsification is None:
                raise RuntimeError("interrupt after theory")

        with self.assertRaisesRegex(RuntimeError, "interrupt after theory"):
            council.review(
                record,
                "Off-policy teammate drift",
                on_progress=interrupt_after_theory,
            )

        reviewed = council.review(record, "Off-policy teammate drift")

        self.assertTrue(reviewed.conceptually_promotable())
        self.assertEqual(model.roles.count("theorist"), 1)
        self.assertEqual(model.roles.count("falsifier"), 1)
        self.assertEqual(model.roles.count("experimentalist"), 1)
        self.assertEqual(model.roles.count("novelty-reviewer"), 1)


if __name__ == "__main__":
    unittest.main()
