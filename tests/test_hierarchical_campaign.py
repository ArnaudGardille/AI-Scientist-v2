from __future__ import annotations

import copy
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from ai_scientist.constrained.bundle_evaluator import BundleEvaluation
from ai_scientist.constrained.campaign import CampaignConfig, HierarchicalCampaign
from ai_scientist.constrained.council import ResearchLens
from ai_scientist.constrained.research_schema import (
    ResearchPhase,
    ResearchRecord,
    RevisionAudit,
)
from launch_hierarchical_campaign import result_payload
from tests.test_candidate_bundle import VALID
from tests.test_diverse_selection import record


class FakeCouncil:
    def generate(self, problem, lenses):
        del problem
        return [record(f"Hypothesis {lens.name}", lens.name, lens.instruction) for lens in lenses]

    def review(self, value, problem, *, on_progress=None):
        del problem
        if on_progress is not None:
            on_progress(value)
        return value


class RejectingCouncil(FakeCouncil):
    def review(self, value, problem, *, on_progress=None):
        del problem
        value.novelty = replace(
            value.novelty,
            likely_incremental=True,
            novelty_score=0.2,
        )
        if on_progress is not None:
            on_progress(value)
        return value


class RevisionCouncil(RejectingCouncil):
    def revise(
        self,
        parent,
        problem,
        *,
        failed_checks,
        revision_index,
    ):
        del problem, failed_checks, revision_index
        revised = record(
            "Revised conditional estimator",
            parent.hypothesis.family,
            "Use a support-aware conditional control variate with an explicit fallback.",
        )
        revised.revision_audit = RevisionAudit(
            preserves_objective=True,
            substantive_mechanism_change=True,
            addresses_reviewed_failure=True,
            objective_analysis="The child retains target-policy replay efficiency as its objective.",
            mechanism_analysis="The child adds a support-aware control variate and fallback mechanism.",
            failure_resolution_analysis="The fallback directly resolves the reviewed support failure.",
        )
        revised.provenance.append(
            {
                "role": "concept-reviser",
                "action": "revised",
                "request_id": "revision:test",
                "parent_hypothesis_id": parent.hypothesis.hypothesis_id,
            }
        )
        return revised

    def review(self, value, problem, *, on_progress=None):
        if value.hypothesis.title == "Revised conditional estimator":
            return FakeCouncil.review(
                self,
                value,
                problem,
                on_progress=on_progress,
            )
        return super().review(
            value,
            problem,
            on_progress=on_progress,
        )


class InterruptingCouncil:
    def __init__(self):
        self.template = record(
            "Checkpointed hypothesis",
            "estimation",
            "Condition on teammate action and resume each scientific role.",
        )
        self.generate_calls = 0

    def generate(self, problem, lenses):
        del problem, lenses
        self.generate_calls += 1
        return [ResearchRecord(hypothesis=self.template.hypothesis)]

    def review(self, value, problem, *, on_progress=None):
        del problem
        value.theory = self.template.theory
        value.phase = ResearchPhase.THEORY
        if on_progress is not None:
            on_progress(value)
        raise RuntimeError("simulated interruption after theory")


class ResumingCouncil(InterruptingCouncil):
    def generate(self, problem, lenses):
        del problem, lenses
        raise AssertionError("checkpointed hypothesis must not be generated again")

    def review(self, value, problem, *, on_progress=None):
        del problem
        if value.theory != self.template.theory:
            raise AssertionError("theory checkpoint was not restored")
        value.falsification = self.template.falsification
        value.phase = ResearchPhase.FALSIFICATION
        if on_progress is not None:
            on_progress(value)
        value.experiment = self.template.experiment
        value.phase = ResearchPhase.EXPERIMENT
        if on_progress is not None:
            on_progress(value)
        value.novelty = self.template.novelty
        if on_progress is not None:
            on_progress(value)
        return value


class FakeImplementationModel:
    def __init__(self):
        self.calls = 0

    def complete(self, *, role, system, prompt, schema, request_id=None):
        del role, system, prompt, schema, request_id
        self.calls += 1
        files = dict(VALID)
        files["state.py"] = f"REVISION = {self.calls}\n"
        return {"hypothesis_summary": f"revision {self.calls}", "files": files}


class FakeEvaluator:
    def __init__(self):
        self.calls = 0
        self.stage_inputs = []

    def evaluate(self, bundle, stage_inputs=None):
        bundle.validate()
        self.calls += 1
        self.stage_inputs.append(stage_inputs)
        return BundleEvaluation(
            status="ok",
            stage_results={"learning": {"status": "ok", "primary_score": self.calls / 10}},
            passed_stages=3,
            complete=True,
        )


class FakeFinalEvaluator:
    def __init__(self):
        self.calls = 0

    def evaluate(self, bundle, stage_inputs=None):
        self.assert_no_inputs(stage_inputs)
        bundle.validate()
        self.calls += 1
        return BundleEvaluation(
            status="ok",
            stage_results={"held_out": {"status": "ok", "primary_score": -1000.0}},
            passed_stages=1,
            complete=True,
        )

    @staticmethod
    def assert_no_inputs(stage_inputs):
        if stage_inputs is not None:
            raise AssertionError("held-out evaluator received search experiment inputs")


class RejectingFinalEvaluator(FakeFinalEvaluator):
    def evaluate(self, bundle, stage_inputs=None):
        self.assert_no_inputs(stage_inputs)
        bundle.validate()
        self.calls += 1
        return BundleEvaluation(
            status="not_promoted:heldout_learning",
            stage_results={
                "held_out": {"status": "ok", "primary_score": -0.2}
            },
            passed_stages=0,
            complete=False,
        )


class HierarchicalCampaignTest(unittest.TestCase):
    def test_campaign_materializes_auditable_tree_and_expands_multiple_families(self):
        with tempfile.TemporaryDirectory() as tmp:
            evaluator = FakeEvaluator()
            final_evaluator = FakeFinalEvaluator()
            campaign = HierarchicalCampaign(
                council=FakeCouncil(),
                implementation_model=FakeImplementationModel(),
                evaluator=evaluator,
                final_evaluator=final_evaluator,
                config=CampaignConfig(
                    output_dir=Path(tmp),
                    initial_capacity=2,
                    max_nodes=4,
                    revisions_per_expansion=1,
                    max_per_family=1,
                    finalist_count=1,
                    experiment_stage_name="designed_diagnostic",
                ),
            )
            best = campaign.run(
                "test problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),
                        ResearchLens("exploration", "preserve uncertain joint action coverage")),
            )
            self.assertEqual(len(campaign.nodes), 4)
            self.assertEqual(best.node_id, 3)
            self.assertTrue(best.evaluation.complete)
            self.assertIsNotNone(best.final_evaluation)
            self.assertEqual(final_evaluator.calls, 1)
            self.assertTrue(all(
                inputs and "designed_diagnostic" in inputs
                for inputs in evaluator.stage_inputs
            ))
            self.assertEqual(len({id(node.record) for node in campaign.nodes}), 4)
            for node in campaign.nodes:
                self.assertEqual(node.record.phase, ResearchPhase.SCREENING)
                self.assertEqual(
                    node.record.empirical_results["learning"]["primary_score"],
                    node.evaluation.stage_results["learning"]["primary_score"],
                )
            self.assertTrue((Path(tmp) / "journal.json").is_file())
            journal = (Path(tmp) / "journal.json").read_text()
            self.assertIn('"accepted": true', journal)
            self.assertTrue((Path(tmp) / "node_0000" / "candidate" / "sampler.py").is_file())
            self.assertTrue(any(
                (Path(tmp) / f"node_{node.node_id:04d}" / "final_evaluation.json").is_file()
                for node in campaign.nodes
            ))
            self.assertEqual({node.record.hypothesis.family for node in campaign.nodes[:2]}, {
                "estimation", "exploration"
            })

    def test_rejected_concepts_are_persisted_with_explicit_gate_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            implementation_model = FakeImplementationModel()
            campaign = HierarchicalCampaign(
                council=RejectingCouncil(),
                implementation_model=implementation_model,
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=Path(tmp),
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )

            best = campaign.run(
                "test problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),),
            )

            self.assertIsNone(best)
            self.assertEqual(implementation_model.calls, 0)
            concept_paths = list((Path(tmp) / "concepts").glob("*.json"))
            self.assertEqual(len(concept_paths), 1)
            concept = json.loads(concept_paths[0].read_text())
            self.assertFalse(concept["gate"]["promotable"])
            self.assertEqual(
                concept["gate"]["failed_checks"],
                ["novelty_score", "material_novelty"],
            )
            self.assertIn("record", concept)
            self.assertEqual(concept["record"]["novelty"]["novelty_score"], 0.2)

            journal = json.loads((Path(tmp) / "journal.json").read_text())
            self.assertEqual(journal["nodes"], [])
            self.assertEqual(len(journal["concepts"]), 1)
            self.assertEqual(
                journal["concepts"][0]["failed_checks"],
                ["novelty_score", "material_novelty"],
            )

    def test_resumes_after_a_role_boundary_without_regenerating_the_hypothesis(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            first_council = InterruptingCouncil()
            first = HierarchicalCampaign(
                council=first_council,
                implementation_model=FakeImplementationModel(),
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=output_dir,
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )
            lenses = (ResearchLens("estimation", "condition on teammate action"),)

            with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                first.run("test problem", lenses=lenses)

            checkpoint_path = output_dir / "checkpoints" / "concept_0000.json"
            checkpoint = json.loads(checkpoint_path.read_text())
            self.assertIsNotNone(checkpoint["record"]["theory"])
            self.assertIsNone(checkpoint["record"]["falsification"])
            self.assertEqual(first_council.generate_calls, 1)

            implementation_model = FakeImplementationModel()
            resumed = HierarchicalCampaign(
                council=ResumingCouncil(),
                implementation_model=implementation_model,
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=output_dir,
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )
            best = resumed.run("test problem", lenses=lenses)

            self.assertIsNotNone(best)
            self.assertEqual(implementation_model.calls, 1)
            restored = json.loads(checkpoint_path.read_text())
            self.assertIsNotNone(restored["record"]["novelty"])
            self.assertEqual(len(resumed.concepts), 1)

    def test_bounded_concept_revision_repasses_the_gate_before_implementation(self):
        with tempfile.TemporaryDirectory() as tmp:
            implementation_model = FakeImplementationModel()
            campaign = HierarchicalCampaign(
                council=RevisionCouncil(),
                implementation_model=implementation_model,
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=Path(tmp),
                    initial_capacity=1,
                    max_nodes=1,
                    max_concept_revisions=1,
                ),
            )

            best = campaign.run(
                "test problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),),
            )

            self.assertIsNotNone(best)
            self.assertEqual(implementation_model.calls, 1)
            self.assertEqual(len(campaign.concepts), 2)
            self.assertFalse(campaign.concepts[0].gate["promotable"])
            self.assertTrue(campaign.concepts[1].gate["promotable"])
            journal = json.loads((Path(tmp) / "journal.json").read_text())
            self.assertEqual(
                journal["concepts"][1]["parent_hypothesis_id"],
                campaign.concepts[0].record.hypothesis.hypothesis_id,
            )
            failed_audit = copy.deepcopy(campaign.concepts[1].record)
            failed_audit.revision_audit = replace(
                failed_audit.revision_audit,
                addresses_reviewed_failure=False,
            )
            self.assertIn(
                "revision_addresses_failure",
                failed_audit.conceptual_gate()["failed_checks"],
            )

    def test_resume_rejects_changed_problem_or_lenses(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            campaign = HierarchicalCampaign(
                council=FakeCouncil(),
                implementation_model=FakeImplementationModel(),
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=output_dir,
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )
            campaign.run(
                "original problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),),
            )

            changed = HierarchicalCampaign(
                council=FakeCouncil(),
                implementation_model=FakeImplementationModel(),
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=output_dir,
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                changed.run(
                    "changed problem",
                    lenses=(ResearchLens("estimation", "condition on teammate action"),),
                )

            changed_protocol = HierarchicalCampaign(
                council=FakeCouncil(),
                implementation_model=FakeImplementationModel(),
                evaluator=FakeEvaluator(),
                final_evaluator=None,
                config=CampaignConfig(
                    output_dir=output_dir,
                    initial_capacity=1,
                    max_nodes=1,
                    resume_protocol={"model": "different"},
                ),
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                changed_protocol.run(
                    "original problem",
                    lenses=(ResearchLens("estimation", "condition on teammate action"),),
                )

    def test_rejects_using_heldout_evaluation_for_model_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "confirmatory"):
                CampaignConfig(
                    output_dir=Path(tmp),
                    initial_capacity=2,
                    max_nodes=2,
                    finalist_count=2,
                ).validate()

    def test_failed_heldout_is_reported_as_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            campaign = HierarchicalCampaign(
                council=FakeCouncil(),
                implementation_model=FakeImplementationModel(),
                evaluator=FakeEvaluator(),
                final_evaluator=RejectingFinalEvaluator(),
                config=CampaignConfig(
                    output_dir=Path(tmp),
                    initial_capacity=1,
                    max_nodes=1,
                ),
            )
            best = campaign.run(
                "test problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),),
            )

            payload = result_payload(best)

            self.assertFalse(payload["accepted"])
            self.assertEqual(payload["status"], "rejected:heldout")
            self.assertEqual(
                payload["heldout_status"], "not_promoted:heldout_learning"
            )


if __name__ == "__main__":
    unittest.main()
