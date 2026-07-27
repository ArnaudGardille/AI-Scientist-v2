from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.constrained.bundle_evaluator import BundleEvaluation
from ai_scientist.constrained.campaign import CampaignConfig, HierarchicalCampaign
from ai_scientist.constrained.council import ResearchLens
from launch_hierarchical_campaign import result_payload
from tests.test_candidate_bundle import VALID
from tests.test_diverse_selection import record


class FakeCouncil:
    def generate(self, problem, lenses):
        del problem
        return [record(f"Hypothesis {lens.name}", lens.name, lens.instruction) for lens in lenses]

    def review(self, value, problem):
        del problem
        return value


class FakeImplementationModel:
    def __init__(self):
        self.calls = 0

    def complete(self, *, role, system, prompt, schema):
        del role, system, prompt, schema
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
                inputs and "mechanism" in inputs for inputs in evaluator.stage_inputs
            ))
            self.assertEqual(len({id(node.record) for node in campaign.nodes}), 4)
            for node in campaign.nodes:
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
