from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.constrained.bundle_evaluator import BundleEvaluation
from ai_scientist.constrained.campaign import CampaignConfig, HierarchicalCampaign
from ai_scientist.constrained.council import ResearchLens
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
            stage_results={"held_out": {"status": "ok", "primary_score": 1.0 / self.calls}},
            passed_stages=1,
            complete=True,
        )

    @staticmethod
    def assert_no_inputs(stage_inputs):
        if stage_inputs is not None:
            raise AssertionError("held-out evaluator received search experiment inputs")


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
                    finalist_count=2,
                ),
            )
            best = campaign.run(
                "test problem",
                lenses=(ResearchLens("estimation", "condition on teammate action"),
                        ResearchLens("exploration", "preserve uncertain joint action coverage")),
            )
            self.assertEqual(len(campaign.nodes), 4)
            self.assertTrue(best.evaluation.complete)
            self.assertIsNotNone(best.final_evaluation)
            self.assertEqual(final_evaluator.calls, 2)
            self.assertTrue(all(
                inputs and "mechanism" in inputs for inputs in evaluator.stage_inputs
            ))
            self.assertTrue((Path(tmp) / "journal.json").is_file())
            self.assertTrue((Path(tmp) / "node_0000" / "candidate" / "sampler.py").is_file())
            self.assertTrue(any(
                (Path(tmp) / f"node_{node.node_id:04d}" / "final_evaluation.json").is_file()
                for node in campaign.nodes
            ))
            self.assertEqual({node.record.hypothesis.family for node in campaign.nodes[:2]}, {
                "estimation", "exploration"
            })


if __name__ == "__main__":
    unittest.main()
