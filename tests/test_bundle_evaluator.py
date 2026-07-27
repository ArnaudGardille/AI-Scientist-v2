from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_scientist.constrained.bundle import CandidateBundle
from ai_scientist.constrained.bundle_evaluator import (
    BundleEvaluation,
    BundleEvaluatorConfig,
    WorktreeBundleEvaluator,
)
from ai_scientist.constrained.search import EvaluationStage
from tests.test_candidate_bundle import VALID


class BundleEvaluatorTest(unittest.TestCase):
    def _repository(self, root: Path) -> Path:
        repo = root / "repo"
        (repo / "candidate").mkdir(parents=True)
        for name, content in VALID.items():
            (repo / "candidate" / name).write_text(content)
        (repo / "frozen.txt").write_text("immutable\n")
        (repo / "evaluate.py").write_text(
            """import argparse, json, os, re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output'); p.add_argument('--config'); p.add_argument('--tamper', action='store_true'); a=p.parse_args()
state=Path('candidate/state.py')
text=state.read_text() if state.exists() else 'SCORE = 0'
score=float(re.search(r'SCORE\\s*=\\s*([0-9.]+)', text).group(1))
config=json.loads(Path(a.config).read_text()) if a.config and Path(a.config).exists() else {}
if a.tamper: Path('frozen.txt').write_text('changed')
Path(a.output).write_text(json.dumps({'status':'ok','primary_score':score,'config':config,'secret_visible':'ANTHROPIC_API_KEY' in os.environ}))
"""
        )
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "-qm", "fixture",
            ],
            cwd=repo,
            check=True,
        )
        return repo

    def _config(
        self,
        repo: Path,
        *,
        tamper: bool = False,
        contributes_to_priority: bool = True,
    ) -> BundleEvaluatorConfig:
        command = [
            "{python}", "evaluate.py", "--output", "{output}",
            "--config", "{stage_config}",
        ]
        if tamper:
            command.append("--tamper")
        return BundleEvaluatorConfig(
            repository=repo,
            base_ref="HEAD",
            candidate_path=Path("candidate"),
            working_subdir=Path("."),
            frozen_paths=(Path("frozen.txt"), Path("evaluate.py")),
            stages=(
                EvaluationStage(
                    "gate",
                    tuple(command),
                    minimum_score=1.0,
                    contributes_to_priority=contributes_to_priority,
                ),
            ),
            shared_python=Path(sys.executable),
        )

    def test_bundle_is_evaluated_and_promoted_in_disposable_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repository(Path(tmp))
            files = dict(VALID)
            files["state.py"] = "SCORE = 2.0\n"
            with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "must-not-leak"}):
                result = WorktreeBundleEvaluator(self._config(repo)).evaluate(
                    CandidateBundle(files),
                    stage_inputs={"gate": {"coverage_floor": 0.02}},
                )
            self.assertEqual(result.status, "ok")
            self.assertTrue(result.complete)
            self.assertEqual(result.passed_stages, 1)
            self.assertEqual(
                result.stage_results["gate"]["config"], {"coverage_floor": 0.02}
            )
            self.assertFalse(result.stage_results["gate"]["secret_visible"])
            self.assertEqual((repo / "frozen.txt").read_text(), "immutable\n")

    def test_frozen_mutation_invalidates_otherwise_successful_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repository(Path(tmp))
            result = WorktreeBundleEvaluator(self._config(repo, tamper=True)).evaluate(
                CandidateBundle(dict(VALID))
            )
            self.assertEqual(result.status, "integrity_failure")
            self.assertFalse(result.complete)
            self.assertEqual((repo / "frozen.txt").read_text(), "immutable\n")

    def test_designer_feedback_redacts_targets_seeds_and_per_seed_outcomes(self):
        evaluation = BundleEvaluation(
            status="ok",
            stage_results={
                "operator": {
                    "status": "ok",
                    "primary_score": -0.1,
                    "paired_seeds": [7, 8],
                    "scenarios": {
                        "private": {
                            "true_backup": 12.0,
                            "mean_estimate": 11.0,
                            "bias": -1.0,
                            "mse": 1.5,
                            "variance": 0.5,
                            "paired_utility_delta": [1.0, 2.0],
                        }
                    },
                },
                "causal_drift": {
                    "status": "ok",
                    "primary_score": 0.1,
                    "paired_seeds": [9, 10],
                    "candidate_utility": {
                        "drift": [0.8, 0.9],
                        "no_drift": [0.7, 0.8],
                    },
                    "immediate_normalized_regret": 1.0,
                    "controls": {
                        "uniform": {
                            "drift_delta": [0.2, 0.3],
                            "causal_interaction": [0.1, 0.2],
                            "drift_screening_bound": 0.15,
                            "interaction_screening_bound": 0.05,
                            "no_drift_screening_bound": -0.01,
                        }
                    },
                },
            },
            passed_stages=1,
            complete=True,
        )

        feedback = json.dumps(evaluation.designer_feedback(), sort_keys=True)

        self.assertNotIn("true_backup", feedback)
        self.assertNotIn("paired_seeds", feedback)
        self.assertNotIn("paired_utility_delta", feedback)
        self.assertNotIn("candidate_utility", feedback)
        self.assertNotIn("drift_delta", feedback)
        self.assertNotIn('"causal_interaction"', feedback)
        self.assertNotIn("mean_estimate", feedback)
        self.assertNotIn('"bias"', feedback)
        self.assertIn('"mse": 1.5', feedback)
        self.assertIn('"interaction_screening_bound": 0.05', feedback)
        self.assertIn('"immediate_normalized_regret": 1.0', feedback)

    def test_diagnostic_stage_score_does_not_change_candidate_priority(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._repository(Path(tmp))
            files = dict(VALID)
            files["state.py"] = "SCORE = 2.0\n"
            result = WorktreeBundleEvaluator(
                self._config(repo, contributes_to_priority=False)
            ).evaluate(CandidateBundle(files))

            self.assertTrue(result.complete)
            self.assertEqual(result.ranking_score, 0.0)
            self.assertEqual(result.priority(), 1000.0)


if __name__ == "__main__":
    unittest.main()
