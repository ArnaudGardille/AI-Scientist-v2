from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ai_scientist.constrained.search import (
    ConstrainedBFTS,
    EvaluationStage,
    Proposal,
    SearchConfig,
)


class FixedProposer:
    def propose(self, parent, *, count, task_context):
        del parent, count, task_context
        return [Proposal("increase score", "SCORE = 2.0\n")]


class TamperingProposer:
    def propose(self, parent, *, count, task_context):
        del parent, count, task_context
        return [
            Proposal(
                "tamper",
                "from pathlib import Path\n"
                "Path('frozen.txt').write_text('changed')\n"
                "SCORE = 99.0\n",
            )
        ]


class ConstrainedSearchTest(unittest.TestCase):
    def _make_repo(self, root: Path) -> Path:
        repo = root / "repo"
        repo.mkdir()
        (repo / "candidate.py").write_text("SCORE = 0.0\n")
        (repo / "frozen.txt").write_text("immutable\n")
        (repo / "evaluate.py").write_text(
            """import argparse, json
from candidate import SCORE
p=argparse.ArgumentParser(); p.add_argument('--output'); a=p.parse_args()
open(a.output,'w').write(json.dumps({'status':'ok','primary_score':SCORE}))
"""
        )
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            [
                "git", "-c", "user.name=Test", "-c", "user.email=test@example.com",
                "commit", "-qm", "root",
            ],
            cwd=repo,
            check=True,
        )
        return repo

    def _config(self, repo: Path, output_dir: Path) -> SearchConfig:
        return SearchConfig(
            repository=repo,
            base_ref="HEAD",
            candidate_path=Path("candidate.py"),
            working_subdir=Path("."),
            frozen_paths=(Path("frozen.txt"), Path("evaluate.py")),
            stages=(
                EvaluationStage(
                    "score", (sys.executable, "evaluate.py", "--output", "{output}")
                ),
            ),
            output_dir=output_dir,
            branching_factor=1,
            max_nodes=2,
        )

    def test_best_first_evaluates_candidate_in_isolated_worktree(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(Path(tmp))
            config = self._config(repo, Path(tmp) / "out")
            search = ConstrainedBFTS(config, FixedProposer(), "test")
            best = search.search()
            self.assertEqual(best.score, 1002.0)
            self.assertEqual(best.node_id, 1)
            journal = json.loads((config.output_dir / "journal.json").read_text())
            self.assertEqual(journal["best_node_id"], 1)
            self.assertEqual((repo / "candidate.py").read_text(), "SCORE = 0.0\n")

    def test_frozen_file_tampering_cannot_win(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = self._make_repo(Path(tmp))
            config = self._config(repo, Path(tmp) / "out")
            search = ConstrainedBFTS(config, TamperingProposer(), "test")
            best = search.search()
            self.assertEqual(best.node_id, 0)
            self.assertEqual(search.nodes[1].status, "integrity_failure")
            self.assertEqual((repo / "frozen.txt").read_text(), "immutable\n")


if __name__ == "__main__":
    unittest.main()
