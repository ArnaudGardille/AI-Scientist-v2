from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from launch_hierarchical_campaign import _hash_git_paths


class LauncherProtocolTest(unittest.TestCase):
    def test_frozen_hash_uses_the_pinned_git_tree_not_worktree_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            repository = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@example.com"],
                cwd=repository,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=repository,
                check=True,
            )
            frozen = repository / "frozen"
            frozen.mkdir()
            (frozen / "benchmark.py").write_text("VALUE = 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "frozen"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "initial"],
                cwd=repository,
                check=True,
            )
            first_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            first_hash = _hash_git_paths(repository, first_sha, ["frozen"])

            (frozen / "benchmark.py").write_text("VALUE = 999\n", encoding="utf-8")
            (frozen / "benchmark.pyc").write_bytes(b"untracked")
            dirty_hash = _hash_git_paths(repository, first_sha, ["frozen"])

            self.assertEqual(dirty_hash, first_hash)

            subprocess.run(["git", "add", "frozen/benchmark.py"], cwd=repository, check=True)
            subprocess.run(
                ["git", "commit", "-qm", "change benchmark"],
                cwd=repository,
                check=True,
            )
            second_sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                text=True,
                capture_output=True,
                check=True,
            ).stdout.strip()
            second_hash = _hash_git_paths(repository, second_sha, ["frozen"])

            self.assertNotEqual(second_hash, first_hash)


if __name__ == "__main__":
    unittest.main()
