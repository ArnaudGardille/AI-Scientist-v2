from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ai_scientist.constrained.bundle import BundleValidationError, CandidateBundle


VALID = {
    "__init__.py": "from .estimator import estimate_operator\nfrom .sampler import sample_replay\n",
    "estimator.py": (
        "import numpy as np\n"
        "from conformal_marl.autoresearch.api import Estimate\n"
        "def estimate_operator(batch, rng):\n"
        "    return Estimate(float(np.mean(batch.outcomes)))\n"
    ),
    "sampler.py": (
        "from conformal_marl.samplers import register_sampler, uniform_sampler\n"
        "@register_sampler('autoresearch_candidate')\n"
        "def sample_replay(buffer, state, rng, ctx, batch_size):\n"
        "    return uniform_sampler(buffer, state, rng, ctx, batch_size)\n"
    ),
}


class CandidateBundleTest(unittest.TestCase):
    def test_valid_bundle_materializes_only_allowed_files(self):
        bundle = CandidateBundle(dict(VALID), "uniform baseline")
        bundle.validate()
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "candidate"
            bundle.materialize(target)
            self.assertEqual({path.name for path in target.iterdir()}, set(VALID))

    def test_forbidden_import_and_file_io_are_rejected(self):
        for bad_line in (
            "import os\n",
            "def x():\n    open('secret')\n",
            "from ..frozen import operator_benchmark\n",
            "from conformal_marl.autoresearch.frozen import operator_benchmark\n",
            "def x():\n    return globals()['__builtins__']\n",
            "import conformal_marl.samplers as samplers\nsamplers.uniform_sampler = None\n",
            (
                "from conformal_marl.samplers import register_sampler\n"
                "@register_sampler('uniform')\n"
                "def bad(*args):\n"
                "    return args\n"
            ),
        ):
            files = dict(VALID)
            files["estimator.py"] = bad_line
            with self.assertRaises(BundleValidationError):
                CandidateBundle(files).validate()

    def test_missing_or_extra_files_are_rejected(self):
        missing = dict(VALID)
        del missing["sampler.py"]
        with self.assertRaisesRegex(BundleValidationError, "missing"):
            CandidateBundle(missing).validate()
        extra = dict(VALID)
        extra["escape.py"] = "x = 1\n"
        with self.assertRaisesRegex(BundleValidationError, "forbidden files"):
            CandidateBundle(extra).validate()


if __name__ == "__main__":
    unittest.main()
