"""Validated multi-file candidate bundle for generated MARL methods."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


ALLOWED_FILES = frozenset({"__init__.py", "estimator.py", "sampler.py", "target_policy.py", "state.py"})
REQUIRED_FILES = frozenset({"__init__.py", "estimator.py", "sampler.py"})
ALLOWED_IMPORT_ROOTS = frozenset({
    "__future__", "dataclasses", "typing", "math", "numpy", "jax", "flax", "chex",
    "conformal_marl",
})
DENIED_CALLS = frozenset({
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "system", "popen", "run", "call", "check_call", "check_output",
    "load", "save", "savez", "savetxt", "fromfile", "tofile",
})


class BundleValidationError(ValueError):
    pass


@dataclass(frozen=True)
class CandidateBundle:
    files: dict[str, str]
    hypothesis_summary: str = ""

    def validate(self) -> None:
        names = set(self.files)
        if not REQUIRED_FILES <= names:
            raise BundleValidationError(
                f"candidate bundle is missing {sorted(REQUIRED_FILES - names)}"
            )
        if not names <= ALLOWED_FILES:
            raise BundleValidationError(f"candidate bundle contains forbidden files: {sorted(names - ALLOWED_FILES)}")
        if sum(len(content.encode()) for content in self.files.values()) > 80_000:
            raise BundleValidationError("candidate bundle exceeds 80 KB")
        for name, content in self.files.items():
            path = PurePosixPath(name)
            if path.name != name or path.is_absolute() or ".." in path.parts:
                raise BundleValidationError(f"unsafe candidate path: {name!r}")
            if not isinstance(content, str) or not content.strip():
                raise BundleValidationError(f"candidate file {name!r} is empty")
            self._validate_python(name, content)

    @staticmethod
    def _validate_python(name: str, content: str) -> None:
        try:
            tree = ast.parse(content, filename=name)
        except SyntaxError as exc:
            raise BundleValidationError(f"invalid Python in {name}: {exc}") from exc
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
                if any(root not in ALLOWED_IMPORT_ROOTS for root in roots):
                    raise BundleValidationError(f"forbidden import in {name}: {roots}")
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                root = (node.module or "").split(".", 1)[0]
                if root not in ALLOWED_IMPORT_ROOTS:
                    raise BundleValidationError(f"forbidden import in {name}: {root}")
            elif isinstance(node, ast.Call):
                function = node.func
                called = function.id if isinstance(function, ast.Name) else (
                    function.attr if isinstance(function, ast.Attribute) else ""
                )
                if called in DENIED_CALLS:
                    raise BundleValidationError(f"forbidden call in {name}: {called}")
            elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
                raise BundleValidationError(f"dunder attribute access is forbidden in {name}")

    @classmethod
    def from_directory(cls, directory: Path) -> "CandidateBundle":
        files = {
            path.name: path.read_text(encoding="utf-8")
            for path in directory.iterdir()
            if path.is_file() and path.name in ALLOWED_FILES
        }
        bundle = cls(files=files, hypothesis_summary="existing baseline")
        bundle.validate()
        return bundle

    def materialize(self, directory: Path) -> None:
        self.validate()
        directory.mkdir(parents=True, exist_ok=True)
        for path in directory.iterdir():
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                import shutil

                shutil.rmtree(path)
        for name, content in self.files.items():
            (directory / name).write_text(content, encoding="utf-8")
