"""Launch the role-separated, diversity-aware BFTS research campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from ai_scientist.constrained.bundle_evaluator import (
    BundleEvaluatorConfig,
    WorktreeBundleEvaluator,
)
from ai_scientist.constrained.campaign import CampaignConfig, HierarchicalCampaign
from ai_scientist.constrained.council import ResearchCouncil
from ai_scientist.constrained.council import ResearchLens
from ai_scientist.constrained.search import EvaluationStage
from ai_scientist.constrained.structured_model import ClaudeCodeStructuredModel


def _stages(raw_stages: list[dict]) -> tuple[EvaluationStage, ...]:
    return tuple(
        EvaluationStage(
            name=stage["name"],
            command=tuple(stage["command"]),
            minimum_score=stage.get("minimum_score"),
            required_status=stage.get("required_status", "ok"),
            contributes_to_priority=stage.get("contributes_to_priority", True),
        )
        for stage in raw_stages
    )


def _evaluator(raw: dict, stages: tuple[EvaluationStage, ...]) -> WorktreeBundleEvaluator:
    return WorktreeBundleEvaluator(BundleEvaluatorConfig(
        repository=Path(raw["repository"]).resolve(),
        base_ref=raw["base_ref"],
        candidate_path=Path(raw["candidate_path"]),
        working_subdir=Path(raw["working_subdir"]),
        frozen_paths=tuple(Path(path) for path in raw["frozen_paths"]),
        stages=stages,
        timeout_seconds=raw.get("timeout_seconds", 3600),
        # Do not resolve the virtualenv's Python symlink: its original path is how
        # CPython discovers pyvenv.cfg and the environment's site-packages.
        shared_python=Path(raw["shared_python"]).absolute() if raw.get("shared_python") else None,
    ))


def _hash_git_paths(repository: Path, commit_sha: str, paths: list[str]) -> str:
    tree = subprocess.run(
        [
            "git",
            "-C",
            str(repository),
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            commit_sha,
            "--",
            *sorted(paths),
        ],
        capture_output=True,
        check=True,
    ).stdout
    return hashlib.sha256(tree).hexdigest()


def load_campaign(config_path: Path, model_name: str) -> HierarchicalCampaign:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    repository = Path(raw["repository"]).resolve()
    base_ref_sha = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", f"{raw['base_ref']}^{{commit}}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    resolved_raw = {**raw, "base_ref": base_ref_sha}
    stages = _stages(raw["stages"])
    evaluator = _evaluator(resolved_raw, stages)
    final_evaluator = (
        _evaluator(resolved_raw, _stages(raw["final_stages"]))
        if raw.get("final_stages")
        else None
    )
    model = ClaudeCodeStructuredModel(
        model=model_name.removeprefix("claude-code/"),
        timeout_seconds=raw.get("model_timeout_seconds", 900),
        max_calls=raw.get("max_model_calls", 128),
        max_wallclock_seconds=raw.get("max_campaign_wallclock_seconds", 21_600),
        max_retries=raw.get("model_max_retries", 2),
        retry_base_seconds=raw.get("model_retry_base_seconds", 1.0),
        telemetry_path=Path(raw["output_dir"]).resolve() / "model_telemetry.json",
    )
    campaign_raw = raw.get("campaign", {})
    claude_version = subprocess.run(
        ["claude", "--version"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    resume_protocol = {
        "model_name": model_name,
        "claude_cli_version": claude_version,
        "python_version": sys.version,
        "target_base_ref_sha": base_ref_sha,
        "frozen_snapshot_sha256": _hash_git_paths(
            repository,
            base_ref_sha,
            raw["frozen_paths"],
        ),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "configuration": {
            key: value for key, value in raw.items() if key != "output_dir"
        },
    }
    return HierarchicalCampaign(
        council=ResearchCouncil(model),
        implementation_model=model,
        evaluator=evaluator,
        final_evaluator=final_evaluator,
        config=CampaignConfig(
            output_dir=Path(raw["output_dir"]).resolve(),
            initial_capacity=campaign_raw.get("initial_capacity", 5),
            max_nodes=campaign_raw.get("max_nodes", 15),
            revisions_per_expansion=campaign_raw.get("revisions_per_expansion", 2),
            max_per_family=campaign_raw.get("max_per_family", 2),
            min_conceptual_distance=campaign_raw.get("min_conceptual_distance", 0.2),
            finalist_count=campaign_raw.get("finalist_count", 1),
            experiment_stage_name=campaign_raw.get(
                "experiment_stage_name", "mechanism"
            ),
            max_concept_revisions=campaign_raw.get("max_concept_revisions", 0),
            resume_protocol=resume_protocol,
        ),
    )


def result_payload(best) -> dict:
    heldout = best.final_evaluation if best else None
    accepted = bool(heldout and heldout.complete)
    return {
        "best_node_id": best.node_id if best else None,
        "status": (
            "accepted"
            if accepted
            else "rejected:heldout"
            if heldout
            else best.evaluation.status
            if best
            else "no_promotable_hypothesis"
        ),
        "accepted": accepted,
        "screening_status": best.evaluation.status if best else None,
        "screening_priority": best.evaluation.priority() if best else None,
        "heldout_status": heldout.status if heldout else None,
        "heldout_priority": heldout.priority() if heldout else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="claude-code/sonnet")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    problem = Path(raw["task_context_file"]).read_text(encoding="utf-8")
    lenses = tuple(
        ResearchLens(item["name"], item["instruction"])
        for item in raw.get("research_lenses", [])
    )
    campaign = load_campaign(args.config, args.model)
    best = campaign.run(problem, lenses=lenses) if lenses else campaign.run(problem)
    print(json.dumps(result_payload(best)))


if __name__ == "__main__":
    main()
