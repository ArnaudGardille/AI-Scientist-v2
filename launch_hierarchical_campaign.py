"""Launch the role-separated, diversity-aware BFTS research campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_scientist.constrained.bundle_evaluator import (
    BundleEvaluatorConfig,
    WorktreeBundleEvaluator,
)
from ai_scientist.constrained.campaign import CampaignConfig, HierarchicalCampaign
from ai_scientist.constrained.council import ResearchCouncil
from ai_scientist.constrained.search import EvaluationStage
from ai_scientist.constrained.structured_model import ClaudeCodeStructuredModel


def load_campaign(config_path: Path, model_name: str) -> HierarchicalCampaign:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    stages = tuple(
        EvaluationStage(
            name=stage["name"],
            command=tuple(stage["command"]),
            minimum_score=stage.get("minimum_score"),
            required_status=stage.get("required_status", "ok"),
        )
        for stage in raw["stages"]
    )
    evaluator = WorktreeBundleEvaluator(BundleEvaluatorConfig(
        repository=Path(raw["repository"]).resolve(),
        base_ref=raw["base_ref"],
        candidate_path=Path(raw["candidate_path"]),
        working_subdir=Path(raw["working_subdir"]),
        frozen_paths=tuple(Path(path) for path in raw["frozen_paths"]),
        stages=stages,
        timeout_seconds=raw.get("timeout_seconds", 3600),
        shared_python=Path(raw["shared_python"]).resolve() if raw.get("shared_python") else None,
    ))
    model = ClaudeCodeStructuredModel(
        model=model_name.removeprefix("claude-code/"),
        timeout_seconds=raw.get("model_timeout_seconds", 900),
    )
    campaign_raw = raw.get("campaign", {})
    return HierarchicalCampaign(
        council=ResearchCouncil(model),
        implementation_model=model,
        evaluator=evaluator,
        config=CampaignConfig(
            output_dir=Path(raw["output_dir"]).resolve(),
            initial_capacity=campaign_raw.get("initial_capacity", 5),
            max_nodes=campaign_raw.get("max_nodes", 15),
            revisions_per_expansion=campaign_raw.get("revisions_per_expansion", 2),
            max_per_family=campaign_raw.get("max_per_family", 2),
            min_conceptual_distance=campaign_raw.get("min_conceptual_distance", 0.2),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="claude-code/sonnet")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    problem = Path(raw["task_context_file"]).read_text(encoding="utf-8")
    campaign = load_campaign(args.config, args.model)
    best = campaign.run(problem)
    print(json.dumps({
        "best_node_id": best.node_id if best else None,
        "status": best.evaluation.status if best else "no_promotable_hypothesis",
        "priority": best.evaluation.priority() if best else None,
    }))


if __name__ == "__main__":
    main()
