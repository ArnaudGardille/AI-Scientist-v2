"""Launch constrained BFTS against a repository-defined numeric benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_scientist.constrained.llm_proposer import AIScientistProposer
from ai_scientist.constrained.search import ConstrainedBFTS, EvaluationStage, SearchConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    stages = tuple(
        EvaluationStage(
            name=stage["name"],
            command=tuple(stage["command"]),
            minimum_score=stage.get("minimum_score"),
            required_status=stage.get("required_status", "ok"),
        )
        for stage in raw["stages"]
    )
    config = SearchConfig(
        repository=Path(raw["repository"]).resolve(),
        base_ref=raw["base_ref"],
        candidate_path=Path(raw["candidate_path"]),
        working_subdir=Path(raw["working_subdir"]),
        frozen_paths=tuple(Path(path) for path in raw["frozen_paths"]),
        stages=stages,
        output_dir=Path(raw["output_dir"]).resolve(),
        branching_factor=raw.get("branching_factor", 3),
        max_nodes=raw.get("max_nodes", 12),
        timeout_seconds=raw.get("timeout_seconds", 3600),
    )
    task_context = Path(raw["task_context_file"]).read_text(encoding="utf-8")
    best = ConstrainedBFTS(
        config, AIScientistProposer(args.model), task_context
    ).search()
    print(json.dumps({"best_node_id": best.node_id, "score": best.score, "status": best.status}))


if __name__ == "__main__":
    main()
