"""AI Scientist model adapter for constrained candidate proposals."""

from __future__ import annotations

import json
import re

from ai_scientist.constrained.search import CandidateNode, Proposal
from ai_scientist.llm import create_client, get_batch_responses_from_llm


class AIScientistProposer:
    def __init__(self, model: str, temperature: float = 0.7) -> None:
        self.client, self.model = create_client(model)
        self.temperature = temperature

    @staticmethod
    def _parse(response: str) -> Proposal:
        match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
        payload = json.loads(match.group(1) if match else response)
        return Proposal(hypothesis=payload["hypothesis"], code=payload["code"])

    def propose(
        self, parent: CandidateNode, *, count: int, task_context: str
    ) -> list[Proposal]:
        system = (
            "You are an ML researcher improving one constrained Python candidate. "
            "Return only JSON with string fields 'hypothesis' and 'code'. Never alter "
            "the benchmark, metrics, seeds, budgets, environment, or evaluation command."
        )
        prompt = f"""{task_context}

Parent hypothesis: {parent.hypothesis}
Parent status: {parent.status}
Parent score: {parent.score}
Frozen evaluator feedback: {parent.feedback}

Current candidate.py:
```python
{parent.code}
```

Propose one scientifically motivated, minimal candidate.py replacement. Preserve both
the operator-estimator API and registered learning sampler API. Explain the hypothesis
in one sentence inside the JSON field, not outside the JSON.
"""
        responses, _ = get_batch_responses_from_llm(
            prompt,
            self.client,
            self.model,
            system,
            temperature=self.temperature,
            n_responses=count,
        )
        proposals = []
        for response in responses:
            try:
                proposals.append(self._parse(response))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
        return proposals
