"""AI Scientist model adapter for constrained candidate proposals."""

from __future__ import annotations

import json
import os
import re

import anthropic
import openai

from ai_scientist.constrained.search import CandidateNode, Proposal


class AIScientistProposer:
    def __init__(self, model: str, temperature: float = 0.7) -> None:
        self.model = model
        self.temperature = temperature
        if model.startswith("ollama/"):
            self.client = openai.OpenAI(
                api_key=os.environ.get("OLLAMA_API_KEY", "ollama"),
                base_url="http://localhost:11434/v1",
            )
        elif model.startswith("claude-"):
            self.client = anthropic.Anthropic()
        elif model.startswith("bedrock/"):
            self.client = anthropic.AnthropicBedrock()
            self.model = model.split("/")[-1]
        else:
            self.client = openai.OpenAI()

    @staticmethod
    def _parse(response: str) -> Proposal:
        match = re.search(r"```json\s*(\{.*\})\s*```", response, re.DOTALL)
        payload = json.loads(match.group(1) if match else response)
        return Proposal(hypothesis=payload["hypothesis"], code=payload["code"])

    def _complete(self, system: str, prompt: str) -> str:
        if isinstance(self.client, (anthropic.Anthropic, anthropic.AnthropicBedrock)):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                temperature=self.temperature,
                system=system,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text
        response = self.client.chat.completions.create(
            model=self.model.removeprefix("ollama/"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content

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
        proposals = []
        for _ in range(count):
            try:
                response = self._complete(system, prompt)
                proposals.append(self._parse(response))
            except (KeyError, TypeError, json.JSONDecodeError, openai.OpenAIError):
                continue
        return proposals
