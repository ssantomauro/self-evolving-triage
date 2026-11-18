"""Asks Claude to propose a better spec (instructions + one definition per label) from recent mistakes.

The proposal is only a candidate: the learner keeps it only if it scores better on held-out issues.
Responses are cached in data/issues.db so a rerun makes no new Claude calls.
Needs Anthropic credentials (ANTHROPIC_API_KEY or an `ant auth login` profile).
"""
from __future__ import annotations

import hashlib
import json

import anthropic
from pydantic import BaseModel

from triage import db

MODEL = "claude-opus-5-5"


class LabelDefinition(BaseModel):
    label: str
    definition: str


class SpecProposal(BaseModel):
    instructions: str
    definitions: list[LabelDefinition]


PROMPT = """You maintain the labeling rules of an automatic issue triage bot for the GitHub repository {repo}.
A decision model reads each new issue and, for every label, decides whether it applies, using only:
- the general instructions, and
- a one-line definition per label.

Current instructions:
{instructions}

Current label definitions:
{definitions}

Below are recent issues where the bot disagreed with the maintainers. "missed" = maintainers applied the label
but the bot didn't; "wrong" = the bot applied it but maintainers didn't.

{mistakes}

Rewrite the instructions and the definitions so the bot matches how THIS repository's maintainers actually
label issues. Rules:
- One definition per label listed above, same label names. Keep each definition under 40 words.
- Describe the maintainers' real convention (what counts, and what doesn't count when two labels are easy to confuse).
- Generalize from patterns; don't mention specific issue numbers or titles.
- Keep the instructions under 80 words."""


class Rewriter:
    def __init__(self):
        self.client = anthropic.Anthropic()
        self.conn = db.connect()
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0

    def propose(self, repo: str, instructions: str, definitions: dict[str, str], mistakes: str) -> SpecProposal:
        prompt = PROMPT.format(
            repo=repo, instructions=instructions, mistakes=mistakes,
            definitions="\n".join(f"- {l}: {d or '(no definition)'}" for l, d in definitions.items()),
        )
        key = hashlib.sha256(f"{MODEL}\n{prompt}".encode()).hexdigest()
        hit = self.conn.execute("SELECT response FROM llm_cache WHERE key=?", (key,)).fetchone()
        if hit:
            return SpecProposal.model_validate_json(hit["response"])

        response = self.client.beta.messages.parse(
            model=MODEL,
            max_tokens=16000,
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            extra_body={"fallbacks": "default"},     # re-run on a fallback model if this one declines
            messages=[{"role": "user", "content": prompt}],
            output_format=SpecProposal,
        )
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"Claude did not return a proposal (stop_reason={response.stop_reason})")
        self.calls += 1
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens
        proposal = response.parsed_output
        self.conn.execute("INSERT OR REPLACE INTO llm_cache VALUES (?,?)", (key, proposal.model_dump_json()))
        self.conn.commit()
        return proposal
