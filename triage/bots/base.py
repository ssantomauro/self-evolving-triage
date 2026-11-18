"""What every triage bot looks like to the replay harness."""
from __future__ import annotations

from dataclasses import dataclass, field

from triage.dataset import Issue, Repo


@dataclass
class Prediction:
    labels: set[str]                                        # labels the bot applies
    uncertain: set[str] = field(default_factory=set)        # labels it refuses to call (flagged for a human)
    probs: dict[str, float] = field(default_factory=dict)   # optional, for analysis
    failed: bool = False                                    # the decision service gave no answer


class Bot:
    """One bot instance per repo. The harness calls predict() on a chunk of issues, then learn() with the
    maintainers' real labels for that same chunk, so a bot never sees the answer before deciding."""

    name = "bot"
    chunk_size = 1        # how many issues the harness hands over at once (lets API bots batch calls)

    def __init__(self, repo: Repo):
        self.repo = repo

    def predict(self, issues: list[Issue]) -> list[Prediction]:
        raise NotImplementedError

    def learn(self, issues: list[Issue]) -> None:
        """Called after predict() with the same issues; issue.truth is now fair game."""

    def snapshot(self) -> dict:
        """Human-readable state for the report (e.g. the current spec)."""
        return {}
