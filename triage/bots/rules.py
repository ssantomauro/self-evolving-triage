"""Version A: hand-written keyword rules, the way most teams set up an auto-labeler.

Two sources of patterns:
  1. A generic table for the usual issue types (bug, feature, docs, question, ...), written once.
  2. Keywords taken from each label's own name ("area/solver" -> "solver", "IO CSV" -> "csv").
It never changes while running: improving it means a developer editing this file.
"""
from __future__ import annotations

import re

from triage.bots.base import Bot, Prediction

GENERIC = {
    "bug": [r"\bbug\b", r"\berror\b", r"exception", r"traceback", r"\bcrash", r"doesn'?t work", r"\bbroken\b",
            r"\bfails?\b", r"unexpected", r"incorrect", r"wrong"],
    "feature": [r"feature request", r"\bfeature\b", r"would be nice", r"\bsupport for\b", r"\badd (a|an|support)\b",
                r"\ballow\b", r"\bproposal\b", r"\benhancement\b"],
    "docs": [r"\bdocs?\b", r"documentation", r"readme", r"\btypo\b", r"docstring example"],
    "question": [r"^how (do|can|to|should)\b", r"\bquestion\b", r"\bis it possible\b", r"\bhelp\b"],
    "performance": [r"\bslow\b", r"performance", r"\bspeed\b", r"memory usage", r"takes too long"],
    "regression": [r"regression", r"used to work", r"worked in", r"since (version|v)?\s?\d"],
    "crash": [r"\bcrash", r"segfault", r"internal error", r"cannot parse", r"failed to parse"],
    "maintenance": [r"\bci\b", r"github actions", r"\btests?\b", r"\bbump\b", r"\brefactor"],
    "deprecation": [r"deprecat"],
    "packaging": [r"\bwheel\b", r"\bpackag", r"\binstall(ation)?\b", r"\bpypi\b"],
    "configuration": [r"pyproject", r"\bconfig", r"\boption\b", r"\bflag\b"],
    "style": [r"\bformat(ting|ted|s)?\b", r"\bstyle\b"],
}

# Which generic group a label belongs to, matched on words in the label name.
GROUP_HINTS = [
    ("bug", ["bug"]), ("feature", ["feature", "enhancement"]), ("docs", ["doc", "documentation"]),
    ("question", ["question", "support"]), ("performance", ["performance"]), ("regression", ["regression"]),
    ("crash", ["crash"]), ("maintenance", ["maintenance"]), ("deprecation", ["deprecation"]),
    ("packaging", ["packaging"]), ("configuration", ["configuration"]), ("style", ["style"]),
]

PREFIX = re.compile(r"^(t|c|f|type|kind|area|topic|project|s|r)\s*[:/-]\s*", re.I)
STOP = {"v1", "v2", "request", "issue", "type", "error", "messages", "plugin"}


def _patterns_for(label: str) -> list[str]:
    lower = label.lower()
    for group, hints in GROUP_HINTS:
        if any(h in lower for h in hints):
            return GENERIC[group]
    words = [w for w in re.split(r"[\s\-_/:]+", PREFIX.sub("", lower)) if w and w not in STOP]
    return [rf"\b{re.escape(w.rstrip('s'))}" for w in words]


class RulesBot(Bot):
    name = "A. Rules"

    def __init__(self, repo):
        super().__init__(repo)
        self.rules = {l: [re.compile(p, re.I | re.M) for p in _patterns_for(l)] for l in repo.labels}

    def predict(self, issues):
        out = []
        for issue in issues:
            text = issue.text
            hits = {l for l in issue.available if any(p.search(text) for p in self.rules[l])}
            out.append(Prediction(hits))
        return out

    def snapshot(self):
        return {l: [p.pattern for p in ps] for l, ps in self.rules.items()}
