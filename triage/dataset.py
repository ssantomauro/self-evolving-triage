"""Turn downloaded issues into a replayable, time-ordered stream per repo."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from triage import config, db

MAX_BODY_CHARS = 1500


@dataclass
class Label:
    name: str
    description: str
    created_at: str


@dataclass
class Issue:
    repo: str
    number: int
    title: str
    body: str
    created_at: str
    truth: frozenset[str]          # labels from the repo's label set that existed when the issue was opened
    available: tuple[str, ...]     # labels from the label set that existed at that time (what a bot may choose from)

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


@dataclass
class Repo:
    name: str
    labels: dict[str, Label]
    issues: list[Issue] = field(default_factory=list)


def clean_body(body: str) -> str:
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)          # issue-template comments
    body = re.sub(r"```.*?```", "[code block]", body, flags=re.S) if len(body) > MAX_BODY_CHARS else body
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return body[:MAX_BODY_CHARS]


def load_repo(name: str, label_names: list[str]) -> Repo:
    conn = db.connect()
    labels = {
        r["name"]: Label(r["name"], r["description"], r["created_at"])
        for r in conn.execute("SELECT * FROM labels WHERE repo=?", (name,))
    }
    missing = [l for l in label_names if l not in labels]
    if missing:
        raise ValueError(f"{name}: labels not found in repo: {missing}")
    label_set = {l: labels[l] for l in label_names}

    repo = Repo(name, label_set)
    for r in conn.execute("SELECT * FROM issues WHERE repo=? ORDER BY created_at", (name,)):
        available = tuple(l for l, meta in label_set.items() if meta.created_at <= r["created_at"])
        final = set(filter(None, r["labels"].split("\n")))
        truth = frozenset(final & set(available))
        if not truth:
            continue    # untriaged (or only workflow labels): nothing to score against
        repo.issues.append(Issue(name, r["number"], r["title"], clean_body(r["body"]), r["created_at"], truth, available))
    return repo


def load_all(limit: int | None = None, only: list[str] | None = None) -> list[Repo]:
    cfg = config.load()
    repos = []
    for r in cfg["repos"]:
        if "labels" not in r or (only and r["repo"] not in only):
            continue
        repo = load_repo(r["repo"], r["labels"])
        if limit:
            repo.issues = repo.issues[:limit]
        repos.append(repo)
    return repos
