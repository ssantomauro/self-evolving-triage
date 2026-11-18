"""Version C: Sage with a fixed spec. Each label becomes a tag described by its name and the repo's own
label description. Nothing changes while running: this measures what the decision model does on its own."""
from __future__ import annotations

from triage.bots.base import Bot, Prediction
from triage.sage import SageClient

DEFAULT_INSTRUCTIONS = (
    "This is a new GitHub issue filed in the {repo} repository. Decide which labels the maintainers "
    "would apply to it. A label applies only when the issue is clearly about that topic."
)


class SageStaticBot(Bot):
    name = "C. Sage (fixed)"
    chunk_size = 32

    def __init__(self, repo, client: SageClient):
        super().__init__(repo)
        self.client = client
        self.instructions = DEFAULT_INSTRUCTIONS.format(repo=repo.name)
        self.definitions = {l: meta.description for l, meta in repo.labels.items()}
        self.label_ids = {l: f"l{i}" for i, l in enumerate(repo.labels)}   # stable tag ids for the wire
        self.id_labels = {v: k for k, v in self.label_ids.items()}

    # --- request building (shared with the learner) ---
    def tag_name(self, label: str, definitions: dict[str, str]) -> str:
        d = definitions.get(label, "").strip()
        return f"{label}: {d}" if d else label

    def group(self, issue, content: str, instructions: str | None = None, definitions: dict | None = None) -> dict:
        definitions = definitions if definitions is not None else self.definitions
        return {
            "content": content,
            "questions": [{
                "id": "labels",
                "kind": "tags",
                "instructions": instructions or self.instructions,
                "tags": [{"id": self.label_ids[l], "name": self.tag_name(l, definitions)} for l in issue.available],
            }],
        }

    def parse(self, answers) -> Prediction:
        if answers is None:                       # dry run
            return Prediction(set())
        a = answers[0]
        if not a.get("ok"):
            return Prediction(set(), uncertain=set(self.repo.labels), failed=True)
        labels, uncertain, probs = set(), set(), {}
        for t in a["result"]["result"]["tags"]:
            label = self.id_labels[t["id"]]
            probs[label] = t["probability"]
            if t["applies"] is True:
                labels.add(label)
            elif t["applies"] is None:
                uncertain.add(label)
        return Prediction(labels, uncertain, probs)

    def predict(self, issues):
        groups = [self.group(i, i.text) for i in issues]
        return [self.parse(a) for a in self.client.decide_groups(groups)]

    def snapshot(self):
        return {"instructions": self.instructions, "definitions": self.definitions}
