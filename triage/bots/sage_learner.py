"""Version D: Sage + a learner. Same decision call as version C, but three things evolve from what the
maintainers actually do, with no human and no code change:

1. Memory: each issue is sent with the most similar past issues from this repo and the labels they got.
2. Cutoffs: once a label has enough history, the probability cutoff for applying it is tuned to this repo.
3. Spec: every REWRITE_EVERY issues, Claude proposes new instructions and label definitions from recent
   mistakes. The proposal goes live only if it beats the current spec on held-out recent issues.
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from triage.bots.base import Prediction
from triage.bots.sage_static import SageStaticBot

K_EXAMPLES = 4
EXAMPLE_CHARS = 220
REFIT_EVERY = 50
MIN_HISTORY_FOR_CUTOFF = 60       # decisions seen for a label before its cutoff is tuned
MIN_POSITIVES_FOR_CUTOFF = 6
REWRITE_EVERY = 150
WRITE_WINDOW = (-150, -60)        # issues whose mistakes Claude sees
HOLDOUT = 60                      # most recent issues, never shown to Claude, used to accept/reject
MIN_GAIN = 0.01

MEMORY_HEADER = ("\n\n---\nFor reference only (do not label these): similar past issues in this repository "
                 "and the labels the maintainers gave them.\n")


class SageLearnerBot(SageStaticBot):
    name = "D. Sage + learner"

    def __init__(self, repo, client, rewriter=None):
        super().__init__(repo, client)
        self.rewriter = rewriter
        self.seen = []                                  # issues whose truth we know, in order
        self.history: list[tuple[dict, frozenset]] = [] # (probs, truth) per seen issue, for cutoffs/mistakes
        self.cutoffs: dict[str, float] = {}
        self.vectorizer = None
        self.matrix = None
        self.since_refit = 0
        self.since_rewrite = 0
        self.spec_log = [{"at_issue": 0, "event": "initial", "instructions": self.instructions,
                          "definitions": dict(self.definitions)}]

    # ---- memory ---------------------------------------------------------------------------------------
    def _content(self, issue, upto: int | None = None) -> str:
        """Issue text plus similar issues among self.seen[:upto]."""
        pool = len(self.seen) if upto is None else upto
        if self.vectorizer is None or pool == 0:
            return issue.text
        n = min(pool, self.matrix.shape[0])
        sims = cosine_similarity(self.vectorizer.transform([issue.text]), self.matrix[:n])[0]
        best = [i for i in np.argsort(-sims)[:K_EXAMPLES] if sims[i] > 0.05]
        if not best:
            return issue.text
        lines = []
        for i in best:
            past = self.seen[i]
            snippet = " ".join(past.body.split())[:EXAMPLE_CHARS]
            lines.append(f"- labels [{', '.join(sorted(past.truth))}] {past.title} — {snippet}")
        return issue.text + MEMORY_HEADER + "\n".join(lines)

    def _refit(self):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=30000, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([i.text for i in self.seen])
        self.since_refit = 0

    # ---- decisions ------------------------------------------------------------------------------------
    def _apply_cutoffs(self, pred: Prediction) -> Prediction:
        labels, uncertain = set(pred.labels), set(pred.uncertain)
        for label, cut in self.cutoffs.items():
            p = pred.probs.get(label)
            if p is None:
                continue
            labels.discard(label)
            uncertain.discard(label)
            if p >= cut + 0.05:
                labels.add(label)
            elif p > cut - 0.05:
                uncertain.add(label)
        return Prediction(labels, uncertain, pred.probs)

    def predict(self, issues):
        groups = [self.group(i, self._content(i)) for i in issues]
        return [self._apply_cutoffs(self.parse(a)) for a in self.client.decide_groups(groups)]

    # ---- learning -------------------------------------------------------------------------------------
    def learn(self, issues):
        # Re-read the probabilities we just used (cached, so free) to keep a decision history.
        groups = [self.group(i, self._content(i)) for i in issues]
        for issue, answers in zip(issues, self.client.decide_groups(groups)):
            self.history.append((self.parse(answers).probs, issue.truth))
        self.seen += issues
        self.since_refit += len(issues)
        self.since_rewrite += len(issues)
        if self.vectorizer is None or self.since_refit >= REFIT_EVERY:
            self._refit()
            self._tune_cutoffs()
        if self.rewriter and self.since_rewrite >= REWRITE_EVERY and len(self.seen) >= REWRITE_EVERY:
            self.since_rewrite = 0
            self._try_rewrite()

    def _tune_cutoffs(self):
        for label in self.repo.labels:
            pairs = [(p[label], label in t) for p, t in self.history if label in p]
            if len(pairs) < MIN_HISTORY_FOR_CUTOFF or sum(y for _, y in pairs) < MIN_POSITIVES_FOR_CUTOFF:
                continue
            probs = np.array([p for p, _ in pairs])
            ys = np.array([y for _, y in pairs])
            best, best_f1 = None, -1.0
            for cut in np.arange(0.10, 0.91, 0.05):
                pred = probs >= cut
                tp = (pred & ys).sum()
                f1 = 2 * tp / (pred.sum() + ys.sum()) if (pred.sum() + ys.sum()) else 0
                if f1 > best_f1:
                    best, best_f1 = float(cut), f1
            self.cutoffs[label] = round(best, 2)

    def _mistakes_text(self, start: int, end: int) -> str:
        out = []
        for issue, (probs, truth) in zip(self.seen[start:end], self.history[start:end]):
            applied = {l for l, p in probs.items() if p >= self.cutoffs.get(l, 0.5)}
            missed, wrong = truth - applied, applied - truth
            if missed or wrong:
                body = " ".join(issue.body.split())[:300]
                out.append(f"### {issue.title}\n{body}\nmaintainers: {sorted(truth)} | "
                           f"missed: {sorted(missed)} | wrong: {sorted(wrong)}")
        return "\n\n".join(out[-40:]) or "(no mistakes)"

    def _score(self, instructions: str, definitions: dict, issues: list, offset: int) -> float:
        """Micro-F1 of a spec on held-out issues, each with memory drawn only from issues before it."""
        groups = [self.group(i, self._content(i, upto=offset + k), instructions, definitions)
                  for k, i in enumerate(issues)]
        tp = fp = fn = 0
        for issue, answers in zip(issues, self.client.decide_groups(groups)):
            pred = self._apply_cutoffs(self.parse(answers)).labels
            tp += len(pred & issue.truth)
            fp += len(pred - issue.truth)
            fn += len(issue.truth - pred)
        return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0

    def _try_rewrite(self):
        n = len(self.seen)
        mistakes = self._mistakes_text(n + WRITE_WINDOW[0], n + WRITE_WINDOW[1])
        try:
            proposal = self.rewriter.propose(self.repo.name, self.instructions, self.definitions, mistakes)
        except Exception as e:  # keep the experiment running; the log records it
            self.spec_log.append({"at_issue": n, "event": "rewrite_failed", "error": str(e)[:300]})
            return
        new_defs = dict(self.definitions)
        for d in proposal.definitions:
            if d.label in new_defs:
                new_defs[d.label] = d.definition
        holdout = self.seen[n - HOLDOUT:]
        current = self._score(self.instructions, self.definitions, holdout, n - HOLDOUT)
        candidate = self._score(proposal.instructions, new_defs, holdout, n - HOLDOUT)
        accepted = candidate >= current + MIN_GAIN
        self.spec_log.append({"at_issue": n, "event": "accepted" if accepted else "rejected",
                              "holdout_f1_current": round(current, 4), "holdout_f1_candidate": round(candidate, 4),
                              "instructions": proposal.instructions, "definitions": new_defs})
        if accepted:
            self.instructions, self.definitions = proposal.instructions, new_defs

    def snapshot(self):
        return {"instructions": self.instructions, "definitions": self.definitions,
                "cutoffs": self.cutoffs, "spec_log": self.spec_log}
