"""Version B: classic machine learning. TF-IDF features + one logistic regression per label,
retrained on everything seen so far in this repo every RETRAIN_EVERY issues. No AI, nearly free."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from triage.bots.base import Bot, Prediction

RETRAIN_EVERY = 50
MIN_POSITIVES = 3      # a label needs at least this many examples before it can be predicted


class MLBot(Bot):
    name = "B. Classic ML"

    def __init__(self, repo):
        super().__init__(repo)
        self.texts: list[str] = []
        self.truths: list[frozenset[str]] = []
        self.vectorizer = None
        self.models: dict[str, LogisticRegression] = {}
        self.since_train = 0

    def predict(self, issues):
        if not self.models:
            return [Prediction(set()) for _ in issues]
        X = self.vectorizer.transform([i.text for i in issues])
        out = []
        for row, issue in zip(X, issues):
            probs = {l: float(m.predict_proba(row)[0, 1]) for l, m in self.models.items() if l in issue.available}
            out.append(Prediction({l for l, p in probs.items() if p >= 0.5}, probs=probs))
        return out

    def learn(self, issues):
        self.texts += [i.text for i in issues]
        self.truths += [i.truth for i in issues]
        self.since_train += len(issues)
        if self.since_train >= RETRAIN_EVERY or not self.models:
            self._train()

    def _train(self):
        self.since_train = 0
        if len(self.texts) < 10:
            return
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)
        X = self.vectorizer.fit_transform(self.texts)
        self.models = {}
        for label in self.repo.labels:
            y = np.array([label in t for t in self.truths])
            if y.sum() >= MIN_POSITIVES and (~y).sum() >= MIN_POSITIVES:
                self.models[label] = LogisticRegression(max_iter=1000, class_weight="balanced", C=4.0).fit(X, y)

    def snapshot(self):
        return {"trained_labels": sorted(self.models), "examples": len(self.texts)}
