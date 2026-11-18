"""Score the recorded runs. Everything here reads results/runs/*; nothing calls an API.

    python -m triage.metrics          # prints the summary table and writes results/report.md + charts
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from triage import dataset  # noqa: E402
from triage.db import ROOT  # noqa: E402
from triage.replay import RUNS, slug  # noqa: E402

OUT = ROOT / "results"
BIN = 100            # issues per point on the learning curve
COLD_START = 50      # "first issues" window
NEW_LABEL_USES = 15  # first N real uses of a label created mid-history
BOT_FILES = {"A. Rules": "rules.py", "B. Classic ML": "ml.py", "C. Sage (fixed)": "sage_static.py",
             "D. Sage + learner": "sage_learner.py"}
COLORS = {"A. Rules": "#9a9a9a", "B. Classic ML": "#3a7bd5", "C. Sage (fixed)": "#e8a33d",
          "D. Sage + learner": "#d1495b"}


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else 0.0


def counts(rows):
    tp = fp = fn = 0
    for r in rows:
        t, p = set(r["truth"]), set(r["pred"])
        tp += len(t & p); fp += len(p - t); fn += len(t - p)
    return tp, fp, fn


def load_runs() -> dict[str, dict[str, list[dict]]]:
    runs: dict[str, dict[str, list[dict]]] = defaultdict(dict)
    for state in sorted(RUNS.glob("*/*.state.json")):
        meta = json.loads(state.read_text())
        rows = [json.loads(l) for l in open(state.with_name(state.name.replace(".state.json", ".jsonl")))]
        runs[meta["bot"]][meta["repo"]] = rows
    return dict(sorted(runs.items()))


def new_label_f1(rows: list[dict], repo: dataset.Repo) -> tuple[int, int, int]:
    """Scores only labels created after the repo's first replayed issue, over each label's first
    NEW_LABEL_USES real uses (the period where nobody has examples of it yet)."""
    start = repo.issues[0].created_at
    tp = fp = fn = 0
    for label, meta in repo.labels.items():
        if meta.created_at <= start:
            continue
        uses = 0
        for r in rows:
            if label not in r["available"]:
                continue
            if uses >= NEW_LABEL_USES:
                break
            t, p = label in r["truth"], label in r["pred"]
            tp += t and p; fp += p and not t; fn += t and not p
            uses += t
    return tp, fp, fn


def summarize(runs):
    repos = {r.name: r for r in dataset.load_all()}
    table = []
    for bot, by_repo in runs.items():
        all_rows = [r for rows in by_repo.values() for r in rows]
        cold = [r for rows in by_repo.values() for r in rows[:COLD_START]]
        late = [r for rows in by_repo.values() for r in rows[len(rows) // 2:]]
        nl = np.sum([new_label_f1(rows, repos[name]) for name, rows in by_repo.items()], axis=0)
        flagged = sum(bool(r["uncertain"]) for r in all_rows) / len(all_rows)
        loc = _loc(BOT_FILES.get(bot))
        table.append({
            "bot": bot,
            "f1": f1(*counts(all_rows)),
            "f1_first50": f1(*counts(cold)),
            "f1_second_half": f1(*counts(late)),
            "f1_new_labels": f1(*nl),
            "flagged": flagged,
            "loc": loc,
            "per_repo": {name: f1(*counts(rows)) for name, rows in by_repo.items()},
        })
    return table


def _loc(filename):
    if not filename:
        return None
    path = ROOT / "triage" / "bots" / filename
    return sum(1 for l in path.read_text().splitlines() if l.strip() and not l.strip().startswith(("#", '"""')))


def learning_curve(runs, path: Path):
    plt.figure(figsize=(8, 4.5))
    for bot, by_repo in runs.items():
        n = min(len(rows) for rows in by_repo.values())
        xs, ys = [], []
        for b in range(0, n - BIN + 1, BIN):
            window = [r for rows in by_repo.values() for r in rows[b:b + BIN]]
            xs.append(b + BIN)
            ys.append(f1(*counts(window)))
        plt.plot(xs, ys, marker="o", ms=3, label=bot, color=COLORS.get(bot))
    plt.xlabel(f"Issues seen per repo (each point = the next {BIN} issues, all repos)")
    plt.ylabel("Label F1 (micro)")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.title("Does the bot get better as it sees more of the repo's history?")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def write_report(table, runs):
    OUT.mkdir(exist_ok=True)
    learning_curve(runs, OUT / "learning_curve.png")
    lines = ["# Results", "", "![learning curve](learning_curve.png)", "",
             "| Version | Label F1 (all) | First 50 issues | Second half | New labels (first uses) | "
             "Issues flagged for a human | Lines of decision logic |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for t in table:
        lines.append(f"| {t['bot']} | {t['f1']:.3f} | {t['f1_first50']:.3f} | {t['f1_second_half']:.3f} | "
                     f"{t['f1_new_labels']:.3f} | {t['flagged']:.1%} | {t['loc'] or '—'} |")
    lines += ["", "## Per repo (label F1)", ""]
    repos = sorted({r for t in table for r in t["per_repo"]})
    lines.append("| Version | " + " | ".join(repos) + " |")
    lines.append("|---|" + "---:|" * len(repos))
    for t in table:
        lines.append(f"| {t['bot']} | " + " | ".join(f"{t['per_repo'].get(r, float('nan')):.3f}" for r in repos) + " |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n")


def main():
    runs = load_runs()
    table = summarize(runs)
    write_report(table, runs)
    print(f"{'version':<20} {'F1':>6} {'first50':>8} {'2nd half':>9} {'new lbl':>8} {'flagged':>8} {'LOC':>5}")
    for t in table:
        print(f"{t['bot']:<20} {t['f1']:>6.3f} {t['f1_first50']:>8.3f} {t['f1_second_half']:>9.3f} "
              f"{t['f1_new_labels']:>8.3f} {t['flagged']:>8.1%} {t['loc'] or '':>5}")
    for t in table:
        print(f"  {t['bot']:<20} " + "  ".join(f"{r.split('/')[1]}={v:.3f}" for r, v in t["per_repo"].items()))
    print(f"\nWrote {OUT / 'report.md'} and learning_curve.png")


if __name__ == "__main__":
    main()
