# Self-evolving triage

**Software that rewrites its own rules from how people use it, with no developer changing the code.**

A decision model ([Levanto Sage 2](https://docs.levanto.ai)) makes every decision. An LLM (Claude Opus 5.5)
reads the mistakes and proposes better rules in plain English. The software tests each proposal on its own
recent history and keeps it only if it works.

> **Goal: experiment with an approach, not build the best code.** This repo tests whether a decision model and
> an LLM can work together so that software improves itself from how people use it. The code exists only to run
> that experiment and make the result visible.

The experiment builds that loop and lets it run on real data: an issue-triage bot that labels new GitHub issues,
replayed over the history of five public repositories. The write-up with diagrams and every rewrite proposal is
in [`docs/index.html`](docs/index.html).

## Out of scope

- **The best issue labeller.** Issue labelling is just a convenient place to watch the loop work. None of the
  four versions is tuned to win: the rules are a typical keyword table, the ML model is a standard TF-IDF +
  logistic regression baseline, and the Sage versions use short, generic instructions. Each could be more
  accurate with more work, and a well-tuned ML model may beat every version here. That's fine.
- **The best code.** This is experiment code, not a product or a library. It isn't optimized, hardened or meant
  for production, and the design choices favour being easy to read and reproduce over being complete.
- **A benchmark.** The numbers show that the loop works. They aren't a ranking of decision models against ML.

## Four ways to build the same feature

| Version | How it decides | What changes over time |
|---|---|---|
| **A. Rules** | Keyword patterns, like a typical auto-labeler config | Nothing. A developer has to edit the patterns. |
| **B. Classic ML** | TF-IDF + logistic regression per label | Weights, retrained every 50 issues on what maintainers did |
| **C. Sage (fixed)** | One Sage 2 `tags` decision per issue, from label names and descriptions | Nothing. Shows what the decision model does alone. |
| **D. Sage + learner** | The same Sage decision as C | Its **rules**: memory of similar past issues, a cutoff per label, and instructions and label definitions rewritten by Claude from recent mistakes, kept only if they beat the current rules on the last 60 issues |

In D the decision logic is **data** (instructions, label definitions, cutoffs), not code. The program never
changes; its behaviour does. Every proposal is logged with the scores that accepted or rejected it, in
[`results/runs/D_Sage_plus_learner/*.state.json`](results/runs/D_Sage_plus_learner).

## What happened

6,643 issues from pandas, black, pydantic, pip and poetry, decided in date order. Each version decides first
and only then sees the maintainers' labels.

| Version | Label F1 | First 50 issues | New labels | Who changes it |
|---|---:|---:|---:|---|
| A. Rules | 0.469 | 0.503 | 0.229 | a developer |
| B. Classic ML | 0.760 | 0.552 | 0.068 | retraining |
| C. Sage (fixed) | 0.661 | 0.650 | 0.383 | nobody |
| D. Sage + learner | 0.778 | 0.663 | 0.619 | itself |

- C and D use the same model and start from the same instructions. C stays flat around 0.65; D climbs to about
  0.80 once its rules start rewriting themselves.
- Claude proposed 40 rewrites. 12 went live and 28 were rejected by the test on recent issues, some of which
  would have cut the score by a third.
- Classic ML does well here too: issue labelling has plenty of examples and fixed categories. D's edge is that it
  works from the first issue, handles labels it has never seen, and every change it makes to itself is readable.
- Cost: about $5.50 of Sage for ~16,900 decisions, plus 40 Claude calls.

Full numbers: [`results/report.md`](results/report.md).

## How the replay works

1. `triage.fetch` downloads the 1,500 most recent issues (before 2026-05-01) of each repo, with their final
   labels and the date each label was created.
2. `triage.replay` feeds issues to each bot **in date order**. The bot decides, and only then sees what the
   maintainers did. A label counts only if it existed when the issue was opened.
3. `triage.metrics` scores the recorded decisions and draws the chart. It never calls an API.

Only content labels (type and area) are used, about 12 per repo. Workflow labels like "needs triage" are left
out because they depend on what happens after an issue is opened. The label sets are in
[`config/repos.toml`](config/repos.toml).

## Code map

| File | What it is |
|---|---|
| `triage/bots/rules.py` | Version A |
| `triage/bots/ml.py` | Version B |
| `triage/bots/sage_static.py` | Version C: builds the Sage request |
| `triage/bots/sage_learner.py` | Version D: memory, cutoffs and the rewrite-and-test loop |
| `triage/rewriter.py` | The Claude call that proposes new rules |
| `triage/sage.py` | Sage client with batching, retries and a response cache |
| `triage/replay.py` | The harness: decide, then learn, issue by issue |

## Run it

```bash
make install                  # creates .venv (or: make install PYTHON=python3.13)
gh auth login                 # GitHub data is read with your gh login
make fetch                    # ~1 min
make classic                  # versions A and B, free
make estimate                 # what C and D would cost, no calls

export SAGE_API_KEY=...       # https://platform.levanto.ai
export ANTHROPIC_API_KEY=...  # for version D's rewrites
make pilot                    # versions C and D on the first 200 issues per repo
make sage                     # versions C and D on everything (~$6 of Sage)
make report                   # results/report.md + results/learning_curve.png
```

Every Sage and Claude response is cached in `data/issues.db`, so an interrupted run resumes where it stopped
and re-plotting costs nothing.

## Limits

- The loop is only as good as its signal. Final labels are the ground truth; maintainers sometimes relabel later.
- The rewrite test uses the 60 most recent issues, which catches bad proposals but can miss close calls.
- A replay of five Python repositories, not a live product. Add repos in `config/repos.toml`.
- Issue text isn't redistributed here; `make fetch` downloads it from GitHub. The run files contain issue
  numbers, labels and probabilities only.
