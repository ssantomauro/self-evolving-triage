# Self-evolving triage

**Can software improve itself from how its users behave, with no developer changing the code?**

This experiment builds the same feature four ways and replays real history through each of them: an
issue-triage bot that labels new GitHub issues. Five public repositories play the role of five customers of a
SaaS product. Each one starts from zero, and each has its own labeling conventions that no version is told about.

| Version | How it decides | How it changes over time |
|---|---|---|
| **A. Rules** | Keyword patterns, like a typical auto-labeler config | It doesn't. A developer has to edit the patterns. |
| **B. Classic ML** | TF-IDF + logistic regression per label | Retrained every 50 issues on what maintainers did |
| **C. Sage (fixed)** | One [Levanto Sage 2](https://docs.levanto.ai) `tags` decision per issue, from the label names and descriptions | It doesn't. Measures what the decision model does alone. |
| **D. Sage + learner** | Same decision as C | Its spec evolves: similar past issues as memory, cutoffs tuned per label, and label definitions rewritten by Claude from mistakes, kept only if they beat the current spec on held-out issues |

In version D the decision logic is **data** (instructions, label definitions, cutoffs) stored next to the
code, not code. The program never changes; its behavior does. Every spec change is logged with the score
that justified it, in `results/runs/D_Sage_plus_learner/*.state.json`.

## How the replay works

1. `triage.fetch` downloads the 1,500 most recent issues (before 2026-05-01) of each repo, with their final
   labels and the date each label was created.
2. `triage.replay` feeds issues to a bot **in date order**. The bot decides, and only then sees what the
   maintainers did. A label counts only if it already existed when the issue was opened, so no version can
   guess a label that didn't exist yet.
3. `triage.metrics` scores all recorded decisions and draws the charts. It never calls an API.

Only content labels (type and area) are used, about 12 per repo. Workflow labels like "needs triage" or
"duplicate" are left out because they depend on what happens after an issue is opened. The exact label sets
are in [`config/repos.toml`](config/repos.toml).

## What's measured

- **Label F1** over the whole history and per block of 100 issues (the learning curve)
- **First 50 issues** of each repo: how good a version is before it has any history
- **New labels**: labels a repo created partway through (e.g. psf/black's 2021 reorganization), over their first 15 uses
- **Issues flagged for a human**: Sage answers `null` when it isn't sure; the other versions always guess
- **Cost** and **lines of decision logic**

## Run it

```bash
make install                  # creates .venv (uses python3; or: make install PYTHON=python3.13)
gh auth login                 # GitHub data is read with your gh login
make fetch                    # ~1 min
make classic                  # versions A and B, free, ~1-2 min
make estimate                 # what C and D would cost, no calls

export SAGE_API_KEY=lv_live_...   # https://platform.levanto.ai
export ANTHROPIC_API_KEY=...      # for version D's spec rewrites (or `ant auth login`)
make pilot                    # ~$1: versions C and D on the first 200 issues per repo
make sage                     # versions C and D, ~$10 in total
make report                   # results/report.md + results/learning_curve.png
```

Every Sage and Claude response is cached in `data/issues.db`, so rerunning or re-plotting costs nothing.

## Results

See [`results/report.md`](results/report.md). Classic baseline so far (versions A and B, all 5 repos, ~6,600 issues):

| Version | Label F1 | First 50 issues | New labels |
|---|---:|---:|---:|
| A. Rules | 0.469 | 0.503 | 0.229 |
| B. Classic ML | 0.760 | 0.552 | 0.068 |

Classic ML is a strong baseline once it has a few hundred examples. Its weak spots are the first issues of a
new repo and labels it has no examples of yet.

## Caveats

- Final labels are used as ground truth; maintainers sometimes relabel later, and some labels come from issue templates.
- Issue text isn't redistributed in this repo; `make fetch` downloads it from GitHub.
- Five Python-ecosystem repos is a small sample. Add more in `config/repos.toml`.
