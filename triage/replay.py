"""Replay each repo's issue history through the bots, in date order, and record every decision.

    python -m triage.replay --bots A,B                 # classic versions only (free)
    python -m triage.replay --bots C,D --dry-run       # estimate Sage tokens and cost, no calls
    python -m triage.replay --bots C,D --limit 200     # small paid pilot: first 200 issues per repo
    python -m triage.replay --bots A,B,C,D             # everything

Writes results/runs/<bot>/<repo>.jsonl (one line per issue) and results/runs/<bot>/<repo>.state.json.
"""
from __future__ import annotations

import argparse
import json
import time

from triage import dataset
from triage.bots.ml import MLBot
from triage.bots.rules import RulesBot
from triage.db import ROOT

RUNS = ROOT / "results" / "runs"
SAGE_INPUT_PRICE = 0.05 / 1_000_000      # $/token, Free and Developer plans (2026-11-16)
SAGE_OUTPUT_PRICE = 1.0 / 1_000_000


def slug(s: str) -> str:
    return s.replace("/", "__").replace(" ", "_").replace(".", "").replace("+", "plus")


def make_bot(code: str, repo, client, rewriter):
    if code == "A":
        return RulesBot(repo)
    if code == "B":
        return MLBot(repo)
    if code == "C":
        from triage.bots.sage_static import SageStaticBot
        return SageStaticBot(repo, client)
    if code == "D":
        from triage.bots.sage_learner import SageLearnerBot
        return SageLearnerBot(repo, client, rewriter)
    raise ValueError(code)


def run(bot, repo, out_dir, dry_run=False) -> None:
    started = time.time()
    rows = []
    issues = repo.issues
    for start in range(0, len(issues), bot.chunk_size):
        chunk = issues[start:start + bot.chunk_size]
        preds = bot.predict(chunk)                 # decide first...
        bot.learn(chunk)                           # ...then reveal what the maintainers did
        for issue, p in zip(chunk, preds):
            rows.append({
                "number": issue.number, "created_at": issue.created_at,
                "truth": sorted(issue.truth), "pred": sorted(p.labels), "uncertain": sorted(p.uncertain),
                "available": list(issue.available),
                "probs": {k: round(v, 4) for k, v in p.probs.items()},
            })
        print(f"  {bot.name:<20} {repo.name:<22} {min(start + bot.chunk_size, len(issues))}/{len(issues)}", end="\r")
    print()
    if dry_run:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{slug(repo.name)}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    state = {"bot": bot.name, "repo": repo.name, "seconds": round(time.time() - started, 1), **bot.snapshot()}
    (out_dir / f"{slug(repo.name)}.state.json").write_text(json.dumps(state, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="A,B")
    ap.add_argument("--repos", default="", help="comma-separated owner/name; default: all in config")
    ap.add_argument("--limit", type=int, default=None, help="only the first N issues of each repo")
    ap.add_argument("--dry-run", action="store_true", help="estimate Sage tokens/cost without calling it")
    ap.add_argument("--no-rewrite", action="store_true", help="version D without Claude spec rewrites")
    args = ap.parse_args()

    codes = [c.strip().upper() for c in args.bots.split(",") if c.strip()]
    repos = dataset.load_all(limit=args.limit, only=[r for r in args.repos.split(",") if r] or None)

    client = rewriter = None
    if {"C", "D"} & set(codes):
        from triage.sage import SageClient
        client = SageClient(dry_run=args.dry_run)
        if "D" in codes and not args.no_rewrite and not args.dry_run:
            import os
            from pathlib import Path
            if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                    or (Path.home() / ".config" / "anthropic").exists()):
                raise SystemExit("Version D rewrites its spec with Claude, but no Anthropic credentials were found.\n"
                                 "Set ANTHROPIC_API_KEY (or run `ant auth login`), or add --no-rewrite to run D "
                                 "with memory and cutoffs only.")
            from triage.rewriter import Rewriter
            rewriter = Rewriter()

    for code in codes:
        for repo in repos:
            bot = make_bot(code, repo, client, rewriter)
            run(bot, repo, RUNS / slug(bot.name), dry_run=args.dry_run)

    if client:
        s = client.stats
        if args.dry_run:
            est = s["est_input_tokens"]
            print(f"\nDry run: {s['groups']} issue decisions, {s['cached']} already cached.")
            print(f"Estimated new Sage input tokens: {est:,} ≈ ${est * SAGE_INPUT_PRICE:.2f} "
                  f"(version D's spec rewrites add roughly 2 x 60 issues per rewrite on top of this)")
        else:
            cost = s["input_tokens"] * SAGE_INPUT_PRICE + s["output_tokens"] * SAGE_OUTPUT_PRICE
            print(f"\nSage: {s['calls']} calls, {s['cached']} cached decisions, "
                  f"{s['input_tokens']:,} input tokens, ≈ ${cost:.2f} this run")
    failed = [e for f in (RUNS / "D_Sage_plus_learner").glob("*.state.json")
              for e in json.loads(f.read_text()).get("spec_log", []) if e["event"] == "rewrite_failed"]
    if "D" in codes and failed and not args.dry_run:
        print(f"WARNING: {len(failed)} spec rewrites failed, e.g.: {failed[0]['error']}")
    if rewriter:
        print(f"Claude rewrites: {rewriter.calls} calls, {rewriter.input_tokens:,} in / {rewriter.output_tokens:,} out")


if __name__ == "__main__":
    main()
