"""Minimal Levanto Sage client for /decide/batch, with an on-disk cache and token accounting.

Reads the key from SAGE_API_KEY. Every response is cached in data/issues.db keyed by the exact request,
so rerunning the experiment (or re-plotting) costs nothing and gives identical results.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from triage import db

BASE_URL = os.environ.get("SAGE_BASE_URL", "https://sage.levanto.ai")
# Smaller, fewer concurrent requests: big batches hit 504 timeouts and parallel ones hit 429 rate limits.
# Override with SAGE_GROUPS_PER_CALL / SAGE_PARALLEL_CALLS if your plan allows more.
GROUPS_PER_CALL = int(os.environ.get("SAGE_GROUPS_PER_CALL", 4))
PARALLEL_CALLS = int(os.environ.get("SAGE_PARALLEL_CALLS", 2))
REQUEST_TIMEOUT = int(os.environ.get("SAGE_TIMEOUT", 45))   # seconds; a normal batch takes 1-3 s


class OutOfUsage(RuntimeError):
    pass


class SageUnavailable(RuntimeError):
    pass


FAILED = [{"ok": False, "error": "Sage unavailable after retries (not cached; rerun to retry)"}]


class SageClient:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.key = os.environ.get("SAGE_API_KEY")
        if not self.key and not dry_run:
            sys.exit("SAGE_API_KEY is not set. Export your lv_live_ key, or run with --dry-run to estimate cost.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        self.conn = db.connect()
        self.lock = threading.Lock()
        self._dry_seen: set[str] = set()
        self.stats = {"groups": 0, "cached": 0, "calls": 0, "input_tokens": 0, "output_tokens": 0, "est_input_tokens": 0}

    # ---- public -------------------------------------------------------------------------------------
    def decide_groups(self, groups: list[dict]) -> list[list[dict]]:
        """groups: [{"content": str, "questions": [...]}]. Returns, per group, the list of answers
        (each {"ok": bool, "result": {...}} as returned by /decide/batch)."""
        keys = [self._key(g) for g in groups]
        results: list = [None] * len(groups)
        todo = []
        for i, k in enumerate(keys):
            hit = self.conn.execute("SELECT response FROM sage_cache WHERE key=?", (k,)).fetchone()
            if hit:
                results[i] = json.loads(hit["response"])
                self.stats["cached"] += 1
            else:
                todo.append(i)
        self.stats["groups"] += len(groups)

        if self.dry_run:
            for i in todo:
                if keys[i] not in self._dry_seen:          # same request asked twice is paid once
                    self._dry_seen.add(keys[i])
                    self.stats["est_input_tokens"] += _estimate_tokens(groups[i])
                results[i] = None
            return results

        batches = [todo[j:j + GROUPS_PER_CALL] for j in range(0, len(todo), GROUPS_PER_CALL)]
        with ThreadPoolExecutor(PARALLEL_CALLS) as pool:
            for done in pool.map(lambda b: self._resilient([(i, groups[i]) for i in b]), batches):
                for i, ans, tin, tout in done:
                    results[i] = ans
                    if ans is not FAILED:
                        with self.lock:
                            self.conn.execute("INSERT OR REPLACE INTO sage_cache VALUES (?,?,?,?)",
                                              (keys[i], json.dumps(ans), tin, tout))
        self.conn.commit()
        return results

    def _resilient(self, items: list[tuple[int, dict]]) -> list[tuple]:
        """Send a batch; if it keeps failing, send its issues one by one. An issue that still fails is
        returned as FAILED (and not cached), so one bad request can't stop the whole replay."""
        try:
            answers, usage = self._call([g for _, g in items], attempts=2 if len(items) > 1 else 3)
            n = max(len(items), 1)
            return [(i, a, usage.get("input_tokens", 0) // n, usage.get("output_tokens", 0) // n)
                    for (i, _), a in zip(items, answers)]
        except SageUnavailable:
            if len(items) > 1:
                print(f"\n  Sage: batch of {len(items)} keeps failing; sending its issues one by one", file=sys.stderr)
                return [row for item in items for row in self._resilient([item])]
            with self.lock:
                self.stats["failed"] = self.stats.get("failed", 0) + 1
            print("\n  Sage: one issue failed after retries; skipping it (rerun to retry)", file=sys.stderr)
            return [(items[0][0], FAILED, 0, 0)]

    def cached_usage(self, groups: list[dict]) -> tuple[int, int]:
        """Tokens that the cached responses for these groups cost when they were first fetched."""
        tin = tout = 0
        for g in groups:
            row = self.conn.execute("SELECT input_tokens, output_tokens FROM sage_cache WHERE key=?",
                                    (self._key(g),)).fetchone()
            if row:
                tin += row["input_tokens"]
                tout += row["output_tokens"]
        return tin, tout

    # ---- internals ----------------------------------------------------------------------------------
    @staticmethod
    def _key(group: dict) -> str:
        return hashlib.sha256(json.dumps(group, sort_keys=True).encode()).hexdigest()

    def _call(self, groups: list[dict], attempts: int = 3) -> tuple[list, dict]:
        """`attempts` limits tries after the server fails (5xx, timeouts, dropped connections).
        Rate limiting (429) means "wait", not "broken": it doesn't use up attempts."""
        body = {"requests": groups}
        failures = waits = 0
        while failures < attempts:
            t0 = time.time()
            try:
                r = self.session.post(f"{BASE_URL}/decide/batch", json=body, timeout=(10, REQUEST_TIMEOUT))
            except requests.RequestException as e:
                failures += 1
                kind = "timed out" if isinstance(e, requests.Timeout) else f"connection error ({type(e).__name__})"
                self._note(f"request {kind} after {time.time() - t0:.0f}s (try {failures}/{attempts})")
                time.sleep(min(30, 2 ** failures))
                continue
            if r.status_code == 200:
                data = r.json()
                usage = (data.get("meta") or {}).get("usage") or {}
                with self.lock:
                    self.stats["calls"] += 1
                    self.stats["input_tokens"] += usage.get("input_tokens", 0)
                    self.stats["output_tokens"] += usage.get("output_tokens", 0)
                return [res["answers"] for res in data["results"]], usage
            if r.status_code == 402:
                raise OutOfUsage("Sage returned 402: this period's included usage is used up.")
            if r.status_code in (401, 400, 413, 422):
                raise RuntimeError(f"Sage {r.status_code}: {r.text[:500]}")
            if r.status_code == 429 and waits < 30:
                waits += 1
                wait = max(float(r.headers.get("Retry-After", 0) or 0), min(30, 2 ** min(waits, 5)))
                self._note(f"429 rate limited; waiting {wait:.0f}s")
                time.sleep(wait)
                continue
            failures += 1
            self._note(f"{r.status_code} after {time.time() - t0:.0f}s (try {failures}/{attempts})")
            time.sleep(min(30, 2 ** failures))
        raise SageUnavailable("Sage kept failing")

    def _note(self, msg: str) -> None:
        with self.lock:
            self.stats["retries"] = self.stats.get("retries", 0) + 1
        print(f"\n  Sage: {msg}", file=sys.stderr)


def _estimate_tokens(group: dict) -> int:
    """Rough offline estimate (~4 chars per token). Tags read the content once per tag."""
    content = len(group["content"]) / 4
    total = 0
    for q in group["questions"]:
        instr = len(q.get("instructions") or "") / 4
        if q["kind"] == "tags":
            total += sum(content + instr + len(t["name"]) / 4 + 10 for t in q["tags"])
        else:
            total += content + instr + 10
    return int(total)
