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
GROUPS_PER_CALL = 8       # issues per /decide/batch request
PARALLEL_CALLS = 4


class OutOfUsage(RuntimeError):
    pass


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
            for idxs, answers, usage in pool.map(lambda b: (b, *self._call([groups[i] for i in b])), batches):
                share_in = usage.get("input_tokens", 0) // max(len(idxs), 1)
                share_out = usage.get("output_tokens", 0) // max(len(idxs), 1)
                for i, ans in zip(idxs, answers):
                    results[i] = ans
                    with self.lock:
                        self.conn.execute("INSERT OR REPLACE INTO sage_cache VALUES (?,?,?,?)",
                                          (keys[i], json.dumps(ans), share_in, share_out))
        self.conn.commit()
        return results

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

    def _call(self, groups: list[dict]) -> tuple[list, dict]:
        body = {"requests": groups}
        for attempt in range(8):
            try:
                r = self.session.post(f"{BASE_URL}/decide/batch", json=body, timeout=120)
            except requests.RequestException as e:
                wait = min(60, 2 ** attempt)
                print(f"  Sage connection error ({e}); retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
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
            if r.status_code in (401, 400, 413):
                raise RuntimeError(f"Sage {r.status_code}: {r.text[:500]}")
            wait = float(r.headers.get("Retry-After", min(60, 2 ** attempt)))
            print(f"  Sage {r.status_code}; retrying in {wait:.0f}s", file=sys.stderr)
            time.sleep(wait)
        raise RuntimeError("Sage kept failing")


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
