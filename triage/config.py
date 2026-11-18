from __future__ import annotations

import tomllib

from triage.db import ROOT


def load() -> dict:
    with open(ROOT / "config" / "repos.toml", "rb") as f:
        return tomllib.load(f)
