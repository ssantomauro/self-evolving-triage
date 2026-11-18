from __future__ import annotations

try:
    import tomllib                      # Python 3.11+
except ModuleNotFoundError:             # Python 3.9 / 3.10
    import tomli as tomllib

from triage.db import ROOT


def load() -> dict:
    with open(ROOT / "config" / "repos.toml", "rb") as f:
        return tomllib.load(f)
