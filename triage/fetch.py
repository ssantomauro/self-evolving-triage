"""Download issue history (text + final labels) and label metadata from public GitHub repos.

Usage: python -m triage.fetch [owner/repo ...]   (defaults to the repos in config/repos.toml)

Writes to data/issues.db. Uses your `gh` CLI login for the token. Read-only.
"""
from __future__ import annotations

import subprocess
import sys
import time

import requests

from triage import config, db

GQL = "https://api.github.com/graphql"

ISSUES_QUERY = """
query($owner:String!, $name:String!, $after:String) {
  repository(owner:$owner, name:$name) {
    issues(first:100, after:$after, orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title body createdAt closedAt state stateReason authorAssociation
        labels(first:30) { nodes { name } }
      }
    }
  }
}"""

LABELS_QUERY = """
query($owner:String!, $name:String!, $after:String) {
  repository(owner:$owner, name:$name) {
    labels(first:100, after:$after) {
      pageInfo { hasNextPage endCursor }
      nodes { name description createdAt }
    }
  }
}"""


def _token() -> str:
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


def _gql(session: requests.Session, query: str, variables: dict) -> dict:
    for attempt in range(6):
        r = session.post(GQL, json={"query": query, "variables": variables}, timeout=60)
        if r.status_code == 200 and "errors" not in r.json():
            return r.json()["data"]
        wait = 2 ** attempt * 5
        print(f"  GitHub {r.status_code}, retrying in {wait}s: {r.text[:200]}", file=sys.stderr)
        time.sleep(wait)
    raise RuntimeError("GitHub GraphQL kept failing")


def fetch_repo(session: requests.Session, repo: str, max_issues: int, created_before: str) -> None:
    owner, name = repo.split("/")
    conn = db.connect()

    after = None
    labels = []
    while True:
        page = _gql(session, LABELS_QUERY, {"owner": owner, "name": name, "after": after})["repository"]["labels"]
        labels += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    conn.execute("DELETE FROM labels WHERE repo=?", (repo,))
    conn.executemany(
        "INSERT INTO labels(repo, name, description, created_at) VALUES (?,?,?,?)",
        [(repo, l["name"], l["description"] or "", l["createdAt"]) for l in labels],
    )

    conn.execute("DELETE FROM issues WHERE repo=?", (repo,))
    after, kept = None, 0
    while kept < max_issues:
        page = _gql(session, ISSUES_QUERY, {"owner": owner, "name": name, "after": after})["repository"]["issues"]
        rows = []
        for n in page["nodes"]:
            if n["createdAt"] >= created_before:   # too recent: labels may not be settled yet
                continue
            rows.append((
                repo, n["number"], n["title"], n["body"] or "", n["createdAt"], n["closedAt"],
                n["state"], n["stateReason"], n["authorAssociation"],
                "\n".join(l["name"] for l in n["labels"]["nodes"]),
            ))
        rows = rows[: max_issues - kept]
        conn.executemany("INSERT INTO issues VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
        kept += len(rows)
        print(f"  {repo}: {kept} issues", end="\r")
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
    conn.commit()
    print(f"  {repo}: {kept} issues, {len(labels)} labels")


def main() -> None:
    cfg = config.load()
    repos = sys.argv[1:] or [r["repo"] for r in cfg["repos"]]
    session = requests.Session()
    session.headers["Authorization"] = f"bearer {_token()}"
    for repo in repos:
        fetch_repo(session, repo, cfg["fetch"]["max_issues"], cfg["fetch"]["created_before"])


if __name__ == "__main__":
    main()
