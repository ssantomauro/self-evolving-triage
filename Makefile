PYTHON ?= python3              # any Python 3.9+; 3.11+ recommended
PY := .venv/bin/python

.PHONY: install fetch classic estimate pilot sage report experiment

install:          ## create .venv and install dependencies
	$(PYTHON) -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt

fetch:            ## download issue history from GitHub (needs `gh auth login`)
	$(PY) -m triage.fetch

classic:          ## versions A and B: free, runs locally
	$(PY) -m triage.replay --bots A,B

estimate:         ## Sage token/cost estimate for C and D, no calls
	$(PY) -m triage.replay --bots C,D --dry-run

pilot:            ## small paid run: first 200 issues per repo, versions C and D (~$1; D needs Anthropic credentials)
	$(PY) -m triage.replay --bots C,D --limit 200

sage:             ## versions C and D on everything (needs SAGE_API_KEY and Anthropic credentials, ~$10)
	$(PY) -m triage.replay --bots C,D

report:           ## scores + charts from recorded runs, no API calls
	$(PY) -m triage.metrics

experiment: fetch classic sage report
