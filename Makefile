PY ?= python3

.PHONY: install fetch classic estimate pilot sage report experiment

install:
	$(PY) -m pip install -r requirements.txt

fetch:            ## download issue history from GitHub (needs `gh auth login`)
	$(PY) -m triage.fetch

classic:          ## versions A and B: free, runs locally
	$(PY) -m triage.replay --bots A,B

estimate:         ## Sage token/cost estimate for C and D, no calls
	$(PY) -m triage.replay --bots C,D --dry-run

pilot:            ## small paid run: first 100 issues per repo, version C only (~$0.15)
	$(PY) -m triage.replay --bots C --limit 100

sage:             ## versions C and D on everything (needs SAGE_API_KEY and Anthropic credentials, ~$10)
	$(PY) -m triage.replay --bots C,D

report:           ## scores + charts from recorded runs, no API calls
	$(PY) -m triage.metrics

experiment: fetch classic sage report
