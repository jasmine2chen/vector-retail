# Runbook — vector-retail

## Quickstart

```bash
git clone https://github.com/jasmine2chen/vector-retail
cd vector-retail
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
python -m vector_retail.main
```

## Common Issues

### "ANTHROPIC_API_KEY not set"
Add to `.env`: `ANTHROPIC_API_KEY=sk-ant-...`

### "No module named 'numpy'"
Run: `pip install -r requirements.txt`

### "zsh: command not found: python"
Use `python3`. Or activate your venv: `source venv/bin/activate`

### Circuit breaker OPEN on yfinance
yfinance is rate-limited. Wait 60 seconds for cooldown, or use `DEPLOYMENT_SLOT=shadow` to test without live data.

### HITL ticket created — no response
Expected behaviour for trades > $25,000 or confidence < 75%.
Check `result["hitl_tickets"]` for ticket details.

## Monitoring

Key log fields to alert on:
- `circuit_opened` — market data source degraded
- `audit_chain_integrity_failed` — potential tampering
- `pii_redacted` with high count — unusual PII in input
- `llm_call_failed` — LLM API issue
- `all_sources_failed` — both data sources down

## Blue/Green Deployment

```bash
# Deploy green
DEPLOYMENT_SLOT=green python -m vector_retail.main

# Compare shadow eval scores
# PROMOTE if mean shadow eval score >= 0.80
```

## Configuration Changes

All thresholds in `config/policy_rules.json`.
Changes require:
1. PR with compliance sign-off
2. Version bump in `"version"` field
3. Deploy with new version — logged in every audit event
