# Vector Retail — Production Finance AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet-blueviolet.svg)](https://www.anthropic.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **production-grade multi-agent financial advisor** built with LangGraph, Anthropic Claude, and a layered compliance architecture. Designed for retail investors, it provides portfolio analysis, risk assessment, tax observations, and rebalancing guidance — all gated by regulatory compliance checks, human-in-the-loop review, and a self-reflective meta-critic.

> **Disclaimer:** This software is for informational and educational purposes only. It does not constitute investment advice. Always consult qualified professionals before making financial decisions.

---

## Architecture

```
                         ┌──────────────┐
                         │  User Query  │
                         └──────┬───────┘
                                │
                     ┌──────────▼──────────┐
                     │  Security Layer     │
                     │  JWT · RBAC · PII   │
                     └──────────┬──────────┘
                                │
                     ┌──────────▼──────────┐
                     │   Data Oracle       │
                     │  Dual-source quotes │
                     │  Circuit breakers   │
                     └──────────┬──────────┘
                                │
             ┌──────────────────┼──────────────────┐
             │ Fan-out to 5 parallel agents        │
     ┌───────┼───────┬─────────┬─────────┐         │
     ▼       ▼       ▼         ▼         ▼         │
 Portfolio Market   Risk     Tax    Rebalance       │
 Analysis  Intel  Assessment Optim.  Agent          │
     │       │       │         │         │         │
     └───────┼───────┴─────────┴─────────┘         │
             │ Fan-in                               │
             ▼                                      │
   ┌─────────────────┐                              │
   │   Meta-Critic   │  Self-reflective audit       │
   │  Hallucination  │  of ALL agent outputs        │
   │  + Compliance   │  before any response         │
   └────────┬────────┘                              │
            ▼                                       │
   ┌─────────────────┐                              │
   │  HITL Gate      │  Priority-scored             │
   │  Human review   │  escalation tickets          │
   └────────┬────────┘                              │
            ▼                                       │
   ┌─────────────────┐                              │
   │  Synthesizer    │  Grounded, cited response    │
   │  + Disclaimer   │  with jurisdiction disclaimers│
   └─────────────────┘                              │
```

### Layer Architecture

| Layer | Component | Responsibility |
|-------|-----------|---------------|
| 0 | Security (`security/`) | JWT auth, RBAC, PII redaction |
| 1 | Data Oracle (`data/`) | Dual-source market data, circuit breakers, caching |
| 3 | Specialist Agents (`agents/`) | Portfolio, Market, Risk, Tax, Rebalance |
| 4 | Meta-Critic | Self-reflective audit of all agent outputs |
| 5 | HITL Gate (`evaluation/`) | Human-in-the-loop escalation with SLA tiers |
| 6 | Synthesizer | Grounded response synthesis with citations |
| 8 | Orchestrator | LangGraph state machine, shadow evaluation |

---

## Key Features

### Multi-Agent Orchestration
- **5 parallel specialist agents** with fan-out/fan-in via LangGraph
- **Meta-critic layer** auditing all outputs before any response is delivered
- **Confidence-based routing** — low confidence sessions auto-escalate to human review

### Compliance & Governance
- **SEC Reg BI / FINRA 2111** suitability checks on every session
- **NIST AI Risk Management Framework** alignment (Govern, Map, Measure, Manage)
- **OWASP LLM Top 10** mitigations (LLM01, LLM02, LLM06, LLM08, LLM09)
- **SOC 2 Type II** ready: SHA-256 hash-chained, tamper-evident audit trail
- **GDPR/CCPA** PII minimisation and redaction

### Production Resilience
- **Dual-source data validation** with configurable divergence thresholds
- **Circuit breakers** on all external API calls
- **Exponential backoff** via tenacity
- **Quote caching** with staleness detection
- **Graceful degradation** when data sources fail

### Evaluation & Deployment
- **Shadow evaluation** scoring 10% of traffic against quality ground-truth
- **Blue/green deployment** validation with automatic promote/hold recommendations
- **Human-in-the-loop** gate with priority tiers (Critical → Low) and SLA tracking

---

## Tech Stack

| Category | Technology |
|----------|-----------|
| Orchestration | LangGraph |
| LLM | Anthropic Claude Sonnet |
| Data Validation | Pydantic v2 |
| Market Data | yfinance (primary) + Alpha Vantage (secondary) |
| Logging | structlog (OpenTelemetry-compatible) |
| Resilience | tenacity, custom circuit breakers |
| Numerical | NumPy |
| Auth | python-jose (JWT) |
| CI/CD | GitHub Actions, Docker, Kubernetes |
| Testing | pytest, ruff, black, mypy, bandit, safety |

---

## Quick Start

```bash
# Clone and set up
git clone https://github.com/jasmine2chen/vector-retail.git
cd vector-retail
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env → add your ANTHROPIC_API_KEY

# Run
python -m vector_retail.main
```

### Example Output

```
═══════════════════════════════════════════════════════════════════
  VECTOR RETAIL — Production Finance AI Agent v2.0
═══════════════════════════════════════════════════════════════════

Session ID       : a1b2c3d4-...
Deployment Slot  : blue
Policy Version   : 2.0.0
Total Latency    : 4230ms
Audit Events     : 18
Chain Integrity  : ✓ VALID
HITL Escalated   : No
Shadow Eval      : 0.92

Agent Confidences:
  portfolio_analysis      : ████████░░ 85%
  market_intel            : ████████░░ 80%
  risk_assessment         : ███████░░░ 78%
  tax_optimization        : ██████░░░░ 65%
  rebalance               : ████████░░ 82%

Meta Confidence  : ████████░░ 81%

─────────────────────────────────────────────────────────────────
RESPONSE:
─────────────────────────────────────────────────────────────────
Your portfolio is performing well overall with a total value of $48,250...
```

---

## Testing

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run unit tests with coverage
pytest tests/ -v --cov=src/vector_retail --cov-report=term

# Lint + format
ruff check src/ tests/
black --check src/ tests/

# Security scan
bandit -r src/ -ll
safety check -r requirements.txt

# Type checking
mypy src/
```

---

## Configuration

All risk thresholds and policy rules live in [`config/policy_rules.json`](config/policy_rules.json) — no magic numbers in application code. Changes require version bumps and compliance sign-off.

Key configurable parameters:
- `trade_value_hitl_threshold_usd` — trade value triggering human review ($25,000 default)
- `max_single_position_pct` — concentration limits by risk profile
- `max_sector_exposure_pct` — sector exposure limits by risk profile
- `min_confidence_for_auto_response` — confidence floor for auto-response (0.75)
- `shadow_eval_sample_rate` — traffic sampling rate for quality scoring (10%)

---

## Project Structure

```
vector-retail/
├── src/vector_retail/
│   ├── agents/           # 5 parallel specialist agents + meta-critic + synthesizer
│   │   ├── base.py       # Base class with LLM call, PII redaction, error handling
│   │   ├── portfolio.py  # Portfolio analysis with concentration checks
│   │   ├── market.py     # Market intelligence (yfinance)
│   │   ├── risk.py       # VaR, max drawdown, plain-language risk explanation
│   │   ├── tax.py        # Tax-loss harvesting, wash-sale warnings
│   │   ├── rebalance.py  # Drift calculation, HITL-gated trades
│   │   ├── meta_critic.py # Self-reflective audit of all agent outputs
│   │   └── synthesizer.py # Grounded response with citations & disclaimers
│   ├── core/             # Domain models, policy engine, audit trail
│   ├── data/             # Data oracle with dual-source verification
│   ├── evaluation/       # HITL gate + shadow evaluation framework
│   ├── security/         # PII redaction, RBAC, JWT validation
│   ├── orchestrator.py   # LangGraph state machine
│   └── main.py           # Entry point and demo runner
├── tests/                # Unit + integration tests
├── config/               # Externalised policy rules
├── docs/                 # Compliance docs and runbook
├── k8s/                  # Kubernetes deployment manifests
├── scripts/              # Health check and utilities
└── .github/workflows/    # CI pipeline (lint, test, security scan)
```

---

## Documentation

- [Compliance & Regulatory Alignment](docs/COMPLIANCE.md) — SEC Reg BI, NIST AI RMF, OWASP, SOC 2, GDPR/CCPA
- [Runbook](docs/RUNBOOK.md) — Quickstart, troubleshooting, monitoring, deployment

---

## License

[MIT License](LICENSE) — For informational and educational purposes only. Not investment advice.
