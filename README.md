# Vector Retail — Production Finance AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet-blueviolet.svg)](https://www.anthropic.com)
[![LangFuse](https://img.shields.io/badge/observability-LangFuse-teal.svg)](https://langfuse.com)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT%20%7C%20HuggingFace-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB%20%7C%20SEC%20EDGAR-red.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **production-grade multi-agent financial advisor** demonstrating applied AI engineering across the full stack: multi-agent orchestration, evaluation-driven development, LLM observability, prompt security, and compliance governance.

Built with LangGraph, Anthropic Claude Sonnet, and a layered compliance architecture aligned with SEC Reg BI, FINRA, NIST AI RMF, and SOC 2 Type II.

> **Disclaimer:** For informational and educational purposes only. Not investment advice.

---

## What this project demonstrates

This project is a deliberate showcase of **production AI engineering** skills across every layer that separates demos from deployed systems:

| Skill Area | Implementation |
|---|---|
| **Multi-agent orchestration** | 6 parallel specialist agents + meta-critic, fan-out/fan-in via LangGraph |
| **Agentic design patterns** | Reflection loop, Tool use, Planning, Multi-agent collaboration (Ng's 4 patterns) |
| **RAG / Vector DB** | ChromaDB + BAAI/bge-small-en-v1.5 embeddings; SEC EDGAR 10-K/10-Q ingestion pipeline; sentiment agent grounded in company filings |
| **NLP model evaluation** | FinBERT vs TF-IDF vs Claude zero-shot on Financial PhraseBank; primary metrics: Neg-Recall, MCC, P95 latency, cost per 10K (banking/finance industry standards) |
| **Evaluation-driven development** | LLM-as-judge shadow evaluator + heuristic scoring + blue/green deployment |
| **LLM observability** | LangFuse traces per agent call — latency, tokens, confidence, prompt version |
| **Prompt engineering** | Versioned YAML prompt registry; prompts as deployable artifacts with rollback |
| **LLM security** | Prompt injection defense (OWASP LLM Top 10 #1), PII redaction, RBAC |
| **Confidence scoring** | Auditable multiplicative penalty chain; drives HITL routing and reflection |
| **Human-in-the-loop** | Priority-tiered escalation gate with SLA tracking and LangGraph resume |
| **Production resilience** | Dual-source data, circuit breakers, exponential backoff, graceful degradation |
| **Compliance & audit** | SHA-256 hash-chained audit trail, policy-engine-driven rules, SEC/FINRA alignment |
| **MLOps / deployment** | LangGraph MemorySaver checkpointing, shadow eval→blue/green promotion, K8s manifests |
| **Testing** | Business logic compliance assertions (not just structural) — proves the rules fire |

---

## Architecture

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  Security Layer  │  JWT · RBAC · PII Redaction          │
│                  │  Prompt Injection Defense (OWASP #1) │
└─────────────────────────────┬───────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Data Oracle      │
                    │  yfinance primary  │
                    │  Alpha Vantage 2°  │  ← dual-source cross-validation
                    │  Circuit breakers  │
                    └─────────┬──────────┘
                              │ fan-out
        ┌──────────────────┬──┴──────┬──────────┬────────────┐
        ▼         ▼        ▼         ▼          ▼            ▼
  Portfolio    Market     Risk       Tax      Rebalance  Sentiment
  Analysis     Intel   Assessment  Optim.      Agent     Analysis
  (conc. chk) (no pred)(VaR/dd) (TLH/wash) (drift/HITL)(FinBERT)
                                                          ▲
                                              ┌───────────┴──────────┐
                                              │  ChromaDB (RAG)      │
                                              │  SEC EDGAR 10-K/10-Q │  ← offline ingestion
                                              │  BAAI/bge-small-en   │
                                              └──────────────────────┘
        │         │        │         │          │            │
        └─────────┴────────┴────┬────┴──────────┴────────────┘
                            │ fan-in
                   ┌────────▼────────┐
                   │   Meta-Critic   │  ← self-reflective audit
                   │  hallucination  │     confidence banding
                   │  policy sweep   │     Reflection pattern
                   └────────┬────────┘
                            │
               ┌────────────▼─────────────┐
               │  Confidence Router       │
               │  ≥ 0.85  → synthesis     │  ← auto-response
               │  0.75–0.85 → synthesis   │  ← reflection loop (critique fed back)
               │  < 0.75  → HITL gate     │  ← human escalation
               └──────────────────────────┘
                            │
                   ┌────────▼────────┐
                   │   Synthesizer   │  ← versioned prompts
                   │  grounded resp  │     revision-aware
                   │  + disclaimer   │     jurisdiction-specific
                   └────────┬────────┘
                            │
               ┌────────────▼────────────────┐
               │  Shadow Evaluator (10%)      │
               │  Pass 1: heuristic scoring   │
               │  Pass 2: LLM-as-judge rubric │  ← 4-dim compliance eval
               │  → blue/green decision       │
               └─────────────────────────────┘
```

---

## AI Engineering Highlights

### 1. Agentic Design Patterns (Andrew Ng's 4 Patterns)

**Reflection** — The meta-critic agent audits all 6 specialist outputs before delivery. When overall confidence falls in the 0.75–0.85 band, its critique is fed back to the synthesizer for a revision pass. LangGraph conditional edge routes medium-concern sessions through this loop rather than straight to the client.

**Tool Use** — The Data Oracle exposes `get_verified_quote` and `get_portfolio_quotes` as callable tools, with dual-source cross-validation and circuit breakers. Each agent declares which tools it used in `AgentResult.data_sources`.

**Planning** — The orchestrator decomposes each user query across 6 specialist agents concurrently (LangGraph fan-out), then converges results at the meta-critic (fan-in) before synthesis.

**Multi-Agent Collaboration** — The 6 parallel agents share a common `GraphState` contract and a hash-chained audit trail. The meta-critic reads all outputs and the synthesizer assembles the final response from all findings.

```python
# Conditional routing — reflection vs HITL vs auto
graph.add_conditional_edges(
    "meta_critic",
    route_after_meta_critic,
    {"synthesis": "synthesis", "hitl_gate": "hitl_gate"},
)
```

---

### 2. Evaluation-Driven Development

Shadow evaluation runs on 10% of traffic with **two independent scoring passes**:

**Pass 1 — Heuristic** (fast, deterministic): regex checks for disclaimer presence, price prediction language, risk disclosure, data attribution.

**Pass 2 — LLM-as-Judge**: a focused compliance LLM call scores on a 4-dimension rubric with JSON output — factual grounding, regulatory compliance, risk disclosure, and user suitability. Returns a score, rationale, and specific flags.

**Weighted blend**: 40% heuristic + 60% LLM judge → `overall_score`. Drives blue/green promotion: score ≥ 0.80 → PROMOTE.

```python
# From evaluation/shadow_eval.py
{
  "factual_grounding": 0.90,
  "regulatory_compliance": 0.95,
  "risk_disclosure": 0.85,
  "user_suitability": 0.88,
  "overall": 0.90,
  "rationale": "Response is well-grounded in provided portfolio data...",
  "flags": []
}
```

Business logic tests assert on compliance outcomes — not just that the pipeline runs:
```python
# Tests prove the rules actually fire
def test_unverified_kyc_triggers_policy_flag(...)        # → KYC_FAIL flag
def test_concentrated_position_triggers_flag(...)        # → CONCENTRATION flag
def test_large_trade_triggers_hitl_escalation(...)       # → hitl_escalated=True
def test_audit_chain_integrity_always_holds(...)         # → chain_integrity=True
def test_response_contains_regulatory_disclaimer(...)    # → SEC Reg BI compliance
```

---

### 3. NLP Model Evaluation & Selection (FinBERT)

The `SentimentAnalysisAgent` is the 6th parallel specialist — it scores recent news headlines for each holding using [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial text (Araci, 2019). The model choice was validated empirically before integration.

**Baseline comparison** — [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb) benchmarks three approaches on [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) (`sentences_allagree`, n=2,264):

| Model | Neg-Recall | MCC | P95 Latency | Cost / 10K samples |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.71 | ~0.67 | < 2ms | $0.00 |
| **FinBERT** (`ProsusAI/finbert`) | **~0.96** | **~0.96** | **~55ms** | **$0.00** |
| Claude Sonnet (zero-shot) | ~0.87 | ~0.86 | ~980ms | ~$0.15 |

Metrics reflect banking/finance industry standards: **Negative-class Recall** (primary — missing a bearish signal is a fiduciary failure under SEC Reg BI); **MCC** (SR 11-7 model validation standard, robust under the ~60% neutral class imbalance in financial text); **P95 Latency** (SLA tail, not mean — what actually breaches service agreements); **Cost per 10K** (production unit economics at 10K daily sessions × 40 headlines).

McNemar's test confirms the FinBERT vs TF-IDF gap is statistically significant (p < 0.001). FinBERT selected: highest Neg-Recall (0.96) and MCC (0.96), $0 inference cost at any scale, offline-capable, no external API dependency at runtime.

**Production integration:**
- Thread-safe singleton with double-checked locking — model loads once per process (~500MB, cached to `~/.cache/huggingface/`)
- **Batch inference** across all symbols in a single forward pass (4× throughput vs per-headline calls)
- **Exponential recency decay** — `weight = exp(-0.15 × rank)` so the most recent headline has 3× the weight of the 8th
- Bearish signal threshold: `negative_prob > 0.40` → `SENTIMENT_BEARISH` policy flag
- Graceful degradation: model download failure, yfinance outage, and no-news scenarios all return valid `AgentResult` with penalised confidence

```python
# Per-symbol result surfaced in AgentResult.findings
{
  "TSLA": {
    "positive": 0.08, "negative": 0.75, "neutral": 0.17,
    "dominant": "negative", "n_headlines": 5, "is_bearish": True
  }
}
# Policy flag automatically raised:
# "SENTIMENT_BEARISH: TSLA — negative sentiment 75% across 5 headlines (threshold: 40%)"
```

Research dependencies (jupyter, datasets, scikit-learn, scipy) are kept in [`requirements-research.txt`](requirements-research.txt), separate from the production image.

---

### 4. RAG — Company Fundamentals Retrieval (ChromaDB + SEC EDGAR)

The `SentimentAnalysisAgent` is augmented with a **Retrieval-Augmented Generation** layer that grounds LLM commentary in company-specific SEC filings — risk factors from 10-K annual reports and MD&A sections from 10-Q quarterly reports.

**Why RAG here?** News headlines (FinBERT's input) are short and high-frequency; they lack the structured forward-looking language regulators care about. SEC filings contain the exact risk disclosures, liquidity commentary, and management assessments that back up or contradict a bearish signal. Grounding LLM output in primary source documents reduces hallucination risk and satisfies SR 11-7's documentation requirements.

**Pipeline: `scripts/ingest_fundamentals.py`**

```
SEC EDGAR API  →  SECEdgarClient  →  FilingChunker  →  FundamentalsEmbedder  →  ChromaDB
(10-K / 10-Q)     (CIK lookup,       (HTML→text,       (embed + upsert,         (cosine HNSW)
                   rate-limited       400-word chunks    batch=50,
                   0.12s/req)         50-word overlap)   idempotent)
```

Run once before first use (idempotent — safe to re-run):
```bash
python scripts/ingest_fundamentals.py --symbols AAPL MSFT TSLA NVDA
python scripts/ingest_fundamentals.py --symbols AAPL --form-type 10-K 10-Q  # both form types
python scripts/ingest_fundamentals.py --from-holdings holdings.json          # from holdings file
```

**Runtime: `data/retriever.py`**

- Thread-safe singleton (same double-checked locking pattern as FinBERT)
- Embedding model: `BAAI/bge-small-en-v1.5` — 130MB, CPU-only, no API key, top MTEB retrieval score for its size class
- Query is **sentiment-anchored**: bearish signals trigger `"risk factors earnings pressure regulatory"` queries; bullish signals trigger `"growth outlook"` queries — fetching the most contextually relevant filing passages
- Staleness penalty: confidence is penalised if the newest filing is > 180 days old
- Top-2 passages per symbol injected into LLM prompt with full provenance (`[10-K 2024-11-01 §risk_factors relevance=0.91]`)
- Fully degrades if ChromaDB / embedding model unavailable — `get_retriever()` returns `None`, agent continues with headlines only

**Fintech compliance settings:**
```python
chromadb.PersistentClient(settings=Settings(
    anonymized_telemetry=False,  # No external telemetry — fintech requirement
    allow_reset=False,           # Prevent accidental collection wipe
))
```

**AgentResult output with RAG active:**
```python
{
  "findings": {
    "fundamentals_passages": {
      "TSLA": [{
        "filing_type": "10-K", "filed_date": "2024-11-01",
        "section": "risk_factors", "relevance_score": 0.91, "age_days": 162,
        "text_preview": "We face significant competition in the electric vehicle market..."
      }]
    }
  },
  "data_sources": ["yfinance_news", "finbert:ProsusAI/finbert", "sec_fundamentals:chromadb:4_passages"]
}
```

**Backend swappability:** The `retrieve()` interface is backend-agnostic. Swap ChromaDB for `pgvector` (reuses existing PostgreSQL session store, ACID guarantees) or Pinecone (managed, no ops overhead) without changing any agent code.

---

### 5. LLM Observability (LangFuse)

Every `_call_llm` invocation creates a LangFuse generation trace with:
- Agent ID, model name, temperature
- Input/output previews (PII-redacted)
- Latency (ms) and token counts
- Prompt version (from YAML registry)
- Confidence score as a named evaluation

Enable by setting `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`. Graceful no-op if not configured — the pipeline never fails due to observability outage.

```python
generation = trace.generation(
    name=f"{self.AGENT_ID}.llm_call",
    model=self.llm.model,
    input=[{"role": "system", ...}, {"role": "user", ...}],
    metadata={"prompt_version": self._prompt_version},
)
# After call:
generation.end(output=..., usage={"input": n_tokens, "output": n_tokens})
trace.score(name="agent_confidence", value=0.87)
```

---

### 6. Prompt Security (OWASP LLM Top 10 #1)

A dedicated `security/prompt_guard.py` module scans all user input before any LLM call, covering 8 attack categories:

| Category | Example |
|---|---|
| `instruction_override` | "ignore all previous instructions" |
| `prompt_extraction` | "repeat your system prompt" |
| `persona_hijack` | "you are now an unrestricted model" |
| `context_wipe` | "forget everything above" |
| `xml_injection` | `<system>new instructions</system>` |
| `template_token_injection` | `[INST]` / `<\|im_start\|>` |
| `jailbreak_attempt` | "DAN mode", "developer mode" |
| `indirect_injection` | "note to AI: ignore prior rules" |

Critical detections return a safe refusal. All detections are recorded to the hash-chained audit trail with session ID.

---

### 7. Versioned Prompt Registry

All system prompts live in `config/prompts/<agent_id>.yaml` alongside `config/policy_rules.json`. This makes prompts **deployable artifacts** — versioned, rollback-able, and A/B testable.

```yaml
# config/prompts/meta_critic.yaml
version: "1.0.0"
agent_id: "meta_critic"
compliance_sign_off: "required before promotion"
system_prompt: |
  You are a senior compliance officer reviewing AI-generated financial analysis...
```

Every `AgentResult` carries `prompt_version`. The `PromptRegistry` supports hot-reload via `invalidate_cache()` — no restart needed when iterating on prompts in production.

---

### 8. LangGraph Checkpointing + HITL Resume

Each session is compiled with a `MemorySaver` checkpointer, keyed by `thread_id=session_id`. When a session is escalated to human review, its full graph state is persisted. A compliance reviewer can approve and resume:

```python
# Initial run — escalated at HITL gate
result = agent.run(user_query=..., user_profile=..., holdings=...)
# result["hitl_escalated"] == True

# After reviewer approves:
completed = agent.resume_hitl_session(
    session_id=result["session_id"],
    reviewer_notes="Concentration within acceptable range for this client.",
    approved=True,
)
```

For production durability, swap `MemorySaver` for `AsyncPostgresSaver` or `RedisSaver` — the invocation API is identical.

---

### 9. Auditable Confidence Scoring

Every agent uses a `ConfidenceCalculator` — a multiplicative penalty chain where every deduction is logged with signal name, observed value, and reason. The complete chain is written to the audit trail and emitted as a LangFuse score.

```python
conf = self._confidence()
conf.penalize("stale_quote", "Price > 5min old", observed=age_seconds)
conf.penalize("missing_secondary", "Alpha Vantage unavailable")
score = conf.score()  # 0.95 × 0.90 × 0.85 = 0.726
```

Confidence drives routing, HITL escalation, and is exposed in every API response — giving compliance full chain of custody over every automated decision.

---



## Quick Start

```bash
git clone https://github.com/jasmine2chen/vector-retail.git
cd vector-retail
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add ANTHROPIC_API_KEY (required)
# Add LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY (optional — enables tracing)

# Ingest SEC filings into ChromaDB (one-time; safe to re-run)
# Downloads ~130MB embedding model on first run; requires internet access
python scripts/ingest_fundamentals.py --symbols AAPL MSFT TSLA NVDA
# Skip this step to run without fundamentals context (agent degrades gracefully)

python -m vector_retail.main
```



### Example session output

```
═══════════════════════════════════════════════════════════════════
  VECTOR RETAIL — Production Finance AI Agent v2.0
═══════════════════════════════════════════════════════════════════

Session ID        : a1b2c3d4-...
Deployment Slot   : blue
Policy Version    : 2.0.0
Prompt Versions   : portfolio=1.0.0  sentiment_analysis=1.0.0  meta_critic=1.0.0  synthesizer=1.0.0
Total Latency     : 4,890ms
Audit Events      : 24
Chain Integrity   : ✓ VALID
HITL Escalated    : No
Reflection Applied: No
Shadow Eval       : 0.91  (heuristic=0.88, llm_judge=0.93)

Agent Confidences:
  portfolio_analysis      : ████████░░  85%
  market_intel            : ████████░░  80%
  risk_assessment         : ███████░░░  78%
  tax_optimization        : ██████░░░░  65%
  rebalance               : ████████░░  82%
  sentiment_analysis      : ███████░░░  76%  [FinBERT · 32 headlines · 0 bearish]

Meta Confidence   : ████████░░  80%

─────────────────────────────────────────────────────────────────
RESPONSE:
─────────────────────────────────────────────────────────────────
Your portfolio is performing well overall, with a current total
value of $48,250 across 3 holdings (based on yfinance data)...

---
*This content is for informational purposes only and does not
constitute investment advice under SEC Regulation Best Interest...*
```

---

## Testing

```bash
# All tests
pytest tests/ -v

# Integration tests — compliance business logic
pytest tests/integration/ -v

# Unit tests — includes FinBERT unit tests (fully mocked, < 1s)
pytest tests/unit/ -v

# With coverage
pytest tests/ --cov=src/vector_retail --cov-report=term

# Security scan
bandit -r src/ -ll
safety check -r requirements.txt
```

All external dependencies are mocked in tests — no network calls, no API keys required, no FinBERT download, no ChromaDB:
- **Integration tests**: `ChatAnthropic`, `yf.Ticker` (oracle + sentiment), and `_load_finbert` are all patched. Sentiment agent runs in graceful no-data mode.
- **Unit tests**: FinBERT pipeline is mocked deterministically; `get_retriever` is patched to return `None` (RAG degrades gracefully); all degradation paths tested in < 1s.

The integration tests assert on **compliance outcomes**, not just pipeline execution:

```bash
PASSED tests/integration/test_orchestrator_pipeline.py::TestComplianceBusinessLogic::test_unverified_kyc_triggers_policy_flag
PASSED tests/integration/test_orchestrator_pipeline.py::TestComplianceBusinessLogic::test_concentrated_position_triggers_concentration_flag
PASSED tests/integration/test_orchestrator_pipeline.py::TestComplianceBusinessLogic::test_large_trade_triggers_hitl_escalation
PASSED tests/integration/test_orchestrator_pipeline.py::TestComplianceBusinessLogic::test_audit_chain_integrity_always_holds
PASSED tests/integration/test_orchestrator_pipeline.py::TestComplianceBusinessLogic::test_response_contains_regulatory_disclaimer
```

Sentiment agent unit tests cover all degradation paths (FinBERT mocked — no network, no GPU, sub-1s):

```bash
PASSED tests/unit/test_sentiment.py::TestSentimentScore::test_dominant_label_positive
PASSED tests/unit/test_sentiment.py::TestSentimentScore::test_bearish_flag_triggers_above_threshold
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_bullish_headlines_produce_positive_sentiment
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_bearish_headlines_produce_policy_flag
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_no_news_returns_gracefully_with_penalised_confidence
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_finbert_failure_degrades_gracefully
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_multi_symbol_batch_inference
PASSED tests/unit/test_sentiment.py::TestSentimentAgent::test_confidence_penalised_for_thin_news
```

---

## Configuration

All risk thresholds live in [`config/policy_rules.json`](config/policy_rules.json). All system prompts live in [`config/prompts/`](config/prompts/). No magic numbers or hardcoded strings in application code.

```jsonc
// config/policy_rules.json
{
  "version": "2.0.0",
  "trade_value_hitl_threshold_usd": 25000,       // Trades > $25k → human review
  "min_confidence_for_auto_response": 0.75,       // < 0.75 → HITL escalation
  "shadow_eval_sample_rate": 0.10,                // 10% traffic sampling
  "max_single_position_pct": {
    "conservative": 0.05,
    "moderate": 0.10,                             // FINRA Rule 2111 enforcement
    "aggressive": 0.20
  }
}
```

---

## Project Structure

```
vector-retail/
├── src/vector_retail/
│   ├── agents/
│   │   ├── base.py            # LangFuse tracing, prompt injection scan, PII redaction
│   │   ├── portfolio.py       # Concentration checks (FINRA 2111)
│   │   ├── market.py          # Market intel (no price predictions)
│   │   ├── risk.py            # VaR (historical simulation), max drawdown
│   │   ├── tax.py             # Tax-loss harvesting, wash-sale (30-day rule)
│   │   ├── rebalance.py       # Drift detection, HITL-gated trades
│   │   ├── sentiment.py       # FinBERT + RAG (SEC filings) news sentiment (6th agent)
│   │   ├── meta_critic.py     # Self-reflective audit + reflection routing
│   │   └── synthesizer.py     # Revision-aware response synthesis
│   ├── core/
│   │   ├── models.py          # Pydantic v2 contracts (AgentResult, GraphState, ...)
│   │   ├── policy.py          # PolicyEngine — SEC/FINRA rule enforcement
│   │   ├── prompts.py         # Versioned prompt registry (YAML-backed)
│   │   └── audit.py           # SHA-256 hash-chained audit trail (SOC 2)
│   ├── data/
│   │   ├── oracle.py          # Dual-source market data, caching, staleness
│   │   ├── circuit_breaker.py # Resilience pattern for API calls
│   │   ├── retriever.py       # ChromaDB singleton — semantic search over SEC filings
│   │   └── embedder.py        # SEC EDGAR ingestion: fetch → chunk → embed → upsert
│   ├── evaluation/
│   │   ├── hitl.py            # HITL escalation gate + SLA-tiered tickets
│   │   └── shadow_eval.py     # LLM-as-judge + heuristic scoring + blue/green
│   ├── security/
│   │   ├── pii.py             # PII redaction (regex; Presidio-ready)
│   │   ├── rbac.py            # RBAC with least-privilege role model
│   │   └── prompt_guard.py    # Prompt injection defense (OWASP LLM #1)
│   ├── orchestrator.py        # LangGraph graph, conditional routing, checkpointing
│   └── server.py              # FastAPI server
├── config/
│   ├── policy_rules.json      # Versioned compliance thresholds
│   └── prompts/               # Versioned agent system prompts (YAML)
│       ├── sentiment_analysis.yaml
│       ├── meta_critic.yaml
│       ├── synthesizer.yaml
│       └── ...
├── notebooks/
│   └── model_evaluation.ipynb # FinBERT vs TF-IDF vs Claude on Financial PhraseBank
├── scripts/
│   └── ingest_fundamentals.py # Offline SEC EDGAR ingestion CLI (run before first use)
├── tests/
│   ├── unit/                  # Oracle, PII, policy, shadow eval, sentiment (mocked FinBERT + RAG)
│   └── integration/           # End-to-end compliance business logic tests
├── docs/
│   ├── COMPLIANCE.md          # SEC Reg BI, FINRA, NIST AI RMF, SOC 2, GDPR
│   └── RUNBOOK.md             # Deployment, monitoring, troubleshooting
├── requirements.txt           # Production dependencies
├── requirements-research.txt  # Notebook / research dependencies (not in Docker image)
├── k8s/                       # Kubernetes manifests (rolling update strategy)
└── .github/workflows/         # CI: lint, test, security scan, coverage
```

---

## Compliance Alignment

| Framework | How it's implemented |
|---|---|
| **SEC Reg BI** | Suitability checks per session, jurisdiction-specific disclaimers, no specific advice without caveats |
| **FINRA Rule 2111** | Concentration limits enforced by PolicyEngine against client risk profile |
| **NIST AI RMF** | Govern (versioned policies/prompts), Map (reasoning chains), Measure (shadow eval), Manage (meta-critic + HITL) |
| **OWASP LLM Top 10** | LLM01: prompt injection guard; LLM02: output validation; LLM06: PII redaction; LLM08: RBAC |
| **SOC 2 Type II** | SHA-256 hash-chained audit trail; export hooks for S3 Object Lock / Azure WORM |
| **GDPR / CCPA** | Name stored as first name + last initial only; PII stripped before LLM calls |

---

## License

[MIT License](LICENSE) — For informational and educational purposes only. Not investment advice.
