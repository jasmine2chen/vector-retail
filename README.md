# Vector Retail — Production Finance AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Anthropic Claude](https://img.shields.io/badge/LLM-Claude%20Sonnet-blueviolet.svg)](https://www.anthropic.com)
[![LangFuse](https://img.shields.io/badge/observability-LangFuse-teal.svg)](https://langfuse.com)
[![FinBERT](https://img.shields.io/badge/NLP-FinBERT%20%7C%20HuggingFace-yellow.svg)](https://huggingface.co/ProsusAI/finbert)
[![ChromaDB](https://img.shields.io/badge/RAG-ChromaDB%20%7C%20SEC%20EDGAR-red.svg)](https://www.trychroma.com)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A **production-grade multi-agent financial advisor** built with LangGraph, Anthropic Claude Sonnet, FinBERT, and ChromaDB. Demonstrates applied AI engineering across multi-agent orchestration, NLP model evaluation, retrieval-augmented generation, and LLM observability.

> **Disclaimer:** For informational and educational purposes only. Not investment advice.

---

## What this project demonstrates

| Skill Area | Implementation |
|---|---|
| **Multi-agent orchestration** | 5 parallel specialist agents + meta-critic, fan-out/fan-in via LangGraph |
| **Agentic design patterns** | Reflection loop, Tool use, Planning, Multi-agent collaboration (Ng's 4 patterns) |
| **RAG / Vector DB** | ChromaDB + BAAI/bge-small-en-v1.5 embeddings; SEC EDGAR 10-K/10-Q ingestion pipeline; sentiment agent grounded in company filings |
| **NLP model evaluation** | FinBERT vs TF-IDF vs Claude zero-shot; primary metrics: Neg-Recall, MCC, P95 latency, cost per 10K |
| **LLM observability** | LangFuse traces per agent call — latency, tokens, confidence, prompt version |
| **Prompt engineering** | Versioned YAML prompt registry; prompts as deployable artifacts with rollback |
| **Production resilience** | Dual-source data, circuit breakers, exponential backoff, graceful degradation |
| **Human-in-the-loop** | Priority-tiered escalation gate with SLA tracking and LangGraph checkpoint resume |

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
        ▼         ▼        ▼         ▼            ▼
  Portfolio    Market     Risk    Rebalance  Sentiment
  Analysis     Intel   Assessment   Agent    Analysis
  (conc. chk) (no pred)(VaR/dd) (drift/HITL)(FinBERT)
                                                          ▲
                                              ┌───────────┴──────────┐
                                              │  ChromaDB (RAG)      │
                                              │  SEC EDGAR 10-K/10-Q │  ← offline ingestion
                                              │  BAAI/bge-small-en   │
                                              └──────────────────────┘
        │         │        │         │            │
        └─────────┴────────┴────┬────┴────────────┘
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
               │  Pass 2: LLM-as-judge rubric │  ← 4-dim eval
               │  → blue/green decision       │
               └─────────────────────────────┘
```

---

## NLP Model Evaluation & Selection (FinBERT)

The `SentimentAnalysisAgent` is the 5th parallel specialist — it scores recent news headlines for each holding using [FinBERT](https://huggingface.co/ProsusAI/finbert), a BERT model fine-tuned on financial text (Araci, 2019). The model choice was validated empirically before integration.

**Baseline comparison** — [`notebooks/model_evaluation.ipynb`](notebooks/model_evaluation.ipynb) benchmarks three approaches on [Financial PhraseBank](https://huggingface.co/datasets/financial_phrasebank) (`sentences_allagree`, n=2,264):

| Model | Neg-Recall | MCC | P95 Latency | Cost / 10K samples |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | ~0.71 | ~0.67 | < 2ms | $0.00 |
| **FinBERT** (`ProsusAI/finbert`) | **~0.96** | **~0.96** | **~55ms** | **$0.00** |
| Claude Sonnet (zero-shot) | ~0.87 | ~0.86 | ~980ms | ~$0.15 |

**Why these metrics:** Negative-class Recall is the primary metric — missing a bearish signal is a fiduciary failure. MCC is robust to the ~60% neutral class imbalance in financial text (SR 11-7 model validation standard). P95 latency measures SLA tail, not mean. Cost per 10K reflects production unit economics.

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

---

## RAG — Company Fundamentals Retrieval (ChromaDB + SEC EDGAR)

The `SentimentAnalysisAgent` is augmented with a **Retrieval-Augmented Generation** layer that grounds LLM commentary in company-specific SEC filings — risk factors from 10-K annual reports and MD&A sections from 10-Q quarterly reports.

**Why RAG here?** News headlines are short and high-frequency; they lack the structured forward-looking language that backs up or contradicts a bearish signal. SEC filings contain the exact risk disclosures, liquidity commentary, and management assessments that ground LLM output in primary source documents.

**Ingestion pipeline:**

```
SEC EDGAR API  →  SECEdgarClient  →  FilingChunker  →  FundamentalsEmbedder  →  ChromaDB
(10-K / 10-Q)     (CIK lookup,       (HTML→text,       (embed + upsert,         (cosine HNSW)
                   rate-limited       400-word chunks    batch=50,
                   0.12s/req)         50-word overlap)   idempotent)
```

```bash
python scripts/ingest_fundamentals.py --symbols AAPL MSFT TSLA NVDA
python scripts/ingest_fundamentals.py --symbols AAPL --form-type 10-K 10-Q
python scripts/ingest_fundamentals.py --from-holdings holdings.json
```

**Runtime retrieval (`data/retriever.py`):**
- Thread-safe singleton — same double-checked locking pattern as FinBERT
- Embedding model: `BAAI/bge-small-en-v1.5` — 130MB, CPU-only, no API key, top MTEB retrieval score for its size class
- **Sentiment-anchored queries**: a bearish FinBERT signal rewrites the ChromaDB query toward `"risk factors earnings pressure regulatory"` — fetching the most contextually relevant filing passages
- Staleness penalty: confidence penalised if newest filing > 180 days old
- Top-2 passages per symbol injected into LLM prompt with full provenance: `[10-K 2024-11-01 §risk_factors relevance=0.91]`
- Fully degrades if ChromaDB unavailable — `get_retriever()` returns `None`, agent continues with headlines only

```python
{
  "data_sources": ["yfinance_news", "finbert:ProsusAI/finbert", "sec_fundamentals:chromadb:4_passages"],
  "findings": {
    "fundamentals_passages": {
      "TSLA": [{"filing_type": "10-K", "filed_date": "2024-11-01",
                "section": "risk_factors", "relevance_score": 0.91}]
    }
  }
}
```

**Backend swappability:** The `retrieve()` interface is backend-agnostic. Swap ChromaDB for `pgvector` or Pinecone without changing any agent code.

---

## LLM Observability (LangFuse)

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

## Versioned Prompt Registry

All system prompts live in `config/prompts/<agent_id>.yaml`. This makes prompts **deployable artifacts** — versioned, rollback-able, and A/B testable without a code deploy.

```yaml
# config/prompts/meta_critic.yaml
version: "1.0.0"
agent_id: "meta_critic"
system_prompt: |
  You are a senior compliance officer reviewing AI-generated financial analysis...
```

Every `AgentResult` carries `prompt_version`. The `PromptRegistry` supports hot-reload via `invalidate_cache()` — no restart needed when iterating on prompts in production.

---

## License

[MIT License](LICENSE) — For informational and educational purposes only. Not investment advice.
