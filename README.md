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
| **Multi-agent orchestration** | 6 parallel specialist agents + meta-critic, fan-out/fan-in via LangGraph |
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
               │  Pass 2: LLM-as-judge rubric │  ← 4-dim eval
               │  → blue/green decision       │
               └─────────────────────────────┘


---

## License

[MIT License](LICENSE) — For informational and educational purposes only. Not investment advice.
