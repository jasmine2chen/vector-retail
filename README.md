# 🏦 Vector Retail: Institutional-Grade Finance AI Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)]([https://www.python.org/downloads/](https://www.python.org/downloads/))
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful-orange.svg)](https://python.langchain.com/docs/langgraph/)

A production-ready AI agent designed to answer complex financial queries over corporate filings with **zero math hallucination** and strict regulatory compliance.

## 🧠 Architectural Philosophy
Most GenAI demos fail in finance because they rely on LLMs to perform arithmetic and loosely cite sources. This architecture restricts the LLM to Semantic Extraction and Natural Language Synthesis. All critical logic is handled by deterministic code.

### Key Engineering Features
1. **Deterministic Math Layer:** Extracts variables into strict `Pydantic` schemas, normalizes units, and routes them to a Python `Decimal` tool to prevent floating-point drift. Includes strict guards against **Division by Zero** and **Currency Mismatches**.
2. **Hard Citation Enforcement:** The generation node executes a "Hard Fail" if the LLM hallucinates a source ID.
3. **Controlled Vocabulary (Enums):** Financial metrics are bound to strict Enums, preventing the LLM from inventing metrics.
4. **Data Access Layer Abstraction:** Decoupled vector retrieval logic for scalability.

## 🚀 Quick Start

    make install
    # Create a .env file with OPENAI_API_KEY=sk-...
    make run

Run the "Torture Suite" tests to validate deterministic safety:

    make test
