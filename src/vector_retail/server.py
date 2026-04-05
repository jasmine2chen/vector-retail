"""
server.py
Production HTTP API for Vector Retail Financial AI Agent.

Exposes the orchestrator as a REST service:
  POST /v1/advise  — run a full advisory session
  GET  /health     — liveness / readiness probe

Run locally:
    uvicorn vector_retail.server:app --host 0.0.0.0 --port 8080
"""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from .core.enums import (
    DeploymentSlot,
)
from .core.models import PortfolioHolding, UserProfile
from .orchestrator import VectorRetailAgent

log = structlog.get_logger("server")

# ── Startup / shutdown ───────────────────────────────────────────────────────

_agent: VectorRetailAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise the agent once on startup, reuse for every request."""
    global _agent
    slot = os.getenv("DEPLOYMENT_SLOT", "blue")
    _agent = VectorRetailAgent(
        deployment_slot=DeploymentSlot(slot),
    )
    log.info("agent_ready", deployment_slot=slot)
    yield
    log.info("shutting_down")


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vector Retail — Financial AI Agent",
    description=(
        "Production-grade multi-agent financial advisor. "
        "SEC Reg BI · FINRA · NIST AI RMF compliant."
    ),
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Api-Key"],
)

# ── Auth ─────────────────────────────────────────────────────────────────────

api_key_header = APIKeyHeader(name="X-Api-Key", auto_error=False)


async def verify_api_key(
    api_key: str | None = Security(api_key_header),
) -> str:
    """
    Validate the API key. In production, check against a database or
    Secrets Manager. Disabled when REQUIRE_API_KEY is not set (dev mode).
    """
    required_key = os.getenv("API_KEY")
    if not required_key:
        return "dev_mode"
    if not api_key or api_key != required_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return api_key


# ── Request / Response models ────────────────────────────────────────────────

class HoldingRequest(BaseModel):
    """Single portfolio holding in the request body."""
    symbol: str = Field(..., example="AAPL")
    quantity: int = Field(..., gt=0, example=50)
    cost_basis_per_share: float = Field(..., gt=0, example=145.00)
    purchase_date: str = Field(..., example="2022-03-15")
    sector: str = Field(..., example="Technology")
    asset_class: str = Field(..., example="equity")


class AdviseRequest(BaseModel):
    """POST /v1/advise request body."""
    query: str = Field(
        ...,
        min_length=5,
        example="How is my portfolio performing? Should I rebalance?",
    )
    user: UserProfile
    holdings: list[HoldingRequest] = Field(..., min_length=1)

    model_config = {"json_schema_extra": {
        "example": {
            "query": "How is my portfolio performing?",
            "user": {
                "name": "Jane Doe",
                "risk_tolerance": "moderate",
                "account_type": "individual",
                "jurisdiction": "us",
                "kyc_verified": True,
                "investment_horizon_years": 15,
            },
            "holdings": [
                {
                    "symbol": "AAPL",
                    "quantity": 50,
                    "cost_basis_per_share": 145.00,
                    "purchase_date": "2022-03-15",
                    "sector": "Technology",
                    "asset_class": "equity",
                }
            ],
        }
    }}


class AdviseResponse(BaseModel):
    """POST /v1/advise response body."""
    session_id: str | None = None
    response: str | None = None
    agent_confidences: dict[str, float] | None = None
    meta_confidence: float | None = None
    hitl_escalated: bool = False
    policy_flags: list[str] = []
    total_latency_ms: float | None = None
    policy_version: str | None = None
    audit_trail_length: int | None = None
    audit_chain_integrity: bool | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    """GET /health response body."""
    status: str
    version: str
    deployment_slot: str
    uptime_seconds: float


_start_time = time.time()

# ── Endpoints ────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health():
    """Liveness / readiness probe for ALB and Kubernetes."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        deployment_slot=os.getenv("DEPLOYMENT_SLOT", "blue"),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.post(
    "/v1/advise",
    response_model=AdviseResponse,
    tags=["advisory"],
    summary="Run a full advisory session",
    description="Executes all 6 specialist agents (incl. FinBERT sentiment), meta-critic audit, "
    "HITL gate, and response synthesis.",
)
async def advise(
    body: AdviseRequest,
    request: Request,
    api_key: str = Security(verify_api_key),
):
    """
    Primary endpoint. Accepts a user profile, holdings, and natural-language
    query. Returns a structured advisory response with confidence scores,
    policy flags, and full audit metadata.
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialised")

    # Convert request holdings → domain model
    holdings = [
        PortfolioHolding(
            symbol=h.symbol,
            quantity=h.quantity,
            cost_basis_per_share=h.cost_basis_per_share,
            purchase_date=h.purchase_date,
            sector=h.sector,
            asset_class=h.asset_class,
        )
        for h in body.holdings
    ]

    client_ip = request.client.host if request.client else "unknown"
    log.info(
        "advise_request",
        user_id=body.user.user_id,
        query_length=len(body.query),
        holdings_count=len(holdings),
        client_ip=client_ip,
    )

    try:
        result: dict[str, Any] = _agent.run(
            user_query=body.query,
            user_profile=body.user,
            holdings=holdings,
            auth_token=api_key,
            role="retail_client",
        )
    except Exception as exc:
        log.error("advise_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Advisory session failed") from exc

    return AdviseResponse(
        session_id=result.get("session_id"),
        response=result.get("response"),
        agent_confidences=result.get("agent_confidences"),
        meta_confidence=result.get("meta_confidence"),
        hitl_escalated=result.get("hitl_escalated", False),
        policy_flags=result.get("policy_flags", []),
        total_latency_ms=result.get("total_latency_ms"),
        policy_version=result.get("policy_version"),
        audit_trail_length=result.get("audit_trail_length"),
        audit_chain_integrity=result.get("audit_chain_integrity"),
        error=result.get("error"),
    )
