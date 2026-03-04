import logging
import json
from typing import Annotated, List, Dict, Optional, Literal
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from enum import Enum
import operator
from datetime import datetime, timezone

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.core.config import settings
from app.db.vector_store import fetch_documents, Document

# --- OBSERVABILITY ---
class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "msg": record.msg,
            "props": getattr(record, "props", {})
        }
        return json.dumps(log_obj)

logger = logging.getLogger("bank_agent")
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)

def log_audit(event_type: str, **kwargs):
    logger.info(event_type, extra={"props": kwargs})

# --- DOMAIN MODELS ---
class MetricLabel(str, Enum):
    TOTAL_REVENUE = "total_revenue"
    NET_INCOME = "net_income"
    OPERATING_EXPENSES = "operating_expenses"
    GROSS_MARGIN = "gross_margin"

class FinancialMetric(BaseModel):
    label: MetricLabel
    value: Decimal
    unit_scale: Literal["thousands", "millions", "billions", "none"]
    currency: str
    source_doc_id: str
    class Config:
        json_encoders = {Decimal: str}

class CalculationResult(BaseModel):
    label: str
    value: Decimal
    formula: str

class RegulatoryResponse(BaseModel):
    answer: str
    citations: List[str]
    confidence_score: float

class AgentState(BaseModel):
    request_id: str
    query: str
    messages: Annotated[List[BaseMessage], operator.add]
    valid_docs: List[Document] = Field(default_factory=list)
    normalized_metrics: List[FinancialMetric] = Field(default_factory=list)
    calculations: List[CalculationResult] = Field(default_factory=list)
    verification_status: Literal["pending", "verified", "rejected", "approved"] = "pending"
    rejection_reason: Optional[str] = None

# --- DETERMINISTIC LOGIC ---
def normalize_magnitude(metric: FinancialMetric) -> FinancialMetric:
    multipliers = {
        "thousands": Decimal("1000"),
        "millions": Decimal("1000000"),
        "billions": Decimal("1000000000"),
        "none": Decimal("1")
    }
    factor = multipliers.get(metric.unit_scale, Decimal("1"))
    normalized_value = metric.value * factor
    log_audit("Unit Normalization", original=str(metric.value), final=str(normalized_value))
    return FinancialMetric(
        label=metric.label, value=normalized_value, unit_scale="none",
        currency=metric.currency, source_doc_id=metric.source_doc_id
    )

# --- GRAPH NODES ---
def node_guardrail(state: AgentState) -> Dict:
    log_audit("Node: Guardrail", query=state.query)
    finance_keywords = ["revenue", "profit", "margin", "tax", "income"]
    if not any(k in state.query.lower() for k in finance_keywords):
        return {"verification_status": "rejected", "rejection_reason": "Out of domain query."}
    return {"verification_status": "pending"}

def node_retrieve_validate(state: AgentState) -> Dict:
    log_audit("Node: Retrieval")
    docs = fetch_documents(state.query)
    valid_docs = [
        d for d in docs 
        if d.retrieval_score >= settings.MIN_REL_SCORE and d.date_filed.year == settings.FISCAL_YEAR_FILTER
    ]
    if not valid_docs:
        return {"verification_status": "rejected", "rejection_reason": "No valid documents found."}
    return {"valid_docs": valid_docs, "verification_status": "verified"}

def node_extract_normalize(state: AgentState) -> Dict:
    log_audit("Node: Extraction")
    llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0)
    class ExtractionSchema(BaseModel):
        metrics: List[FinancialMetric]
    try:
        structured_llm = llm.with_structured_output(ExtractionSchema)
        context = "\n".join([f"[{d.doc_id}] {d.content}" for d in state.valid_docs])
        result = structured_llm.invoke(f"Extract metrics from:\n{context}")
        clean = [normalize_magnitude(m) for m in result.metrics]
        return {"normalized_metrics": clean}
    except Exception as e:
        return {"verification_status": "rejected", "rejection_reason": f"Extraction Failed: {str(e)}"}

def node_compute_logic(state: AgentState) -> Dict:
    log_audit("Node: Compute")
    metrics = state.normalized_metrics
    rev = next((m for m in metrics if m.label == MetricLabel.TOTAL_REVENUE), None)
    inc = next((m for m in metrics if m.label == MetricLabel.NET_INCOME), None)
    
    results = []
    if rev and inc:
        if rev.currency != inc.currency:
            return {"verification_status": "rejected", "rejection_reason": "Currency Mismatch Detected."}
        if rev.value == Decimal("0"):
            return {"verification_status": "rejected", "rejection_reason": "Division by Zero Error."}
        try:
            val = (inc.value / rev.value).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
            results.append(CalculationResult(label="Net Profit Margin", value=val, formula=f"{inc.value}/{rev.value}"))
        except InvalidOperation:
            log_audit("Math Error")
    return {"calculations": results}

def node_synthesize_strict(state: AgentState) -> Dict:
    log_audit("Node: Synthesis")
    llm = ChatOpenAI(model=settings.MODEL_NAME, temperature=0)
    structured_llm = llm.with_structured_output(RegulatoryResponse)
    
    math_ctx = "\n".join([f"{c.label}: {c.value}" for c in state.calculations])
    doc_ctx = "\n".join([f"[{d.doc_id}] {d.content}" for d in state.valid_docs])
    resp = structured_llm.invoke(f"Context: {doc_ctx}\nMath: {math_ctx}")
    
    valid_ids = {d.doc_id for d in state.valid_docs}
    if any(c for c in resp.citations if c not in valid_ids):
        return {"verification_status": "rejected", "rejection_reason": "Hallucinated Citation"}
    return {"messages": [AIMessage(content=resp.answer)]}

def node_human_approval(state: AgentState): pass

def build_bank_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("guardrail", node_guardrail)
    workflow.add_node("retrieve", node_retrieve_validate)
    workflow.add_node("extract", node_extract_normalize)
    workflow.add_node("compute", node_compute_logic)
    workflow.add_node("synthesize", node_synthesize_strict)
    workflow.add_node("human_review", node_human_approval)
    
    workflow.set_entry_point("guardrail")
    workflow.add_conditional_edges("guardrail", lambda x: "retrieve" if x.verification_status != "rejected" else END)
    workflow.add_conditional_edges("retrieve", lambda x: "extract" if x.verification_status == "verified" else END)
    workflow.add_conditional_edges("extract", lambda x: "compute" if x.verification_status != "rejected" else END)
    workflow.add_conditional_edges("compute", lambda x: "synthesize" if x.verification_status != "rejected" else END)
    workflow.add_conditional_edges("synthesize", lambda x: "human_review" if x.verification_status != "rejected" else END)
    workflow.add_edge("human_review", END)
    
    return workflow.compile(checkpointer=MemorySaver(), interrupt_before=["human_review"])
