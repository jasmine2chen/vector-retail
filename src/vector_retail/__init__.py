"""
vector_retail — Production Finance AI Agent v2.0

Multi-agent financial advisor with LangGraph orchestration,
compliance-first architecture, and human-in-the-loop review.
"""

__version__ = "2.0.0"

from .core.enums import AccountType, DeploymentSlot, Jurisdiction, RiskTolerance
from .core.models import AgentResult, PortfolioHolding, UserProfile
from .orchestrator import VectorRetailAgent

__all__ = [
    "VectorRetailAgent",
    "UserProfile",
    "PortfolioHolding",
    "AgentResult",
    "RiskTolerance",
    "AccountType",
    "Jurisdiction",
    "DeploymentSlot",
]
