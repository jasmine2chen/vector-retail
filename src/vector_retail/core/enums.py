"""
core/enums.py
Centralised domain enumerations.
All modules import from here — never redefined elsewhere.
"""

from enum import StrEnum


class RiskTolerance(StrEnum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AccountType(StrEnum):
    INDIVIDUAL = "individual"
    IRA = "ira"
    ROTH_IRA = "roth_ira"
    JOINT = "joint"
    TRUST = "trust"


class Jurisdiction(StrEnum):
    US = "us"
    CA = "ca"
    EU = "eu"
    UK = "uk"


class HITLPriority(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeploymentSlot(StrEnum):
    BLUE = "blue"
    GREEN = "green"
    SHADOW = "shadow"
