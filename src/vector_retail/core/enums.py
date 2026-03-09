"""
core/enums.py
Centralised domain enumerations.
All modules import from here — never redefined elsewhere.
"""
from enum import Enum


class RiskTolerance(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class AccountType(str, Enum):
    INDIVIDUAL = "individual"
    IRA = "ira"
    ROTH_IRA = "roth_ira"
    JOINT = "joint"
    TRUST = "trust"


class Jurisdiction(str, Enum):
    US = "us"
    CA = "ca"
    EU = "eu"
    UK = "uk"


class HITLPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class DeploymentSlot(str, Enum):
    BLUE = "blue"
    GREEN = "green"
    SHADOW = "shadow"
