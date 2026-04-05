"""
core/prompts.py
Versioned Prompt Registry.

All LLM system prompts are loaded from config/prompts/<agent_id>.yaml.
This makes prompts deployable artifacts — versioned, rollback-able,
and A/B testable — exactly like policy_rules.json for thresholds.

Prompt YAML format:
    version: "1.0.0"
    agent_id: "portfolio_analysis"
    description: "One-line description of this prompt's role"
    system_prompt: |
      You are a portfolio analyst...

If a YAML file is missing or malformed, falls back to the inline default
embedded in each agent class. This ensures the system never fails due to
a missing prompt file.

Production extensions:
  - Swap _load_prompt() to fetch from a prompt management service (LangSmith,
    PromptLayer, or a custom prompt registry API).
  - Add a SHA-256 hash of the prompt content to AgentResult for forensic audit.
  - Gate prompt promotion behind shadow_eval score threshold (same as blue/green).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger("prompt_registry")

_PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "config" / "prompts"

# ── In-memory cache (populated at first access per agent) ────────────────────
_cache: dict[str, dict[str, Any]] = {}


def _load_prompt(agent_id: str) -> dict[str, Any]:
    """Load and cache a prompt config from YAML. Returns {} on any failure."""
    if agent_id in _cache:
        return _cache[agent_id]

    path = _PROMPTS_DIR / f"{agent_id}.yaml"
    if not path.exists():
        log.debug("prompt_yaml_not_found", agent_id=agent_id, path=str(path))
        _cache[agent_id] = {}
        return {}

    try:
        import yaml  # deferred import — only needed if YAML files exist

        with open(path) as f:
            data = yaml.safe_load(f) or {}
        _cache[agent_id] = data
        log.debug(
            "prompt_loaded",
            agent_id=agent_id,
            version=data.get("version", "unknown"),
        )
        return data
    except Exception as exc:
        log.warning("prompt_yaml_load_failed", agent_id=agent_id, error=str(exc))
        _cache[agent_id] = {}
        return {}


def get_system_prompt(agent_id: str, fallback: str = "") -> str:
    """
    Return the versioned system prompt for an agent.

    Falls back to `fallback` string if no YAML file exists, so agents
    can keep an inline default without breaking on missing files.
    """
    data = _load_prompt(agent_id)
    return data.get("system_prompt", fallback).strip()


def get_prompt_version(agent_id: str) -> str:
    """Return the version string for an agent's current prompt (e.g. '1.2.0')."""
    return _load_prompt(agent_id).get("version", "inline")


def invalidate_cache(agent_id: str | None = None) -> None:
    """
    Invalidate the prompt cache.
    Call after a hot-reload of prompt YAML files (e.g. in a /reload admin endpoint).
    Pass agent_id=None to flush all cached prompts.
    """
    if agent_id is None:
        _cache.clear()
        log.info("prompt_cache_cleared_all")
    else:
        _cache.pop(agent_id, None)
        log.info("prompt_cache_cleared", agent_id=agent_id)
