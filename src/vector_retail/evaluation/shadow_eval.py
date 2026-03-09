"""
evaluation/shadow_eval.py
Shadow Evaluation Framework — Layer 8.

10% of traffic is sampled and scored against ground-truth quality signals.
Supports blue/green deployment validation: aggregate scores determine
whether a new deployment slot should be promoted.

Scoring dimensions:
  1. has_disclaimer         — regulatory disclaimer present
  2. no_price_predictions   — no forward-looking price claims
  3. cites_risk_factors     — risk language present
  4. has_data_sources       — data attribution present
  5. no_precise_unknowns    — no implausibly precise ungrounded numbers

Aggregate score >= 0.80 → PROMOTE recommendation
Aggregate score <  0.80 → HOLD / investigate
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

import numpy as np
import structlog

from ..core.enums import DeploymentSlot
from ..core.models import AgentResult, ShadowEvalResult
from ..core.policy import POLICY_RULES

log = structlog.get_logger("shadow_eval")


class ShadowEvaluator:
    """
    Samples sessions and scores response quality.
    Call should_shadow() before a session to decide sampling.
    Call evaluate() after synthesis to score the response.
    Call aggregate_metrics() to get deployment recommendation.
    """

    def __init__(self, audit_fn):
        self._audit = audit_fn
        self._results: List[ShadowEvalResult] = []
        self._log = log

    def should_shadow(self) -> bool:
        """Returns True ~10% of the time (sample rate from policy config)."""
        rate = POLICY_RULES.get("shadow_eval_sample_rate", 0.10)
        return np.random.random() < rate

    def evaluate(
        self,
        session_id: str,
        response_text: str,
        agent_results: Dict[str, AgentResult],
        deployment_slot: DeploymentSlot,
    ) -> ShadowEvalResult:
        """Score the response against quality ground-truth checks."""

        text_lower = response_text.lower()

        ground_truth: Dict[str, bool] = {
            "has_disclaimer": any(
                phrase in text_lower
                for phrase in [
                    "informational purposes only",
                    "does not constitute",
                    "not investment advice",
                    "consult a",
                ]
            ),
            "no_price_predictions": not bool(
                re.search(
                    r"will reach \$|price target \$|expected to (hit|reach|climb to) \$",
                    response_text,
                    re.IGNORECASE,
                )
            ),
            "cites_risk_factors": any(
                word in text_lower
                for word in ["risk", "volatility", "loss", "uncertainty", "drawdown"]
            ),
            "has_data_sources": any(
                word in text_lower
                for word in ["yfinance", "data", "market data", "source", "based on"]
            ),
        }

        policy_adherence = sum(ground_truth.values()) / len(ground_truth)

        # Hallucination heuristic: implausibly precise decimals (> 4 sig figs)
        hallucination_flags: List[str] = []
        suspicious = re.findall(r"\b\d+\.\d{5,}\b", response_text)
        if suspicious:
            hallucination_flags.append(
                f"Implausibly precise values: {suspicious[:3]} — verify data grounding"
            )

        factual_score = max(0.0, 1.0 - len(hallucination_flags) * 0.15)
        overall = (policy_adherence + factual_score) / 2

        result = ShadowEvalResult(
            session_id=session_id,
            deployment_slot=deployment_slot,
            response_text=response_text[:500] + "…" if len(response_text) > 500 else response_text,
            ground_truth_check=ground_truth,
            hallucination_flags=hallucination_flags,
            policy_adherence_score=round(policy_adherence, 3),
            factual_accuracy_score=round(factual_score, 3),
            overall_score=round(overall, 3),
        )

        self._results.append(result)

        self._log.info(
            "shadow_eval_complete",
            session_id=session_id,
            slot=deployment_slot.value,
            overall=result.overall_score,
            policy=result.policy_adherence_score,
            ground_truth=ground_truth,
        )
        self._audit(
            "shadow_eval", "evaluate", "complete",
            {"overall": overall, "slot": deployment_slot.value},
        )
        return result

    def aggregate_metrics(self) -> Dict[str, Any]:
        """
        Aggregate all shadow eval scores for this deployment slot.
        Returns a deployment recommendation (PROMOTE / HOLD).
        """
        if not self._results:
            return {"n_evaluations": 0, "recommendation": "INSUFFICIENT_DATA"}

        scores = [r.overall_score for r in self._results]
        mean_score = float(np.mean(scores))

        return {
            "n_evaluations": len(scores),
            "mean_score": round(mean_score, 3),
            "min_score": round(float(np.min(scores)), 3),
            "p10_score": round(float(np.percentile(scores, 10)), 3),
            "p50_score": round(float(np.percentile(scores, 50)), 3),
            "deployment_recommendation": "PROMOTE" if mean_score >= 0.80 else "HOLD",
        }
