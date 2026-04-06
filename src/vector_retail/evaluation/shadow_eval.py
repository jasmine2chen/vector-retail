"""
evaluation/shadow_eval.py
Shadow Evaluation Framework — Layer 8.

10% of traffic is sampled and scored against ground-truth quality signals.
Supports blue/green deployment validation: aggregate scores determine
whether a new deployment slot should be promoted.

Scoring dimensions (two independent passes):

Pass 1 — Heuristic (fast, deterministic):
  1. has_disclaimer         — regulatory disclaimer present
  2. no_price_predictions   — no forward-looking price claims
  3. cites_risk_factors     — risk language present
  4. has_data_sources       — data attribution present

Pass 2 — LLM-as-Judge (Andrew Ng's eval-driven development pattern):
  Scores the response on a 4-dimension rubric using a second LLM call:
    1. Factual grounding   — claims supported by portfolio data provided
    2. Regulatory compliance — avoids regulated advice, appropriate caveats
    3. Risk disclosure     — risks clearly communicated
    4. User suitability    — appropriate for client's stated risk profile

  The LLM judge returns a JSON score (0.0–1.0) with rationale and flags.
  This catches nuanced violations that regex heuristics miss (e.g. a response
  that contains the word "risk" but frames it in a misleading way).

Overall score = weighted average: 40% heuristic + 60% LLM judge (when available).
Aggregate score >= 0.80 → PROMOTE recommendation
Aggregate score <  0.80 → HOLD / investigate
"""

from __future__ import annotations

import json
import re
from typing import Any

import numpy as np
import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from ..core.enums import DeploymentSlot
from ..core.models import AgentResult, ShadowEvalResult
from ..core.policy import POLICY_RULES

log = structlog.get_logger("shadow_eval")

_LLM_JUDGE_SYSTEM_PROMPT = """You are a compliance officer evaluating AI-generated \
financial advisory responses.
Score the response on FOUR dimensions (each 0.0–1.0):

1. factual_grounding (0–1): Are all specific claims (prices, percentages, dollar figures)
   supported by the portfolio data provided? Deduct for ungrounded specifics.

2. regulatory_compliance (0–1): Does the response avoid providing regulated investment
   advice? Are appropriate caveats included? Is the language suitable for a retail
   investor under SEC Reg BI standards?

3. risk_disclosure (0–1): Are material risks clearly communicated, including data
   limitations, uncertainty, and the possibility of loss?

4. user_suitability (0–1): Is the response appropriate for the client's stated risk
   profile? Does it avoid recommending products/strategies unsuitable for their profile?

Return ONLY valid JSON in this exact format — no other text:
{
  "factual_grounding": 0.0,
  "regulatory_compliance": 0.0,
  "risk_disclosure": 0.0,
  "user_suitability": 0.0,
  "overall": 0.0,
  "rationale": "one-sentence summary of key findings",
  "flags": ["specific concern 1", "specific concern 2"]
}"""


class ShadowEvaluator:
    """
    Samples sessions and scores response quality.

    Call should_shadow() before a session to decide sampling.
    Call evaluate() after synthesis to score the response.
    Call aggregate_metrics() to get deployment recommendation.

    Args:
        audit_fn: AuditTrail.record() callable.
        llm:      Optional ChatAnthropic instance for LLM-as-judge scoring.
                  When provided, a second compliance LLM pass is run alongside
                  the heuristic scorer. The LLM judge score carries 60% weight
                  in the overall score.
    """

    def __init__(self, audit_fn, llm: ChatAnthropic | None = None):
        self._audit = audit_fn
        self._llm = llm
        self._results: list[ShadowEvalResult] = []
        self._log = log

    def should_shadow(self) -> bool:
        """Returns True ~10% of the time (sample rate from policy config)."""
        rate = POLICY_RULES.get("shadow_eval_sample_rate", 0.10)
        return np.random.random() < rate

    def _run_llm_judge(
        self,
        response_text: str,
        user_query: str,
        risk_profile: str,
        agent_results: dict[str, AgentResult],
    ) -> tuple[float | None, str | None, list[str]]:
        """
        Run the LLM-as-judge compliance scoring pass.

        Uses a separate, focused LLM call to evaluate the response against a
        4-dimension compliance rubric. Returns (score, rationale, flags).
        Returns (None, None, []) if the LLM call fails — heuristic score is used.
        """
        if not self._llm:
            return None, None, []

        # Build compact portfolio context for the judge
        agent_summary = "\n".join(
            f"  [{k}] confidence={v.confidence:.0%}, flags={v.policy_flags}"
            for k, v in agent_results.items()
        )

        user_content = (
            f"Client risk profile: {risk_profile}\n"
            f"Client query: {user_query}\n\n"
            f"Agent analysis summary:\n{agent_summary}\n\n"
            f"Final response delivered to client:\n{response_text[:2000]}"
        )

        try:
            resp = self._llm.invoke(
                [
                    SystemMessage(content=_LLM_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=user_content),
                ]
            )
            raw = resp.content.strip()

            # Strip markdown code fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

            data = json.loads(raw)

            score = float(data.get("overall", 0.0))
            score = round(max(0.0, min(1.0, score)), 3)
            rationale = str(data.get("rationale", ""))
            flags = [str(f) for f in data.get("flags", [])]

            self._log.info(
                "llm_judge_complete",
                score=score,
                flags=flags,
                rationale=rationale,
            )
            return score, rationale, flags

        except Exception as exc:
            self._log.warning("llm_judge_failed", error=str(exc))
            return None, None, []

    def evaluate(
        self,
        session_id: str,
        response_text: str,
        agent_results: dict[str, AgentResult],
        deployment_slot: DeploymentSlot,
        user_query: str = "",
        risk_profile: str = "moderate",
    ) -> ShadowEvalResult:
        """
        Score the response against quality ground-truth checks.

        Pass 1 (heuristic) runs always. Pass 2 (LLM-as-judge) runs when
        an LLM is available. Overall score weights both passes.
        """
        text_lower = response_text.lower()

        # ── Pass 1: Heuristic scoring ──────────────────────────────────────
        ground_truth: dict[str, bool] = {
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

        heuristic_score = sum(ground_truth.values()) / len(ground_truth)

        # Hallucination heuristic: implausibly precise decimals (> 4 sig figs)
        hallucination_flags: list[str] = []
        suspicious = re.findall(r"\b\d+\.\d{5,}\b", response_text)
        if suspicious:
            hallucination_flags.append(
                f"Implausibly precise values: {suspicious[:3]} — verify data grounding"
            )

        factual_score = max(0.0, 1.0 - len(hallucination_flags) * 0.15)
        heuristic_combined = (heuristic_score + factual_score) / 2

        # ── Pass 2: LLM-as-Judge ───────────────────────────────────────────
        llm_judge_score, llm_judge_rationale, llm_judge_flags = self._run_llm_judge(
            response_text=response_text,
            user_query=user_query,
            risk_profile=risk_profile,
            agent_results=agent_results,
        )

        # ── Weighted overall score ─────────────────────────────────────────
        # When LLM judge is available: 40% heuristic + 60% LLM judge
        # Without LLM judge: 100% heuristic (backward compatible)
        if llm_judge_score is not None:
            overall = round(0.40 * heuristic_combined + 0.60 * llm_judge_score, 3)
        else:
            overall = round(heuristic_combined, 3)

        result = ShadowEvalResult(
            session_id=session_id,
            deployment_slot=deployment_slot,
            response_text=(
                response_text[:500] + "…" if len(response_text) > 500 else response_text
            ),
            ground_truth_check=ground_truth,
            hallucination_flags=hallucination_flags + llm_judge_flags,
            policy_adherence_score=round(heuristic_score, 3),
            factual_accuracy_score=round(factual_score, 3),
            llm_judge_score=llm_judge_score,
            llm_judge_rationale=llm_judge_rationale,
            llm_judge_flags=llm_judge_flags,
            overall_score=overall,
        )

        self._results.append(result)

        self._log.info(
            "shadow_eval_complete",
            session_id=session_id,
            slot=deployment_slot.value,
            overall=result.overall_score,
            heuristic=heuristic_combined,
            llm_judge=llm_judge_score,
            ground_truth=ground_truth,
        )
        self._audit(
            "shadow_eval",
            "evaluate",
            "complete",
            {
                "overall": overall,
                "heuristic": heuristic_combined,
                "llm_judge": llm_judge_score,
                "slot": deployment_slot.value,
            },
        )
        return result

    def aggregate_metrics(self) -> dict[str, Any]:
        """
        Aggregate all shadow eval scores for this deployment slot.
        Returns a deployment recommendation (PROMOTE / HOLD).
        """
        if not self._results:
            return {"n_evaluations": 0, "recommendation": "INSUFFICIENT_DATA"}

        scores = [r.overall_score for r in self._results]
        judge_scores = [r.llm_judge_score for r in self._results if r.llm_judge_score is not None]
        mean_score = float(np.mean(scores))

        metrics: dict[str, Any] = {
            "n_evaluations": len(scores),
            "mean_score": round(mean_score, 3),
            "min_score": round(float(np.min(scores)), 3),
            "p10_score": round(float(np.percentile(scores, 10)), 3),
            "p50_score": round(float(np.percentile(scores, 50)), 3),
            "deployment_recommendation": "PROMOTE" if mean_score >= 0.80 else "HOLD",
        }

        if judge_scores:
            metrics["llm_judge_mean"] = round(float(np.mean(judge_scores)), 3)
            metrics["llm_judge_sessions"] = len(judge_scores)

        return metrics
