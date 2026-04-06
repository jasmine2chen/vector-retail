"""
tests/unit/test_shadow_eval.py
Unit tests for shadow evaluation scoring.
"""

from vector_retail.core.enums import DeploymentSlot
from vector_retail.evaluation.shadow_eval import ShadowEvaluator


def _noop_audit(*args, **kwargs):
    pass


class TestShadowEvaluator:

    def _make_eval(self) -> ShadowEvaluator:
        return ShadowEvaluator(audit_fn=_noop_audit)

    def test_good_response_scores_high(self):
        evaluator = self._make_eval()
        good_response = (
            "Based on market data sources, your portfolio shows some risk and volatility. "
            "There is uncertainty in current conditions and potential for loss. "
            "This is informational purposes only and does not constitute investment advice."
        )
        result = evaluator.evaluate("sess-001", good_response, {}, DeploymentSlot.BLUE)
        assert result.overall_score >= 0.80

    def test_bad_response_scores_low(self):
        evaluator = self._make_eval()
        bad_response = "AAPL will reach $500! Price target $600. Buy immediately."
        result = evaluator.evaluate("sess-002", bad_response, {}, DeploymentSlot.BLUE)
        assert result.policy_adherence_score < 0.75

    def test_disclaimer_check(self):
        evaluator = self._make_eval()
        with_disclaimer = "This does not constitute investment advice."
        without_disclaimer = "Buy more stocks."
        r1 = evaluator.evaluate("s1", with_disclaimer, {}, DeploymentSlot.BLUE)
        r2 = evaluator.evaluate("s2", without_disclaimer, {}, DeploymentSlot.BLUE)
        assert r1.ground_truth_check["has_disclaimer"] is True
        assert r2.ground_truth_check["has_disclaimer"] is False

    def test_no_price_prediction_check(self):
        evaluator = self._make_eval()
        with_prediction = "AAPL will reach $300 by year end."
        without_prediction = "AAPL has shown volatility recently."
        r1 = evaluator.evaluate("s3", with_prediction, {}, DeploymentSlot.BLUE)
        r2 = evaluator.evaluate("s4", without_prediction, {}, DeploymentSlot.BLUE)
        assert r1.ground_truth_check["no_price_predictions"] is False
        assert r2.ground_truth_check["no_price_predictions"] is True

    def test_aggregate_metrics_empty(self):
        evaluator = self._make_eval()
        metrics = evaluator.aggregate_metrics()
        assert metrics["recommendation"] == "INSUFFICIENT_DATA"

    def test_aggregate_metrics_promote(self):
        evaluator = self._make_eval()
        good = (
            "Based on data sources, portfolio shows risk and uncertainty. "
            "This does not constitute investment advice. Volatility may cause loss."
        )
        for i in range(5):
            evaluator.evaluate(f"sess-{i}", good, {}, DeploymentSlot.BLUE)
        metrics = evaluator.aggregate_metrics()
        assert metrics["n_evaluations"] == 5
        assert metrics["deployment_recommendation"] == "PROMOTE"

    def test_sampling_rate_approximately_correct(self):
        """Shadow eval should sample roughly 10% of sessions."""
        evaluator = self._make_eval()
        samples = sum(1 for _ in range(5000) if evaluator.should_shadow())
        rate = samples / 5000
        assert 0.06 < rate < 0.15, f"Sampling rate {rate:.2%} outside expected 6-15% range"
