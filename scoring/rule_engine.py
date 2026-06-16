from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

from models.contracts import CrossSellOpportunity, FactorContribution, RecommendationOutput
from utils.helpers import (
    PRODUCTS,
    clamp,
    contact_cost,
    decision_from_score,
    default_channel,
    gross_expected_value,
    net_expected_value,
    next_step_from_decision,
    priority_band_from_score,
    product_already_held,
    product_eligibility_issues,
    safe_divide,
    suppressing_issues,
    target_gap,
    uplift_score,
)


APP_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = APP_DIR / "scoring" / "recommendation_engine_config.json"


class RuleBasedRecommendationEngine:
    name = "Rule-Based Recommendation Engine"

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        with self.config_path.open("r", encoding="utf-8") as handle:
            self.config = json.load(handle)

    def _metric_value(self, app: CrossSellOpportunity, metric: str, product: str) -> float:
        if metric == "target_gap":
            return float(target_gap(app, product))
        if metric == "segment_wealthy":
            return float(app.segment in {"Affluent", "HNI"})
        value = getattr(app, metric, 0)
        try:
            return float(value)
        except Exception:
            return 0.0

    def _score_rule(self, metric_name: str, value: float, rules: List[Dict]) -> Tuple[float, str]:
        for band in rules:
            min_val = float(band.get("min", float("-inf")))
            max_val = float(band.get("max", float("inf")))
            if min_val <= value <= max_val:
                return float(band.get("points", 0)), str(band.get("rationale", metric_name))
        return 0.0, f"No rule matched for {metric_name}."

    def _evaluate_product(self, app: CrossSellOpportunity, product: str) -> RecommendationOutput:
        base_score = float(self.config["base_score"])
        product_profile = self.config["product_profiles"].get(product, {})
        score = base_score + float(product_profile.get("base_adjustment", 0))

        contributions: List[FactorContribution] = []
        for rule in self.config["rules"]:
            metric_name = str(rule["metric"])
            value = self._metric_value(app, metric_name, product)
            points, rationale = self._score_rule(metric_name, value, rule["bands"])
            score += points
            contributions.append(
                FactorContribution(
                    factor=metric_name,
                    impact_direction="Positive" if points >= 0 else "Negative",
                    points=round(points, 2),
                    description=rationale,
                )
            )

        for product_rule in self.config.get("product_specific_rules", {}).get(product, []):
            metric_name = str(product_rule["metric"])
            value = self._metric_value(app, metric_name, product)
            points, rationale = self._score_rule(metric_name, value, product_rule["bands"])
            score += points
            contributions.append(
                FactorContribution(
                    factor=f"{product}:{metric_name}",
                    impact_direction="Positive" if points >= 0 else "Negative",
                    points=round(points, 2),
                    description=rationale,
                )
            )

        score = clamp(score, float(self.config["score_bounds"]["min"]), float(self.config["score_bounds"]["max"]))
        already_held = bool(product_already_held(app, product))
        eligibility_issues = product_eligibility_issues(app, product)
        suppression_reasons = suppressing_issues(eligibility_issues)
        if already_held:
            contributions.append(
                FactorContribution(
                    factor="Existing product",
                    impact_direction="Negative",
                    points=-120.0,
                    description=f"Customer already holds {product}, so this should not be routed as a cross-sell lead.",
                )
            )
        for issue in eligibility_issues:
            if already_held and issue == f"Customer already holds {product}.":
                continue
            contributions.append(
                FactorContribution(
                    factor="Eligibility",
                    impact_direction="Negative",
                    points=-90.0 if issue in suppression_reasons else -45.0,
                    description=issue,
                )
            )

        # Relationship / engagement / fit / value lenses for UI cards.
        relationship_score = round(
            clamp(
                100
                * (
                    0.30 * safe_divide(app.relationship_tenure_months, 96)
                    + 0.25 * app.existing_customer_flag
                    + 0.20 * app.salary_customer_flag
                    + 0.25 * safe_divide(app.product_holding_count, 5)
                ),
                0,
                100,
            ),
            1,
        )
        engagement_score = round(
            clamp(
                100
                * (
                    0.35 * safe_divide(app.digital_login_count_30d, 24)
                    + 0.20 * safe_divide(app.app_sessions_30d, 20)
                    + 0.20 * safe_divide(app.rm_interactions_90d, 6)
                    + 0.15 * safe_divide(app.avg_debit_txn_count_3m, 36)
                    + 0.10 * safe_divide(app.avg_credit_txn_count_3m, 18)
                ),
                0,
                100,
            ),
            1,
        )
        product_fit_score = round(
            clamp(
                100
                * (
                    0.35 * target_gap(app, product)
                    + 0.20 * safe_divide(max(app.bureau_score - 600, 0), 220)
                    + 0.20 * safe_divide(app.monthly_income, 250000)
                    + 0.25 * safe_divide(app.avg_monthly_balance, 350000)
                ),
                0,
                100,
            ),
            1,
        )
        value_score = round(
            clamp(
                100
                * (
                    0.45 * safe_divide(app.avg_monthly_balance, 400000)
                    + 0.30 * safe_divide(app.monthly_income, 250000)
                    + 0.25 * (1.0 if app.segment in {"Affluent", "HNI"} else 0.45 if app.segment == "Mass" else 0.25)
                ),
                0,
                100,
            ),
            1,
        )

        probability_bounds = self.config["probability_mapping"]
        propensity_probability = round(
            clamp(
                probability_bounds["min_prob"]
                + (score - self.config["score_bounds"]["min"])
                / (self.config["score_bounds"]["max"] - self.config["score_bounds"]["min"])
                * (probability_bounds["max_prob"] - probability_bounds["min_prob"]),
                0.01,
                0.95,
            ),
            4,
        )

        suppression_flag = int(bool(suppression_reasons))

        action_score = score
        if suppression_flag:
            action_score = min(action_score, 500.0)
        elif already_held or eligibility_issues:
            action_score = min(action_score, 520.0)

        decision = "Hold / Defer" if (already_held or eligibility_issues) and not suppression_flag else decision_from_score(action_score, suppression_flag)
        channel = default_channel(decision, propensity_probability, app.rm_interactions_90d, app.channel_preference)
        if (already_held or eligibility_issues) and not suppression_flag:
            channel = "No Outreach"
        if (
            not already_held
            and decision != "Suppress"
            and product_profile.get("preferred_channel") == "RM / Branch"
            and app.rm_interactions_90d >= 2
        ):
            channel = "RM / Branch"
        priority_band = priority_band_from_score(action_score)
        next_step = self.config["next_step_mapping"].get(decision, next_step_from_decision(decision))

        ranked = sorted(contributions, key=lambda c: abs(c.points), reverse=True)
        positives = [c.description for c in ranked if c.impact_direction == "Positive"][:3]
        negatives = [c.description for c in ranked if c.impact_direction == "Negative"][:3]

        value_multiplier = float(product_profile.get("value_multiplier", 1.0))
        gross_revenue = 0.0 if suppression_flag or already_held or eligibility_issues else gross_expected_value(propensity_probability, product, value_multiplier)
        net_revenue = 0.0 if suppression_flag or already_held or eligibility_issues else net_expected_value(propensity_probability, product, channel, value_multiplier)
        expected_revenue = net_revenue

        return RecommendationOutput(
            engine_name=self.name,
            recommended_product=product,
            recommended_channel=channel,
            priority_score=round(action_score, 1),
            propensity_probability=propensity_probability,
            priority_band=priority_band,
            decision=decision,
            recommended_next_step=next_step,
            expected_value=expected_revenue,
            relationship_score=relationship_score,
            engagement_score=engagement_score,
            product_fit_score=product_fit_score,
            value_score=value_score,
            suppression_flag=suppression_flag,
            gross_expected_value=gross_revenue,
            contact_cost=contact_cost(channel),
            net_expected_value=net_revenue,
            uplift_score=uplift_score(propensity_probability),
            top_positive_reasons=positives,
            top_negative_reasons=negatives,
            factor_contributions=ranked[:10],
            notes=["Config-driven recommendation generated from commercial fit, engagement, and fatigue rules."],
        )

    def evaluate(self, app: CrossSellOpportunity) -> RecommendationOutput:
        candidates = (
            [product for product in PRODUCTS if not product_already_held(app, product)]
            if app.target_product == "Auto Select"
            else [app.target_product]
        )
        if not candidates:
            candidates = PRODUCTS
        candidate_outputs = [self._evaluate_product(app, product) for product in candidates]
        decision_rank = {
            "RM Priority Lead": 4,
            "Campaign Target": 3,
            "Digital Nurture": 2,
            "Hold / Defer": 1,
            "Suppress": 0,
        }
        ranked = sorted(
            candidate_outputs,
            key=lambda x: (decision_rank.get(x.decision, 0), x.expected_value, x.propensity_probability, x.priority_score),
            reverse=True,
        )
        best = ranked[0]
        alternates = [
            {
                "product": item.recommended_product,
                "probability": item.propensity_probability,
                "priority_score": item.priority_score,
                "decision": item.decision,
                "expected_value": item.expected_value,
            }
            for item in ranked[1:4]
        ]
        return replace(best, alternate_recommendations=alternates)
