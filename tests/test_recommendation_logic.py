from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from models.contracts import CrossSellOpportunity
from scoring.ml_engine import MLRecommendationEngine
from scoring.rule_engine import RuleBasedRecommendationEngine
from utils.data_loader import generate_synthetic_dataset, validate_history_schema


def strong_opportunity(**overrides) -> CrossSellOpportunity:
    payload = {
        "opportunity_id": "OPTEST",
        "customer_id": "CUTEST",
        "monthly_income": 300000.0,
        "segment": "HNI",
        "relationship_tenure_months": 100,
        "employment_type": "Professional",
        "city_tier": "Tier 1",
        "existing_customer_flag": 1,
        "salary_customer_flag": 1,
        "kyc_complete_flag": 1,
        "product_holding_count": 4,
        "has_credit_card_flag": 0,
        "has_personal_loan_flag": 0,
        "has_home_loan_flag": 0,
        "has_insurance_flag": 0,
        "avg_monthly_balance": 600000.0,
        "avg_debit_txn_count_3m": 35,
        "avg_credit_txn_count_3m": 15,
        "digital_login_count_30d": 25,
        "app_sessions_30d": 22,
        "branch_visits_90d": 5,
        "rm_interactions_90d": 5,
        "last_campaign_contact_days": 90,
        "last_campaign_response": "Accepted",
        "prior_offer_accept_count_12m": 3,
        "recent_service_issue_flag": 0,
        "bureau_score": 830,
        "lifecycle_stage": "Established",
        "channel_preference": "RM / Branch",
        "target_product": "Auto Select",
    }
    payload.update(overrides)
    return CrossSellOpportunity(**payload)


class CrossSellRecommendationLogicTests(unittest.TestCase):
    def test_rule_auto_select_excludes_already_held_product(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(has_credit_card_flag=1)
        )

        self.assertNotEqual(output.recommended_product, "Credit Card")

    def test_rule_forced_already_held_product_is_not_actionable(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(has_credit_card_flag=1, target_product="Credit Card")
        )

        self.assertEqual(output.decision, "Hold / Defer")
        self.assertEqual(output.recommended_channel, "No Outreach")
        self.assertEqual(output.expected_value, 0.0)
        self.assertLessEqual(output.priority_score, 520.0)

    def test_rule_suppressed_lead_has_no_actionable_expected_value(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(last_campaign_contact_days=2)
        )

        self.assertEqual(output.decision, "Suppress")
        self.assertEqual(output.expected_value, 0.0)
        self.assertLessEqual(output.priority_score, 500.0)

    def test_rule_missing_consent_is_suppressed(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(consent_flag=0)
        )

        self.assertEqual(output.decision, "Suppress")
        self.assertEqual(output.recommended_channel, "No Outreach")
        self.assertEqual(output.net_expected_value, 0.0)
        self.assertTrue(any("consent" in reason.lower() for reason in output.top_negative_reasons))

    def test_rule_forced_unsuitable_product_is_held(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(
                target_product="Personal Loan",
                monthly_income=42000.0,
                bureau_score=620,
                segment="Mass",
                avg_monthly_balance=22000.0,
                rm_interactions_90d=0,
                branch_visits_90d=0,
            )
        )

        self.assertEqual(output.decision, "Hold / Defer")
        self.assertEqual(output.recommended_channel, "No Outreach")
        self.assertEqual(output.expected_value, 0.0)
        self.assertTrue(any("personal loan eligibility" in reason.lower() for reason in output.top_negative_reasons))

    def test_rule_clean_lead_returns_net_economics(self):
        output = RuleBasedRecommendationEngine().evaluate(
            strong_opportunity(target_product="Wealth Upgrade")
        )

        self.assertGreater(output.gross_expected_value, 0.0)
        self.assertGreater(output.contact_cost, 0.0)
        self.assertEqual(output.expected_value, output.net_expected_value)
        self.assertAlmostEqual(output.net_expected_value, output.gross_expected_value - output.contact_cost, places=2)
        self.assertGreater(output.uplift_score, 0.0)

    def test_ml_fallback_forced_already_held_product_is_not_actionable(self):
        with tempfile.TemporaryDirectory() as model_dir:
            output = MLRecommendationEngine(model_dir=model_dir).evaluate(
                strong_opportunity(has_credit_card_flag=1, target_product="Credit Card")
            )

        self.assertEqual(output.decision, "Hold / Defer")
        self.assertEqual(output.recommended_channel, "No Outreach")
        self.assertEqual(output.expected_value, 0.0)
        self.assertLessEqual(output.priority_score, 520.0)

    def test_generated_dataset_matches_demo_schema(self):
        with tempfile.TemporaryDirectory() as data_dir:
            df = generate_synthetic_dataset(Path(data_dir) / "history.csv", n_rows=25, seed=7)

        self.assertEqual(validate_history_schema(df), [])
        self.assertIn("historical_net_expected_value", df.columns)
        self.assertIn("holdout_control_flag", df.columns)


if __name__ == "__main__":
    unittest.main()
