from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from models.contracts import CrossSellOpportunity, FactorContribution, RecommendationOutput
from utils.helpers import (
    PRODUCTS,
    clamp,
    decision_from_score,
    default_channel,
    expected_value,
    next_step_from_decision,
    priority_band_from_score,
    probability_to_priority_score,
    safe_divide,
    target_gap,
)


class MLRecommendationEngine:
    name = "Machine Learning Recommendation Engine"

    def __init__(self, model_dir: str = "artifacts"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline_path = self.model_dir / "cross_sell_pipeline.joblib"
        self.pipeline: Optional[Pipeline] = None
        self.feature_columns = [
            "age",
            "segment",
            "monthly_income",
            "relationship_tenure_months",
            "employment_type",
            "city_tier",
            "existing_customer_flag",
            "salary_customer_flag",
            "kyc_complete_flag",
            "product_holding_count",
            "has_credit_card_flag",
            "has_personal_loan_flag",
            "has_home_loan_flag",
            "has_insurance_flag",
            "avg_monthly_balance",
            "avg_debit_txn_count_3m",
            "avg_credit_txn_count_3m",
            "digital_login_count_30d",
            "app_sessions_30d",
            "branch_visits_90d",
            "rm_interactions_90d",
            "last_campaign_contact_days",
            "last_campaign_response",
            "prior_offer_accept_count_12m",
            "recent_service_issue_flag",
            "bureau_score",
            "lifecycle_stage",
            "channel_preference",
            "target_product",
            "target_gap",
            "engagement_index",
            "relationship_index",
            "value_index",
        ]
        self.numeric_features = [
            "age",
            "monthly_income",
            "relationship_tenure_months",
            "existing_customer_flag",
            "salary_customer_flag",
            "kyc_complete_flag",
            "product_holding_count",
            "has_credit_card_flag",
            "has_personal_loan_flag",
            "has_home_loan_flag",
            "has_insurance_flag",
            "avg_monthly_balance",
            "avg_debit_txn_count_3m",
            "avg_credit_txn_count_3m",
            "digital_login_count_30d",
            "app_sessions_30d",
            "branch_visits_90d",
            "rm_interactions_90d",
            "last_campaign_contact_days",
            "prior_offer_accept_count_12m",
            "recent_service_issue_flag",
            "bureau_score",
            "target_gap",
            "engagement_index",
            "relationship_index",
            "value_index",
        ]
        self.categorical_features = [
            "segment",
            "employment_type",
            "city_tier",
            "last_campaign_response",
            "lifecycle_stage",
            "channel_preference",
            "target_product",
        ]

    def _prepare_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["target_gap"] = out.apply(lambda r: float(target_gap(r, str(r["target_product"]))), axis=1)
        out["engagement_index"] = (
            0.32 * out["digital_login_count_30d"]
            + 0.22 * out["app_sessions_30d"]
            + 0.18 * out["rm_interactions_90d"] * 3
            + 0.16 * out["avg_debit_txn_count_3m"]
            + 0.12 * out["avg_credit_txn_count_3m"]
        )
        out["relationship_index"] = (
            0.38 * out["relationship_tenure_months"]
            + 18 * out["existing_customer_flag"]
            + 10 * out["salary_customer_flag"]
            + 12 * out["product_holding_count"]
        )
        out["value_index"] = (
            out["avg_monthly_balance"] * 0.0016
            + out["monthly_income"] * 0.0009
            + np.where(out["segment"].eq("HNI"), 48, np.where(out["segment"].eq("Affluent"), 22, 0))
        )
        return out

    def train(self, historical_df: pd.DataFrame) -> Dict[str, float]:
        train_df = historical_df.copy()
        train_df = train_df[train_df["converted_flag"].notna()].copy()
        if train_df.empty:
            raise ValueError("Training data must contain non-null converted_flag values.")

        train_df = self._prepare_frame(train_df)
        X = train_df[self.feature_columns]
        y = train_df["converted_flag"].astype(int)

        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_transformer, self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features),
        ])

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=800, class_weight="balanced")),
        ])
        pipeline.fit(X, y)
        self.pipeline = pipeline
        joblib.dump(pipeline, self.pipeline_path)

        return {
            "rows_trained": float(len(train_df)),
            "conversion_rate": float(y.mean()),
        }

    def load(self) -> bool:
        if self.pipeline_path.exists():
            self.pipeline = joblib.load(self.pipeline_path)
            return True
        return False

    def _score_one(self, app: CrossSellOpportunity, product: str) -> RecommendationOutput:
        if self.pipeline is None:
            self.load()

        row = pd.DataFrame([{**app.to_dict(), "target_product": product}])
        prepared = self._prepare_frame(row)

        engagement_score = round(clamp(prepared["engagement_index"].iloc[0] / 15.0, 0, 100), 1)
        relationship_score = round(clamp(prepared["relationship_index"].iloc[0] / 1.6, 0, 100), 1)
        product_fit_score = round(
            clamp(
                100
                * (
                    0.35 * prepared["target_gap"].iloc[0]
                    + 0.30 * safe_divide(max(app.bureau_score - 600, 0), 220)
                    + 0.20 * safe_divide(app.monthly_income, 250000)
                    + 0.15 * safe_divide(app.avg_monthly_balance, 350000)
                ),
                0,
                100,
            ),
            1,
        )
        value_score = round(clamp(prepared["value_index"].iloc[0] / 5.0, 0, 100), 1)
        suppression_flag = int(app.recent_service_issue_flag == 1 or app.kyc_complete_flag == 0 or app.last_campaign_contact_days <= 4)

        if self.pipeline is None:
            prob = clamp(
                0.08
                + 0.12 * safe_divide(app.digital_login_count_30d, 20)
                + 0.08 * safe_divide(app.rm_interactions_90d, 6)
                + 0.10 * safe_divide(app.avg_monthly_balance, 300000)
                + 0.10 * safe_divide(max(app.bureau_score - 650, 0), 180)
                + 0.08 * safe_divide(app.prior_offer_accept_count_12m, 3)
                + 0.12 * prepared["target_gap"].iloc[0]
                - 0.16 * app.recent_service_issue_flag
                - 0.10 * int(app.last_campaign_contact_days < 10),
                0.02,
                0.90,
            )
            feature_importance = [
                {"feature": "engagement_index", "relative_importance": 0.28, "signed_contribution": round(engagement_score / 100, 3)},
                {"feature": "target_gap", "relative_importance": 0.22, "signed_contribution": round(prepared["target_gap"].iloc[0] * 0.8, 3)},
                {"feature": "value_index", "relative_importance": 0.18, "signed_contribution": round(value_score / 100, 3)},
                {"feature": "service_risk", "relative_importance": 0.14, "signed_contribution": -0.45 if suppression_flag else 0.0},
            ]
            notes = ["No trained model loaded; using a fallback surrogate propensity estimate."]
        else:
            prob = float(self.pipeline.predict_proba(prepared[self.feature_columns])[0, 1])
            model = self.pipeline.named_steps["model"]
            pre = self.pipeline.named_steps["preprocessor"]
            transformed = pre.transform(prepared[self.feature_columns])
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            feature_names = pre.get_feature_names_out()
            contribution_values = transformed[0] * model.coef_[0]
            order = np.argsort(np.abs(contribution_values))[::-1][:10]
            feature_importance = [
                {
                    "feature": str(feature_names[i]).replace("num__", "").replace("cat__", ""),
                    "relative_importance": round(float(abs(model.coef_[0][i])), 4),
                    "signed_contribution": round(float(contribution_values[i]), 4),
                }
                for i in order
            ]
            notes = ["Prediction generated by logistic regression trained on historical campaign outcomes."]

        score = probability_to_priority_score(prob, value_index=min(value_score / 100.0, 1.0))
        if suppression_flag:
            score = min(score, 500.0)
        priority_band = priority_band_from_score(score)
        decision = decision_from_score(score, suppression_flag)
        channel = default_channel(decision, prob, app.rm_interactions_90d, app.channel_preference)
        confidence = round(abs(prob - 0.5) * 2, 3)
        exp_value = expected_value(prob, product, 1.0 + value_score / 200.0)

        factor_contributions = [
            FactorContribution(
                factor=item["feature"],
                impact_direction="Positive" if item["signed_contribution"] >= 0 else "Negative",
                points=round(abs(item["signed_contribution"]) * 100, 2),
                description=(
                    f"{item['feature']} improved the model view of this lead."
                    if item["signed_contribution"] >= 0
                    else f"{item['feature']} reduced the model view of this lead."
                ),
            )
            for item in feature_importance[:8]
        ]
        ranked_contribs = sorted(factor_contributions, key=lambda x: x.points, reverse=True)
        positives = [c.description for c in ranked_contribs if c.impact_direction == "Positive"][:3]
        negatives = [c.description for c in ranked_contribs if c.impact_direction == "Negative"][:3]

        return RecommendationOutput(
            engine_name=self.name,
            recommended_product=product,
            recommended_channel=channel,
            priority_score=score,
            propensity_probability=round(prob, 4),
            priority_band=priority_band,
            decision=decision,
            recommended_next_step=next_step_from_decision(decision),
            expected_value=exp_value,
            relationship_score=relationship_score,
            engagement_score=engagement_score,
            product_fit_score=product_fit_score,
            value_score=value_score,
            suppression_flag=suppression_flag,
            top_positive_reasons=positives,
            top_negative_reasons=negatives,
            factor_contributions=ranked_contribs,
            feature_importance=feature_importance,
            confidence=confidence,
            notes=notes,
        )

    def evaluate(self, app: CrossSellOpportunity) -> RecommendationOutput:
        candidates = PRODUCTS if app.target_product == "Auto Select" else [app.target_product]
        scored = [self._score_one(app, product) for product in candidates]
        ranked = sorted(
            scored,
            key=lambda x: (x.decision != "Suppress", x.propensity_probability, x.expected_value, x.priority_score),
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
