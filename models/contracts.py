from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CrossSellOpportunity:
    opportunity_id: str
    customer_id: str
    employee_id: str = ""
    branch_id: str = ""
    campaign_id: str = ""
    customer_name: str = ""
    age: int = 30
    segment: str = "Mass"
    monthly_income: float = 0.0
    relationship_tenure_months: int = 0
    employment_type: str = "Salaried"
    city_tier: str = "Tier 1"
    existing_customer_flag: int = 1
    salary_customer_flag: int = 0
    kyc_complete_flag: int = 1
    product_holding_count: int = 1
    has_credit_card_flag: int = 0
    has_personal_loan_flag: int = 0
    has_home_loan_flag: int = 0
    has_insurance_flag: int = 0
    avg_monthly_balance: float = 0.0
    avg_debit_txn_count_3m: int = 0
    avg_credit_txn_count_3m: int = 0
    digital_login_count_30d: int = 0
    app_sessions_30d: int = 0
    branch_visits_90d: int = 0
    rm_interactions_90d: int = 0
    last_campaign_contact_days: int = 180
    last_campaign_response: str = "No Response"
    prior_offer_accept_count_12m: int = 0
    recent_service_issue_flag: int = 0
    bureau_score: int = 720
    lifecycle_stage: str = "Emerging"
    channel_preference: str = "Digital"
    target_product: str = "Auto Select"
    consent_flag: int = 1
    contactable_flag: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FactorContribution:
    factor: str
    impact_direction: str
    points: float
    description: str


@dataclass
class RecommendationOutput:
    engine_name: str
    recommended_product: str
    recommended_channel: str
    priority_score: float
    propensity_probability: float
    priority_band: str
    decision: str
    recommended_next_step: str
    expected_value: float
    relationship_score: float
    engagement_score: float
    product_fit_score: float
    value_score: float
    suppression_flag: int
    gross_expected_value: float = 0.0
    contact_cost: float = 0.0
    net_expected_value: float = 0.0
    uplift_score: float = 0.0
    top_positive_reasons: List[str] = field(default_factory=list)
    top_negative_reasons: List[str] = field(default_factory=list)
    factor_contributions: List[FactorContribution] = field(default_factory=list)
    feature_importance: List[Dict[str, Any]] = field(default_factory=list)
    confidence: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    alternate_recommendations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
