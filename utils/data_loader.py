from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models.contracts import CrossSellOpportunity, RecommendationOutput
from scoring.rule_engine import RuleBasedRecommendationEngine
from utils.helpers import PRODUCTS, extract_numeric_id, format_entity_id, next_entity_id, normalize_binary

APP_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = APP_DIR / "data" / "historical_cross_sell_opportunities.csv"
DATASET_VERSION = "cross-sell-demo-v2"
BRANCHES = [format_entity_id("BR", i) for i in range(1, 3)]
EMPLOYEES_BY_BRANCH = {
    BRANCHES[0]: [format_entity_id("EM", i) for i in range(1, 6)],
    BRANCHES[1]: [format_entity_id("EM", i) for i in range(6, 11)],
}
CAMPAIGN_CODES = {
    "Credit Card": "CP1001",
    "Personal Loan": "CP1002",
    "Insurance": "CP1003",
    "Wealth Upgrade": "CP1004",
}
REQUIRED_COLUMNS = [
    "opportunity_id",
    "customer_id",
    "employee_id",
    "branch_id",
    "campaign_id",
    "customer_name",
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
    "consent_flag",
    "contactable_flag",
    "assessment_timestamp",
    "recorded_engine",
    "historical_priority_score",
    "historical_propensity_probability",
    "historical_priority_band",
    "historical_decision",
    "historical_recommended_product",
    "historical_channel",
    "historical_next_step",
    "historical_expected_value",
    "historical_gross_expected_value",
    "historical_contact_cost",
    "historical_net_expected_value",
    "historical_uplift_score",
    "contacted_flag",
    "holdout_control_flag",
    "converted_flag",
    "realized_value",
]


def validate_history_schema(df: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        issues.append("Missing columns: " + ", ".join(missing))

    numeric_columns = [
        "age",
        "monthly_income",
        "relationship_tenure_months",
        "product_holding_count",
        "avg_monthly_balance",
        "digital_login_count_30d",
        "last_campaign_contact_days",
        "bureau_score",
        "historical_priority_score",
        "historical_propensity_probability",
        "historical_expected_value",
        "historical_gross_expected_value",
        "historical_contact_cost",
        "historical_net_expected_value",
        "historical_uplift_score",
        "contacted_flag",
        "holdout_control_flag",
        "converted_flag",
        "realized_value",
    ]
    for column in numeric_columns:
        if column in df.columns:
            numeric = pd.to_numeric(df[column], errors="coerce")
            if numeric.isna().all() and df[column].notna().any():
                issues.append(f"Column {column} must be numeric.")

    if "bureau_score" in df.columns:
        scores = pd.to_numeric(df["bureau_score"], errors="coerce")
        if ((scores < 0) | (scores > 900)).any():
            issues.append("Bureau score must be between 0 and 900.")
    if "monthly_income" in df.columns and (pd.to_numeric(df["monthly_income"], errors="coerce") < 0).any():
        issues.append("Monthly income cannot be negative.")
    for column in ["consent_flag", "contactable_flag", "contacted_flag", "holdout_control_flag"]:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce").dropna()
            if (~values.isin([0, 1])).any():
                issues.append(f"Column {column} must contain 0/1 values.")

    return issues


def _choice(rng: np.random.Generator, values, probs=None):
    return rng.choice(values, p=probs)


def _weighted_choice(rng: np.random.Generator, mapping: Dict[str, float]) -> str:
    values = list(mapping.keys())
    probs = np.array(list(mapping.values()), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(values, p=probs))


def _assign_branch_and_employee(i: int, rng: np.random.Generator) -> Tuple[str, str]:
    branch_id = BRANCHES[(i - 1) % len(BRANCHES)]
    employee_id = str(rng.choice(EMPLOYEES_BY_BRANCH[branch_id]))
    return branch_id, employee_id


def _sample_profile(rng: np.random.Generator) -> dict:
    employment_type = _weighted_choice(
        rng,
        {"Salaried": 0.50, "Self-Employed": 0.17, "Professional": 0.18, "Business Owner": 0.09, "Contract": 0.06},
    )
    age = int(np.clip(rng.normal(38, 11), 21, 69))
    income_base = {
        "Salaried": 82000,
        "Self-Employed": 108000,
        "Professional": 136000,
        "Business Owner": 162000,
        "Contract": 62000,
    }[employment_type]
    monthly_income = float(np.clip(rng.normal(income_base + max(age - 30, 0) * 2200, income_base * 0.22), 28000, 420000))

    if monthly_income >= 220000:
        segment = _weighted_choice(rng, {"HNI": 0.62, "Affluent": 0.38})
    elif monthly_income >= 105000:
        segment = _weighted_choice(rng, {"Affluent": 0.58, "Mass": 0.42})
    else:
        segment = _weighted_choice(rng, {"Mass": 0.80, "Affluent": 0.20})

    relationship_tenure_months = int(np.clip(rng.normal(42 if segment != "Mass" else 28, 26), 1, 180))
    existing_customer_flag = int(rng.random() < (0.90 if relationship_tenure_months >= 12 else 0.55))
    salary_customer_flag = int(rng.random() < (0.68 if employment_type in {"Salaried", "Professional"} else 0.22))
    kyc_complete_flag = int(rng.random() < (0.985 if existing_customer_flag else 0.92))

    city_tier = _weighted_choice(
        rng,
        {"Tier 1": 0.56 if monthly_income >= 100000 else 0.32, "Tier 2": 0.30, "Tier 3": 0.14 if monthly_income >= 100000 else 0.38},
    )
    lifecycle_stage = _weighted_choice(
        rng,
        {
            "Emerging": 0.28 if age < 30 else 0.12,
            "Growth": 0.38,
            "Established": 0.32 if age >= 34 else 0.18,
            "Mature": 0.12 if age >= 48 else 0.04,
        },
    )
    channel_preference = _weighted_choice(
        rng,
        {
            "Digital": 0.56 if age < 42 else 0.36,
            "Hybrid": 0.30,
            "RM / Branch": 0.14 if segment == "Mass" else 0.34,
        },
    )

    avg_monthly_balance = float(
        np.clip(
            rng.normal(monthly_income * (1.2 if segment == "Mass" else 2.1 if segment == "Affluent" else 3.8), monthly_income * 0.8),
            12000,
            850000,
        )
    )
    digital_login_count_30d = int(np.clip(rng.normal(12 if channel_preference != "RM / Branch" else 6, 6), 0, 36))
    app_sessions_30d = int(np.clip(rng.normal(10 if channel_preference == "Digital" else 7, 5), 0, 28))
    avg_debit_txn_count_3m = int(np.clip(rng.normal(18 if salary_customer_flag else 12, 8), 1, 55))
    avg_credit_txn_count_3m = int(np.clip(rng.normal(7 if existing_customer_flag else 5, 3), 1, 20))
    branch_visits_90d = int(np.clip(rng.poisson(1.8 if channel_preference == "RM / Branch" else 0.8), 0, 8))
    rm_interactions_90d = int(np.clip(rng.poisson(2.6 if segment in {"Affluent", "HNI"} else 0.9), 0, 9))
    last_campaign_contact_days = int(np.clip(rng.gamma(shape=2.1, scale=26.0), 0, 210))
    last_campaign_response = _weighted_choice(
        rng,
        {
            "Accepted": 0.08,
            "Clicked": 0.17,
            "Opened": 0.18,
            "No Response": 0.46,
            "Declined": 0.11,
        },
    )
    prior_offer_accept_count_12m = int(np.clip(rng.poisson(0.6 if last_campaign_response in {"Accepted", "Clicked"} else 0.25), 0, 4))
    recent_service_issue_flag = int(rng.random() < (0.05 if segment in {"Affluent", "HNI"} else 0.08))
    consent_flag = int(rng.random() < (0.96 if existing_customer_flag else 0.84))
    contactable_flag = int(rng.random() < (0.97 if channel_preference != "RM / Branch" else 0.92))
    bureau_score = int(
        np.clip(
            rng.normal(
                715
                + (18 if employment_type in {"Professional", "Salaried"} else 6)
                + (10 if existing_customer_flag else 0)
                + (14 if segment == "HNI" else 6 if segment == "Affluent" else 0)
                - (28 if recent_service_issue_flag else 0),
                42,
            ),
            560,
            860,
        )
    )

    has_credit_card_flag = int(rng.random() < (0.52 if salary_customer_flag else 0.24))
    has_personal_loan_flag = int(rng.random() < (0.18 if monthly_income >= 65000 else 0.10))
    has_home_loan_flag = int(rng.random() < (0.22 if age >= 32 and monthly_income >= 90000 else 0.06))
    has_insurance_flag = int(rng.random() < (0.28 if age >= 34 or segment in {"Affluent", "HNI"} else 0.11))
    product_holding_count = int(1 + has_credit_card_flag + has_personal_loan_flag + has_home_loan_flag + has_insurance_flag)

    return {
        "age": age,
        "segment": segment,
        "monthly_income": round(monthly_income, 2),
        "relationship_tenure_months": relationship_tenure_months,
        "employment_type": employment_type,
        "city_tier": city_tier,
        "existing_customer_flag": existing_customer_flag,
        "salary_customer_flag": salary_customer_flag,
        "kyc_complete_flag": kyc_complete_flag,
        "product_holding_count": product_holding_count,
        "has_credit_card_flag": has_credit_card_flag,
        "has_personal_loan_flag": has_personal_loan_flag,
        "has_home_loan_flag": has_home_loan_flag,
        "has_insurance_flag": has_insurance_flag,
        "avg_monthly_balance": round(avg_monthly_balance, 2),
        "avg_debit_txn_count_3m": avg_debit_txn_count_3m,
        "avg_credit_txn_count_3m": avg_credit_txn_count_3m,
        "digital_login_count_30d": digital_login_count_30d,
        "app_sessions_30d": app_sessions_30d,
        "branch_visits_90d": branch_visits_90d,
        "rm_interactions_90d": rm_interactions_90d,
        "last_campaign_contact_days": last_campaign_contact_days,
        "last_campaign_response": last_campaign_response,
        "prior_offer_accept_count_12m": prior_offer_accept_count_12m,
        "recent_service_issue_flag": recent_service_issue_flag,
        "consent_flag": consent_flag,
        "contactable_flag": contactable_flag,
        "bureau_score": bureau_score,
        "lifecycle_stage": lifecycle_stage,
        "channel_preference": channel_preference,
    }


def _recommended_target_product(profile: dict, rng: np.random.Generator) -> str:
    product_scores = {
        "Credit Card": 0.34 + 0.12 * (1 - profile["has_credit_card_flag"]) + 0.10 * profile["salary_customer_flag"] + 0.08 * (profile["digital_login_count_30d"] >= 10),
        "Personal Loan": 0.26 + 0.10 * (1 - profile["has_personal_loan_flag"]) + 0.08 * (profile["monthly_income"] >= 70000) + 0.04 * (profile["bureau_score"] >= 700),
        "Insurance": 0.22 + 0.10 * (1 - profile["has_insurance_flag"]) + 0.08 * (profile["age"] >= 32) + 0.06 * (profile["branch_visits_90d"] >= 1),
        "Wealth Upgrade": 0.18 + 0.14 * (profile["segment"] in {"Affluent", "HNI"}) + 0.10 * (profile["avg_monthly_balance"] >= 180000) + 0.06 * (profile["rm_interactions_90d"] >= 2),
    }
    products = list(product_scores.keys())
    probs = np.array(list(product_scores.values()), dtype=float)
    probs = probs / probs.sum()
    return str(rng.choice(products, p=probs))


def _conversion_probability(app: CrossSellOpportunity, evaluation: RecommendationOutput, rng: np.random.Generator) -> float:
    product = evaluation.recommended_product
    response_bonus = {
        "Accepted": 0.055,
        "Clicked": 0.035,
        "Opened": 0.015,
        "No Response": -0.005,
        "Declined": -0.035,
    }[app.last_campaign_response]
    product_bias = {
        "Credit Card": 0.010 if app.salary_customer_flag else -0.005,
        "Personal Loan": 0.015 if app.bureau_score >= 710 else -0.020,
        "Insurance": 0.020 if app.age >= 32 else -0.015,
        "Wealth Upgrade": 0.025 if app.segment in {"Affluent", "HNI"} else -0.030,
    }[product]
    branch_bias = 0.008 if app.branch_id == BRANCHES[0] else -0.004
    rm_bias = (extract_numeric_id(app.employee_id) % 5) * 0.003
    probability = np.clip(
        0.035
        + 0.23 * evaluation.propensity_probability
        + response_bonus
        + product_bias
        + branch_bias
        + rm_bias
        + (0.012 if app.channel_preference == evaluation.recommended_channel else 0)
        + 0.012 * min(app.prior_offer_accept_count_12m, 3)
        - (0.055 if app.recent_service_issue_flag else 0)
        - (0.040 if app.last_campaign_contact_days < 12 else 0)
        + rng.normal(0, 0.018),
        0.02,
        0.38,
    )
    return float(probability)


def golden_demo_opportunities(start_index: int = 1) -> list[CrossSellOpportunity]:
    scenarios = [
        {
            "customer_name": "Golden - HNI Wealth Priority",
            "age": 45,
            "segment": "HNI",
            "monthly_income": 360000.0,
            "relationship_tenure_months": 96,
            "employment_type": "Professional",
            "city_tier": "Tier 1",
            "existing_customer_flag": 1,
            "salary_customer_flag": 1,
            "kyc_complete_flag": 1,
            "product_holding_count": 3,
            "has_credit_card_flag": 1,
            "has_personal_loan_flag": 0,
            "has_home_loan_flag": 1,
            "has_insurance_flag": 0,
            "avg_monthly_balance": 720000.0,
            "avg_debit_txn_count_3m": 36,
            "avg_credit_txn_count_3m": 14,
            "digital_login_count_30d": 22,
            "app_sessions_30d": 20,
            "branch_visits_90d": 4,
            "rm_interactions_90d": 6,
            "last_campaign_contact_days": 95,
            "last_campaign_response": "Clicked",
            "prior_offer_accept_count_12m": 2,
            "recent_service_issue_flag": 0,
            "bureau_score": 830,
            "lifecycle_stage": "Established",
            "channel_preference": "RM / Branch",
            "target_product": "Auto Select",
            "consent_flag": 1,
            "contactable_flag": 1,
        },
        {
            "customer_name": "Golden - Service Suppression",
            "age": 39,
            "segment": "Affluent",
            "monthly_income": 190000.0,
            "relationship_tenure_months": 48,
            "employment_type": "Salaried",
            "city_tier": "Tier 1",
            "existing_customer_flag": 1,
            "salary_customer_flag": 1,
            "kyc_complete_flag": 1,
            "product_holding_count": 2,
            "has_credit_card_flag": 0,
            "has_personal_loan_flag": 0,
            "has_home_loan_flag": 1,
            "has_insurance_flag": 0,
            "avg_monthly_balance": 320000.0,
            "avg_debit_txn_count_3m": 22,
            "avg_credit_txn_count_3m": 9,
            "digital_login_count_30d": 18,
            "app_sessions_30d": 14,
            "branch_visits_90d": 2,
            "rm_interactions_90d": 3,
            "last_campaign_contact_days": 75,
            "last_campaign_response": "Opened",
            "prior_offer_accept_count_12m": 1,
            "recent_service_issue_flag": 1,
            "bureau_score": 760,
            "lifecycle_stage": "Growth",
            "channel_preference": "Hybrid",
            "target_product": "Auto Select",
            "consent_flag": 1,
            "contactable_flag": 1,
        },
        {
            "customer_name": "Golden - Already Holds Card",
            "age": 34,
            "segment": "Mass",
            "monthly_income": 95000.0,
            "relationship_tenure_months": 30,
            "employment_type": "Salaried",
            "city_tier": "Tier 1",
            "existing_customer_flag": 1,
            "salary_customer_flag": 1,
            "kyc_complete_flag": 1,
            "product_holding_count": 2,
            "has_credit_card_flag": 1,
            "has_personal_loan_flag": 0,
            "has_home_loan_flag": 0,
            "has_insurance_flag": 0,
            "avg_monthly_balance": 145000.0,
            "avg_debit_txn_count_3m": 18,
            "avg_credit_txn_count_3m": 8,
            "digital_login_count_30d": 16,
            "app_sessions_30d": 12,
            "branch_visits_90d": 1,
            "rm_interactions_90d": 2,
            "last_campaign_contact_days": 50,
            "last_campaign_response": "No Response",
            "prior_offer_accept_count_12m": 0,
            "recent_service_issue_flag": 0,
            "bureau_score": 725,
            "lifecycle_stage": "Growth",
            "channel_preference": "Digital",
            "target_product": "Credit Card",
            "consent_flag": 1,
            "contactable_flag": 1,
        },
        {
            "customer_name": "Golden - Contact Fatigue",
            "age": 42,
            "segment": "Affluent",
            "monthly_income": 210000.0,
            "relationship_tenure_months": 66,
            "employment_type": "Business Owner",
            "city_tier": "Tier 1",
            "existing_customer_flag": 1,
            "salary_customer_flag": 0,
            "kyc_complete_flag": 1,
            "product_holding_count": 2,
            "has_credit_card_flag": 0,
            "has_personal_loan_flag": 0,
            "has_home_loan_flag": 0,
            "has_insurance_flag": 1,
            "avg_monthly_balance": 410000.0,
            "avg_debit_txn_count_3m": 28,
            "avg_credit_txn_count_3m": 10,
            "digital_login_count_30d": 20,
            "app_sessions_30d": 15,
            "branch_visits_90d": 2,
            "rm_interactions_90d": 4,
            "last_campaign_contact_days": 3,
            "last_campaign_response": "Clicked",
            "prior_offer_accept_count_12m": 2,
            "recent_service_issue_flag": 0,
            "bureau_score": 790,
            "lifecycle_stage": "Established",
            "channel_preference": "Hybrid",
            "target_product": "Auto Select",
            "consent_flag": 1,
            "contactable_flag": 1,
        },
        {
            "customer_name": "Golden - Suitability Hold",
            "age": 28,
            "segment": "Mass",
            "monthly_income": 42000.0,
            "relationship_tenure_months": 10,
            "employment_type": "Contract",
            "city_tier": "Tier 2",
            "existing_customer_flag": 1,
            "salary_customer_flag": 0,
            "kyc_complete_flag": 1,
            "product_holding_count": 1,
            "has_credit_card_flag": 0,
            "has_personal_loan_flag": 0,
            "has_home_loan_flag": 0,
            "has_insurance_flag": 0,
            "avg_monthly_balance": 28000.0,
            "avg_debit_txn_count_3m": 8,
            "avg_credit_txn_count_3m": 3,
            "digital_login_count_30d": 6,
            "app_sessions_30d": 4,
            "branch_visits_90d": 0,
            "rm_interactions_90d": 0,
            "last_campaign_contact_days": 90,
            "last_campaign_response": "No Response",
            "prior_offer_accept_count_12m": 0,
            "recent_service_issue_flag": 0,
            "bureau_score": 610,
            "lifecycle_stage": "Emerging",
            "channel_preference": "Digital",
            "target_product": "Personal Loan",
            "consent_flag": 1,
            "contactable_flag": 1,
        },
    ]

    opportunities: list[CrossSellOpportunity] = []
    rng = np.random.default_rng(911)
    for offset, scenario in enumerate(scenarios):
        row_number = start_index + offset
        branch_id, employee_id = _assign_branch_and_employee(row_number, rng)
        target_product = str(scenario["target_product"])
        campaign_product = "Credit Card" if target_product == "Auto Select" else target_product
        opportunities.append(
            CrossSellOpportunity(
                opportunity_id=format_entity_id("OP", row_number),
                customer_id=format_entity_id("CU", row_number),
                employee_id=employee_id,
                branch_id=branch_id,
                campaign_id=CAMPAIGN_CODES.get(campaign_product, "CP9000"),
                **scenario,
            )
        )
    return opportunities


def build_opportunity_from_row(row: pd.Series) -> CrossSellOpportunity:
    payload = {col: row.get(col) for col in CrossSellOpportunity.__dataclass_fields__.keys()}
    for field in [
        "existing_customer_flag",
        "salary_customer_flag",
        "kyc_complete_flag",
        "has_credit_card_flag",
        "has_personal_loan_flag",
        "has_home_loan_flag",
        "has_insurance_flag",
        "recent_service_issue_flag",
    ]:
        payload[field] = normalize_binary(payload.get(field, 0))
    for field in ["consent_flag", "contactable_flag"]:
        payload[field] = normalize_binary(payload.get(field, 1))
    payload["age"] = int(payload.get("age", 30))
    payload["relationship_tenure_months"] = int(payload.get("relationship_tenure_months", 0))
    payload["product_holding_count"] = int(payload.get("product_holding_count", 1))
    payload["avg_debit_txn_count_3m"] = int(payload.get("avg_debit_txn_count_3m", 0))
    payload["avg_credit_txn_count_3m"] = int(payload.get("avg_credit_txn_count_3m", 0))
    payload["digital_login_count_30d"] = int(payload.get("digital_login_count_30d", 0))
    payload["app_sessions_30d"] = int(payload.get("app_sessions_30d", 0))
    payload["branch_visits_90d"] = int(payload.get("branch_visits_90d", 0))
    payload["rm_interactions_90d"] = int(payload.get("rm_interactions_90d", 0))
    payload["last_campaign_contact_days"] = int(payload.get("last_campaign_contact_days", 0))
    payload["prior_offer_accept_count_12m"] = int(payload.get("prior_offer_accept_count_12m", 0))
    payload["bureau_score"] = int(payload.get("bureau_score", 720))
    for field in ["monthly_income", "avg_monthly_balance"]:
        payload[field] = float(payload.get(field, 0.0))
    return CrossSellOpportunity(**payload)


def _normalize_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "opportunity_id" in out.columns:
        out["opportunity_id"] = [format_entity_id("OP", i + 1) if not str(v).startswith("OP") else v for i, v in enumerate(out["opportunity_id"])]
    if "customer_id" in out.columns:
        out["customer_id"] = [format_entity_id("CU", i + 1) if not str(v).startswith("CU") else v for i, v in enumerate(out["customer_id"])]
    if "branch_id" in out.columns:
        out["branch_id"] = out["branch_id"].apply(lambda v: v if str(v).startswith("BR") else BRANCHES[0])
    if "employee_id" in out.columns:
        normalized_employees = []
        for idx, value in enumerate(out["employee_id"]):
            if str(value).startswith("EM"):
                normalized_employees.append(value)
            else:
                normalized_employees.append(format_entity_id("EM", (idx % 10) + 1))
        out["employee_id"] = normalized_employees
    if "campaign_id" in out.columns:
        out["campaign_id"] = out.apply(lambda r: CAMPAIGN_CODES.get(r.get("target_product", "Credit Card"), "CP1001") if not str(r.get("campaign_id", "")).startswith("CP") else r.get("campaign_id"), axis=1)
    return out


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in REQUIRED_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan
    out = out[REQUIRED_COLUMNS]
    if "historical_expected_value" in out.columns:
        expected_value = pd.to_numeric(out["historical_expected_value"], errors="coerce")
        for value_col in ["historical_gross_expected_value", "historical_net_expected_value"]:
            out[value_col] = pd.to_numeric(out[value_col], errors="coerce").fillna(expected_value)
        out["historical_contact_cost"] = pd.to_numeric(out["historical_contact_cost"], errors="coerce").fillna(0.0)
        probability = pd.to_numeric(out["historical_propensity_probability"], errors="coerce")
        out["historical_uplift_score"] = pd.to_numeric(out["historical_uplift_score"], errors="coerce").fillna((probability - 0.08).clip(-1, 1))
    out["holdout_control_flag"] = pd.to_numeric(out["holdout_control_flag"], errors="coerce").fillna(0).astype(int)
    return out


def _history_row(
    app: CrossSellOpportunity,
    evaluation: RecommendationOutput,
    contacted_flag: int,
    converted_flag: int,
    realized_value: float,
    holdout_control_flag: int,
    assessment_timestamp: datetime,
) -> dict[str, object]:
    return {
        **app.to_dict(),
        "assessment_timestamp": assessment_timestamp.isoformat(),
        "recorded_engine": evaluation.engine_name,
        "historical_priority_score": evaluation.priority_score,
        "historical_propensity_probability": evaluation.propensity_probability,
        "historical_priority_band": evaluation.priority_band,
        "historical_decision": evaluation.decision,
        "historical_recommended_product": evaluation.recommended_product,
        "historical_channel": evaluation.recommended_channel,
        "historical_next_step": evaluation.recommended_next_step,
        "historical_expected_value": evaluation.expected_value,
        "historical_gross_expected_value": evaluation.gross_expected_value,
        "historical_contact_cost": evaluation.contact_cost,
        "historical_net_expected_value": evaluation.net_expected_value,
        "historical_uplift_score": evaluation.uplift_score,
        "contacted_flag": contacted_flag,
        "holdout_control_flag": holdout_control_flag,
        "converted_flag": converted_flag,
        "realized_value": realized_value,
    }


def _simulate_outcome(
    app: CrossSellOpportunity,
    evaluation: RecommendationOutput,
    rng: np.random.Generator,
) -> tuple[int, int, float, int]:
    conversion_prob = _conversion_probability(app, evaluation, rng)
    targetable_flag = int(evaluation.decision in {"RM Priority Lead", "Campaign Target", "Digital Nurture"})
    holdout_control_flag = int(targetable_flag == 1 and rng.random() < 0.12)
    contacted_flag = int(targetable_flag == 1 and holdout_control_flag == 0)
    organic_probability = max(0.01, conversion_prob * 0.28)
    converted_flag = int(
        (contacted_flag == 1 and rng.random() < conversion_prob)
        or (holdout_control_flag == 1 and rng.random() < organic_probability)
    )
    value_base = evaluation.gross_expected_value or evaluation.expected_value
    realized_value = round(value_base * (1.0 + rng.uniform(0.08, 0.34)), 2) if converted_flag else 0.0
    return contacted_flag, converted_flag, realized_value, holdout_control_flag


def generate_synthetic_dataset(
    path: Path = DATA_PATH,
    n_rows: int = 900,
    seed: int = 42,
    include_golden: bool = True,
) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    engine = RuleBasedRecommendationEngine()

    rows = []
    for i in range(1, n_rows + 1):
        branch_id, employee_id = _assign_branch_and_employee(i, rng)
        profile = _sample_profile(rng)
        target_product = _recommended_target_product(profile, rng)
        app = CrossSellOpportunity(
            opportunity_id=format_entity_id("OP", i),
            customer_id=format_entity_id("CU", i),
            employee_id=employee_id,
            branch_id=branch_id,
            campaign_id=CAMPAIGN_CODES[target_product],
            customer_name=f"Customer {i}",
            target_product=target_product,
            **profile,
        )
        evaluation = engine.evaluate(app)
        contacted_flag, converted_flag, realized_value, holdout_control_flag = _simulate_outcome(app, evaluation, rng)
        assessment_timestamp = datetime(2025, 11, 1) + pd.to_timedelta(int(rng.integers(0, 160)), unit="D")

        rows.append(_history_row(app, evaluation, contacted_flag, converted_flag, realized_value, holdout_control_flag, assessment_timestamp))

    if include_golden:
        for opportunity in golden_demo_opportunities(n_rows + 1):
            evaluation = engine.evaluate(opportunity)
            contacted_flag, converted_flag, realized_value, holdout_control_flag = _simulate_outcome(opportunity, evaluation, rng)
            assessment_timestamp = datetime(2026, 4, 15) + pd.to_timedelta(int(rng.integers(0, 30)), unit="D")
            rows.append(
                _history_row(
                    opportunity,
                    evaluation,
                    contacted_flag,
                    converted_flag,
                    realized_value,
                    holdout_control_flag,
                    assessment_timestamp,
                )
            )

    df = pd.DataFrame(rows)
    df = _ensure_schema(_normalize_ids(df))
    df.to_csv(path, index=False)
    return df


def load_or_create_data(path: Path = DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return generate_synthetic_dataset(path)

    df = pd.read_csv(path)
    df = _ensure_schema(_normalize_ids(df))

    integer_cols = [
        "age",
        "relationship_tenure_months",
        "product_holding_count",
        "avg_debit_txn_count_3m",
        "avg_credit_txn_count_3m",
        "digital_login_count_30d",
        "app_sessions_30d",
        "branch_visits_90d",
        "rm_interactions_90d",
        "last_campaign_contact_days",
        "prior_offer_accept_count_12m",
        "bureau_score",
        "contacted_flag",
        "holdout_control_flag",
    ]
    binary_cols = [
        "existing_customer_flag",
        "salary_customer_flag",
        "kyc_complete_flag",
        "has_credit_card_flag",
        "has_personal_loan_flag",
        "has_home_loan_flag",
        "has_insurance_flag",
        "recent_service_issue_flag",
        "consent_flag",
        "contactable_flag",
    ]
    numeric_cols = [
        "monthly_income",
        "avg_monthly_balance",
        "historical_priority_score",
        "historical_propensity_probability",
        "historical_expected_value",
        "historical_gross_expected_value",
        "historical_contact_cost",
        "historical_net_expected_value",
        "historical_uplift_score",
        "realized_value",
    ]

    for col in integer_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in binary_cols:
        df[col] = df[col].apply(normalize_binary)
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "converted_flag" in df.columns:
        df["converted_flag"] = pd.to_numeric(df["converted_flag"], errors="coerce")

    return df


def append_assessment_to_history(path: Path, opportunity: CrossSellOpportunity, output: RecommendationOutput) -> pd.DataFrame:
    path = Path(path)
    df = load_or_create_data(path)
    row = {
        **opportunity.to_dict(),
        "campaign_id": CAMPAIGN_CODES.get(output.recommended_product, opportunity.campaign_id or "CP1001"),
        "target_product": opportunity.target_product,
        "assessment_timestamp": datetime.utcnow().isoformat(),
        "recorded_engine": output.engine_name,
        "historical_priority_score": output.priority_score,
        "historical_propensity_probability": output.propensity_probability,
        "historical_priority_band": output.priority_band,
        "historical_decision": output.decision,
        "historical_recommended_product": output.recommended_product,
        "historical_channel": output.recommended_channel,
        "historical_next_step": output.recommended_next_step,
        "historical_expected_value": output.expected_value,
        "historical_gross_expected_value": output.gross_expected_value,
        "historical_contact_cost": output.contact_cost,
        "historical_net_expected_value": output.net_expected_value,
        "historical_uplift_score": output.uplift_score,
        "contacted_flag": int(output.decision in {"RM Priority Lead", "Campaign Target", "Digital Nurture"}),
        "holdout_control_flag": 0,
        "converted_flag": np.nan,
        "realized_value": np.nan,
    }
    updated = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    updated = _ensure_schema(_normalize_ids(updated))
    updated.to_csv(path, index=False)
    return updated


def branch_employee_options() -> Dict[str, list[str]]:
    return EMPLOYEES_BY_BRANCH


def allocate_new_ids(df: pd.DataFrame) -> Dict[str, str]:
    return {
        "opportunity_id": next_entity_id(df["opportunity_id"], "OP"),
        "customer_id": next_entity_id(df["customer_id"], "CU"),
    }
