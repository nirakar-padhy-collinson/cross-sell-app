from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from models.contracts import CrossSellOpportunity, RecommendationOutput
from scoring.rule_engine import RuleBasedRecommendationEngine
from utils.helpers import PRODUCTS, extract_numeric_id, format_entity_id, next_entity_id, normalize_binary

DATA_PATH = Path("data/historical_cross_sell_opportunities.csv")
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
    "contacted_flag",
    "converted_flag",
    "realized_value",
]


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
        "Accepted": 0.16,
        "Clicked": 0.08,
        "Opened": 0.03,
        "No Response": -0.01,
        "Declined": -0.08,
    }[app.last_campaign_response]
    product_bias = {
        "Credit Card": 0.02 if app.salary_customer_flag else -0.01,
        "Personal Loan": 0.03 if app.bureau_score >= 710 else -0.03,
        "Insurance": 0.04 if app.age >= 32 else -0.02,
        "Wealth Upgrade": 0.06 if app.segment in {"Affluent", "HNI"} else -0.05,
    }[product]
    branch_bias = 0.03 if app.branch_id == BRANCHES[0] else -0.01
    rm_bias = (extract_numeric_id(app.employee_id) % 5) * 0.006
    probability = np.clip(
        evaluation.propensity_probability
        + response_bonus
        + product_bias
        + branch_bias
        + rm_bias
        + (0.03 if app.channel_preference == evaluation.recommended_channel else 0)
        - (0.15 if app.recent_service_issue_flag else 0)
        - (0.08 if app.last_campaign_contact_days < 12 else 0)
        + rng.normal(0, 0.035),
        0.01,
        0.92,
    )
    return float(probability)


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
    return out


def generate_synthetic_dataset(path: Path = DATA_PATH, n_rows: int = 900, seed: int = 42) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    engine = RuleBasedRecommendationEngine(config_path=str(Path(__file__).resolve().parents[1] / "scoring" / "recommendation_engine_config.json"))

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
        conversion_prob = _conversion_probability(app, evaluation, rng)
        contacted_flag = int(evaluation.decision in {"RM Priority Lead", "Campaign Target", "Digital Nurture"})
        converted_flag = int(contacted_flag == 1 and rng.random() < conversion_prob)
        realized_value = round(evaluation.expected_value * (1.0 + rng.uniform(0.08, 0.34)), 2) if converted_flag else 0.0
        assessment_timestamp = datetime(2025, 11, 1) + pd.to_timedelta(int(rng.integers(0, 160)), unit="D")

        rows.append(
            {
                **app.to_dict(),
                "assessment_timestamp": assessment_timestamp.isoformat(),
                "recorded_engine": engine.name,
                "historical_priority_score": evaluation.priority_score,
                "historical_propensity_probability": evaluation.propensity_probability,
                "historical_priority_band": evaluation.priority_band,
                "historical_decision": evaluation.decision,
                "historical_recommended_product": evaluation.recommended_product,
                "historical_channel": evaluation.recommended_channel,
                "historical_next_step": evaluation.recommended_next_step,
                "historical_expected_value": evaluation.expected_value,
                "contacted_flag": contacted_flag,
                "converted_flag": converted_flag,
                "realized_value": realized_value,
            }
        )

    df = pd.DataFrame(rows)
    df = _ensure_schema(_normalize_ids(df))
    df.to_csv(path, index=False)
    return df


def load_or_create_data(path: Path = DATA_PATH) -> pd.DataFrame:
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
    ]
    numeric_cols = [
        "monthly_income",
        "avg_monthly_balance",
        "historical_priority_score",
        "historical_propensity_probability",
        "historical_expected_value",
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

    df.to_csv(path, index=False)
    return df


def append_assessment_to_history(path: Path, opportunity: CrossSellOpportunity, output: RecommendationOutput) -> pd.DataFrame:
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
        "contacted_flag": int(output.decision in {"RM Priority Lead", "Campaign Target", "Digital Nurture"}),
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
