from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

PRODUCTS = ["Credit Card", "Personal Loan", "Insurance", "Wealth Upgrade"]
DECISIONS = ["RM Priority Lead", "Campaign Target", "Digital Nurture", "Hold / Defer", "Suppress"]


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    return numerator / denominator if denominator else default


def format_entity_id(prefix: str, number: int) -> str:
    return f"{prefix}{int(number):04d}"


def extract_numeric_id(value: str, fallback: int = 0) -> int:
    if not value:
        return fallback
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return int(digits) if digits else fallback


def next_entity_id(values: Iterable[str], prefix: str) -> str:
    max_id = max((extract_numeric_id(v) for v in values if str(v).startswith(prefix)), default=0)
    return format_entity_id(prefix, max_id + 1)


def product_held_flag_name(product: str) -> Optional[str]:
    mapping = {
        "Credit Card": "has_credit_card_flag",
        "Personal Loan": "has_personal_loan_flag",
        "Insurance": "has_insurance_flag",
        "Wealth Upgrade": None,
    }
    return mapping.get(product)


def product_already_held(record: object, product: str) -> int:
    flag_name = product_held_flag_name(product)
    if flag_name is None:
        return 0
    return int(getattr(record, flag_name, 0))


def target_gap(record: object, product: str) -> int:
    return int(not product_already_held(record, product))


def probability_to_priority_score(probability: float, value_index: float = 0.0) -> float:
    base = 320 + 520 * clamp(probability, 0.0, 1.0)
    value_bump = 60 * clamp(value_index, 0.0, 1.0)
    return round(clamp(base + value_bump, 300, 900), 1)


def priority_band_from_score(score: float) -> str:
    if score >= 790:
        return "Prime"
    if score >= 700:
        return "High"
    if score >= 610:
        return "Moderate"
    if score >= 520:
        return "Low"
    return "Suppressed"


def decision_from_score(score: float, suppression_flag: int) -> str:
    if suppression_flag:
        return "Suppress"
    if score >= 790:
        return "RM Priority Lead"
    if score >= 700:
        return "Campaign Target"
    if score >= 610:
        return "Digital Nurture"
    return "Hold / Defer"


def next_step_from_decision(decision: str) -> str:
    return {
        "RM Priority Lead": "Route to RM within 48 hours with personalized offer context.",
        "Campaign Target": "Push into active campaign list with offer personalization and treatment tags.",
        "Digital Nurture": "Enroll in app / email nurture journey and monitor engagement for 14 days.",
        "Hold / Defer": "Keep off priority lists and re-score after the next engagement refresh.",
        "Suppress": "Do not contact now. Resolve service / compliance issues before reactivation.",
    }.get(decision, "Review strategy before activation.")


def default_channel(decision: str, probability: float, rm_interactions_90d: int, channel_preference: str) -> str:
    if decision == "Suppress":
        return "No Outreach"
    if decision == "RM Priority Lead" or rm_interactions_90d >= 3:
        return "RM / Branch"
    if channel_preference == "Digital" and probability >= 0.22:
        return "Digital"
    if channel_preference == "Hybrid":
        return "Hybrid"
    return "Outbound Campaign"


def product_base_value(product: str) -> float:
    return {
        "Credit Card": 5200.0,
        "Personal Loan": 14500.0,
        "Insurance": 9800.0,
        "Wealth Upgrade": 18200.0,
    }.get(product, 7000.0)


def expected_value(probability: float, product: str, value_multiplier: float = 1.0) -> float:
    return round(product_base_value(product) * clamp(probability, 0.0, 1.0) * max(0.4, value_multiplier), 2)


def normalize_binary(value: object) -> int:
    if isinstance(value, str):
        return int(value.strip().lower() in {"1", "true", "yes", "y"})
    return int(bool(value))


def top_reasons(contributions: List[dict], direction: str, limit: int = 3) -> List[str]:
    selected = [c["description"] for c in contributions if c["impact_direction"] == direction]
    return selected[:limit]


def weighted_score(values: List[float], weights: List[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return 0.0
    return float(np.average(np.asarray(values, dtype=float), weights=np.asarray(weights, dtype=float)))
