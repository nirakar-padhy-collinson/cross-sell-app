from __future__ import annotations

import numpy as np
import pandas as pd


def historical_value_column(df: pd.DataFrame) -> str:
    if "historical_net_expected_value" in df.columns and df["historical_net_expected_value"].notna().any():
        return "historical_net_expected_value"
    return "historical_expected_value"


def targetable_mask(df: pd.DataFrame) -> pd.Series:
    return df["historical_decision"].isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"])


def conversion_lift(df: pd.DataFrame) -> dict[str, float | None]:
    labeled = df[df["converted_flag"].notna()].copy()
    if labeled.empty or "holdout_control_flag" not in labeled:
        return {"contacted_conversion": None, "holdout_conversion": None, "absolute_lift": None, "relative_lift": None}
    contacted = labeled[labeled["contacted_flag"] == 1]
    holdout = labeled[labeled["holdout_control_flag"] == 1]
    if contacted.empty or holdout.empty:
        return {"contacted_conversion": None, "holdout_conversion": None, "absolute_lift": None, "relative_lift": None}
    contacted_conversion = float(contacted["converted_flag"].mean())
    holdout_conversion = float(holdout["converted_flag"].mean())
    absolute_lift = contacted_conversion - holdout_conversion
    relative_lift = absolute_lift / holdout_conversion if holdout_conversion else None
    return {
        "contacted_conversion": round(contacted_conversion, 4),
        "holdout_conversion": round(holdout_conversion, 4),
        "absolute_lift": round(absolute_lift, 4),
        "relative_lift": round(relative_lift, 4) if relative_lift is not None else None,
    }


def top_decile_lift(df: pd.DataFrame) -> dict[str, float | None]:
    labeled = df[df["converted_flag"].notna()].copy()
    if labeled.empty or "historical_priority_score" not in labeled:
        return {"top_decile_conversion": None, "portfolio_conversion": None, "lift": None}
    cutoff = labeled["historical_priority_score"].quantile(0.9)
    top_decile = labeled[labeled["historical_priority_score"] >= cutoff]
    if top_decile.empty:
        return {"top_decile_conversion": None, "portfolio_conversion": None, "lift": None}
    top_rate = float(top_decile["converted_flag"].mean())
    portfolio_rate = float(labeled["converted_flag"].mean())
    lift = top_rate / portfolio_rate if portfolio_rate else None
    return {
        "top_decile_conversion": round(top_rate, 4),
        "portfolio_conversion": round(portfolio_rate, 4),
        "lift": round(lift, 4) if lift is not None else None,
    }


def product_performance(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    value_column = historical_value_column(df)
    return (
        df.groupby("historical_recommended_product", as_index=False)
        .agg(
            opportunities=("opportunity_id", "count"),
            targetable_rate=("historical_decision", lambda x: x.isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"]).mean()),
            suppress_rate=("historical_decision", lambda x: (x == "Suppress").mean()),
            conversion_rate=("converted_flag", "mean"),
            avg_propensity=("historical_propensity_probability", "mean"),
            avg_net_expected_value=(value_column, "mean"),
            realized_value=("realized_value", "mean"),
        )
        .sort_values(["avg_net_expected_value", "opportunities"], ascending=[False, False])
    )


def rm_capacity_allocation(df: pd.DataFrame, max_leads_per_rm: int = 25) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    candidates = df[targetable_mask(df)].copy()
    value_column = historical_value_column(candidates)
    candidates = candidates.sort_values(["employee_id", value_column, "historical_priority_score"], ascending=[True, False, False])
    candidates["rm_rank"] = candidates.groupby("employee_id").cumcount() + 1
    candidates["within_capacity"] = candidates["rm_rank"] <= max_leads_per_rm
    return (
        candidates.groupby(["employee_id", "branch_id"], as_index=False)
        .agg(
            eligible_leads=("opportunity_id", "count"),
            allocated_leads=("within_capacity", "sum"),
            avg_net_expected_value=(value_column, "mean"),
            expected_pipeline_value=(value_column, "sum"),
        )
        .sort_values(["expected_pipeline_value", "allocated_leads"], ascending=[False, False])
    )


def fairness_proxy_table(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    columns = columns or ["segment", "city_tier", "employment_type", "channel_preference"]
    rows: list[dict[str, object]] = []
    for column in columns:
        if column not in df.columns:
            continue
        for segment, group in df.groupby(column, dropna=False):
            rows.append(
                {
                    "proxy_attribute": column,
                    "segment": str(segment),
                    "opportunities": len(group),
                    "targetable_rate": float(targetable_mask(group).mean()),
                    "suppress_rate": float((group["historical_decision"] == "Suppress").mean()),
                    "conversion_rate": float(group["converted_flag"].mean()) if "converted_flag" in group else np.nan,
                    "avg_propensity": float(group["historical_propensity_probability"].mean()) if "historical_propensity_probability" in group else np.nan,
                }
            )
    return pd.DataFrame(rows)


def portfolio_report(df: pd.DataFrame) -> dict[str, object]:
    value_column = historical_value_column(df)
    return {
        "opportunities": int(len(df)),
        "targetable_rate": round(float(targetable_mask(df).mean()), 4) if len(df) else 0.0,
        "suppress_rate": round(float((df["historical_decision"] == "Suppress").mean()), 4) if len(df) else 0.0,
        "avg_net_expected_value": round(float(df[value_column].mean()), 2) if value_column in df and len(df) else 0.0,
        "conversion_lift": conversion_lift(df),
        "top_decile_lift": top_decile_lift(df),
        "product_performance": product_performance(df).to_dict(orient="records"),
        "rm_capacity_allocation": rm_capacity_allocation(df).to_dict(orient="records"),
        "fairness_proxy": fairness_proxy_table(df).to_dict(orient="records"),
    }
