from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Dict

import altair as alt
import pandas as pd
import streamlit as st

from assets.styles import CUSTOM_CSS
from models.contracts import CrossSellOpportunity, RecommendationOutput
from scoring.ml_engine import MLRecommendationEngine
from scoring.rule_engine import RuleBasedRecommendationEngine
from utils.data_loader import (
    CAMPAIGN_CODES,
    append_assessment_to_history,
    allocate_new_ids,
    branch_employee_options,
    build_opportunity_from_row,
    load_or_create_data,
)

st.set_page_config(page_title="Cross-Sell Decision Studio", layout="wide", initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent
DATA_FILE = APP_DIR / "data" / "historical_cross_sell_opportunities.csv"


@st.cache_data(show_spinner=False)
def get_data(path_str: str) -> pd.DataFrame:
    return load_or_create_data(Path(path_str))


@st.cache_resource(show_spinner=False)
def get_ml_engine(path_str: str) -> MLRecommendationEngine:
    df = load_or_create_data(Path(path_str))
    engine = MLRecommendationEngine(model_dir=str(APP_DIR / "artifacts"))
    if not engine.load():
        engine.train(df)
    return engine


def get_engines(path_str: str) -> Dict[str, object]:
    return {
        "Rule-Based Recommendation Engine": RuleBasedRecommendationEngine(),
        "Machine Learning Recommendation Engine": get_ml_engine(path_str),
    }


def fmt_currency(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"₹{float(value):,.0f}"


def fmt_pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100 * float(value):.1f}%"


def historical_value_column(df: pd.DataFrame) -> str:
    if "historical_net_expected_value" in df.columns and df["historical_net_expected_value"].notna().any():
        return "historical_net_expected_value"
    return "historical_expected_value"


def decision_badge(decision: str) -> str:
    css = {
        "RM Priority Lead": "badge-priority",
        "Campaign Target": "badge-target",
        "Digital Nurture": "badge-nurture",
        "Hold / Defer": "badge-hold",
        "Suppress": "badge-suppress",
    }.get(decision, "badge-hold")
    return f'<span class="decision-badge {css}">{decision}</span>'


def render_stat_card(label: str, value: str, description: str):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="stat-card-label">{label}</div>
            <div class="stat-card-value">{value}</div>
            <div class="stat-card-description">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_driver_item(text: str, tone: str):
    st.markdown(
        f"""
        <div class="driver-item driver-{tone}">{text}</div>
        """,
        unsafe_allow_html=True,
    )


def confidence_label(confidence: float | None, probability: float) -> str:
    effective = confidence if confidence is not None else abs(probability - 0.5) * 2
    if effective >= 0.72:
        return "High confidence"
    if effective >= 0.45:
        return "Medium confidence"
    return "Needs review"


def render_status_badges(items: list[tuple[str, str]]) -> None:
    badge_html = " ".join(
        f"<span style='display:inline-block;margin:0 0.35rem 0.45rem 0;padding:0.35rem 0.65rem;border-radius:999px;"
        f"background:{color};color:#fff;font-size:0.78rem;font-weight:700;'>{label}</span>"
        for label, color in items
    )
    st.markdown(badge_html, unsafe_allow_html=True)


def product_held(opportunity: CrossSellOpportunity, product: str) -> bool:
    return {
        "Credit Card": bool(opportunity.has_credit_card_flag),
        "Personal Loan": bool(opportunity.has_personal_loan_flag),
        "Insurance": bool(opportunity.has_insurance_flag),
        "Wealth Upgrade": False,
    }.get(product, False)


def cross_sell_governance_badges(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> list[tuple[str, str]]:
    green = "#3E7D57"
    amber = "#C59A49"
    red = "#C44F45"
    blue = "#2E6896"
    cooldown_ok = opportunity.last_campaign_contact_days > 30
    product_gap = not product_held(opportunity, output.recommended_product)
    return [
        ("Consent Available" if opportunity.consent_flag else "No Consent", green if opportunity.consent_flag else red),
        ("Contactable" if opportunity.contactable_flag else "Not Contactable", green if opportunity.contactable_flag else red),
        ("KYC Complete" if opportunity.kyc_complete_flag else "KYC Block", green if opportunity.kyc_complete_flag else amber),
        ("Cooldown Passed" if cooldown_ok else "Cooldown Active", green if cooldown_ok else amber),
        ("Product Gap Confirmed" if product_gap else "Already Holds Product", green if product_gap else amber),
        ("Suppression Clear" if output.suppression_flag == 0 else "Suppression Active", green if output.suppression_flag == 0 else red),
        (confidence_label(output.confidence, output.propensity_probability), blue if confidence_label(output.confidence, output.propensity_probability) != "Needs review" else amber),
    ]


def cross_sell_alerts(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> list[str]:
    alerts: list[str] = []
    if not opportunity.consent_flag:
        alerts.append("Consent is unavailable; proactive outreach should remain suppressed.")
    if not opportunity.contactable_flag:
        alerts.append("No approved contact route is available for this customer.")
    if not opportunity.kyc_complete_flag:
        alerts.append("KYC is incomplete; immediate fulfillment should be blocked.")
    if opportunity.recent_service_issue_flag:
        alerts.append("Recent service issue is active; resolve service recovery before sales outreach.")
    if opportunity.last_campaign_contact_days <= 30:
        alerts.append("Campaign cooldown is active; avoid contact fatigue.")
    if product_held(opportunity, output.recommended_product):
        alerts.append(f"Customer already holds {output.recommended_product}; use the why-not panel before activating.")
    if output.decision in {"Hold / Defer", "Suppress"}:
        alerts.append(f"Current action is {output.decision}; banker should not treat this as an active sales lead.")
    return alerts


def cross_sell_banker_brief(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> str:
    positive = output.top_positive_reasons[0] if output.top_positive_reasons else "No dominant supportive driver surfaced."
    adverse = output.top_negative_reasons[0] if output.top_negative_reasons else "No material adverse driver surfaced."
    return (
        f"{opportunity.customer_id} is recommended for **{output.decision}** via **{output.recommended_channel}** "
        f"with **{output.recommended_product}** as the proposition. Primary support: {positive} "
        f"Primary caution: {adverse} Banker action: {output.recommended_next_step}"
    )


def render_cross_sell_structured_brief(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> None:
    watchouts = cross_sell_alerts(opportunity, output) or ["No material consent, KYC, fatigue, or suitability watchout is active."]
    brief = pd.DataFrame(
        [
            {"Section": "Executive Summary", "Brief": f"{opportunity.customer_id} is recommended for {output.decision} with {output.recommended_product} via {output.recommended_channel}."},
            {"Section": "Key Rationale", "Brief": (output.top_positive_reasons or ["No dominant supportive driver surfaced."])[0]},
            {"Section": "Recommended Banker Action", "Brief": output.recommended_next_step},
            {"Section": "Compliance Watchouts", "Brief": " ".join(watchouts[:2])},
        ]
    )
    render_table(brief, max_rows_visible=4)


def cross_sell_decision_packet_html(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> str:
    positives = "".join(f"<li>{item}</li>" for item in output.top_positive_reasons[:3]) or "<li>No dominant supportive driver surfaced.</li>"
    negatives = "".join(f"<li>{item}</li>" for item in output.top_negative_reasons[:3]) or "<li>No material adverse driver surfaced.</li>"
    alerts = "".join(f"<li>{item}</li>" for item in cross_sell_alerts(opportunity, output)) or "<li>No material active alert.</li>"
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Cross-Sell Decision Packet {opportunity.opportunity_id}</title>
<style>body{{font-family:Arial,sans-serif;line-height:1.45;color:#1f2933;margin:32px}} .kpi{{display:inline-block;margin:6px 12px 6px 0;padding:10px 14px;border:1px solid #d8dee7;border-radius:8px}}</style></head>
<body>
<h1>Growth Decision Packet</h1>
<p><strong>Opportunity:</strong> {opportunity.opportunity_id} | <strong>Customer:</strong> {opportunity.customer_id} | <strong>Engine:</strong> {output.engine_name}</p>
<div class="kpi"><strong>Decision</strong><br>{output.decision}</div>
<div class="kpi"><strong>Product</strong><br>{output.recommended_product}</div>
<div class="kpi"><strong>Channel</strong><br>{output.recommended_channel}</div>
<div class="kpi"><strong>Propensity</strong><br>{output.propensity_probability:.1%}</div>
<h2>AI Banker Brief</h2><p>{cross_sell_banker_brief(opportunity, output)}</p>
<h2>Recommended Action</h2><p>{output.recommended_next_step}</p>
<h2>Supportive Drivers</h2><ul>{positives}</ul>
<h2>Adverse Drivers / Watchouts</h2><ul>{negatives}{alerts}</ul>
<h2>Governance Status</h2><p>Consent: {bool(opportunity.consent_flag)}; Contactable: {bool(opportunity.contactable_flag)}; KYC: {bool(opportunity.kyc_complete_flag)}; Suppression flag: {output.suppression_flag}</p>
</body></html>"""


def render_cross_sell_packet_download(opportunity: CrossSellOpportunity, output: RecommendationOutput) -> None:
    st.download_button(
        "Download Decision Packet",
        data=cross_sell_decision_packet_html(opportunity, output).encode("utf-8"),
        file_name=f"{opportunity.opportunity_id}_growth_decision_packet.html",
        mime="text/html",
        use_container_width=True,
    )


def render_cross_sell_banker_notes(opportunity: CrossSellOpportunity) -> None:
    with st.expander("Policy Override / Banker Notes", expanded=False):
        override_requested = st.checkbox("Override requested", key=f"xs_override_{opportunity.opportunity_id}")
        escalation_reason = st.selectbox(
            "Escalation reason",
            ["None", "RM exception", "Consent remediation", "Product suitability review", "Campaign strategy review"],
            key=f"xs_escalation_{opportunity.opportunity_id}",
        )
        banker_note = st.text_area("Banker note", key=f"xs_note_{opportunity.opportunity_id}", placeholder="Add relationship context, outreach rationale, or escalation notes.")
        if override_requested or escalation_reason != "None" or banker_note:
            st.info("Workflow note captured in session for the banker review pack.")


def render_cross_sell_similar_cases(opportunity: CrossSellOpportunity, df: pd.DataFrame | None) -> None:
    if df is None or df.empty:
        return
    with st.expander("Similar Customer Comparison", expanded=False):
        candidates = df.copy()
        candidates["similarity_gap"] = (
            (candidates["bureau_score"] - opportunity.bureau_score).abs() / 100
            + (candidates["monthly_income"] - opportunity.monthly_income).abs() / max(opportunity.monthly_income, 1)
            + (candidates["avg_monthly_balance"] - opportunity.avg_monthly_balance).abs() / max(opportunity.avg_monthly_balance, 1)
        )
        similar = candidates.sort_values("similarity_gap").head(6)[
            ["opportunity_id", "customer_id", "historical_recommended_product", "historical_decision", "historical_channel", "historical_propensity_probability", "segment"]
        ].copy()
        similar["historical_propensity_probability"] = similar["historical_propensity_probability"].map(fmt_pct)
        render_table(similar, max_rows_visible=6)


def render_cross_sell_model_health(df: pd.DataFrame) -> None:
    st.markdown("### Model & Campaign Health")
    rows = len(df)
    data_quality = "Passed" if rows and df[["opportunity_id", "historical_decision", "historical_priority_score"]].notna().all().all() else "Review"
    targetable_rate = df["historical_decision"].isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"]).mean() if rows else 0
    suppress_rate = df["historical_decision"].eq("Suppress").mean() if rows else 0
    h1, h2, h3, h4 = st.columns(4, gap="medium")
    with h1:
        render_stat_card("Model Health", "Stable", "Latest artifact loaded")
    with h2:
        render_stat_card("Policy Version", "growth-policy-v2", "Active guardrails")
    with h3:
        render_stat_card("Data Quality", data_quality, f"{rows:,} records")
    with h4:
        render_stat_card("Targetable / Suppress", f"{targetable_rate:.0%} / {suppress_rate:.0%}", "Activation balance")


def render_cross_sell_task_queues(df: pd.DataFrame, key_prefix: str = "xs") -> None:
    st.markdown("### Banker Task Queues")
    queues = [
        ("RM Priority Queue", df[df["historical_decision"].eq("RM Priority Lead")]),
        ("Suppression Review", df[df["historical_decision"].eq("Suppress")]),
        ("Consent Missing", df[df["consent_flag"].eq(0)] if "consent_flag" in df else df.iloc[0:0]),
        ("Cooldown Hold", df[df["last_campaign_contact_days"].le(30)] if "last_campaign_contact_days" in df else df.iloc[0:0]),
    ]
    qcols = st.columns(4, gap="medium")
    for col, (label, queue_df) in zip(qcols, queues):
        with col:
            render_stat_card(label, f"{len(queue_df):,}", "Open records")
    selected_queue = st.selectbox("Open Queue", [label for label, _ in queues], key=f"{key_prefix}_task_queue")
    queue_lookup = {label: qdf for label, qdf in queues}
    queue_view = queue_lookup[selected_queue].head(8)[
        ["opportunity_id", "customer_id", "employee_id", "historical_recommended_product", "historical_decision", "historical_channel", "historical_propensity_probability"]
    ].copy()
    if queue_view.empty:
        st.success("No records currently require this queue action.")
    else:
        queue_view["historical_propensity_probability"] = queue_view["historical_propensity_probability"].map(fmt_pct)
        render_table(queue_view, max_rows_visible=8)


def render_cross_sell_suite_command(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">Banking Decisioning Suite</div>', unsafe_allow_html=True)
    st.caption("A command landing for growth decisioning, lead activation, and outreach governance.")
    render_cross_sell_model_health(df)
    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_cross_sell_task_queues(df, key_prefix="xs_suite")
    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    st.markdown("### Suite Capabilities")
    capability_df = pd.DataFrame(
        [
            {"Module": "Credit Decisioning", "Capability": "Underwriting assessment, policy explainability, what-if simulation, and audit packet export."},
            {"Module": "Growth Decisioning", "Capability": "Lead prioritization, relationship context, consent controls, and activation queues."},
            {"Module": "Governance Layer", "Capability": "Champion/challenger comparison, decision traceability, model health, and banker notes."},
        ]
    )
    render_table(capability_df, max_rows_visible=3)


def render_relationship_context(opportunity: CrossSellOpportunity) -> None:
    st.markdown("### Relationship Context")
    products = []
    if opportunity.has_credit_card_flag:
        products.append("Credit Card")
    if opportunity.has_personal_loan_flag:
        products.append("Personal Loan")
    if opportunity.has_home_loan_flag:
        products.append("Home Loan")
    if opportunity.has_insurance_flag:
        products.append("Insurance")
    relationship = pd.DataFrame(
        [
            {"Signal": "Segment", "Value": opportunity.segment},
            {"Signal": "Products Held", "Value": ", ".join(products) if products else "Core relationship only"},
            {"Signal": "Tenure", "Value": f"{opportunity.relationship_tenure_months} months"},
            {"Signal": "Channel Preference", "Value": opportunity.channel_preference},
            {"Signal": "Digital Activity", "Value": f"{opportunity.digital_login_count_30d} logins / 30D"},
            {"Signal": "RM Engagement", "Value": f"{opportunity.rm_interactions_90d} interactions / 90D"},
        ]
    )
    render_table(relationship, max_rows_visible=6)


def render_decision_journey(output: RecommendationOutput) -> None:
    journey = pd.DataFrame(
        [
            {"Stage": "1. Intake", "Outcome": "Customer, product, and channel context captured"},
            {"Stage": "2. Eligibility", "Outcome": "Consent, KYC, product gap, and fatigue checks applied"},
            {"Stage": "3. Propensity", "Outcome": f"{output.propensity_probability:.1%} conversion propensity"},
            {"Stage": "4. Decision", "Outcome": output.decision},
            {"Stage": "5. Next Action", "Outcome": output.recommended_next_step},
        ]
    )
    render_table(journey, max_rows_visible=5)


def render_champion_challenger(opportunity: CrossSellOpportunity, engines: Dict[str, object]) -> None:
    st.markdown("### Champion vs Challenger")
    rows = []
    for name, engine in engines.items():
        out = engine.evaluate(opportunity)
        rows.append(
            {
                "Engine": name.replace(" Recommendation Engine", ""),
                "Decision": out.decision,
                "Product": out.recommended_product,
                "Channel": out.recommended_channel,
                "Propensity": out.propensity_probability,
                "Priority Score": out.priority_score,
                "Confidence": confidence_label(out.confidence, out.propensity_probability),
            }
        )
    comparison = pd.DataFrame(rows)
    comparison["Propensity"] = comparison["Propensity"].map(fmt_pct)
    comparison["Priority Score"] = comparison["Priority Score"].map(lambda x: f"{float(x):.0f}")
    render_table(comparison, max_rows_visible=2)
    if comparison["Decision"].nunique() > 1 or comparison["Product"].nunique() > 1:
        st.warning("Rules and ML disagree. Treat this as an RM review or campaign strategy checkpoint.")
    else:
        st.success("Rules and ML are aligned on the recommendation.")


def render_why_not(opportunity: CrossSellOpportunity, engine: object, selected_product: str) -> None:
    st.markdown("### Why This Product / Why Not Others")
    rows = []
    for product in CAMPAIGN_CODES.keys():
        product_opportunity = replace(opportunity, target_product=product)
        product_output = engine.evaluate(product_opportunity)
        reason = "Selected recommendation" if product == selected_product else (product_output.top_negative_reasons or product_output.top_positive_reasons or ["No dominant reason"])[0]
        rows.append(
            {
                "Product": product,
                "Action": product_output.decision,
                "Propensity": product_output.propensity_probability,
                "Channel": product_output.recommended_channel,
                "Primary Reason": reason,
            }
        )
    why_df = pd.DataFrame(rows)
    why_df["Propensity"] = why_df["Propensity"].map(fmt_pct)
    render_table(why_df, max_rows_visible=4)


def render_cross_sell_what_if(opportunity: CrossSellOpportunity, engines: Dict[str, object], selected_engine_name: str) -> None:
    with st.expander("What-if Simulator", expanded=False):
        st.caption("Adjust activation levers to see how the recommendation changes. Simulated cases are not saved.")
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        with c1:
            sim_product = st.selectbox("Target Product", ["Auto Select", *list(CAMPAIGN_CODES.keys())], index=["Auto Select", *list(CAMPAIGN_CODES.keys())].index(opportunity.target_product), key=f"xs_sim_product_{opportunity.opportunity_id}")
        with c2:
            sim_contact_days = st.slider("Days Since Contact", 0, 210, int(opportunity.last_campaign_contact_days), key=f"xs_sim_days_{opportunity.opportunity_id}")
        with c3:
            sim_bureau = st.slider("Bureau Score", 0, 900, int(opportunity.bureau_score), key=f"xs_sim_bureau_{opportunity.opportunity_id}")
        with c4:
            sim_digital = st.slider("Digital Logins", 0, 40, int(opportunity.digital_login_count_30d), key=f"xs_sim_digital_{opportunity.opportunity_id}")
        simulated_opportunity = replace(
            opportunity,
            target_product=sim_product,
            last_campaign_contact_days=int(sim_contact_days),
            bureau_score=int(sim_bureau),
            digital_login_count_30d=int(sim_digital),
        )
        simulated_output = engines[selected_engine_name].evaluate(simulated_opportunity)
        w1, w2, w3, w4 = st.columns(4, gap="medium")
        with w1:
            render_stat_card("Simulated Action", simulated_output.decision, "Current engine")
        with w2:
            render_stat_card("Simulated Product", simulated_output.recommended_product, "Selected proposition")
        with w3:
            render_stat_card("Simulated Channel", simulated_output.recommended_channel, "Activation route")
        with w4:
            render_stat_card("Simulated Propensity", fmt_pct(simulated_output.propensity_probability), "Conversion signal")


def render_section_intro(kicker: str, title: str, copy: str):
    st.markdown(
        f"""
        <div class="section-shell">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_intro(kicker: str, title: str, copy: str):
    st.markdown(
        f"""
        <div class="panel-intro">
            <div class="panel-kicker">{kicker}</div>
            <div class="panel-title">{title}</div>
            <div class="panel-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_intro(title: str, copy: str):
    st.markdown(
        f"""
        <div class="sidebar-hero-card">
            <div class="sidebar-hero-title">{title}</div>
            <div class="sidebar-hero-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_section(title: str, copy: str):
    st.markdown(
        f"""
        <div class="sidebar-section">
            <div class="sidebar-section-title">{title}</div>
            <div class="sidebar-section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def dataframe_height(row_count: int, *, max_rows_visible: int = 8) -> int:
    visible_rows = max(1, min(row_count, max_rows_visible))
    return 44 + (visible_rows * 42)


PRIORITY_BAND_ORDER = ["Prime", "High", "Moderate", "Low", "Suppressed"]
PRIORITY_BAND_COLORS = ["#3E7D57", "#2E6896", "#C59A49", "#D67F45", "#C44F45"]
DECISION_ORDER = ["RM Priority Lead", "Campaign Target", "Digital Nurture", "Hold / Defer", "Suppress"]
DECISION_COLORS = ["#3E7D57", "#2E6896", "#C59A49", "#D67F45", "#C44F45"]
CROSS_SELL_SCENARIOS: dict[str, dict[str, object]] = {
    "Custom Intake": {},
    "High-Value Wealth Lead": {
        "customer_name": "Scenario - Wealth Priority",
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
    "Campaign Fatigue": {
        "customer_name": "Scenario - Campaign Fatigue",
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
    "Already Holds Product": {
        "customer_name": "Scenario - Already Holds Card",
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
    "No Consent Suppression": {
        "customer_name": "Scenario - No Consent",
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
        "recent_service_issue_flag": 0,
        "bureau_score": 760,
        "lifecycle_stage": "Growth",
        "channel_preference": "Hybrid",
        "target_product": "Auto Select",
        "consent_flag": 0,
        "contactable_flag": 1,
    },
    "Unsuitable Loan Offer": {
        "customer_name": "Scenario - Suitability Hold",
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
}


def styled_chart(chart: alt.Chart, *, height: int, padding: dict | None = None) -> alt.Chart:
    return (
        chart.properties(height=height, padding=padding or {"left": 8, "right": 18, "top": 8, "bottom": 8})
        .configure_view(strokeOpacity=0)
        .configure_axis(
            domain=False,
            tickColor="#c6b6a3",
            gridColor="rgba(88, 102, 121, 0.12)",
            labelColor="#5b6778",
            titleColor="#5b6778",
            labelFontSize=11,
            titleFontSize=11,
        )
        .configure_legend(
            labelColor="#4f5c6d",
            titleColor="#4f5c6d",
            orient="right",
            symbolType="circle",
            padding=8,
        )
    )


def render_table(
    df: pd.DataFrame,
    *,
    max_rows_visible: int = 8,
    column_config: dict | None = None,
):
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=dataframe_height(len(df), max_rows_visible=max_rows_visible),
        column_config=column_config,
    )


def render_header(df: pd.DataFrame):
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    labeled = df[df["converted_flag"].notna()].copy()
    conversion = labeled["converted_flag"].mean() if len(labeled) else 0.0
    targetable = df["historical_decision"].isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"]).mean() if len(df) else 0.0
    st.markdown(
        """
        <div class="hero">
            <div class="hero-grid">
                <div>
                    <div class="hero-eyebrow">Cross-Sell Decision Studio</div>
                    <h1>Commercial decisioning for next-best-action, lead prioritization, and product growth.</h1>
                    <p>Score customers with configurable rules or machine learning, route the right action to the right channel, and monitor branch, RM, and campaign performance in one polished decisioning workspace.</p>
                </div>
                <div class="hero-panel">
                    <div class="hero-panel-value">Portfolio Snapshot</div>
                    <div class="hero-panel-copy">"""
        + f"Observed conversion {conversion * 100:.1f}% • Targetable rate {targetable * 100:.1f}%"
        + """</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_workflow_guide():
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-title">Workspace Flow</div>
            <div class="workflow-grid">
                <div class="workflow-step">
                    <div class="workflow-step-number">01</div>
                    <div class="workflow-step-title">New Opportunity</div>
                    <p>Capture a customer profile, score commercial potential, and recommend the next best action.</p>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-number">02</div>
                    <div class="workflow-step-title">Portfolio Overview</div>
                    <p>Monitor targeting mix, channel allocation, and team-level opportunity flow.</p>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-number">03</div>
                    <div class="workflow-step-title">Decision Traceability</div>
                    <p>Preserve eligibility signals, suppression reasons, and engine rationale for governance-ready activation.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_cards(df: pd.DataFrame):
    total = len(df)
    targetable = (df["historical_decision"].isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"]).mean() * 100) if total else 0
    rm_priority = (df["historical_decision"].eq("RM Priority Lead").mean() * 100) if total else 0
    suppress_rate = (df["historical_decision"].eq("Suppress").mean() * 100) if total else 0
    labeled = df[df["converted_flag"].notna()].copy()
    realized_conv = labeled["converted_flag"].mean() * 100 if len(labeled) else 0

    cards = [
        ("Total Opportunities", f"{total:,}", "Records in the working portfolio"),
        ("Targetable Rate", f"{targetable:.1f}%", "Priority + campaign + nurture actions"),
        ("RM Priority Rate", f"{rm_priority:.1f}%", "Top leads routed to relationship managers"),
        ("Suppress Rate", f"{suppress_rate:.1f}%", "Leads held out for fatigue / service reasons"),
        ("Observed Conversion", f"{realized_conv:.1f}%", "Realized on labeled historical cases"),
    ]
    cols = st.columns(5, gap="medium")
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )


def render_output(
    output: RecommendationOutput,
    opportunity: CrossSellOpportunity | None = None,
    engines: Dict[str, object] | None = None,
    selected_engine_name: str | None = None,
    history_df: pd.DataFrame | None = None,
):
    st.markdown('<div class="section-banner">Recommendation Summary</div>', unsafe_allow_html=True)
    st.caption("Interpretation guide: higher priority score and propensity indicate a stronger commercial case; the decision and channel translate that signal into the recommended action.")
    if opportunity is not None:
        st.markdown("### AI Banker Brief")
        render_cross_sell_structured_brief(opportunity, output)
        render_cross_sell_packet_download(opportunity, output)
        st.markdown("### Governance & Activation Signals")
        render_status_badges(cross_sell_governance_badges(opportunity, output))
        alerts = cross_sell_alerts(opportunity, output)
        if alerts:
            for alert in alerts:
                st.warning(alert)
        else:
            st.success("No material consent, eligibility, or fatigue alert is active for this opportunity.")
        render_relationship_context(opportunity)
        st.markdown("### Decision Journey")
        render_decision_journey(output)

    c1, c2, c3, c4, c5 = st.columns(5, gap="medium")
    with c1:
        render_stat_card("Decision", output.decision, "Operational action for this lead")
    with c2:
        render_stat_card("Priority Score", f"{output.priority_score:.0f}", "Higher means stronger commercial priority")
    with c3:
        render_stat_card("Propensity", fmt_pct(output.propensity_probability), "Estimated probability of conversion")
    with c4:
        render_stat_card("Recommended Product", output.recommended_product, "Product with the strongest fit under the current engine")
    with c5:
        render_stat_card("Channel", output.recommended_channel, "Suggested route to market")

    st.markdown("### Commercial Lenses")
    l1, l2, l3, l4 = st.columns(4, gap="medium")
    with l1:
        render_stat_card("Relationship", f"{output.relationship_score:.0f}", "Depth and maturity of the banking relationship")
    with l2:
        render_stat_card("Engagement", f"{output.engagement_score:.0f}", "Observed channel and interaction intensity")
    with l3:
        render_stat_card("Product Fit", f"{output.product_fit_score:.0f}", "Suitability for the recommended proposition")
    with l4:
        render_stat_card("Value Score", f"{output.value_score:.0f}", "Commercial upside potential")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown("### Supportive Reasons")
        for item in output.top_positive_reasons or ["No strong supportive drivers surfaced."]:
            render_driver_item(item, "positive")
    with right:
        st.markdown("### Adverse Reasons")
        for item in output.top_negative_reasons or ["No material adverse drivers surfaced."]:
            render_driver_item(item, "negative")

    if output.alternate_recommendations:
        with st.expander("Alternate Recommendations", expanded=False):
            alt_df = pd.DataFrame(output.alternate_recommendations)[["product", "probability", "priority_score", "decision"]]
            alt_df.columns = ["Product", "Probability", "Priority Score", "Decision"]
            alt_df["Probability"] = alt_df["Probability"].map(fmt_pct)
            alt_df["Priority Score"] = alt_df["Priority Score"].map(lambda x: f"{float(x):.0f}")
            render_table(
                alt_df,
                max_rows_visible=4,
            )

    with st.expander("Engine Rationale & Model Detail", expanded=False):
        insight_left, insight_right = st.columns([1.1, 0.9], gap="large")
        with insight_left:
            st.markdown("### Engine Rationale")
            rationale_df = pd.DataFrame([
                {
                    "Factor": c.factor,
                    "Direction": c.impact_direction,
                    "Points": c.points,
                    "Description": c.description,
                }
                for c in output.factor_contributions
            ])
            render_table(rationale_df, max_rows_visible=6)
        with insight_right:
            st.markdown("### Notes")
            for note in output.notes:
                st.info(note)
            if output.feature_importance:
                fi_df = pd.DataFrame(output.feature_importance)
                render_table(fi_df, max_rows_visible=6)
            if output.confidence is not None:
                st.metric("Model Confidence", f"{output.confidence * 100:.1f}%")

    if opportunity is not None and engines is not None:
        render_champion_challenger(opportunity, engines)
        if selected_engine_name is not None:
            render_why_not(opportunity, engines[selected_engine_name], output.recommended_product)
            render_cross_sell_what_if(opportunity, engines, selected_engine_name)
        render_cross_sell_similar_cases(opportunity, history_df)
        render_cross_sell_banker_notes(opportunity)


def render_new_opportunity_tab(df: pd.DataFrame, engines: Dict[str, object], selected_engine_name: str):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Assessment Workspace",
        "New Opportunity Assessment",
        "Capture a customer profile, review the commercial context, and run a recommendation using the active engine. Every submitted assessment is appended to the portfolio history immediately.",
    )
    scenario_name = st.selectbox(
        "Scenario Library",
        list(CROSS_SELL_SCENARIOS.keys()),
        help="Use a guided scenario to quickly show high-value targeting, fatigue, product overlap, consent suppression, or suitability hold.",
    )
    defaults = CROSS_SELL_SCENARIOS[scenario_name]
    if scenario_name != "Custom Intake":
        st.caption(f"Loaded scenario: {scenario_name}. You can still adjust any field before running the assessment.")

    options = branch_employee_options()
    next_ids = allocate_new_ids(df)

    with st.form(f"new_opportunity_form_{scenario_name}", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### IDs")
            opportunity_id = st.text_input("Opportunity ID", value=next_ids["opportunity_id"], help="Unique lead-level identifier for this assessment run.")
            customer_id = st.text_input("Customer ID", value=next_ids["customer_id"], help="Customer-level identifier used to track repeat reviews and historical comparisons.")
            customer_name = st.text_input("Customer Name", value=str(defaults.get("customer_name", "Customer A")), help="Display name shown in the case record.")
            branch_id = st.selectbox("Branch ID", options=list(options.keys()), index=0, help="Originating branch for the lead or customer relationship.")
            employee_id = st.selectbox("Employee ID", options=options[branch_id], index=0, help="RM or employee who owns the lead in this relationship record.")
            target_product_options = ["Auto Select", *list(CAMPAIGN_CODES.keys())]
            default_target_product = str(defaults.get("target_product", "Auto Select"))
            target_product = st.selectbox("Target Product", options=target_product_options, index=target_product_options.index(default_target_product) if default_target_product in target_product_options else 0, help="Choose a product to force evaluation toward that campaign, or leave Auto Select to let the engine recommend the strongest fit.")
            campaign_id = st.text_input("Campaign ID", value=CAMPAIGN_CODES.get(target_product, "CP9000"), help="Campaign reference tied to the selected proposition or outreach motion.")

            st.markdown("#### Borrower / Customer Profile")
            age = st.slider("Age", min_value=21, max_value=75, value=int(defaults.get("age", 36)), help="Basic demographic input used as one of several suitability signals.")
            segment_options = ["Mass", "Affluent", "HNI"]
            segment = st.selectbox("Segment", segment_options, index=segment_options.index(str(defaults.get("segment", "Mass"))), help="Commercial segment classification. Higher-value segments often support richer offers and RM-led servicing.")
            monthly_income = st.number_input("Monthly Income", min_value=20000.0, max_value=500000.0, value=float(defaults.get("monthly_income", 90000.0)), step=5000.0, help="Approximate monthly income used for affordability and commercial potential.")
            relationship_tenure_months = st.slider("Relationship Tenure (Months)", 1, 180, int(defaults.get("relationship_tenure_months", 30)), help="Longer tenure usually signals a deeper, more stable relationship.")
            employment_options = ["Salaried", "Self-Employed", "Professional", "Business Owner", "Contract"]
            employment_type = st.selectbox("Employment Type", employment_options, index=employment_options.index(str(defaults.get("employment_type", "Salaried"))), help="Employment profile can affect risk, fit, and product relevance.")
            city_options = ["Tier 1", "Tier 2", "Tier 3"]
            city_tier = st.selectbox("City Tier", city_options, index=city_options.index(str(defaults.get("city_tier", "Tier 1"))), help="Location tier used as a light proxy for market context.")
            lifecycle_options = ["Emerging", "Growth", "Established", "Mature"]
            lifecycle_stage = st.selectbox("Lifecycle Stage", lifecycle_options, index=lifecycle_options.index(str(defaults.get("lifecycle_stage", "Growth"))), help="Business or customer maturity stage. Growth and established customers often surface stronger cross-sell opportunities.")
            channel_options = ["Digital", "Hybrid", "RM / Branch"]
            channel_preference = st.selectbox("Channel Preference", channel_options, index=channel_options.index(str(defaults.get("channel_preference", "Digital"))), help="Preferred outreach model for activation. This helps interpret the suggested channel.")

        with col2:
            st.markdown("#### Relationship / KYC / Holdings")
            existing_customer_flag = st.toggle("Existing Customer", value=bool(defaults.get("existing_customer_flag", 1)), help="Turn on if the customer already has a relationship with the bank.")
            salary_customer_flag = st.toggle("Salary Customer", value=bool(defaults.get("salary_customer_flag", 1)), help="Salary customers often show stronger engagement and targeting potential.")
            kyc_complete_flag = st.toggle("KYC Complete", value=bool(defaults.get("kyc_complete_flag", 1)), help="Incomplete KYC may limit immediate campaign eligibility or product fulfilment.")
            consent_flag = st.toggle("Consent Available", value=bool(defaults.get("consent_flag", 1)), help="Controls whether proactive outreach is permitted.")
            contactable_flag = st.toggle("Approved Contact Route", value=bool(defaults.get("contactable_flag", 1)), help="Controls whether the customer can be contacted through approved channels.")
            has_credit_card_flag = st.toggle("Already Holds Credit Card", value=bool(defaults.get("has_credit_card_flag", 0)), help="Use current holdings to avoid recommending what the customer already has.")
            has_personal_loan_flag = st.toggle("Already Holds Personal Loan", value=bool(defaults.get("has_personal_loan_flag", 0)), help="Existing product ownership affects fit and cross-sell headroom.")
            has_home_loan_flag = st.toggle("Already Holds Home Loan", value=bool(defaults.get("has_home_loan_flag", 0)), help="Existing secured borrowing may shift the next-best-product recommendation.")
            has_insurance_flag = st.toggle("Already Holds Insurance", value=bool(defaults.get("has_insurance_flag", 0)), help="Existing protection cover can reduce the need for insurance-led outreach.")
            product_holding_count = st.slider("Product Holding Count", 1, 6, int(defaults.get("product_holding_count", 2)), help="How many products the customer already holds with the bank. Higher counts may imply stronger relationship depth.")

            st.markdown("#### Behavior / Engagement / Campaign History")
            avg_monthly_balance = st.number_input("Average Monthly Balance", min_value=1000.0, max_value=1000000.0, value=float(defaults.get("avg_monthly_balance", 145000.0)), step=5000.0, help="Average balance is a strong indicator of wallet depth and commercial value.")
            avg_debit_txn_count_3m = st.slider("Avg Debit Transactions (3M)", 1, 60, int(defaults.get("avg_debit_txn_count_3m", 18)), help="Recent transaction frequency is used as an activity signal.")
            avg_credit_txn_count_3m = st.slider("Avg Credit Transactions (3M)", 1, 25, int(defaults.get("avg_credit_txn_count_3m", 8)), help="Captures income or inward-flow regularity over the past 3 months.")
            digital_login_count_30d = st.slider("Digital Logins (30D)", 0, 40, int(defaults.get("digital_login_count_30d", 12)), help="Higher digital activity can support digital or hybrid outreach decisions.")
            app_sessions_30d = st.slider("App Sessions (30D)", 0, 30, int(defaults.get("app_sessions_30d", 10)), help="Used as another engagement signal alongside login activity.")
            branch_visits_90d = st.slider("Branch Visits (90D)", 0, 10, int(defaults.get("branch_visits_90d", 1)), help="Higher branch usage may support RM or branch-led activation.")
            rm_interactions_90d = st.slider("RM Interactions (90D)", 0, 10, int(defaults.get("rm_interactions_90d", 2)), help="Frequent RM interaction often supports higher-touch channels.")
            last_campaign_contact_days = st.slider("Days Since Last Campaign Contact", 0, 210, int(defaults.get("last_campaign_contact_days", 45)), help="Very recent contact can reduce readiness for another outbound push.")
            response_options = ["Accepted", "Clicked", "Opened", "No Response", "Declined"]
            last_campaign_response = st.selectbox("Last Campaign Response", response_options, index=response_options.index(str(defaults.get("last_campaign_response", "No Response"))), help="Past response behavior helps interpret current propensity and channel fit.")
            prior_offer_accept_count_12m = st.slider("Prior Offer Accepts (12M)", 0, 5, int(defaults.get("prior_offer_accept_count_12m", 0)), help="Past offer acceptance is usually a positive responsiveness signal.")
            recent_service_issue_flag = st.toggle("Recent Service Issue", value=bool(defaults.get("recent_service_issue_flag", 0)), help="Turn on if the customer recently had a service issue that could justify holdout or suppression.")
            bureau_score = st.slider("Bureau Score", 0, 900, int(defaults.get("bureau_score", 725)), help="Credit quality signal. Higher values generally support stronger eligibility and lower risk.")

        submit = st.form_submit_button("Run Assessment", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submit:
        opportunity = CrossSellOpportunity(
            opportunity_id=opportunity_id,
            customer_id=customer_id,
            employee_id=employee_id,
            branch_id=branch_id,
            campaign_id=campaign_id,
            customer_name=customer_name,
            age=age,
            segment=segment,
            monthly_income=monthly_income,
            relationship_tenure_months=relationship_tenure_months,
            employment_type=employment_type,
            city_tier=city_tier,
            existing_customer_flag=int(existing_customer_flag),
            salary_customer_flag=int(salary_customer_flag),
            kyc_complete_flag=int(kyc_complete_flag),
            product_holding_count=product_holding_count,
            has_credit_card_flag=int(has_credit_card_flag),
            has_personal_loan_flag=int(has_personal_loan_flag),
            has_home_loan_flag=int(has_home_loan_flag),
            has_insurance_flag=int(has_insurance_flag),
            avg_monthly_balance=avg_monthly_balance,
            avg_debit_txn_count_3m=avg_debit_txn_count_3m,
            avg_credit_txn_count_3m=avg_credit_txn_count_3m,
            digital_login_count_30d=digital_login_count_30d,
            app_sessions_30d=app_sessions_30d,
            branch_visits_90d=branch_visits_90d,
            rm_interactions_90d=rm_interactions_90d,
            last_campaign_contact_days=last_campaign_contact_days,
            last_campaign_response=last_campaign_response,
            prior_offer_accept_count_12m=prior_offer_accept_count_12m,
            recent_service_issue_flag=int(recent_service_issue_flag),
            bureau_score=bureau_score,
            lifecycle_stage=lifecycle_stage,
            channel_preference=channel_preference,
            target_product=target_product,
            consent_flag=int(consent_flag),
            contactable_flag=int(contactable_flag),
        )
        output = engines[selected_engine_name].evaluate(opportunity)
        append_assessment_to_history(DATA_FILE, opportunity, output)
        st.cache_data.clear()
        st.session_state["latest_output"] = output
        st.session_state["latest_opportunity"] = opportunity
        st.session_state["latest_opportunity_id"] = opportunity_id
        st.success(f"Assessment saved to history and scored using {selected_engine_name}.")
        render_output(output, opportunity, engines, selected_engine_name, df)
    elif st.session_state.get("latest_output"):
        render_output(st.session_state["latest_output"], st.session_state.get("latest_opportunity"), engines, selected_engine_name, df)


# Review workspace supports governance, audit, explainability, and model/rule replay.
def render_case_review_tab(df: pd.DataFrame, engines: Dict[str, object], selected_engine_name: str):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Governance Workspace",
        "Decision Review",
        "Search a historical opportunity or customer, then compare the stored portfolio record against current rule and ML replay using the same customer inputs.",
    )

    search_mode = st.radio("Search By", ["Opportunity ID", "Customer ID"], horizontal=True, help="Choose whether to retrieve a saved case by lead ID or customer ID.")
    search_value = st.text_input("Search", value=st.session_state.get("latest_opportunity_id", ""), help="Enter a full or partial ID to find a historical record.")

    review_df = df.copy()
    if search_value:
        if search_mode == "Opportunity ID":
            review_df = review_df[review_df["opportunity_id"].astype(str).str.contains(search_value, case=False, na=False)]
        else:
            review_df = review_df[review_df["customer_id"].astype(str).str.contains(search_value, case=False, na=False)]

    if review_df.empty:
        st.info("No matching opportunity found yet.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    review_df = review_df.sort_values("assessment_timestamp", ascending=False)
    selected_id = st.selectbox("Select Record", options=review_df["opportunity_id"].tolist(), help="Pick the saved assessment you want to compare against the currently selected engine.")
    record = review_df[review_df["opportunity_id"] == selected_id].iloc[0]
    opportunity = build_opportunity_from_row(record)
    rule_output = engines["Rule-Based Recommendation Engine"].evaluate(opportunity)
    ml_output = engines["Machine Learning Recommendation Engine"].evaluate(opportunity)

    st.markdown("### Recorded Decision vs Current Engine Replay")
    st.caption(
        "Replay uses the same saved customer inputs. It is a current-engine comparison, not a reconstruction of an older model artifact."
    )
    comparison_df = pd.DataFrame(
        [
            ["Decision", record["historical_decision"], rule_output.decision, ml_output.decision],
            ["Recommended Product", record["historical_recommended_product"], rule_output.recommended_product, ml_output.recommended_product],
            ["Priority Score", f"{float(record['historical_priority_score']):.0f}", f"{rule_output.priority_score:.0f}", f"{ml_output.priority_score:.0f}"],
            ["Propensity", fmt_pct(record["historical_propensity_probability"]), fmt_pct(rule_output.propensity_probability), fmt_pct(ml_output.propensity_probability)],
            ["Channel", record["historical_channel"], rule_output.recommended_channel, ml_output.recommended_channel],
            ["Next Step", record["historical_next_step"], rule_output.recommended_next_step, ml_output.recommended_next_step],
        ],
        columns=["Measure", "Recorded History", "Current Rule Replay", "Current ML Replay"],
    )
    render_table(comparison_df, max_rows_visible=6)

    if record["historical_decision"] != rule_output.decision or record["historical_decision"] != ml_output.decision:
        st.warning("At least one current engine replay differs from the recorded decision. Treat this as a campaign governance review cue.")
    else:
        st.success("Recorded decision and current replays are directionally aligned.")

    st.markdown("### Recorded Customer Interaction Context")
    interaction_df = pd.DataFrame(
        [
            {"Item": "Assessment Timestamp", "Value": record.get("assessment_timestamp", "N/A")},
            {"Item": "Last Campaign Response", "Value": record.get("last_campaign_response", "N/A")},
            {"Item": "Days Since Last Contact", "Value": record.get("last_campaign_contact_days", "N/A")},
            {"Item": "Prior Accepted Offers 12M", "Value": record.get("prior_offer_accept_count_12m", "N/A")},
            {"Item": "RM Interactions 90D", "Value": record.get("rm_interactions_90d", "N/A")},
            {"Item": "Branch Visits 90D", "Value": record.get("branch_visits_90d", "N/A")},
            {"Item": "Contacted / Holdout / Converted", "Value": f"{record.get('contacted_flag', 'N/A')} / {record.get('holdout_control_flag', 'N/A')} / {record.get('converted_flag', 'N/A')}"},
            {"Item": "Realized Value", "Value": fmt_currency(record.get("realized_value", 0)) if pd.notna(record.get("realized_value", 0)) else "Pending"},
        ]
    )
    render_table(interaction_df, max_rows_visible=8)

    st.markdown("### Opportunity Detail")
    detail_columns = [
        "opportunity_id", "customer_id", "employee_id", "branch_id", "campaign_id", "customer_name", "age", "segment",
        "monthly_income", "relationship_tenure_months", "employment_type", "city_tier", "existing_customer_flag",
        "salary_customer_flag", "kyc_complete_flag", "product_holding_count", "has_credit_card_flag", "has_personal_loan_flag",
        "has_home_loan_flag", "has_insurance_flag", "avg_monthly_balance", "digital_login_count_30d", "app_sessions_30d",
        "rm_interactions_90d", "last_campaign_contact_days", "last_campaign_response", "prior_offer_accept_count_12m",
        "recent_service_issue_flag", "bureau_score", "lifecycle_stage", "channel_preference", "target_product",
    ]
    detail_df = pd.DataFrame({"Field": detail_columns, "Value": [record[col] for col in detail_columns]})
    render_table(detail_df, max_rows_visible=10)

    st.markdown("### Current Replay Drivers")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Rule-Based Engine")
        for item in rule_output.top_negative_reasons[:3] + rule_output.top_positive_reasons[:3]:
            render_driver_item(item, "negative" if item in rule_output.top_negative_reasons else "positive")
    with right:
        st.markdown("#### Machine Learning Engine")
        for item in ml_output.top_negative_reasons[:3] + ml_output.top_positive_reasons[:3]:
            render_driver_item(item, "negative" if item in ml_output.top_negative_reasons else "positive")
    st.markdown("</div>", unsafe_allow_html=True)


def render_cross_sell_command_center(df: pd.DataFrame) -> None:
    st.markdown("### Portfolio Command Center")
    if df.empty:
        st.info("No records match the current filters.")
        return

    targetable = df["historical_decision"].isin(["RM Priority Lead", "Campaign Target", "Digital Nurture"])
    suppress_rate = df["historical_decision"].eq("Suppress").mean()
    rm_ready = df["historical_decision"].eq("RM Priority Lead").sum()
    avg_propensity = df["historical_propensity_probability"].mean()
    top_product = df["historical_recommended_product"].mode().iloc[0] if not df["historical_recommended_product"].mode().empty else "N/A"

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        render_stat_card("Activation-Ready", f"{targetable.mean():.1%}", "Routable leads")
    with c2:
        render_stat_card("RM Priority Queue", f"{rm_ready:,}", "High-touch actions")
    with c3:
        render_stat_card("Suppression Rate", f"{suppress_rate:.1%}", "Governance discipline")
    with c4:
        render_stat_card("Avg Propensity", fmt_pct(avg_propensity), "Conversion signal")

    suppression_reasons = {
        "Consent / Contactability": int(((df.get("consent_flag", 1) == 0) | (df.get("contactable_flag", 1) == 0)).sum()) if "consent_flag" in df and "contactable_flag" in df else 0,
        "KYC Hold": int((df["kyc_complete_flag"] == 0).sum()) if "kyc_complete_flag" in df else 0,
        "Service Recovery": int((df["recent_service_issue_flag"] == 1).sum()) if "recent_service_issue_flag" in df else 0,
        "Contact Fatigue": int((df["last_campaign_contact_days"] <= 30).sum()) if "last_campaign_contact_days" in df else 0,
    }
    st.info(f"Most common recommended product in this slice: {top_product}.")
    st.info("Use RM Priority Queue for banker allocation and Suppress/Hold lists for governance review.")
    reason_df = pd.DataFrame(
        [{"Suppression Driver": reason, "Records": count} for reason, count in suppression_reasons.items() if count > 0]
    )
    if not reason_df.empty:
        st.markdown("#### Suppression Drivers")
        render_table(reason_df, max_rows_visible=4)

    ready_cols = [
        "opportunity_id",
        "customer_id",
        "employee_id",
        "historical_recommended_product",
        "historical_channel",
        "historical_priority_score",
        "historical_propensity_probability",
    ]
    ready_df = df[targetable].sort_values(["historical_priority_score", "historical_propensity_probability"], ascending=False).head(8)
    if not ready_df.empty:
        st.markdown("#### Next Best Banker Queue")
        queue = ready_df[ready_cols].copy()
        queue["historical_priority_score"] = queue["historical_priority_score"].map(lambda x: f"{float(x):.0f}")
        queue["historical_propensity_probability"] = queue["historical_propensity_probability"].map(fmt_pct)
        render_table(queue, max_rows_visible=8)


def render_portfolio_tab(df: pd.DataFrame):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Portfolio Workspace",
        "Portfolio Overview",
        "Filter the scoring portfolio and monitor targeting mix, channel allocation, and team performance in a polished decisioning view.",
    )

    render_panel_intro(
        "Filters",
        "Portfolio Filters",
        "Use the controls below to focus the dashboard on a specific product mix, branch view, customer base, or decision segment.",
    )
    f1, f2, f3, f4 = st.columns(4, gap="medium")
    with f1:
        product_filter = st.multiselect("Product", options=sorted(df["historical_recommended_product"].dropna().unique().tolist()))
        decision_filter = st.multiselect("Decision", options=DECISION_ORDER)
    with f2:
        band_filter = st.multiselect("Priority Band", options=PRIORITY_BAND_ORDER)
        employment_filter = st.multiselect("Employment Type", options=sorted(df["employment_type"].dropna().unique().tolist()))
    with f3:
        existing_filter = st.selectbox("Existing Customer", options=["All", "Yes", "No"], index=0)
        branch_filter = st.multiselect("Branch ID", options=sorted(df["branch_id"].dropna().unique().tolist()))
    with f4:
        employee_filter = st.multiselect("Employee ID", options=sorted(df["employee_id"].dropna().unique().tolist()))
        channel_filter = st.multiselect("Channel", options=sorted(df["historical_channel"].dropna().unique().tolist()))

    filtered = df.copy()
    if product_filter:
        filtered = filtered[filtered["historical_recommended_product"].isin(product_filter)]
    if decision_filter:
        filtered = filtered[filtered["historical_decision"].isin(decision_filter)]
    if band_filter:
        filtered = filtered[filtered["historical_priority_band"].isin(band_filter)]
    if employment_filter:
        filtered = filtered[filtered["employment_type"].isin(employment_filter)]
    if existing_filter != "All":
        filtered = filtered[filtered["existing_customer_flag"] == int(existing_filter == "Yes")]
    if branch_filter:
        filtered = filtered[filtered["branch_id"].isin(branch_filter)]
    if employee_filter:
        filtered = filtered[filtered["employee_id"].isin(employee_filter)]
    if channel_filter:
        filtered = filtered[filtered["historical_channel"].isin(channel_filter)]

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_cross_sell_model_health(filtered)

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_cross_sell_command_center(filtered)
    render_cross_sell_task_queues(filtered, key_prefix="xs_portfolio")

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_panel_intro(
        "Summary",
        "Portfolio Snapshot",
        "These headline metrics summarize the filtered opportunity set and help leadership quickly gauge scale, targeting intensity, and activation readiness.",
    )
    render_metric_cards(filtered)

    donut_df = (
        filtered.groupby("historical_priority_band", as_index=False)
        .size()
        .rename(columns={"size": "count", "historical_priority_band": "Priority Band"})
    )
    donut_df["Priority Band"] = pd.Categorical(donut_df["Priority Band"], categories=PRIORITY_BAND_ORDER, ordered=True)
    donut_df = donut_df.sort_values("Priority Band")
    hist_df = filtered[["historical_priority_score"]].dropna().rename(columns={"historical_priority_score": "Priority Score"})
    decision_df = filtered.groupby("historical_decision", as_index=False).size().rename(columns={"size": "count", "historical_decision": "Decision"})
    decision_df["Decision"] = pd.Categorical(decision_df["Decision"], categories=DECISION_ORDER, ordered=True)
    decision_df = decision_df.sort_values("Decision")
    product_mix_df = filtered.groupby("historical_recommended_product", as_index=False).size().rename(columns={"historical_recommended_product": "Product", "size": "Opportunities"})
    branch_df = filtered.groupby("branch_id", as_index=False).size().rename(columns={"size": "Applications", "branch_id": "Branch"})
    employee_df = filtered.groupby("employee_id", as_index=False).size().rename(columns={"size": "Applications", "employee_id": "Employee"})

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_panel_intro(
        "Distribution",
        "Portfolio Composition",
        "Review how the filtered book spreads across priority bands, score ranges, actions, and product recommendations.",
    )
    r1c1, r1c2 = st.columns(2, gap="large")
    with r1c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Priority Band Mix")
        st.caption("Shows how the portfolio is distributed across commercial priority tiers.")
        donut = alt.Chart(donut_df).mark_arc(innerRadius=62, outerRadius=106).encode(
            theta=alt.Theta("count:Q"),
            color=alt.Color("Priority Band:N", scale=alt.Scale(domain=PRIORITY_BAND_ORDER, range=PRIORITY_BAND_COLORS), legend=alt.Legend(title="Priority Band")),
            tooltip=["Priority Band", "count"],
        )
        st.altair_chart(styled_chart(donut, height=280), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r1c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Priority Score Distribution")
        st.caption("Histogram of scores in the filtered portfolio. Right-skew generally means a stronger overall opportunity set.")
        hist = alt.Chart(hist_df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Priority Score:Q", bin=alt.Bin(maxbins=18), title="Priority Score"),
            y=alt.Y("count():Q", title="Applications"),
            color=alt.value("#b07a3f"),
            tooltip=[alt.Tooltip("count():Q", title="Applications")],
        )
        st.altair_chart(styled_chart(hist, height=280), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    r2c1, r2c2 = st.columns(2, gap="large")
    with r2c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Decision Mix")
        st.caption("Operational actions recommended by the engine for the filtered set.")
        dec_chart = alt.Chart(decision_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("count:Q", title="Applications"),
            y=alt.Y("Decision:N", sort=DECISION_ORDER, title=None),
            color=alt.Color("Decision:N", scale=alt.Scale(domain=DECISION_ORDER, range=DECISION_COLORS), legend=None),
            tooltip=["Decision", "count"],
        )
        st.altair_chart(styled_chart(dec_chart, height=280), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r2c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Product Targeting Mix")
        st.caption("Recommended product distribution within the filtered portfolio.")
        product_chart = alt.Chart(product_mix_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Opportunities:Q", title="Opportunities"),
            y=alt.Y("Product:N", sort="-x", title=None),
            color=alt.value("#8d5f2d"),
            tooltip=["Product", alt.Tooltip("Opportunities:Q", format=",")],
        )
        st.altair_chart(styled_chart(product_chart, height=280), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_panel_intro(
        "Coverage",
        "Team Activity",
        "Compare how opportunity flow is distributed across branches and employees in the current filtered view.",
    )
    r3c1, r3c2 = st.columns(2, gap="large")
    with r3c1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Branch Workload")
        st.caption("Application count by branch for the currently selected slice.")
        branch_chart = alt.Chart(branch_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Applications:Q", title="Applications"),
            y=alt.Y("Branch:N", sort="-x", title=None),
            color=alt.value("#2e6896"),
            tooltip=["Branch", "Applications"],
        )
        st.altair_chart(styled_chart(branch_chart, height=260), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
    with r3c2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("#### Employee Workload")
        st.caption("Application count by employee across the filtered portfolio.")
        employee_chart = alt.Chart(employee_df).mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5).encode(
            x=alt.X("Employee:N", sort="-y", title=None),
            y=alt.Y("Applications:Q", title="Applications"),
            color=alt.value("#c59a49"),
            tooltip=["Employee", "Applications"],
        )
        st.altair_chart(styled_chart(employee_chart, height=260), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_panel_intro(
        "Reporting",
        "Performance Tables",
        "Use these reporting tables for branch, employee, and record-level drilldown after you narrow the portfolio with filters.",
    )
    st.markdown("### Branch Performance")
    branch_perf = (
        filtered.groupby("branch_id", as_index=False)
        .agg(
            applications=("opportunity_id", "count"),
            avg_priority_score=("historical_priority_score", "mean"),
            avg_propensity=("historical_propensity_probability", "mean"),
            conversion_rate=("converted_flag", "mean"),
        )
        .rename(columns={"branch_id": "Branch ID"})
    )
    branch_perf_display = branch_perf.copy()
    branch_perf_display["avg_priority_score"] = branch_perf_display["avg_priority_score"].map(lambda x: f"{x:.1f}")
    branch_perf_display["avg_propensity"] = branch_perf_display["avg_propensity"].map(fmt_pct)
    branch_perf_display["conversion_rate"] = branch_perf_display["conversion_rate"].map(fmt_pct)
    render_table(
        branch_perf_display,
        max_rows_visible=6,
    )

    st.markdown("### Employee Performance")
    emp_perf = (
        filtered.groupby(["employee_id", "branch_id"], as_index=False)
        .agg(
            opportunities=("opportunity_id", "count"),
            rm_priority_rate=("historical_decision", lambda x: (x == "RM Priority Lead").mean()),
            avg_propensity=("historical_propensity_probability", "mean"),
            conversion_rate=("converted_flag", "mean"),
        )
        .rename(columns={"employee_id": "Employee ID", "branch_id": "Branch ID"})
    )
    emp_perf_display = emp_perf.copy()
    emp_perf_display["rm_priority_rate"] = emp_perf_display["rm_priority_rate"].map(fmt_pct)
    emp_perf_display["avg_propensity"] = emp_perf_display["avg_propensity"].map(fmt_pct)
    emp_perf_display["conversion_rate"] = emp_perf_display["conversion_rate"].map(fmt_pct)
    render_table(
        emp_perf_display,
        max_rows_visible=8,
    )

    st.markdown("### Curated Portfolio Table")
    curated_cols = [
        "opportunity_id",
        "customer_id",
        "branch_id",
        "employee_id",
        "segment",
        "employment_type",
        "historical_recommended_product",
        "historical_decision",
        "historical_priority_band",
        "historical_priority_score",
        "historical_propensity_probability",
        "historical_channel",
        "converted_flag",
    ]
    portfolio_view = filtered[curated_cols].sort_values("historical_priority_score", ascending=False)
    portfolio_display = portfolio_view.copy()
    portfolio_display["historical_priority_score"] = portfolio_display["historical_priority_score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    portfolio_display["historical_propensity_probability"] = portfolio_display["historical_propensity_probability"].map(fmt_pct)
    portfolio_display["converted_flag"] = portfolio_display["converted_flag"].map(
        lambda x: "Yes" if pd.notna(x) and int(x) == 1 else "No / Pending"
    )
    render_table(
        portfolio_display,
        max_rows_visible=10,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    df = get_data(str(DATA_FILE))
    render_header(df)

    with st.sidebar:
        render_sidebar_intro(
            "Cross-Sell Decision",
            "A premium decisioning workspace for lead prioritization, product targeting, and portfolio walkthroughs.",
        )
        render_sidebar_section(
            "Decision Settings",
            "Choose which recommendation engine should be used throughout the workspace.",
        )
        selected_engine_name = st.radio(
            "Engine",
            options=["Rule-Based Recommendation Engine", "Machine Learning Recommendation Engine"],
            index=0,
        )
        st.caption("The selected engine is used in New Assessment, Portfolio Overview, and Decision Review context.")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.caption("Use governed customer data with consent, contactability, and suitability controls before activating outreach.")

    render_workflow_guide()
    engines = get_engines(str(DATA_FILE))

    tab1, tab2, tab3, tab4 = st.tabs(["Suite Command", "New Opportunity", "Portfolio Overview", "Decision Review"])
    with tab1:
        render_cross_sell_suite_command(df)
    with tab2:
        render_new_opportunity_tab(df, engines, selected_engine_name)
    with tab3:
        render_portfolio_tab(df)
    with tab4:
        render_case_review_tab(df, engines, selected_engine_name)


if __name__ == "__main__":
    main()
