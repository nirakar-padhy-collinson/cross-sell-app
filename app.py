from __future__ import annotations

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
    generate_synthetic_dataset,
    load_or_create_data,
)

st.set_page_config(page_title="Cross-Sell Decision Studio", layout="wide", initial_sidebar_state="expanded")

DATA_FILE = Path("data/historical_cross_sell_opportunities.csv")


@st.cache_data(show_spinner=False)
def get_data(path_str: str) -> pd.DataFrame:
    return load_or_create_data(Path(path_str))


@st.cache_resource(show_spinner=False)
def get_ml_engine(path_str: str) -> MLRecommendationEngine:
    df = load_or_create_data(Path(path_str))
    engine = MLRecommendationEngine(model_dir="artifacts")
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
    avg_ev = df["historical_expected_value"].mean() if len(df) else 0.0
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
        + f"Observed conversion {conversion * 100:.1f}% • Avg expected value {fmt_currency(avg_ev)}"
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
                    <div class="workflow-step-title">Lead Review</div>
                    <p>Compare the historical record against the current engine view for a customer or opportunity.</p>
                </div>
                <div class="workflow-step">
                    <div class="workflow-step-number">03</div>
                    <div class="workflow-step-title">Portfolio Overview</div>
                    <p>Monitor targeting mix, expected value, and RM / branch level performance in real time.</p>
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
    avg_expected = df["historical_expected_value"].mean() if total else 0
    labeled = df[df["converted_flag"].notna()].copy()
    realized_conv = labeled["converted_flag"].mean() * 100 if len(labeled) else 0

    cards = [
        ("Total Opportunities", f"{total:,}", "Records in the working portfolio"),
        ("Targetable Rate", f"{targetable:.1f}%", "Priority + campaign + nurture actions"),
        ("RM Priority Rate", f"{rm_priority:.1f}%", "Top leads routed to relationship managers"),
        ("Suppress Rate", f"{suppress_rate:.1f}%", "Leads held out for fatigue / service reasons"),
        ("Average Expected Value", fmt_currency(avg_expected), "Per-lead expected commercial value"),
        ("Observed Conversion", f"{realized_conv:.1f}%", "Realized on labeled historical cases"),
    ]
    cols = st.columns(6, gap="medium")
    for col, (label, value, sub) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-sub">{sub}</div></div>',
                unsafe_allow_html=True,
            )


def render_output(output: RecommendationOutput):
    st.markdown('<div class="section-banner">Recommendation Summary</div>', unsafe_allow_html=True)
    st.caption("Interpretation guide: higher priority score, propensity, and value indicate a stronger commercial case; the decision and channel translate that signal into the recommended action.")
    c1, c2, c3, c4, c5, c6 = st.columns(6, gap="medium")
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
    with c6:
        render_stat_card("Expected Value", fmt_currency(output.expected_value), "Indicative value if activated")

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
        st.markdown("### Alternate Recommendations")
        alt_df = pd.DataFrame(output.alternate_recommendations)
        alt_df.columns = ["Product", "Probability", "Priority Score", "Decision", "Expected Value"]
        alt_df["Probability"] = alt_df["Probability"].map(fmt_pct)
        alt_df["Priority Score"] = alt_df["Priority Score"].map(lambda x: f"{float(x):.0f}")
        alt_df["Expected Value"] = alt_df["Expected Value"].map(fmt_currency)
        render_table(
            alt_df,
            max_rows_visible=4,
        )

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


def render_new_opportunity_tab(df: pd.DataFrame, engines: Dict[str, object], selected_engine_name: str):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Assessment Workspace",
        "New Opportunity Assessment",
        "Capture a customer profile, review the commercial context, and run a recommendation using the active engine. Every submitted assessment is appended to the portfolio history immediately.",
    )

    options = branch_employee_options()
    next_ids = allocate_new_ids(df)

    with st.form("new_opportunity_form", clear_on_submit=False):
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.markdown("#### IDs")
            opportunity_id = st.text_input("Opportunity ID", value=next_ids["opportunity_id"], help="Unique lead-level identifier for this assessment run.")
            customer_id = st.text_input("Customer ID", value=next_ids["customer_id"], help="Customer-level identifier used to track repeat reviews and historical comparisons.")
            customer_name = st.text_input("Customer Name", value="Customer Demo", help="Display name shown in the case record.")
            branch_id = st.selectbox("Branch ID", options=list(options.keys()), index=0, help="Originating branch for the lead or customer relationship.")
            employee_id = st.selectbox("Employee ID", options=options[branch_id], index=0, help="RM or employee who owns the lead in this demo record.")
            target_product = st.selectbox("Target Product", options=["Auto Select", *list(CAMPAIGN_CODES.keys())], index=0, help="Choose a product to force evaluation toward that campaign, or leave Auto Select to let the engine recommend the strongest fit.")
            campaign_id = st.text_input("Campaign ID", value=CAMPAIGN_CODES.get(target_product, "CP9000"), help="Campaign reference tied to the selected proposition or outreach motion.")

            st.markdown("#### Borrower / Customer Profile")
            age = st.slider("Age", min_value=21, max_value=75, value=36, help="Basic demographic input used as one of several suitability signals.")
            segment = st.selectbox("Segment", ["Mass", "Affluent", "HNI"], index=0, help="Commercial segment classification. Higher-value segments often support richer offers and RM-led servicing.")
            monthly_income = st.number_input("Monthly Income", min_value=20000.0, max_value=500000.0, value=90000.0, step=5000.0, help="Approximate monthly income used for affordability and commercial potential.")
            relationship_tenure_months = st.slider("Relationship Tenure (Months)", 1, 180, 30, help="Longer tenure usually signals a deeper, more stable relationship.")
            employment_type = st.selectbox("Employment Type", ["Salaried", "Self-Employed", "Professional", "Business Owner", "Contract"], index=0, help="Employment profile can affect risk, fit, and product relevance.")
            city_tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"], index=0, help="Location tier used as a light proxy for market context.")
            lifecycle_stage = st.selectbox("Lifecycle Stage", ["Emerging", "Growth", "Established", "Mature"], index=1, help="Business or customer maturity stage. Growth and established customers often surface stronger cross-sell opportunities.")
            channel_preference = st.selectbox("Channel Preference", ["Digital", "Hybrid", "RM / Branch"], index=0, help="Preferred outreach model for activation. This helps interpret the suggested channel.")

        with col2:
            st.markdown("#### Relationship / KYC / Holdings")
            existing_customer_flag = st.toggle("Existing Customer", value=True, help="Turn on if the customer already has a relationship with the bank.")
            salary_customer_flag = st.toggle("Salary Customer", value=True, help="Salary customers often show stronger engagement and targeting potential.")
            kyc_complete_flag = st.toggle("KYC Complete", value=True, help="Incomplete KYC may limit immediate campaign eligibility or product fulfilment.")
            has_credit_card_flag = st.toggle("Already Holds Credit Card", value=False, help="Use current holdings to avoid recommending what the customer already has.")
            has_personal_loan_flag = st.toggle("Already Holds Personal Loan", value=False, help="Existing product ownership affects fit and cross-sell headroom.")
            has_home_loan_flag = st.toggle("Already Holds Home Loan", value=False, help="Existing secured borrowing may shift the next-best-product recommendation.")
            has_insurance_flag = st.toggle("Already Holds Insurance", value=False, help="Existing protection cover can reduce the need for insurance-led outreach.")
            product_holding_count = st.slider("Product Holding Count", 1, 6, 2, help="How many products the customer already holds with the bank. Higher counts may imply stronger relationship depth.")

            st.markdown("#### Behavior / Engagement / Campaign History")
            avg_monthly_balance = st.number_input("Average Monthly Balance", min_value=1000.0, max_value=1000000.0, value=145000.0, step=5000.0, help="Average balance is a strong indicator of wallet depth and commercial value.")
            avg_debit_txn_count_3m = st.slider("Avg Debit Transactions (3M)", 1, 60, 18, help="Recent transaction frequency is used as an activity signal.")
            avg_credit_txn_count_3m = st.slider("Avg Credit Transactions (3M)", 1, 25, 8, help="Captures income or inward-flow regularity over the past 3 months.")
            digital_login_count_30d = st.slider("Digital Logins (30D)", 0, 40, 12, help="Higher digital activity can support digital or hybrid outreach decisions.")
            app_sessions_30d = st.slider("App Sessions (30D)", 0, 30, 10, help="Used as another engagement signal alongside login activity.")
            branch_visits_90d = st.slider("Branch Visits (90D)", 0, 10, 1, help="Higher branch usage may support RM or branch-led activation.")
            rm_interactions_90d = st.slider("RM Interactions (90D)", 0, 10, 2, help="Frequent RM interaction often supports higher-touch channels.")
            last_campaign_contact_days = st.slider("Days Since Last Campaign Contact", 0, 210, 45, help="Very recent contact can reduce readiness for another outbound push.")
            last_campaign_response = st.selectbox("Last Campaign Response", ["Accepted", "Clicked", "Opened", "No Response", "Declined"], index=3, help="Past response behavior helps interpret current propensity and channel fit.")
            prior_offer_accept_count_12m = st.slider("Prior Offer Accepts (12M)", 0, 5, 0, help="Past offer acceptance is usually a positive responsiveness signal.")
            recent_service_issue_flag = st.toggle("Recent Service Issue", value=False, help="Turn on if the customer recently had a service issue that could justify holdout or suppression.")
            bureau_score = st.slider("Bureau Score", 0, 900, 725, help="Credit quality signal. Higher values generally support stronger eligibility and lower risk.")

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
        )
        output = engines[selected_engine_name].evaluate(opportunity)
        append_assessment_to_history(DATA_FILE, opportunity, output)
        st.cache_data.clear()
        st.session_state["latest_output"] = output
        st.session_state["latest_opportunity_id"] = opportunity_id
        st.success(f"Assessment saved to history and scored using {selected_engine_name}.")
        render_output(output)
    elif st.session_state.get("latest_output"):
        render_output(st.session_state["latest_output"])


def render_case_review_tab(df: pd.DataFrame, engines: Dict[str, object], selected_engine_name: str):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Review Workspace",
        "Lead / Customer Review",
        "Search a historical opportunity or customer, then compare the stored portfolio record against the current engine view for a more executive-style side-by-side review.",
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
    current_output = engines[selected_engine_name].evaluate(opportunity)

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("### Historical Record")
        hist_summary = pd.DataFrame(
            [
                {"Metric": "Decision", "Value": record["historical_decision"]},
                {"Metric": "Recommended Product", "Value": record["historical_recommended_product"]},
                {"Metric": "Priority Score", "Value": record["historical_priority_score"]},
                {"Metric": "Propensity", "Value": fmt_pct(record["historical_propensity_probability"])},
                {"Metric": "Expected Value", "Value": fmt_currency(record["historical_expected_value"])},
                {"Metric": "Converted", "Value": "Yes" if pd.notna(record["converted_flag"]) and int(record["converted_flag"]) == 1 else "No / Pending"},
            ]
        )
        render_table(hist_summary, max_rows_visible=6)
    with right:
        st.markdown("### Current Engine View")
        current_summary = pd.DataFrame(
            [
                {"Metric": "Decision", "Value": current_output.decision},
                {"Metric": "Recommended Product", "Value": current_output.recommended_product},
                {"Metric": "Priority Score", "Value": f"{current_output.priority_score:.0f}"},
                {"Metric": "Propensity", "Value": fmt_pct(current_output.propensity_probability)},
                {"Metric": "Expected Value", "Value": fmt_currency(current_output.expected_value)},
                {"Metric": "Channel", "Value": current_output.recommended_channel},
            ]
        )
        render_table(current_summary, max_rows_visible=6)

    st.markdown("### Application Detail")
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

    st.markdown("### Current Rationale")
    render_output(current_output)
    st.markdown("</div>", unsafe_allow_html=True)


def render_portfolio_tab(df: pd.DataFrame):
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    render_section_intro(
        "Portfolio Workspace",
        "Portfolio Overview",
        "Filter the scoring portfolio and monitor targeting mix, expected value, and team performance in a more polished decisioning view.",
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
    render_panel_intro(
        "Summary",
        "Portfolio Snapshot",
        "These headline metrics summarize the filtered opportunity set and help leadership quickly gauge scale, targeting intensity, and commercial potential.",
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
    avg_value_df = filtered.groupby("historical_recommended_product", as_index=False)["historical_expected_value"].mean().rename(columns={"historical_recommended_product": "Product", "historical_expected_value": "Average Expected Value"})
    branch_df = filtered.groupby("branch_id", as_index=False).size().rename(columns={"size": "Applications", "branch_id": "Branch"})
    employee_df = filtered.groupby("employee_id", as_index=False).size().rename(columns={"size": "Applications", "employee_id": "Employee"})

    st.markdown('<div class="portfolio-spacer"></div>', unsafe_allow_html=True)
    render_panel_intro(
        "Distribution",
        "Portfolio Composition",
        "Review how the filtered book spreads across priority bands, score ranges, actions, and product-level value concentration.",
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
        st.markdown("#### Product Value Ranking")
        st.caption("Average expected value by recommended product within the filtered portfolio.")
        product_chart = alt.Chart(avg_value_df).mark_bar(cornerRadiusEnd=6).encode(
            x=alt.X("Average Expected Value:Q", title="Average Expected Value"),
            y=alt.Y("Product:N", sort="-x", title=None),
            color=alt.value("#8d5f2d"),
            tooltip=["Product", alt.Tooltip("Average Expected Value:Q", format=",.0f")],
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
            expected_value=("historical_expected_value", "mean"),
            conversion_rate=("converted_flag", "mean"),
        )
        .rename(columns={"branch_id": "Branch ID"})
    )
    branch_perf_display = branch_perf.copy()
    branch_perf_display["avg_priority_score"] = branch_perf_display["avg_priority_score"].map(lambda x: f"{x:.1f}")
    branch_perf_display["avg_propensity"] = branch_perf_display["avg_propensity"].map(fmt_pct)
    branch_perf_display["expected_value"] = branch_perf_display["expected_value"].map(fmt_currency)
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
            expected_value=("historical_expected_value", "mean"),
            conversion_rate=("converted_flag", "mean"),
        )
        .rename(columns={"employee_id": "Employee ID", "branch_id": "Branch ID"})
    )
    emp_perf_display = emp_perf.copy()
    emp_perf_display["rm_priority_rate"] = emp_perf_display["rm_priority_rate"].map(fmt_pct)
    emp_perf_display["avg_propensity"] = emp_perf_display["avg_propensity"].map(fmt_pct)
    emp_perf_display["expected_value"] = emp_perf_display["expected_value"].map(fmt_currency)
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
        "historical_expected_value",
        "converted_flag",
    ]
    portfolio_view = filtered[curated_cols].sort_values("historical_priority_score", ascending=False)
    portfolio_display = portfolio_view.copy()
    portfolio_display["historical_priority_score"] = portfolio_display["historical_priority_score"].map(lambda x: f"{x:.0f}" if pd.notna(x) else "—")
    portfolio_display["historical_propensity_probability"] = portfolio_display["historical_propensity_probability"].map(fmt_pct)
    portfolio_display["historical_expected_value"] = portfolio_display["historical_expected_value"].map(fmt_currency)
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
            options=["Rule-Based Engine", "Machine Learning Recommendation Engine"],
            index=0,
        )
        st.caption("The selected engine is used in New Assessment and Lead Review.")

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        render_sidebar_section(
            "Portfolio Data",
            "Refresh or rebuild the sample portfolio used across the dashboard and reporting views.",
        )

        if st.button("Regenerate Synthetic Portfolio", use_container_width=True):
            generate_synthetic_dataset(DATA_FILE, n_rows=900, seed=42)
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("latest_output", None)
            st.rerun()

        if st.button("Refresh From CSV", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        st.caption("Synthetic demo only. Do not use raw PII or protected attributes in production targeting without governance and consent controls.")

    render_workflow_guide()
    engines = get_engines(str(DATA_FILE))

    tab1, tab2, tab3 = st.tabs(["New Opportunity", "Lead Review", "Portfolio Overview"])
    with tab1:
        render_new_opportunity_tab(df, engines, selected_engine_name)
    with tab2:
        render_case_review_tab(df, engines, selected_engine_name)
    with tab3:
        render_portfolio_tab(df)


if __name__ == "__main__":
    main()
