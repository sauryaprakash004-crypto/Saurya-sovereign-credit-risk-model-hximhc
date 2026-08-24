"""
Sovereign Credit Risk & Debt Sustainability Dashboard (Streamlit).

Run with:
    streamlit run dashboard/app.py

Tabs:
  1. Country Risk Monitor - current stress probability, risk bucket, top
     risk drivers for the selected country/year.
  2. Debt Sustainability Analysis - baseline vs. adverse 5-year debt and
     stress-probability projections for the selected country.
  3. Model Diagnostics - ROC curve, calibration curve, coefficient table,
     cross-validated AUC, for the underlying logistic regression.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_generation import FEATURE_COLUMNS, generate_panel
from src.model import TARGET_COLUMN, coefficient_table, fit_model, predict_proba
from src.scenario_analysis import run_scenario
from src.validation import (
    calibration_points,
    classification_metrics,
    roc_points,
    risk_bucket,
)

st.set_page_config(page_title="Sovereign Credit Risk Dashboard", layout="wide")


@st.cache_data(show_spinner="Generating sovereign macro-fiscal panel ...")
def load_panel():
    return generate_panel()


@st.cache_resource(show_spinner="Fitting logistic regression model ...")
def load_model(panel: pd.DataFrame):
    return fit_model(panel)


panel = load_panel()
model = load_model(panel)
panel = panel.copy()
panel["stress_prob"] = predict_proba(model, panel)
panel["risk_bucket"] = panel["stress_prob"].apply(risk_bucket)

st.title("🏛️ Sovereign Credit Risk & Debt Sustainability Dashboard")
st.caption(
    "IMF-inspired quantitative framework: logistic-regression early-warning model for sovereign "
    "financial stress, with baseline/adverse debt-sustainability scenario analysis. "
    "Data is a statistically-calibrated synthetic panel (70 countries, 2005-2022) — see README for methodology."
)

tab1, tab2, tab3 = st.tabs(["📍 Country Risk Monitor", "📉 Debt Sustainability Analysis", "🧪 Model Diagnostics"])

# ------------------------------------------------------------------ TAB 1
with tab1:
    col_a, col_b = st.columns([1, 3])
    with col_a:
        country = st.selectbox("Country", sorted(panel["country_id"].unique()), key="country_tab1")
        year = st.slider("Year", int(panel["year"].min()), int(panel["year"].max()), int(panel["year"].max()))

    row = panel[(panel["country_id"] == country) & (panel["year"] == year)]
    if row.empty:
        st.warning("No data for this country/year.")
    else:
        row = row.iloc[0]
        with col_b:
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Stress Probability (next year)", f"{row['stress_prob']:.1%}")
            k2.metric("Risk Bucket", row["risk_bucket"])
            k3.metric("Gov. Debt / GDP", f"{row['gov_debt_gdp']:.1f}%")
            k4.metric("Reserves (months of imports)", f"{row['reserves_months_imports']:.1f}")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Risk driver contribution (this observation)")
            scaler = model.scaler
            x_std = scaler.transform(row[FEATURE_COLUMNS].values.reshape(1, -1))[0]
            coefs = model.pipeline.named_steps["logreg"].coef_[0]
            contrib = pd.DataFrame({
                "feature": FEATURE_COLUMNS,
                "contribution_log_odds": x_std * coefs,
            }).sort_values("contribution_log_odds")
            fig = px.bar(
                contrib, x="contribution_log_odds", y="feature", orientation="h",
                color="contribution_log_odds", color_continuous_scale="RdYlGn_r",
                labels={"contribution_log_odds": "Contribution to log-odds of stress"},
            )
            fig.update_layout(coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.subheader(f"{country} — macro-fiscal trajectory")
            hist = panel[panel["country_id"] == country].sort_values("year")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=hist["year"], y=hist["gov_debt_gdp"], name="Gov. Debt/GDP (%)", yaxis="y1"))
            fig2.add_trace(go.Scatter(x=hist["year"], y=hist["stress_prob"] * 100, name="Stress Prob. (%)", yaxis="y2"))
            fig2.update_layout(
                yaxis=dict(title="Gov. Debt/GDP (%)"),
                yaxis2=dict(title="Stress Probability (%)", overlaying="y", side="right"),
                height=420, legend=dict(orientation="h", y=1.1),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Full indicator panel")
        st.dataframe(
            hist[["year"] + FEATURE_COLUMNS + ["stress_prob", "risk_bucket", TARGET_COLUMN]]
            .rename(columns={TARGET_COLUMN: "stress_next_year_actual"})
            .set_index("year"),
            use_container_width=True,
        )

# ------------------------------------------------------------------ TAB 2
with tab2:
    st.subheader("Baseline vs. Adverse Scenario — 5-Year Debt Sustainability Projection")
    country2 = st.selectbox("Country", sorted(panel["country_id"].unique()), key="country_tab2")
    result = run_scenario(model, panel, country2)

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.years, y=result.baseline_debt_path, name="Baseline", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=result.years, y=result.adverse_debt_path, name="Adverse", mode="lines+markers"))
        fig.update_layout(title="Government Debt / GDP (%)", height=420, yaxis_title="% of GDP")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=result.years, y=[p * 100 for p in result.baseline_prob_path], name="Baseline", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=result.years, y=[p * 100 for p in result.adverse_prob_path], name="Adverse", mode="lines+markers"))
        fig.update_layout(title="Financial Stress Probability (%)", height=420, yaxis_title="Probability (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"**Adverse scenario shocks applied:** real GDP growth −2.0pp · effective interest rate +1.5pp · "
        f"primary balance −1.0pp of GDP · one-off FX/stock-flow shock +3.0pp of GDP in year 1.\n\n"
        f"Under the adverse scenario, {country2}'s projected debt/GDP in year {result.years[-1]} is "
        f"**{result.adverse_debt_path[-1]:.1f}%** vs. **{result.baseline_debt_path[-1]:.1f}%** at baseline "
        f"(+{result.adverse_debt_path[-1] - result.baseline_debt_path[-1]:.1f}pp), and the modelled stress "
        f"probability rises from **{result.baseline_prob_path[-1]:.1%}** to **{result.adverse_prob_path[-1]:.1%}**."
    )

# ------------------------------------------------------------------ TAB 3
with tab3:
    st.subheader("Out-of-sample validation (train ≤2019, test 2020–2022, incl. COVID-19 shock)")
    y_test = model.test_df[TARGET_COLUMN].values
    p_test = predict_proba(model, model.test_df)
    metrics = classification_metrics(y_test, p_test)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC (test)", f"{metrics['roc_auc']:.3f}")
    m2.metric("Brier Score (test)", f"{metrics['brier_score']:.3f}")
    m3.metric("Recall (test)", f"{metrics['recall']:.2f}")
    m4.metric("Precision (test)", f"{metrics['precision']:.2f}")

    c1, c2 = st.columns(2)
    with c1:
        fpr, tpr, _ = roc_points(y_test, p_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={metrics['roc_auc']:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        mean_pred, frac_pos = calibration_points(y_test, p_test)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=mean_pred, y=frac_pos, name="Model", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Perfect calibration", line=dict(dash="dash")))
        fig.update_layout(title="Calibration Curve", xaxis_title="Mean Predicted Prob.", yaxis_title="Observed Frequency", height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Logistic regression coefficient table (statsmodels, standardized features)")
    st.dataframe(coefficient_table(model).round(4), use_container_width=True)
