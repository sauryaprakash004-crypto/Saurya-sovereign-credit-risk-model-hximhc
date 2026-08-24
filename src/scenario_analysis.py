"""
Debt sustainability analysis (DSA): baseline vs. adverse macroeconomic
scenario projection, following the standard IMF/World Bank DSA debt
dynamics identity:

    d_t = d_{t-1} * (1 + i_t - g_t - pi_t) / (1 + g_t + pi_t) - pb_t + sfa_t

where
    d   = public debt / GDP (%)
    i   = effective nominal interest rate on debt (%)
    g   = real GDP growth (%)
    pi  = inflation (GDP deflator, %) - included so the denominator uses
          nominal growth, as in standard DSA templates
    pb  = primary fiscal balance / GDP (%), positive = surplus
    sfa = stock-flow adjustment / GDP (%) (e.g. FX valuation effects)

Two 5-year-ahead scenarios are projected for a given country's latest
observed state:
  - Baseline: macro variables continue at their recent (5y trailing) average.
  - Adverse:  a combined growth, interest-rate and fiscal shock calibrated to
    the IMF's standard DSA adverse-scenario magnitudes (growth -2pp,
    effective interest rate +150bp, primary balance -1pp of GDP, one-off FX
    depreciation feeding into external-debt-driven stock-flow adjustment).

The projected macro paths are then fed back through the fitted logistic
model to translate the debt trajectory into a projected financial-stress
probability path - directly linking the DSA to the early-warning model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.model import TrainedModel, predict_proba

HORIZON_YEARS = 5

ADVERSE_SHOCK = {
    "growth_shock_pp": -2.0,        # real GDP growth, percentage points
    "interest_rate_shock_pp": 1.5,  # effective interest rate, percentage points
    "primary_balance_shock_pp": -1.0,  # fiscal_balance_gdp, percentage points
    "fx_sfa_shock_pp": 3.0,         # one-off stock-flow adjustment (%GDP) in year 1
    "inflation_shock_pp": 1.0,
}


@dataclass
class ScenarioResult:
    country_id: str
    baseline_debt_path: list = field(default_factory=list)
    adverse_debt_path: list = field(default_factory=list)
    baseline_prob_path: list = field(default_factory=list)
    adverse_prob_path: list = field(default_factory=list)
    years: list = field(default_factory=list)


def _effective_interest_rate(row: pd.Series) -> float:
    # r - g diff was generated directly; recover an implied nominal rate.
    return row["real_gdp_growth"] + row["real_rate_growth_diff"] + row["inflation"] * 0.3


def _project_debt_path(latest_row: pd.Series, horizon: int, shock: dict | None) -> pd.DataFrame:
    debt = latest_row["gov_debt_gdp"]
    growth = latest_row["real_gdp_growth"]
    inflation = latest_row["inflation"]
    interest = _effective_interest_rate(latest_row)
    primary_balance = latest_row["fiscal_balance_gdp"]  # proxy for primary balance
    ext_debt = latest_row["external_debt_gdp"]
    reserves = latest_row["reserves_months_imports"]
    st_debt_reserves = latest_row["short_term_debt_reserves"]
    ca_balance = latest_row["current_account_gdp"]
    pol_stability = latest_row["political_stability"]

    if shock is not None:
        growth += shock["growth_shock_pp"]
        interest += shock["interest_rate_shock_pp"]
        primary_balance += shock["primary_balance_shock_pp"]
        inflation += shock["inflation_shock_pp"]

    rows = []
    for h in range(1, horizon + 1):
        sfa = shock["fx_sfa_shock_pp"] if (shock is not None and h == 1) else 0.0
        debt = (debt * (1 + (interest - growth - inflation) / 100)) / (1 + (growth + inflation) / 100) \
            - primary_balance + sfa
        debt = max(debt, 0.0)
        # very mild mean-reversion for growth/inflation after the initial shock
        if shock is not None and h == 1:
            growth += 0.4  # partial recovery from year 2 onward
        rows.append({
            "h": h,
            "gov_debt_gdp": debt,
            "external_debt_gdp": max(ext_debt + (debt - latest_row["gov_debt_gdp"]) * 0.4, 0),
            "fiscal_balance_gdp": primary_balance,
            "current_account_gdp": ca_balance - (3.0 if (shock is not None and h == 1) else 0.0),
            "reserves_months_imports": max(reserves - (0.8 if shock is not None else 0.0), 0.1),
            "real_gdp_growth": growth,
            "inflation": inflation,
            "real_rate_growth_diff": interest - growth,
            "short_term_debt_reserves": st_debt_reserves * (1.15 if shock is not None else 1.0),
            "political_stability": pol_stability,
        })
    return pd.DataFrame(rows)


def run_scenario(model: TrainedModel, panel: pd.DataFrame, country_id: str,
                  horizon: int = HORIZON_YEARS) -> ScenarioResult:
    country_df = panel[panel["country_id"] == country_id].sort_values("year")
    latest_row = country_df.iloc[-1]
    latest_year = int(latest_row["year"])

    baseline_path = _project_debt_path(latest_row, horizon, shock=None)
    adverse_path = _project_debt_path(latest_row, horizon, shock=ADVERSE_SHOCK)

    baseline_probs = predict_proba(model, baseline_path)
    adverse_probs = predict_proba(model, adverse_path)

    return ScenarioResult(
        country_id=country_id,
        years=[latest_year + h for h in baseline_path["h"]],
        baseline_debt_path=baseline_path["gov_debt_gdp"].round(2).tolist(),
        adverse_debt_path=adverse_path["gov_debt_gdp"].round(2).tolist(),
        baseline_prob_path=[round(float(p), 4) for p in baseline_probs],
        adverse_prob_path=[round(float(p), 4) for p in adverse_probs],
    )


def run_scenario_for_all(model: TrainedModel, panel: pd.DataFrame, horizon: int = HORIZON_YEARS) -> pd.DataFrame:
    results = []
    for cid in panel["country_id"].unique():
        r = run_scenario(model, panel, cid, horizon)
        results.append({
            "country_id": cid,
            "baseline_debt_gdp_5y": r.baseline_debt_path[-1],
            "adverse_debt_gdp_5y": r.adverse_debt_path[-1],
            "baseline_stress_prob_5y": r.baseline_prob_path[-1],
            "adverse_stress_prob_5y": r.adverse_prob_path[-1],
        })
    df = pd.DataFrame(results)
    df["debt_gdp_delta_pp"] = df["adverse_debt_gdp_5y"] - df["baseline_debt_gdp_5y"]
    df["stress_prob_delta"] = df["adverse_stress_prob_5y"] - df["baseline_stress_prob_5y"]
    return df
