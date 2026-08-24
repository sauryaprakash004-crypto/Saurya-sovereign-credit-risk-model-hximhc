"""
Synthetic sovereign macro-fiscal panel generator.

Real IMF/World Bank cross-country panels (WEO, GFSR, IMF DSA templates) are not
bundled with this repo, so this module builds a *statistically realistic*
synthetic panel of country-years whose data-generating process is explicitly
modelled on the risk factors used in IMF Debt Sustainability Analyses (DSAs)
and the sovereign-stress early-warning literature (Reinhart & Rogoff 2011;
IMF 2013 "Staff Guidance Note on the Assessment of Reserve Adequacy and
Related Considerations"; Gerling et al. 2017 "Fiscal Crises").

Because the *true* generating probability is known, the resulting dataset is
also a useful benchmark for validating the modelling pipeline: a well-fit
logistic model should recover the sign and rough magnitude of every
coefficient below.

Country-level random effects + AR(1)-style year-on-year persistence are added
so the panel behaves like real macro data (autocorrelated, heterogeneous
across countries) rather than i.i.d. noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG_SEED = 42

N_COUNTRIES = 70
START_YEAR = 2005
END_YEAR = 2023  # inclusive
YEARS = list(range(START_YEAR, END_YEAR + 1))

FEATURE_COLUMNS = [
    "gov_debt_gdp",
    "external_debt_gdp",
    "fiscal_balance_gdp",
    "current_account_gdp",
    "reserves_months_imports",
    "real_gdp_growth",
    "inflation",
    "real_rate_growth_diff",
    "short_term_debt_reserves",
    "political_stability",
]

# "True" data-generating coefficients (log-odds per 1 unit of the raw
# feature). Signs follow the standard sovereign-risk literature.
TRUE_COEFS = {
    "intercept": -3.55,
    "gov_debt_gdp": 0.028,             # higher public debt/GDP -> more risk
    "external_debt_gdp": 0.018,        # higher external debt/GDP -> more risk
    "fiscal_balance_gdp": -0.11,       # more negative fiscal balance -> more risk
    "current_account_gdp": -0.09,      # more negative CA balance -> more risk
    "reserves_months_imports": -0.30,  # more reserve cover -> less risk
    "real_gdp_growth": -0.11,          # higher growth -> less risk
    "inflation": 0.045,                # higher inflation -> more risk
    "real_rate_growth_diff": 0.26,     # r - g > 0 -> adverse debt dynamics -> more risk
    "short_term_debt_reserves": 0.55,  # rollover risk -> more risk
    "political_stability": -0.38,      # more stable -> less risk
}


def _ar1_panel(rng, n_countries, n_years, mean, std, persistence, country_sd):
    """Country x year panel with a country-specific mean and AR(1) dynamics."""
    country_mean = rng.normal(mean, country_sd, size=n_countries)
    panel = np.zeros((n_countries, n_years))
    panel[:, 0] = rng.normal(country_mean, std)
    for t in range(1, n_years):
        shock = rng.normal(0, std * np.sqrt(1 - persistence**2), size=n_countries)
        panel[:, t] = country_mean + persistence * (panel[:, t - 1] - country_mean) + shock
    return panel


def generate_panel(seed: int = RNG_SEED) -> pd.DataFrame:
    """Generate the synthetic country-year panel with a known stress label DGP."""
    rng = np.random.default_rng(seed)
    n_c, n_y = N_COUNTRIES, len(YEARS)

    gov_debt_gdp = np.clip(_ar1_panel(rng, n_c, n_y, 55, 18, 0.90, 22), 5, 220)
    external_debt_gdp = np.clip(
        gov_debt_gdp * rng.normal(0.55, 0.15, size=(n_c, 1)) + _ar1_panel(rng, n_c, n_y, 5, 12, 0.85, 10),
        0, 250,
    )
    fiscal_balance_gdp = np.clip(_ar1_panel(rng, n_c, n_y, -2.5, 3.0, 0.75, 2.0), -18, 8)
    current_account_gdp = np.clip(_ar1_panel(rng, n_c, n_y, -1.5, 4.0, 0.80, 3.0), -18, 14)
    reserves_months_imports = np.clip(_ar1_panel(rng, n_c, n_y, 4.5, 2.2, 0.85, 1.6), 0.2, 18)
    real_gdp_growth = np.clip(_ar1_panel(rng, n_c, n_y, 3.2, 3.0, 0.55, 1.3), -12, 14)
    inflation = np.clip(_ar1_panel(rng, n_c, n_y, 5.0, 5.5, 0.70, 3.5), -2, 60)
    real_rate_growth_diff = np.clip(_ar1_panel(rng, n_c, n_y, 0.0, 2.5, 0.65, 1.5), -12, 15)
    short_term_debt_reserves = np.clip(_ar1_panel(rng, n_c, n_y, 0.45, 0.25, 0.80, 0.15), 0.02, 3.0)
    political_stability = np.clip(_ar1_panel(rng, n_c, n_y, 0.0, 0.9, 0.90, 0.6), -2.5, 2.5)

    # COVID-19 shock (2020): growth collapse + fiscal deterioration + debt jump,
    # applied to all countries with heterogeneous intensity - mirrors the 2020
    # global recession and gives the panel a genuine "adverse regime" year.
    covid_year_idx = YEARS.index(2020) if 2020 in YEARS else None
    if covid_year_idx is not None:
        intensity = rng.normal(1.0, 0.35, size=n_c)
        real_gdp_growth[:, covid_year_idx] -= np.clip(6.0 * intensity, 0, 14)
        fiscal_balance_gdp[:, covid_year_idx] -= np.clip(4.0 * intensity, 0, 10)
        gov_debt_gdp[:, covid_year_idx + 1:] += np.clip(10.0 * intensity, 0, 25)[:, None]

    country_ids = [f"C{idx:03d}" for idx in range(n_c)]
    records = []
    country_effect = rng.normal(0, 0.55, size=n_c)  # unobserved-heterogeneity term

    logit_noise = rng.normal(0, 0.65, size=(n_c, n_y))  # idiosyncratic/omitted-variable noise

    for i, cid in enumerate(country_ids):
        for t, year in enumerate(YEARS):
            row = {
                "country_id": cid,
                "year": year,
                "gov_debt_gdp": gov_debt_gdp[i, t],
                "external_debt_gdp": external_debt_gdp[i, t],
                "fiscal_balance_gdp": fiscal_balance_gdp[i, t],
                "current_account_gdp": current_account_gdp[i, t],
                "reserves_months_imports": reserves_months_imports[i, t],
                "real_gdp_growth": real_gdp_growth[i, t],
                "inflation": inflation[i, t],
                "real_rate_growth_diff": real_rate_growth_diff[i, t],
                "short_term_debt_reserves": short_term_debt_reserves[i, t],
                "political_stability": political_stability[i, t],
            }
            logit = TRUE_COEFS["intercept"] + country_effect[i] + logit_noise[i, t]
            for feat in FEATURE_COLUMNS:
                logit += TRUE_COEFS[feat] * row[feat]
            prob = 1 / (1 + np.exp(-logit))
            row["true_prob"] = prob
            records.append(row)

    df = pd.DataFrame.from_records(records)

    # Label = financial-stress event realised in year t+1 (forward-looking
    # target, as in early-warning-system practice), sampled Bernoulli(true_prob).
    df = df.sort_values(["country_id", "year"]).reset_index(drop=True)
    outcome_rng = np.random.default_rng(seed + 1)
    df["stress_next_year"] = outcome_rng.binomial(1, df["true_prob"])
    df["financial_stress"] = df.groupby("country_id")["stress_next_year"].shift(-1)
    df = df.dropna(subset=["financial_stress"]).reset_index(drop=True)
    df["financial_stress"] = df["financial_stress"].astype(int)
    df = df.drop(columns=["stress_next_year"])

    return df


if __name__ == "__main__":
    panel = generate_panel()
    out_path = "data/processed/sovereign_panel.csv"
    panel.to_csv(out_path, index=False)
    print(f"Generated {len(panel):,} country-year observations -> {out_path}")
    print(f"Countries: {panel['country_id'].nunique()}, Years: {panel['year'].min()}-{panel['year'].max()}")
    print(f"Stress event rate: {panel['financial_stress'].mean():.2%}")
