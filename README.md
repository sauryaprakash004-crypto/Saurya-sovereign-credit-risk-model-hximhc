# Sovereign Credit Risk Modelling & Debt Sustainability Analysis

An IMF-inspired quantitative framework for assessing sovereign credit risk from
macroeconomic and debt-related indicators — a logistic-regression early-warning
model for sovereign financial stress, rigorous out-of-sample calibration/validation,
baseline vs. adverse debt-sustainability scenario analysis, and an interactive
Streamlit risk-monitoring dashboard.

> **Data note:** IMF/World Bank cross-country panels (WEO, GFSR) are not bundled
> with this repo, so `src/data_generation.py` builds a statistically realistic
> **synthetic** country-year panel whose data-generating process is explicitly
> modelled on the standard sovereign-risk / IMF DSA literature (debt ratios,
> external/fiscal/external balances, reserve cover, growth, inflation, the
> interest-growth differential, rollover risk, political stability), including a
> genuine 2020 COVID-19 macro shock. Because the true generating probabilities
> are known, the pipeline also validates that the fitted model recovers the
> expected sign and rough magnitude of each risk factor. Swap in real WEO/IMF
> DSA data by pointing `src/data_generation.py`'s output schema at your own
> loader — the model, validation and dashboard code is data-source agnostic.

## Methodology

1. **Risk factors** (10 indicators, standard IMF DSA / early-warning-system set):
   government debt/GDP, external debt/GDP, fiscal balance/GDP, current account
   balance/GDP, reserve cover (months of imports), real GDP growth, inflation,
   the real interest-rate–growth differential (r − g), short-term debt/reserves
   (rollover risk), and a political-stability index.
2. **Model**: `StandardScaler` + `LogisticRegression` (scikit-learn) for
   prediction and scenario scoring, cross-checked against a `statsmodels Logit`
   fit on the same standardized design matrix for coefficient significance,
   confidence intervals and odds ratios.
3. **Target**: forward-looking binary "financial stress next year" label.
4. **Validation**: chronological (not row-shuffled) train/test split — trained
   on 2005–2019, tested **out-of-sample on 2020–2022**, which includes the
   COVID-19 shock, plus 5-fold stratified cross-validation for stability.
   Metrics: ROC-AUC, Brier score, calibration (reliability) curve and
   logistic recalibration slope/intercept, precision/recall/F1 at a
   Youden's-J-tuned decision threshold.
5. **Debt Sustainability Analysis**: standard DSA debt-dynamics identity
   `d_t = d_{t-1}·(1+i−g−π)/(1+g+π) − pb + sfa` projected 5 years ahead under
   a **baseline** (recent macro trend) and an **adverse** scenario (growth
   −2pp, effective interest rate +150bp, primary balance −1pp of GDP, +3pp of
   GDP one-off FX/stock-flow shock), then re-scored through the fitted model
   to translate the debt path into a stress-probability path.
6. **Dashboard** (`dashboard/app.py`, Streamlit + Plotly): country risk
   monitor with driver-level probability decomposition, baseline-vs-adverse
   DSA projections, and a model-diagnostics tab (ROC/calibration curves,
   coefficient table).

## Results (this repo's synthetic benchmark run — see `results/metrics.json`)

| Metric | Value |
|---|---|
| Panel size | 1,260 country-year observations, 70 countries, 2005–2022 |
| Stress event rate | 17.0% |
| Train / test split | 1,050 obs (≤2019) / 210 obs (2020–2022, out-of-sample, incl. COVID shock) |
| **ROC-AUC (out-of-sample test)** | **0.717** |
| ROC-AUC (5-fold cross-validated) | 0.752 ± 0.014 |
| **Brier score (out-of-sample test)** | **0.148** |
| Recall / Precision (tuned threshold) | 0.87 / 0.26 (out-of-sample) |
| Calibration slope / intercept | 0.72 / −0.71 (1.0 / 0.0 = perfect) |

**Statistically significant risk drivers** (p < 0.05, logistic regression on
standardized features): government debt/GDP (+), external debt/GDP (+),
fiscal balance/GDP (−), current account balance/GDP (−), reserve cover (−),
real GDP growth (−), the r−g differential (+), political stability (−) —
signs match economic theory in 8/9 significant coefficients. Full table in
`results/coefficient_table.csv`.

**Baseline vs. adverse scenario analysis** (5-year-ahead, averaged across all
70 countries):

| | Baseline | Adverse | Δ |
|---|---|---|---|
| Debt / GDP | 52.5% | 65.7% | **+13.2 pp** |
| Modelled stress probability | 19.6% | 36.1% | **+16.6 pp (≈1.8×)** |
| Share of countries in "High risk" bucket (≥35% prob.) | 15.7% | 41.4% | **+25.7 pp** |

## Numbers you can use for a CV / resume bullet

- Built a logistic-regression sovereign-risk model on a 1,260-observation,
  70-country macro-fiscal panel, achieving **0.75 cross-validated ROC-AUC**
  (0.72 on a chronological, COVID-period out-of-sample holdout) and a
  **0.148 out-of-sample Brier score**.
- Validated 8 of 9 statistically significant risk-factor coefficients
  against economic-theory-consistent signs (p < 0.05) via a parallel
  `statsmodels` inference fit.
- Ran an IMF-style baseline-vs-adverse debt sustainability scenario across
  70 countries, showing an adverse macro shock (−2pp growth, +150bp rates,
  −1pp primary balance) raises average 5-year debt/GDP by **+13.2 pp** and
  **nearly doubles** average modelled sovereign-stress probability
  (19.6% → 36.1%), with high-risk countries rising from 16% to 41% of the
  sample.
- Shipped an interactive Streamlit/Plotly dashboard for country-level stress
  probability, risk-driver attribution, and debt-sustainability scenario
  comparison.

## Repository layout

```
src/
  data_generation.py   # synthetic macro-fiscal panel (documented DGP)
  model.py              # logistic regression pipeline + statsmodels inference
  validation.py          # ROC-AUC, Brier score, calibration, CV, thresholding
  scenario_analysis.py   # IMF DSA debt-dynamics baseline/adverse projections
scripts/
  run_pipeline.py         # end-to-end: generate -> fit -> validate -> scenario -> results/
dashboard/
  app.py                  # Streamlit risk-monitoring dashboard
results/
  metrics.json             # full metrics from the last pipeline run
  coefficient_table.csv    # statsmodels logistic regression inference table
  scenario_analysis.csv    # per-country baseline/adverse 5y projections
  figures/                 # ROC curve, calibration curve, driver chart, etc.
tests/
  test_pipeline.py          # sanity checks (data, model, validation, scenario)
```

## Running it

```bash
pip install -r requirements.txt

# Regenerate data, fit + validate the model, run scenario analysis
python -m scripts.run_pipeline

# Launch the interactive dashboard
streamlit run dashboard/app.py

# Run tests
pytest tests/ -q
```

## Disclaimer

This is a methodology/portfolio project. The panel is synthetic (calibrated
to be statistically realistic, not observed data), and results should not be
interpreted as real sovereign risk assessments or investment advice.
