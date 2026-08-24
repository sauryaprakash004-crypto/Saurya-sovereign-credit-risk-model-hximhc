"""
End-to-end pipeline: generate data -> fit model -> validate -> scenario
analysis -> persist metrics/figures used by the README and the dashboard.

Usage:
    python -m scripts.run_pipeline
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data_generation import FEATURE_COLUMNS, generate_panel
from src.model import TARGET_COLUMN, fit_model, coefficient_table, predict_proba
from src.scenario_analysis import run_scenario_for_all
from src.validation import (
    best_threshold,
    calibration_points,
    calibration_slope_intercept,
    classification_metrics,
    cross_validated_auc,
    roc_points,
)

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
DATA_PATH = "data/processed/sovereign_panel.csv"


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    print("[1/5] Generating synthetic sovereign macro-fiscal panel ...")
    panel = generate_panel()
    panel.to_csv(DATA_PATH, index=False)
    print(f"    {len(panel):,} country-year obs | {panel['country_id'].nunique()} countries | "
          f"{panel['year'].min()}-{panel['year'].max()} | event rate {panel[TARGET_COLUMN].mean():.2%}")

    print("[2/5] Fitting logistic regression (chronological train/test split) ...")
    model = fit_model(panel)
    print(f"    train n={len(model.train_df):,} (<=2019)  test n={len(model.test_df):,} (2020-2023, incl. COVID shock)")

    coef_table = coefficient_table(model)
    coef_table.to_csv(os.path.join(RESULTS_DIR, "coefficient_table.csv"), index=False)

    print("[3/5] Out-of-sample validation ...")
    y_test = model.test_df[TARGET_COLUMN].values
    p_test = predict_proba(model, model.test_df)
    y_train = model.train_df[TARGET_COLUMN].values
    p_train = predict_proba(model, model.train_df)

    threshold = best_threshold(y_train, p_train)  # tuned on train only, applied out-of-sample
    test_metrics = classification_metrics(y_test, p_test, threshold=threshold)
    train_metrics = classification_metrics(y_train, p_train, threshold=threshold)
    intercept_cal, slope_cal = calibration_slope_intercept(y_test, p_test)

    X_all = panel[FEATURE_COLUMNS].values
    y_all = panel[TARGET_COLUMN].values
    cv_result = cross_validated_auc(model.pipeline, X_all, y_all)

    print(f"    Decision threshold (Youden's J, tuned on train)={threshold:.3f}")
    print(f"    Test ROC-AUC={test_metrics['roc_auc']:.3f}  Brier={test_metrics['brier_score']:.3f}  "
          f"Recall={test_metrics['recall']:.2f}  Precision={test_metrics['precision']:.2f}")
    print(f"    5-fold CV ROC-AUC = {cv_result['mean_auc']:.3f} +/- {cv_result['std_auc']:.3f}")
    print(f"    Calibration slope={slope_cal:.2f} intercept={intercept_cal:.2f} (ideal: slope=1, intercept=0)")

    # --- Figures -----------------------------------------------------
    print("[4/5] Generating figures ...")
    fpr, tpr, _ = roc_points(y_test, p_test)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"Test ROC (AUC={test_metrics['roc_auc']:.3f})", color="#2b6cb0")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Out-of-Sample ROC Curve (2020-2023 holdout)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curve.png"), dpi=140)
    plt.close()

    mean_pred, frac_pos = calibration_points(y_test, p_test)
    plt.figure(figsize=(5, 5))
    plt.plot(mean_pred, frac_pos, "o-", color="#2b6cb0", label="Model")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration (Reliability) Curve - Test Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "calibration_curve.png"), dpi=140)
    plt.close()

    coef_plot = coef_table[coef_table["feature"] != "const"].copy()
    coef_plot = coef_plot.sort_values("coef")
    plt.figure(figsize=(7, 5))
    colors = ["#c53030" if c > 0 else "#2f855a" for c in coef_plot["coef"]]
    plt.barh(coef_plot["feature"], coef_plot["coef"], color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Standardized Logistic Regression Coefficient (log-odds)")
    plt.title("Sovereign Stress Risk Drivers")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"), dpi=140)
    plt.close()

    print("[5/5] Running baseline vs. adverse scenario analysis ...")
    scenario_df = run_scenario_for_all(model, panel)
    scenario_df.to_csv(os.path.join(RESULTS_DIR, "scenario_analysis.csv"), index=False)

    scenario_summary = {
        "avg_baseline_debt_gdp_5y": float(scenario_df["baseline_debt_gdp_5y"].mean()),
        "avg_adverse_debt_gdp_5y": float(scenario_df["adverse_debt_gdp_5y"].mean()),
        "avg_debt_gdp_delta_pp": float(scenario_df["debt_gdp_delta_pp"].mean()),
        "avg_baseline_stress_prob_5y": float(scenario_df["baseline_stress_prob_5y"].mean()),
        "avg_adverse_stress_prob_5y": float(scenario_df["adverse_stress_prob_5y"].mean()),
        "avg_stress_prob_delta": float(scenario_df["stress_prob_delta"].mean()),
        "pct_countries_high_risk_adverse": float((scenario_df["adverse_stress_prob_5y"] >= 0.35).mean()),
        "pct_countries_high_risk_baseline": float((scenario_df["baseline_stress_prob_5y"] >= 0.35).mean()),
    }

    plt.figure(figsize=(6, 4))
    plt.hist(scenario_df["baseline_stress_prob_5y"], bins=20, alpha=0.6, label="Baseline", color="#2b6cb0")
    plt.hist(scenario_df["adverse_stress_prob_5y"], bins=20, alpha=0.6, label="Adverse", color="#c53030")
    plt.xlabel("5-Year-Ahead Stress Probability")
    plt.ylabel("Number of Countries")
    plt.title("Baseline vs. Adverse Scenario: Stress Probability Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "scenario_distribution.png"), dpi=140)
    plt.close()

    metrics = {
        "dataset": {
            "n_observations": int(len(panel)),
            "n_countries": int(panel["country_id"].nunique()),
            "years": f"{panel['year'].min()}-{panel['year'].max()}",
            "event_rate": float(panel[TARGET_COLUMN].mean()),
            "train_n": int(len(model.train_df)),
            "test_n": int(len(model.test_df)),
            "decision_threshold": threshold,
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cross_validation": cv_result,
        "calibration": {"slope": slope_cal, "intercept": intercept_cal},
        "scenario_analysis": scenario_summary,
    }
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone. Metrics written to results/metrics.json")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
