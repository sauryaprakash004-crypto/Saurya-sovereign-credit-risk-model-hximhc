"""Sanity tests for the sovereign risk modelling pipeline."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.data_generation import FEATURE_COLUMNS, generate_panel
from src.model import TARGET_COLUMN, fit_model, predict_proba
from src.scenario_analysis import run_scenario
from src.validation import classification_metrics, cross_validated_auc


def test_panel_shape_and_balance():
    panel = generate_panel()
    assert set(FEATURE_COLUMNS).issubset(panel.columns)
    assert panel[TARGET_COLUMN].isin([0, 1]).all()
    # event rate should be a "rare event" but not degenerate
    rate = panel[TARGET_COLUMN].mean()
    assert 0.05 < rate < 0.35


def test_model_fits_and_predicts_valid_probabilities():
    panel = generate_panel()
    model = fit_model(panel)
    probs = predict_proba(model, model.test_df)
    assert ((probs >= 0) & (probs <= 1)).all()
    assert len(probs) == len(model.test_df)


def test_out_of_sample_auc_beats_random():
    panel = generate_panel()
    model = fit_model(panel)
    y_test = model.test_df[TARGET_COLUMN].values
    p_test = predict_proba(model, model.test_df)
    metrics = classification_metrics(y_test, p_test)
    assert metrics["roc_auc"] > 0.65  # meaningfully better than a coin flip


def test_cross_validated_auc_is_stable():
    panel = generate_panel()
    model = fit_model(panel)
    X = panel[FEATURE_COLUMNS].values
    y = panel[TARGET_COLUMN].values
    result = cross_validated_auc(model.pipeline, X, y)
    assert result["std_auc"] < 0.05


def test_adverse_scenario_raises_debt_and_stress_probability():
    panel = generate_panel()
    model = fit_model(panel)
    country = panel["country_id"].iloc[0]
    result = run_scenario(model, panel, country)
    assert result.adverse_debt_path[-1] >= result.baseline_debt_path[-1]
    assert result.adverse_prob_path[-1] >= result.baseline_prob_path[-1]
