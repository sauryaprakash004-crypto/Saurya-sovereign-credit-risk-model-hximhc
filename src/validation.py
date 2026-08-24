"""
Model calibration & validation utilities: out-of-sample discrimination
(ROC-AUC), probabilistic accuracy (Brier score), reliability/calibration
analysis, and k-fold cross-validation stability checks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "n_obs": int(len(y_true)),
        "n_events": int(np.sum(y_true)),
        "event_rate": float(np.mean(y_true)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def roc_points(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    return fpr, tpr, thresholds


def calibration_points(y_true, y_prob, n_bins: int = 8):
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def calibration_slope_intercept(y_true, y_prob):
    """Logistic recalibration of predicted probs against outcomes.

    Slope ~ 1 and intercept ~ 0 indicate a well-calibrated model.
    """
    import statsmodels.api as sm

    eps = 1e-6
    logit_p = np.log(np.clip(y_prob, eps, 1 - eps) / (1 - np.clip(y_prob, eps, 1 - eps)))
    X = sm.add_constant(logit_p)
    result = sm.Logit(y_true, X).fit(disp=0)
    intercept, slope = result.params
    return float(intercept), float(slope)


def cross_validated_auc(pipeline, X, y, n_splits: int = 5, seed: int = 42) -> dict:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="roc_auc")
    return {"mean_auc": float(scores.mean()), "std_auc": float(scores.std()), "folds": scores.tolist()}


def best_threshold(y_true, y_prob) -> float:
    """Youden's J statistic (max of TPR - FPR) - used only for the reported
    confusion matrix / precision-recall, never to alter predicted probabilities."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def risk_bucket(prob: float) -> str:
    if prob < 0.05:
        return "Low"
    if prob < 0.15:
        return "Moderate"
    if prob < 0.35:
        return "Elevated"
    return "High"
