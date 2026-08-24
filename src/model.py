"""
Logistic-regression sovereign financial-stress model.

Provides:
  - a scikit-learn pipeline (StandardScaler + LogisticRegression) used for
    prediction, scenario scoring and the dashboard, and
  - a statsmodels Logit fit on the same standardized design matrix, used for
    inference (coefficient significance, odds ratios) since scikit-learn does
    not report standard errors / p-values.

The train/test split is chronological (out-of-sample in time, not just
row-shuffled): the model is trained on 2005-2019 and evaluated on unseen
2020-2023 country-years, which happens to include the COVID-19 shock -
i.e. genuine adverse-regime, out-of-sample testing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.data_generation import FEATURE_COLUMNS

TARGET_COLUMN = "financial_stress"
TRAIN_TEST_SPLIT_YEAR = 2019  # train: year <= 2019, test: year > 2019


@dataclass
class TrainedModel:
    pipeline: Pipeline
    sm_result: object
    scaler: StandardScaler
    feature_names: list
    train_df: pd.DataFrame
    test_df: pd.DataFrame


def chronological_split(df: pd.DataFrame, split_year: int = TRAIN_TEST_SPLIT_YEAR):
    train_df = df[df["year"] <= split_year].reset_index(drop=True)
    test_df = df[df["year"] > split_year].reset_index(drop=True)
    return train_df, test_df


def fit_model(df: pd.DataFrame, split_year: int = TRAIN_TEST_SPLIT_YEAR) -> TrainedModel:
    train_df, test_df = chronological_split(df, split_year)

    X_train = train_df[FEATURE_COLUMNS].values
    y_train = train_df[TARGET_COLUMN].values

    # Note: class imbalance (~17% event rate) is handled via a tuned decision
    # threshold (see validation.best_threshold), not via class_weight
    # reweighting - reweighting shifts predict_proba away from the true
    # unconditional probability, which would corrupt the Brier score,
    # calibration curve and the scenario-analysis probabilities below.
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=2000, C=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    # statsmodels fit on the same standardized features, for inference table
    scaler = pipeline.named_steps["scaler"]
    X_train_std = scaler.transform(X_train)
    X_train_sm = sm.add_constant(X_train_std, has_constant="add")
    sm_result = sm.Logit(y_train, X_train_sm).fit(disp=0)

    return TrainedModel(
        pipeline=pipeline,
        sm_result=sm_result,
        scaler=scaler,
        feature_names=FEATURE_COLUMNS,
        train_df=train_df,
        test_df=test_df,
    )


def predict_proba(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    X = df[model.feature_names].values
    return model.pipeline.predict_proba(X)[:, 1]


def coefficient_table(model: TrainedModel) -> pd.DataFrame:
    params = model.sm_result.params
    conf = model.sm_result.conf_int()
    pvalues = model.sm_result.pvalues
    names = ["const"] + model.feature_names
    table = pd.DataFrame({
        "feature": names,
        "coef": params,
        "std_err": model.sm_result.bse,
        "z": model.sm_result.tvalues,
        "p_value": pvalues,
        "ci_lower": conf[:, 0],
        "ci_upper": conf[:, 1],
    })
    table["odds_ratio"] = np.exp(table["coef"])
    return table.reset_index(drop=True)
