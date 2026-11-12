"""Feature engineering + model as one sklearn Pipeline (Task 1).

The datetime is one-hot encoded inside the pipeline, so the saved model takes raw
readings and the endpoint (Task 2) needs no feature code of its own.
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from occupancy.data import DATE_COLUMN, SENSOR_COLUMNS

HOUR_COLUMN = "hour"
DAY_COLUMN = "day_of_week"
HOUR_CATEGORIES = list(range(24))
DAY_CATEGORIES = list(range(7))  # Monday = 0 ... Sunday = 6
DEFAULT_SEED = 42
MODEL_STEP = "model"
N_FEATURES = len(HOUR_CATEGORIES) + len(DAY_CATEGORIES) + len(SENSOR_COLUMNS)


class DatetimeFeatures(BaseEstimator, TransformerMixin):
    """Replace the `date` column by categorical `hour` and `day_of_week` columns."""

    def fit(self, X: pd.DataFrame, y: Any = None) -> DatetimeFeatures:
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        stamps = pd.to_datetime(X[DATE_COLUMN])
        out = X.drop(columns=[DATE_COLUMN])
        out[HOUR_COLUMN] = stamps.dt.hour.to_numpy()
        out[DAY_COLUMN] = stamps.dt.dayofweek.to_numpy()
        return out


def build_pipeline(seed: int = DEFAULT_SEED) -> Pipeline:
    """DatetimeFeatures -> one-hot hour/day + 5 sensors -> XGBClassifier."""
    one_hot = OneHotEncoder(
        categories=[HOUR_CATEGORIES, DAY_CATEGORIES], handle_unknown="ignore", sparse_output=False
    )
    preprocess = ColumnTransformer(
        [
            ("datetime_onehot", one_hot, [HOUR_COLUMN, DAY_COLUMN]),
            ("sensors", "passthrough", list(SENSOR_COLUMNS)),
        ]
    )
    return Pipeline(
        [
            ("datetime", DatetimeFeatures()),
            ("preprocess", preprocess),
            (MODEL_STEP, XGBClassifier(random_state=seed)),
        ]
    )
