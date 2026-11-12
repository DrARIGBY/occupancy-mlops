"""Task 1: loading, the time split and the one-hot datetime features (real data)."""
from __future__ import annotations

import pandas as pd

from occupancy.data import INPUT_COLUMNS, TARGET_COLUMN
from occupancy.features import N_FEATURES, build_pipeline

N_TRAIN, N_TEST = 10_808, 9_752  # datatest + datatraining / datatest2


def test_train_and_test_files_load_with_expected_rows_and_columns(train_test):
    train, test = train_test
    assert (len(train), len(test)) == (N_TRAIN, N_TEST)
    for df in (train, test):
        assert set(INPUT_COLUMNS) | {TARGET_COLUMN} <= set(df.columns)
        assert df["date"].is_monotonic_increasing


def test_split_is_by_time(train_test):
    train, test = train_test
    assert train["date"].max() < test["date"].min()


def test_pipeline_gives_36_features_for_a_sunday_3am_reading(train_test):
    _, test = train_test
    row = test[list(INPUT_COLUMNS)].iloc[[0]].copy()
    row["date"] = pd.Timestamp("2015-02-15 03:00:00")  # a Sunday night
    pipeline = build_pipeline()
    preprocess = pipeline[:-1].fit(test[list(INPUT_COLUMNS)])
    out = preprocess.transform(row)
    assert out.shape == (1, N_FEATURES) == (1, 36)
    assert out[0, 3] == 1.0  # hour 3
    assert out[0, 24 + 6] == 1.0  # Sunday
