"""Shared fixtures. Every test uses the three real UCI files; nothing is synthetic."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from occupancy import train as train_mod
from occupancy.data import DEFAULT_DATA_DIR, load_train_test

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / DEFAULT_DATA_DIR
TRAIN_STRIDE = 5  # every 5th real minute: keeps the full time span, trains in seconds
SMALL_N_ITER = 2


@pytest.fixture(scope="session")
def train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    return load_train_test(RAW_DIR)


@pytest.fixture(scope="session")
def small_model(train_test):
    """A real search on a thinned copy of the train rows (the full run is `make train`)."""
    train, _ = train_test
    return train_mod.random_search(train.iloc[::TRAIN_STRIDE], SMALL_N_ITER).best_estimator_


@pytest.fixture(scope="session")
def model_dir(small_model, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("models")
    train_mod.save(small_model, {}, out)
    return out
