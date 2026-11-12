"""Load the provided UCI Occupancy files (Task 1).

Train on the two earlier files (2-10 Feb 2015) and test on datatest2.txt (11-18 Feb 2015).
This is a split by time, so the model is tested on days after the ones it learned from.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

# --- data location (override the directory with OCC_DATA_DIR) ---
DATA_DIR_ENV = "OCC_DATA_DIR"
DEFAULT_DATA_DIR = "data/raw"  # att:portable-path
TRAIN_FILES: tuple[str, ...] = ("datatest.txt", "datatraining.txt")  # 2-10 Feb 2015
TEST_FILE = "datatest2.txt"  # 11-18 Feb 2015

# --- schema ---
DATE_COLUMN = "date"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SENSOR_COLUMNS: tuple[str, ...] = ("Temperature", "Humidity", "Light", "CO2", "HumidityRatio")
TARGET_COLUMN = "Occupancy"
INPUT_COLUMNS: tuple[str, ...] = (DATE_COLUMN, *SENSOR_COLUMNS)


def data_dir() -> Path:
    """The raw-data directory: $OCC_DATA_DIR, else data/raw."""
    return Path(os.environ.get(DATA_DIR_ENV, DEFAULT_DATA_DIR))


def load_data(paths: Iterable[str | Path]) -> pd.DataFrame:
    """Read the files (first column = row number), parse `date`, sort by time."""
    frames = [pd.read_csv(p, index_col=0) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT)
    return df.sort_values(DATE_COLUMN, kind="stable").reset_index(drop=True)


def load_train_test(directory: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test): the two earlier files, and the later test file."""
    base = Path(directory) if directory is not None else data_dir()
    return load_data([base / name for name in TRAIN_FILES]), load_data([base / TEST_FILE])
