"""Training pipeline (Task 1).

    python -m occupancy.train --data-dir data/raw --out models --n-iter 10

load -> random search over the datetime-one-hot + XGBoost pipeline
(RandomizedSearchCV, TimeSeriesSplit(3), ROC-AUC) -> ROC-AUC on the test file
-> save models/model.joblib + models/metadata.json.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from occupancy.data import INPUT_COLUMNS, TARGET_COLUMN, data_dir, load_train_test
from occupancy.features import DEFAULT_SEED, MODEL_STEP, build_pipeline

# --- model location (override the directory with OCC_MODEL_DIR) ---
MODEL_DIR_ENV = "OCC_MODEL_DIR"
DEFAULT_MODEL_DIR = "models"  # att:portable-path
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"

# --- random search ---
DEFAULT_N_ITER = 10
CV_SPLITS = 3  # with 4-5 splits one validation fold is an empty weekend -> ROC-AUC undefined
SCORING = "roc_auc"
SEARCH_SPACE: dict[str, list[Any]] = {
    "n_estimators": [100, 200, 300, 400],
    "max_depth": [2, 3, 4, 5, 6],
    "learning_rate": [0.03, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.85, 1.0],
}


def model_dir() -> Path:
    """The model directory: $OCC_MODEL_DIR, else models."""
    return Path(os.environ.get(MODEL_DIR_ENV, DEFAULT_MODEL_DIR))


def features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw model inputs (date + 5 sensors) and the 0/1 target."""
    return df[list(INPUT_COLUMNS)], df[TARGET_COLUMN].astype(int)


def random_search(train: pd.DataFrame, n_iter: int, seed: int = DEFAULT_SEED) -> RandomizedSearchCV:
    """Tune the pipeline on time-ordered train rows; refits the best candidate."""
    space = {f"{MODEL_STEP}__{name}": values for name, values in SEARCH_SPACE.items()}
    search = RandomizedSearchCV(
        build_pipeline(seed),
        param_distributions=space,
        n_iter=n_iter,
        scoring=SCORING,
        cv=TimeSeriesSplit(n_splits=CV_SPLITS),
        random_state=seed,
    )
    X, y = features_and_target(train)
    return search.fit(X, y)


def roc_auc(model: Pipeline, df: pd.DataFrame) -> float:
    """ROC-AUC of the model's P(occupied) on `df`."""
    X, y = features_and_target(df)
    return float(roc_auc_score(y, model.predict_proba(X)[:, 1]))


def save(model: Pipeline, metadata: dict[str, Any], out_dir: str | Path) -> None:
    """Write model.joblib and metadata.json into `out_dir`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / MODEL_FILE)
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def run(data_dir_path: str | Path, out_dir: str | Path, n_iter: int = DEFAULT_N_ITER) -> dict[str, Any]:
    """Load -> search -> evaluate -> save; returns the saved metadata."""
    train, test = load_train_test(data_dir_path)
    search = random_search(train, n_iter)
    prefix = f"{MODEL_STEP}__"
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_params": {k.removeprefix(prefix): v for k, v in search.best_params_.items()},
        "cv_roc_auc": float(search.best_score_),
        "test_roc_auc": roc_auc(search.best_estimator_, test),
    }
    save(search.best_estimator_, metadata, out_dir)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m occupancy.train", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=data_dir(),
                        help="folder with the three UCI files (default: $OCC_DATA_DIR or data/raw)")
    parser.add_argument("--out", type=Path, default=model_dir(),
                        help="output folder (default: $OCC_MODEL_DIR or models)")
    parser.add_argument("--n-iter", type=int, default=DEFAULT_N_ITER,
                        help=f"random-search candidates (default {DEFAULT_N_ITER})")
    args = parser.parse_args(argv)
    metadata = run(args.data_dir, args.out, args.n_iter)
    print(f"best params  : {metadata['best_params']}")
    print(f"CV ROC-AUC   : {metadata['cv_roc_auc']:.4f}")
    print(f"test ROC-AUC : {metadata['test_roc_auc']:.4f}")
    print(f"saved        : {args.out / MODEL_FILE}, {args.out / METADATA_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
