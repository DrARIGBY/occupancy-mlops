"""Task 1: random search, ROC-AUC evaluation and the saved artifacts (real data)."""
from __future__ import annotations

import json

import joblib

from occupancy import train as train_mod

from conftest import RAW_DIR


def test_search_gives_a_model_with_predict_proba_and_a_test_roc_auc(train_test, small_model):
    _, test = train_test
    auc = train_mod.roc_auc(small_model, test)
    assert 0.5 < auc <= 1.0


def test_run_saves_a_reloadable_model_and_its_roc_auc(tmp_path, train_test):
    metadata = train_mod.run(RAW_DIR, tmp_path, n_iter=1)
    saved = json.loads((tmp_path / train_mod.METADATA_FILE).read_text(encoding="utf-8"))
    assert saved == metadata
    assert set(saved) == {"created_at", "best_params", "cv_roc_auc", "test_roc_auc"}
    model = joblib.load(tmp_path / train_mod.MODEL_FILE)
    _, test = train_test
    assert train_mod.roc_auc(model, test) == saved["test_roc_auc"]
