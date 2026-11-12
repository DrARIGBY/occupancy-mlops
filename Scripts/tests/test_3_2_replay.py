"""Task 2: the replay event source against the real endpoint app (in-process)."""
from __future__ import annotations

from fastapi.testclient import TestClient

from occupancy import service
from occupancy.data import SENSOR_COLUMNS
from occupancy.replay import replay, row_to_event
from occupancy.train import MODEL_DIR_ENV

N_ROWS = 50


def test_row_to_event_keeps_the_timestamp_and_leaves_out_the_label(train_test):
    _, test = train_test
    event = row_to_event(test.iloc[0].to_dict())
    assert set(event) == {"event_id", "room_id", "reading"}
    assert set(event["reading"]) == {"date", *SENSOR_COLUMNS}
    assert event["reading"]["date"] == str(test.iloc[0]["date"])


def test_replay_sends_real_rows_as_events_without_errors(model_dir, monkeypatch, train_test,
                                                          capsys):
    monkeypatch.setenv(MODEL_DIR_ENV, str(model_dir))
    _, test = train_test
    with TestClient(service.app) as client:
        assert replay(client, test.head(N_ROWS)) == 0
    assert f"events sent: {N_ROWS}, errors: 0" in capsys.readouterr().out
