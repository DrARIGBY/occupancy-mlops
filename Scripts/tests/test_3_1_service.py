"""Task 2: the event-driven endpoint, driven with real datatest2 rows (no server needed)."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from occupancy import service
from occupancy.data import INPUT_COLUMNS
from occupancy.replay import row_to_event
from occupancy.train import MODEL_DIR_ENV

N_ROWS = 20


@pytest.fixture()
def client(model_dir, monkeypatch):
    monkeypatch.setenv(MODEL_DIR_ENV, str(model_dir))
    with TestClient(service.app) as test_client:
        yield test_client


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_bare_address_opens_the_api_page(client):
    # A person who clicks the address uvicorn prints should land on /docs, not a 404.
    reply = client.get("/", follow_redirects=False)
    assert reply.status_code == 307
    assert reply.headers["location"] == service.DOCS_PATH
    assert client.get("/").status_code == 200


def test_events_return_the_model_prediction(client, train_test, small_model):
    _, test = train_test
    rows = test.iloc[:: len(test) // N_ROWS]  # spread over the whole test week
    expected = small_model.predict_proba(rows[list(INPUT_COLUMNS)])[:, 1]
    for record, p in zip(rows.to_dict(orient="records"), expected):
        event = row_to_event(record)
        reply = client.post("/events", json=event)
        assert reply.status_code == 200
        body = reply.json()
        assert body["event_id"] == event["event_id"]
        assert body["probability"] == pytest.approx(p, abs=1e-6)
        assert body["occupied"] == (p >= service.DECISION_THRESHOLD)


def test_event_without_a_sensor_value_is_rejected(client, train_test):
    _, test = train_test
    event = row_to_event(test.iloc[0].to_dict())
    del event["reading"]["CO2"]
    assert client.post("/events", json=event).status_code == 422
