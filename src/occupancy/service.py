"""Event-driven model endpoint (Task 2).

    uvicorn occupancy.service:app --port 8000      (or: make serve)

POST /events  one sensor event {"event_id", "room_id", "reading"} in -> the occupancy
              prediction for that room out
GET  /health  {"status": "ok"} once the model is loaded (used by the CI smoke test)
GET  /        redirects to /docs, the interactive page for trying the endpoints

The model is loaded once at start-up from $OCC_MODEL_DIR (default models). The saved
pipeline builds its own datetime features, so raw readings go straight in.
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from occupancy.data import INPUT_COLUMNS
from occupancy.train import MODEL_FILE, model_dir

DECISION_THRESHOLD = 0.5
HEALTH_OK = "ok"
DOCS_PATH = "/docs"


class Reading(BaseModel):
    """One raw sensor reading; field names equal the training column names."""

    date: datetime
    Temperature: float
    Humidity: float
    Light: float
    CO2: float
    HumidityRatio: float


class Event(BaseModel):
    event_id: str
    room_id: str
    reading: Reading


class Prediction(BaseModel):
    event_id: str
    room_id: str
    probability: float
    occupied: bool


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at start-up."""
    app.state.model = joblib.load(model_dir() / MODEL_FILE)
    yield


app = FastAPI(title="occupancy-service", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """The bare address opens the interactive API page instead of a 404."""
    return RedirectResponse(DOCS_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": HEALTH_OK}


@app.post("/events", response_model=Prediction)
def events(event: Event, request: Request) -> Prediction:
    frame = pd.DataFrame([event.reading.model_dump()])[list(INPUT_COLUMNS)]
    probability = float(request.app.state.model.predict_proba(frame)[0, 1])
    return Prediction(
        event_id=event.event_id,
        room_id=event.room_id,
        probability=probability,
        occupied=probability >= DECISION_THRESHOLD,
    )
