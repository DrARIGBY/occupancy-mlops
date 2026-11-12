"""Event source: replay datatest2.txt as sensor events to the endpoint (Task 2).

    python -m occupancy.replay --url http://localhost:8000 --limit 100   (or: make replay)

Each row becomes one event {"event_id", "room_id", "reading"} posted to POST /events,
and the returned prediction is printed. The label column is not sent.
`replay()` takes any client with `.post(path, json=...)`, so the tests drive the same
code in-process through FastAPI's TestClient.
"""
from __future__ import annotations

import argparse
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
import pandas as pd

from occupancy.data import DATE_COLUMN, SENSOR_COLUMNS, TEST_FILE, data_dir, load_data

DEFAULT_URL = "http://localhost:8000"
ROOM_ID = "room-1"
EVENTS_PATH = "/events"
ALL_ROWS = 0  # --limit 0 means "replay every row"
HTTP_OK = 200


def row_to_event(row: Mapping[str, Any], room_id: str = ROOM_ID) -> dict[str, Any]:
    """One data row -> the POST /events body (the Occupancy label is left out)."""
    reading = {DATE_COLUMN: str(row[DATE_COLUMN])}
    reading.update({name: float(row[name]) for name in SENSOR_COLUMNS})
    return {"event_id": str(uuid.uuid4()), "room_id": room_id, "reading": reading}


def replay(client: Any, rows: pd.DataFrame) -> int:
    """Post every row as an event, print each prediction; returns the number of errors."""
    errors = 0
    for record in rows.to_dict(orient="records"):
        response = client.post(EVENTS_PATH, json=row_to_event(record))
        if response.status_code != HTTP_OK:
            errors += 1
            print(f"{record[DATE_COLUMN]}  HTTP {response.status_code}: {response.text}")
            continue
        body = response.json()
        print(f"{record[DATE_COLUMN]}  {body['room_id']}  "
              f"p={body['probability']:.3f}  occupied={body['occupied']}")
    print(f"events sent: {len(rows)}, errors: {errors}")
    return errors


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m occupancy.replay", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default=DEFAULT_URL, help=f"service URL (default {DEFAULT_URL})")
    parser.add_argument("--limit", type=int, default=ALL_ROWS,
                        help="number of rows to send, 0 = all (default 0)")
    args = parser.parse_args(argv)
    rows = load_data([data_dir() / TEST_FILE])
    if args.limit:
        rows = rows.head(args.limit)
    with httpx.Client(base_url=args.url) as client:
        return 1 if replay(client, rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
