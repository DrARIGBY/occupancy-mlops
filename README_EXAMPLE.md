# Testing the occupancy endpoint: a worked example

This image runs the **occupancy prediction endpoint**. You send it one sensor reading from a room, and it replies with the probability that the room is occupied.

This file walks through one complete test: start the endpoint, check it is up, send a reading, send a bad reading, and replay a week of real data. Every "you should see" output below was recorded from a real run.

A copy of this file is inside the image at `/app/README_EXAMPLE.md`:

```bash
docker run --rm occupancy-serve cat README_EXAMPLE.md
```

## What you need

| Item | Why |
|---|---|
| Docker | to run the images |
| The `occupancy-serve` image | the endpoint (this image) |
| A trained model folder, `models/` with `model.joblib` | the image does not contain a model; you mount one at run time |
| The data folder, `data/raw/` with `datatest2.txt` | only for step 5 (replaying a real week) and for training a model |

**If you don't have a model yet,** train one with the training image. It takes about 10 seconds:

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" occupancy-train --n-iter 10
```

Run all commands from the folder that contains `models/` and `data/`.

## Step 1: start the endpoint

```bash
docker network create occupancy
docker run -d --rm --name serve --network occupancy -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" occupancy-serve
```

The endpoint is now at **http://localhost:8000**. That is the base address; the endpoints are paths added to it:

| Address | What it does |
|---|---|
| `http://localhost:8000` | opens the interactive test page (`/docs`) |
| `http://localhost:8000/health` | `{"status": "ok"}` when the model is loaded |
| `http://localhost:8000/events` | the prediction endpoint; **POST only**, so a browser address bar shows "Method Not Allowed" |

## Step 2: check it is up

```bash
curl http://localhost:8000/health
```

You should see:

```json
{"status":"ok"}
```

If you get "connection refused", wait two seconds and try again. If it still fails, run `docker logs serve`. A missing `models/model.joblib` shows up there.

## Step 3: send one real reading

This is the first minute of the test week, 11 Feb 2015 at 14:48, with the lights on and CO2 high:

```bash
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{
  "event_id": "e-1",
  "room_id": "room-1",
  "reading": {"date": "2015-02-11 14:48:00", "Temperature": 21.76, "Humidity": 31.13,
              "Light": 437.33, "CO2": 1029.67, "HumidityRatio": 0.00502}}'
```

You should see (HTTP 200):

```json
{"event_id":"e-1","room_id":"room-1","probability":0.9989364743232727,"occupied":true}
```

- `probability` is the model's estimate that the room is occupied, from 0 to 1.
- `occupied` is `true` when the probability is 0.5 or more.
- `event_id` and `room_id` are copied back, so the caller can match the answer to the event.

## Step 4: send a bad reading

The same event with `CO2` left out:

```bash
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{
  "event_id": "e-2",
  "room_id": "room-1",
  "reading": {"date": "2015-02-11 14:48:00", "Temperature": 21.76, "Humidity": 31.13,
              "Light": 437.33, "HumidityRatio": 0.00502}}'
```

You should see **HTTP 422**, with the missing field named:

```json
{"detail":[{"type":"missing","loc":["body","reading","CO2"],"msg":"Field required", ...}]}
```

The endpoint checks every event before the model sees it. A missing field, or text where a number should be, is rejected with 422 and never gets a prediction.

## Step 5: replay a real week of events

The image also contains the replay script. It reads `datatest2.txt` (11–18 Feb 2015, 9,752 one-minute readings) and posts each row to `/events` as an event. The true occupancy label is **not** sent.

```bash
docker run --rm --network occupancy -v "$PWD/data:/app/data:ro" -e OCC_DATA_DIR=/app/data/raw \
  occupancy-serve python -m occupancy.replay --url http://serve:8000 --limit 10
```

You should see:

```
2015-02-11 14:48:00  room-1  p=0.999  occupied=True
2015-02-11 14:49:00  room-1  p=0.999  occupied=True
2015-02-11 14:50:00  room-1  p=0.999  occupied=True
2015-02-11 14:51:00  room-1  p=0.999  occupied=True
2015-02-11 14:51:59  room-1  p=0.999  occupied=True
2015-02-11 14:53:00  room-1  p=0.999  occupied=True
2015-02-11 14:54:00  room-1  p=0.999  occupied=True
2015-02-11 14:55:00  room-1  p=0.999  occupied=True
2015-02-11 14:55:59  room-1  p=0.999  occupied=True
2015-02-11 14:57:00  room-1  p=0.999  occupied=True
events sent: 10, errors: 0
```

Use `--limit 0` to send the whole week. The last line should read `events sent: 9752, errors: 0`; it takes about 30 seconds.

`--url http://serve:8000` uses the container name `serve` because the replay runs in its own container on the `occupancy` network. From your own machine the address is `http://localhost:8000`.

## Step 6 (optional): try it in a browser

Open **http://localhost:8000**. It takes you to the interactive API page.

1. Click **POST /events**, then **Try it out**.
2. Paste the event from step 3 into the request body and click **Execute**.
3. The response appears below, the same as in step 3.

## Step 7: stop

```bash
docker stop serve
docker network rm occupancy
```

## Summary of a passing test

| Step | Pass if |
|---|---|
| 2. Health | `{"status":"ok"}` |
| 3. Real reading | HTTP 200, `"occupied": true`, probability about 0.999 |
| 4. Bad reading | HTTP 422 naming `reading.CO2` |
| 5. Replay | `errors: 0` |

For reference, the same model scores a ROC-AUC of 0.9973 on the full test week. When all 9,752 events are sent through the endpoint, its answers are identical to calling the model directly.
