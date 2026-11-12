# Testing Task 3: the CI/CD pipeline, step by step

Task 3 asks for a CI/CD pipeline that **builds, tests and pushes** the training pipeline and the model endpoint to a container registry. This file lists, in order, every step used to check that the pipeline works and that the model still gives the same results once it is inside the containers. Each step says what it does, why it is needed, and what a pass looks like.

All commands run from the root of this repository unless a step says otherwise. On Windows, use Git Bash or WSL.

## Progress

| # | Check | Status | Result from the last run |
|---|---|---|---|
| 1–3 | Fresh copy, Python environment, Docker running | Done | |
| 4 | Pipeline dry run (test → build → smoke test) | **Pass** | 11 tests passed, both images built, smoke test p = 0.99894 |
| 5–7 | Pipeline with push to a local registry | **Pass** | both images pushed with tags `local` and `v1` |
| 8–10 | Pull the pushed images and retrain from them | **Pass** | test ROC-AUC 0.9973, identical to the original |
| 11–12 | Serve the pulled image, send events | **Pass** | `/health` ok, p = 0.99894, bad event → 422 |
| 13 | Replay the whole test week | **Pass** | 9,752 events, 0 errors |
| 14–15 | Check what is inside the images | **Pass** | code only; no data, models, tests or local paths |
| 16 | Clean up | Done | |
| 17 | Push to a public registry (Docker Hub or GHCR) | **Not yet run** | needs your registry account and login |
| 18 | GitHub Actions run | **Not yet run** | needs the repository on GitHub |

Steps 4–15 were run from a fresh copy of this repository's files.

## Before you start

You need:

- **Docker,** running. On macOS or Windows that means Docker Desktop is open.
- **Python 3.12,** for the pipeline's first stage. The unit tests run on your machine, not in a container.
- **`bash` and `curl`.**
- **About 2 GB of free disk space** for the images.

---

## Part A: run the pipeline

### Step 1: start from a fresh clone of the repository

```bash
git clone <repository-url> /tmp/occupancy-check
cd /tmp/occupancy-check
```

- **What it does:** downloads the repository into a new folder and works only there.
- **Why:** it proves the repository contains everything it needs. If a file had been left out, a later step would fail here.
- **Pass:** the clone has `docker/`, `src/`, `data/raw/`, `models/`, `Scripts/`, `requirements.txt` and `pyproject.toml`.

### Step 2: create the Python environment for the test stage

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e ".[dev]"
```

`make setup` does the same.

- **What it does:** creates a private Python 3.12 environment in `.venv` and installs the pinned libraries, the `occupancy` package and pytest.
- **Why:** stage 1 of the pipeline runs pytest, and the pipeline script finds `.venv` by itself.
- **Pass:** both installs finish without errors.

### Step 3: check Docker is running

```bash
docker info > /dev/null && echo "docker is running"
```

- **What it does:** asks the Docker engine for its status.
- **Why:** stages 2–4 need Docker. If it is not running, the pipeline stops at stage 2 with a connection error.
- **Pass:** it prints `docker is running`.

### Step 4: dry run of the pipeline, without pushing

```bash
bash Scripts/ci_local.sh
```

`make ci-local` does the same.

- **What it does:** runs the whole pipeline except the push. It goes through four stages in order and stops at the first failure:
  1. **Test:** pytest runs the 11 unit tests on the real data files. This checks the data loading, the features, training, the endpoint and the replay script.
  2. **Build:** builds `occupancy-train` and `occupancy-serve` from the two Dockerfiles.
  3. **Smoke test:** the newly built **training image** trains a model into a temporary folder. The newly built **endpoint image** starts on that model, and the script sends one real reading (the first row of `datatest2.txt`) to `POST /events`. This proves the two images work together: a model saved by one loads in the other.
  4. **Push:** skipped, because no registry is set.
- **Why:** it checks build and test before involving a registry.
- **Pass:** the script exits with code 0 and prints:

  ```
  ==> 1/4 test: pytest
  11 passed
  ==> 2/4 build: occupancy-train:local + occupancy-serve:local
  ==> 3/4 smoke test: occupancy-train:local -> occupancy-serve:local -> POST /events
  POST /events -> {"event_id":"ci-smoke","room_id":"room-1","probability":0.9989364743232727,"occupied":true}
  ==> 4/4 push: skipped (set PUSH_REGISTRY to push)
  ```

### Step 5: start a local registry

```bash
docker run -d --rm --name registry -p 5001:5000 registry:2
```

- **What it does:** starts a private container registry on your own machine at `localhost:5001`.
- **Why:** it tests the push stage for real, without needing an account or publishing anything. Port 5001 is used because macOS keeps port 5000 for AirPlay.
- **Pass:** Docker prints a container ID.

### Step 6: run the full pipeline, including the push

```bash
OCC_TAG=v1 PUSH_REGISTRY=localhost:5001 bash Scripts/ci_local.sh
```

- **What it does:** runs all four stages again. Stage 4 now tags the images that passed stage 3 as `localhost:5001/occupancy-train:v1` and `localhost:5001/occupancy-serve:v1`, and pushes them.
- **Why:** this is the whole of Task 3: build, test, push. The push uses the **same** images that passed the tests; nothing is rebuilt between testing and pushing.
- **Pass:** exit code 0, and stage 4 prints a `digest: sha256:...` line for each image.

### Step 7: check the registry has both images

```bash
curl localhost:5001/v2/_catalog
curl localhost:5001/v2/occupancy-serve/tags/list
```

- **What it does:** asks the registry what it holds.
- **Why:** it confirms the push really arrived, rather than trusting the script's own output.
- **Pass:**

  ```
  {"repositories":["occupancy-serve","occupancy-train"]}
  {"name":"occupancy-serve","tags":["v1"]}
  ```

---

## Part B: check that the pushed images give the same results

### Step 8: delete your local copies of the pushed images

```bash
docker rmi localhost:5001/occupancy-train:v1 localhost:5001/occupancy-serve:v1
```

- **What it does:** removes the pushed images from your machine. The registry still has them.
- **Why:** the next step must download the images from the registry, the way a new user would. Without this, Docker would reuse the copies already on your machine.
- **Pass:** Docker prints `Untagged:` for both.

### Step 9: pull the images into an empty folder that holds only the data

```bash
mkdir -p /tmp/new-user/data && cp -R data/raw /tmp/new-user/data/
cd /tmp/new-user
docker pull localhost:5001/occupancy-train:v1
docker pull localhost:5001/occupancy-serve:v1
```

- **What it does:** acts as a new user who has the data files and nothing else: no code, no model, no Python environment.
- **Why:** it proves the pushed images are complete and can be used as they are.
- **Pass:** both pulls finish, and the folder contains only `data/`.

### Step 10: retrain with the pulled training image and check its accuracy

```bash
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/models:/app/models" localhost:5001/occupancy-train:v1
```

- **What it does:** runs the whole Task 1 training pipeline inside the container. It loads the data, splits it by time, one-hot encodes the hour and weekday, tunes XGBoost with random search and 3-fold time-series cross-validation, measures ROC-AUC on the unseen test week, and saves the model to `models/`.
- **Why:** this is the **accuracy check**. The containerised pipeline must reproduce the original results exactly. A different library version or a packaging mistake would show up as a different score.
- **Pass:** the scores match the original run:

  ```
  best params  : {'subsample': 0.7, 'n_estimators': 400, 'max_depth': 2, 'learning_rate': 0.03}
  CV ROC-AUC   : 0.9931
  test ROC-AUC : 0.9973
  ```

### Step 11: start the pulled endpoint image on the new model

```bash
docker network create occupancy
docker run -d --rm --name serve --network occupancy -p 8000:8000 \
  -v "$PWD/models:/app/models:ro" localhost:5001/occupancy-serve:v1
curl http://localhost:8000/health
```

- **What it does:** starts the Task 2 endpoint from the pulled image, using the model step 10 just trained.
- **Why:** it checks the pushed endpoint image starts and loads a model made by the pushed training image.
- **Pass:** `{"status":"ok"}`. If you get "connection refused", wait two seconds and try again.

### Step 12: send one good event and one bad event

```bash
curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{"event_id":"e-1","room_id":"room-1","reading":{"date":"2015-02-11 14:48:00","Temperature":21.76,"Humidity":31.13,"Light":437.33,"CO2":1029.67,"HumidityRatio":0.00502}}'

curl -X POST http://localhost:8000/events -H "Content-Type: application/json" -d '{"event_id":"e-2","room_id":"room-1","reading":{"date":"2015-02-11 14:48:00","Temperature":21.76,"Humidity":31.13,"Light":437.33,"HumidityRatio":0.00502}}'
```

- **What it does:** sends a real reading from the test week, then the same reading with CO2 missing.
- **Why:** the first checks that predictions from the pulled image match the original **to every digit**. The second checks that bad input is rejected before it reaches the model.
- **Pass:**
  - **Good event:** HTTP 200 and `"probability":0.9989364743232727,"occupied":true`, the same number as in step 4.
  - **Bad event:** HTTP 422 with `"loc":["body","reading","CO2"],"msg":"Field required"`.

### Step 13: replay the whole test week through the endpoint

```bash
docker run --rm --network occupancy -v "$PWD/data:/app/data:ro" -e OCC_DATA_DIR=/app/data/raw \
  localhost:5001/occupancy-serve:v1 python -m occupancy.replay --url http://serve:8000 --limit 0
```

- **What it does:** reads all 9,752 one-minute readings of 11–18 Feb (`datatest2.txt`) and posts each one to `/events` as an event, the way a room's sensors would. The true occupancy label is not sent.
- **Why:** it checks the endpoint under a realistic run of traffic, not just one request.
- **Pass:** the last line is `events sent: 9752, errors: 0`. It takes about 30 seconds.

---

## Part C: check what the images contain

### Step 14: list the files inside each image

```bash
docker run --rm --entrypoint sh localhost:5001/occupancy-train:v1 -c 'cd /app && find . -type f | grep -v __pycache__ | sort'
docker run --rm localhost:5001/occupancy-serve:v1 sh -c 'cd /app && find . -type f | grep -v __pycache__ | sort'
```

- **What it does:** lists every file under `/app` in each image.
- **Why:** the images will be public, so they must hold only what is needed to run: no data, no models, no tests, no presentation material.
- **Pass:** only `pyproject.toml`, `requirements.txt` and `src/occupancy/*.py`. The serve image also has `README_EXAMPLE.md`. A `build/lib/` copy of the same Python files is left by pip and is harmless.

### Step 15: check for local file paths or personal details

```bash
docker run --rm localhost:5001/occupancy-serve:v1 sh -c 'grep -rIl -E "/Users/|/home/" /app || echo "no local paths"'
grep -rIl -E "/Users/|/home/|@" --exclude-dir=.venv . || echo "nothing found"
```

- **What it does:** searches the image, then the delivered folder, for home-folder paths and `@` signs, which would catch email addresses.
- **Why:** anything that identifies a person or their machine must not be published.
- **Pass:** `no local paths` for the image. In the folder, the only hits should be code:
  - Python decorators such as `@app.post` and `@pytest.fixture`
  - `@echo` in the Makefile
  - action versions such as `actions/checkout@v4` in `ci.yml`

  Read through the list to confirm there's nothing else.

### Step 16: clean up

```bash
docker stop serve registry
docker network rm occupancy
```

- **What it does:** stops the endpoint and the local registry, and removes the test network.
- **Why:** it frees ports 8000 and 5001 and leaves nothing running.
- **Pass:** both containers stop. They were started with `--rm`, so they are also removed.

---

## Part D: still to do (needs your accounts)

### Step 17: push to a public registry

**Docker Hub:**

```bash
docker login
DOCKER_DEFAULT_PLATFORM=linux/amd64 OCC_TAG=v1 PUSH_REGISTRY=docker.io/<your-docker-id> bash Scripts/ci_local.sh
```

**GitHub Container Registry:**

```bash
echo "<GitHub token with write:packages>" | docker login ghcr.io -u <your-github-user> --password-stdin
DOCKER_DEFAULT_PLATFORM=linux/amd64 OCC_TAG=v1 PUSH_REGISTRY=ghcr.io/<your-github-user> bash Scripts/ci_local.sh
```

- **What it does:** the same as step 6, but the images go to a public registry. `DOCKER_DEFAULT_PLATFORM=linux/amd64` builds the images for Intel/AMD machines, which most servers are. Without it, an Apple-silicon Mac builds ARM-only images.
- **Why:** this is the final form of Task 3's "push to a container registry".
- **Pass:** exit code 0. Then repeat steps 8–13 with `docker.io/<your-docker-id>/...` or `ghcr.io/<your-github-user>/...` in place of `localhost:5001/...`, ideally on another machine.

### Step 18: run the pipeline on GitHub

1. Push a commit to `main`, or open a pull request.
2. Open the repository's **Actions** tab.

- **What it does:** `.github/workflows/ci.yml` runs the same `Scripts/ci_local.sh`:
  - **Every pull request and every push to `main`:** test → build → smoke test.
  - **Pushes to `main`:** also pushes the tested images to `ghcr.io/<owner>`, tagged with the commit SHA, using GitHub's built-in token. No secret needs setting up.
- **Why:** this makes Task 3 automatic. Every change is tested, and only tested images are published.
- **Pass:** a green tick on the workflow run, and two packages under the repository's **Packages**. They start private; make them public in each package's settings.
