#!/usr/bin/env bash
# CI/CD pipeline (Task 3): test -> build -> smoke-test the images -> push.
#   make ci-local        (or: bash Scripts/ci_local.sh)
# .github/workflows/ci.yml runs this same script on GitHub. Needs Docker running.
#
# Settings (environment variables):
#   PYTHON           interpreter for pytest (default: project .venv, else python3)
#   DOCKER_BIN       docker CLI (default: docker)
#   OCC_TAG          image tag (default: short git SHA, or 'local')
#   OCC_TRAIN_IMAGE  training image name (default: occupancy-train)
#   OCC_SERVE_IMAGE  endpoint image name (default: occupancy-serve)
#   N_ITER           random-search candidates for the smoke training run (default: 10)
#   CI_PORT          host port for the smoke-test endpoint container (default: 8001)
#   PUSH_REGISTRY    container registry to push the tested images to, e.g. localhost:5000
#                    or ghcr.io/<owner>; unset = no push
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

default_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then echo "$ROOT/.venv/bin/python"
  elif [[ -x "$ROOT/.venv/Scripts/python.exe" ]]; then echo "$ROOT/.venv/Scripts/python.exe"
  else echo python3; fi
}

PYTHON="${PYTHON:-$(default_python)}"
DOCKER="${DOCKER_BIN:-docker}"
TAG="${OCC_TAG:-$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo local)}"
TRAIN_IMAGE="${OCC_TRAIN_IMAGE:-occupancy-train}:$TAG"
SERVE_IMAGE="${OCC_SERVE_IMAGE:-occupancy-serve}:$TAG"
DATA_DIR="$ROOT/data"
TEST_FILE="$DATA_DIR/raw/datatest2.txt"
N_ITER="${N_ITER:-10}"
CI_PORT="${CI_PORT:-8001}"
CONTAINER_PORT=8000
HEALTH_TIMEOUT=90
SERVE_NAME="occupancy-ci-serve-$$"
MODELS_DIR=""

stage() { printf '\n==> %s\n' "$1"; }
die() { printf 'ci_local: %s\n' "$*" >&2; exit 1; }

# --- 1. test: unit tests of the training pipeline and the endpoint -------------------------
run_tests() {
  stage "1/4 test: pytest"
  "$PYTHON" -m pytest
}

# --- 2. build both images, tagged with the git SHA -----------------------------------------
build_images() {
  stage "2/4 build: $TRAIN_IMAGE + $SERVE_IMAGE"
  "$DOCKER" build -f "$ROOT/docker/Dockerfile.train" -t "$TRAIN_IMAGE" "$ROOT"
  "$DOCKER" build -f "$ROOT/docker/Dockerfile.serve" -t "$SERVE_IMAGE" "$ROOT"
}

# First data row of datatest2.txt -> a POST /events body.
smoke_event() {
  awk -F',' '{ gsub(/["\r]/, "") }
    $2 == "Temperature" || NF < 8 { next }
    { printf "{\"event_id\":\"ci-smoke\",\"room_id\":\"room-1\",\"reading\":"
      printf "{\"date\":\"%s\",\"Temperature\":%s,\"Humidity\":%s,\"Light\":%s,", $2, $3, $4, $5
      printf "\"CO2\":%s,\"HumidityRatio\":%s}}\n", $6, $7; exit }' "$1"
}

cleanup() {
  "$DOCKER" rm -f "$SERVE_NAME" >/dev/null 2>&1 || true
  if [[ -n "$MODELS_DIR" ]]; then rm -rf "$MODELS_DIR"; fi
}

# --- 3. test the images: train image -> endpoint image on that model -> POST /events -------
smoke_test() {
  stage "3/4 smoke test: $TRAIN_IMAGE -> $SERVE_IMAGE -> POST /events"
  local base="http://127.0.0.1:$CI_PORT" deadline reply
  MODELS_DIR="$(mktemp -d)"
  trap cleanup EXIT
  "$DOCKER" run --rm -v "$DATA_DIR:/app/data:ro" -v "$MODELS_DIR:/app/models" \
    "$TRAIN_IMAGE" --n-iter "$N_ITER"
  "$DOCKER" run -d --name "$SERVE_NAME" -p "$CI_PORT:$CONTAINER_PORT" \
    -v "$MODELS_DIR:/app/models:ro" "$SERVE_IMAGE" >/dev/null
  deadline=$((SECONDS + HEALTH_TIMEOUT))
  until curl -fsS "$base/health" >/dev/null 2>&1; do
    if (( SECONDS >= deadline )); then
      "$DOCKER" logs "$SERVE_NAME" >&2 || true
      die "endpoint not up after ${HEALTH_TIMEOUT}s - see the container log above"
    fi
    sleep 1
  done
  reply="$(curl -fsS -X POST -H 'Content-Type: application/json' \
    -d "$(smoke_event "$TEST_FILE")" "$base/events")"
  echo "POST /events -> $reply"
  [[ "$reply" == *'"probability"'* ]] || die "/events reply has no prediction: $reply"
}

# --- 4. push the images that passed the tests (no rebuild) ---------------------------------
push_images() {
  if [[ -z "${PUSH_REGISTRY:-}" ]]; then
    stage "4/4 push: skipped (set PUSH_REGISTRY to push)"
    return 0
  fi
  stage "4/4 push: $PUSH_REGISTRY"
  local image
  for image in "$TRAIN_IMAGE" "$SERVE_IMAGE"; do
    "$DOCKER" tag "$image" "$PUSH_REGISTRY/$image"
    "$DOCKER" push "$PUSH_REGISTRY/$image"
  done
}

run_tests
build_images
smoke_test
push_images
