#!/usr/bin/env bash
# The cloud box: the backend API (uvicorn on $PORT, public) and the Brodie worker (LiveKit Agents, outbound only) in one
# container, each under its own restart loop. Railway runs this as the start command (railway.json); a plain VM runs it
# under systemd (team/cloud/README.md). FaceTime also needs a separate authorized Mac/Linux Chert CLI process.
#
# Env it expects: DATABASE_URL (Supabase), LIVEKIT_URL/API_KEY/API_SECRET, GEMINI_API_KEY (+ GEMINI_BASE_URL for the relay),
# AGENT_NAME (the dispatch name; the API mints /join tokens for the same name). Optional: DANCE_LIBRARY_URL (the AIST keys),
# DANCE_PUBLIC_URL (defaults to the Railway domain), DANCE_CALL_HANDLE, JOIN_KEY, WORKER=0 (API only), DANCE_* tuning.
set -u
cd "$(dirname "$0")/../.."
[ -f /opt/venv/bin/activate ] && . /opt/venv/bin/activate        # legacy Nixpacks path; Railpack supplies /app/.venv/bin in PATH
export PYTHONUNBUFFERED=1 PYTHONUTF8=1
export PORT="${PORT:-8787}"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:$PORT}"        # the worker talks to the API next to it
export REQUIRE_DATABASE_URL=1                                       # app.py refuses to start on SQLite here
if [ -z "${DANCE_PUBLIC_URL:-}" ] && [ -n "${RAILWAY_PUBLIC_DOMAIN:-}" ]; then export DANCE_PUBLIC_URL="https://$RAILWAY_PUBLIC_DOMAIN"; fi
export DANCE_MAX_POSES="${DANCE_MAX_POSES:-1}" DANCE_POSE_PX="${DANCE_POSE_PX:-320}"   # solo defaults, as chrome.ps1
export RTCOACH_EVENT_LOG="${RTCOACH_EVENT_LOG:-data/tmp/worker_events.jsonl}" RTCOACH_CUE_LOG="${RTCOACH_CUE_LOG:-data/tmp/worker_cues.jsonl}"
mkdir -p data/tmp/sessions models
echo "[cloud] python: $(python --version 2>&1)  port: $PORT  public: ${DANCE_PUBLIC_URL:-unset}  agent: ${AGENT_NAME:-brodie-coach (default)}"
python team/cloud/fetch_models.py || echo "[cloud] pose models missing: the worker cannot score until they download" >&2
python team/cloud/fetch_library.py || echo "[cloud] AIST keys missing: generated routines and the daily key will fail (set DANCE_LIBRARY_URL)" >&2

loop() {   # loop NAME CMD...: run CMD forever; the pause between restarts doubles while it keeps dying fast, up to a minute
  local name=$1; shift; local pause=2
  while true; do
    local t0; t0=$(date +%s)
    "$@"; local rc=$?
    local ran=$(( $(date +%s) - t0 ))
    echo "[cloud] $name exited with $rc after ${ran}s; restarting in ${pause}s" >&2
    sleep "$pause"
    if [ "$ran" -lt 30 ]; then pause=$(( pause < 60 ? pause * 2 : 60 )); else pause=2; fi
  done
}

# /join and /phone.html query strings contain credentials; keep them out of access logs.
loop api python -m uvicorn app:app --app-dir team/backend --host 0.0.0.0 --port "$PORT" --no-access-log &
API_PID=$!
WORKER_PID=
if [ "${WORKER:-1}" != "0" ]; then
  loop worker python -m team.dance.run_worker --perception off --voice gemini --persona team/dance/persona.md \
      --handlers team/dance/live.py --plan-file team/dance/plan.txt --log-caller --audio-queue-ms 300 --leave-grace-s 5 start &
  WORKER_PID=$!
fi
trap 'echo "[cloud] stopping"; kill $API_PID $WORKER_PID 2>/dev/null; exit 0' TERM INT
wait
