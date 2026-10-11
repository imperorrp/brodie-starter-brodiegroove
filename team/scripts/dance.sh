#!/usr/bin/env bash
# Terminal B in dance mode: the Brodie worker with the dance skill.  team/scripts/dance.sh [chert-dispatch-name] [extra worker flags...]
# env: DANCE_KEY, DANCE_MODE=learn|battle, DANCE_TEMPO, DANCE_RTT, DANCE_DEBUG_MP4 (see team/dance/live.py)
set -e
cd "$(dirname "$0")/../.."
source team/scripts/env.sh
name="${1:-}"; shift || true
exec .venv/bin/python -m team.dance.run_worker --perception off --voice gemini --persona team/dance/persona.md \
  --handlers team/dance/live.py --plan-file team/dance/plan.txt --log-caller --audio-queue-ms 300 --agent-name="$name" "$@" dev
