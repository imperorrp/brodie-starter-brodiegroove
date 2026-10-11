#!/usr/bin/env bash
# Terminal B: the Brodie coach with our skill.  team/scripts/worker.sh [chert-dispatch-name] [extra worker flags...]
# No name (or "") = automatic dispatch (the phone page). With a name = the Chert line (then run chert dev --agent <name>).
set -e
cd "$(dirname "$0")/../.."
source team/scripts/env.sh
name="${1:-}"; shift || true
exec .venv/bin/python -m facetime.worker --persona team/skill/persona.md --plan-file team/skill/plan.txt \
  --handlers team/skill/handlers.py --voice gemini --agent-name="$name" "$@" dev
