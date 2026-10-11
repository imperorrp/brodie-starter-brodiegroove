#!/usr/bin/env bash
# Terminal A: the team backend on http://127.0.0.1:8787
set -e
cd "$(dirname "$0")/../.."
source team/scripts/env.sh
exec .venv/bin/python -m uvicorn app:app --app-dir team/backend --port 8787 --reload
