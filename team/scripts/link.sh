#!/usr/bin/env bash
# Terminal C: a phone link for a fresh room (start the worker first).  team/scripts/link.sh [room-name]
# Prints three lines; open the THIRD on the phone. The link carries a token: don't post it anywhere.
set -e
cd "$(dirname "$0")/../.."
source team/scripts/env.sh
exec .venv/bin/python -m facetime.mint_token --room "${1:-team-$(date +%m%d-%H%M%S)}" --url
