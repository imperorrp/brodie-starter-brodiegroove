# Load .env.local (repo root) into THIS shell. Source it:  source team/scripts/env.sh
_root="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"
if [ -f "$_root/.env.local" ]; then set -a; . "$_root/.env.local"; set +a; else echo ".env.local not found in $_root" >&2; fi
export PYTHONUTF8=1
