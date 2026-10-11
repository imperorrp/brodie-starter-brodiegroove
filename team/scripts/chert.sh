#!/usr/bin/env bash
# The Chert FaceTime CLI through the Windows preload shim (see chert.ps1). On a Mac or Linux just use `chert` directly.
set -e
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$HERE/../.." && pwd)"; cd "$ROOT"
set -a; [ -f .env.local ] && . ./.env.local; set +a
REL="node_modules/@trychert/facetime/dist/src/cli/main.js"
MAIN=""
G="$(npm root -g 2>/dev/null | head -1 | tr -d '\r')"
for c in "${G%/node_modules}/$REL" "${G%\\node_modules}/$REL" "$APPDATA/npm/$REL" "/c/Program Files/nodejs/$REL"; do
  c="$(printf '%s' "$c" | sed 's#\\#/#g')"
  if [ -f "$c" ]; then MAIN="$c"; break; fi
done
[ -n "$MAIN" ] || { echo "Chert CLI not installed for this Node ($(node --version)): npm install -g @trychert/facetime@0.1.0-beta.4" >&2; exit 1; }
exec node -r "$HERE/chert_win_shim.js" "$MAIN" "$@"
