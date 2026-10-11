#!/usr/bin/env bash
# Publish team/web/groove as the gh-pages branch of the PUBLIC fork (GitHub Pages). Force-pushes a generated branch; safe to re-run.
set -euo pipefail
PUBLIC=https://github.com/imperorrp/brodie-starter-brodiegroove.git
root="$(cd "$(dirname "$0")/../.." && pwd)"; site="$root/team/web/groove"; tmp="$(mktemp -d)"
cp -r "$site"/. "$tmp"/ && touch "$tmp/.nojekyll"
git -C "$tmp" init -q -b gh-pages && git -C "$tmp" add -A && git -C "$tmp" -c user.name="$(git config user.name)" -c user.email="$(git config user.email)" commit -q -m "Publish site $(date +%Y-%m-%d_%H:%M)"
git -C "$tmp" push -f "$PUBLIC" gh-pages:gh-pages
rm -rf "$tmp"
echo "Published. Enable once: GitHub repo Settings > Pages > Deploy from a branch > gh-pages / root. URL: https://imperorrp.github.io/brodie-starter-brodiegroove/"
