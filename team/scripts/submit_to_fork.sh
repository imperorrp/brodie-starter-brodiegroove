#!/usr/bin/env bash
# Mac/Linux twin of submit_to_fork.ps1: publish team/ (without docs/) and the facetime edits to the PUBLIC fork.
set -euo pipefail
PUBLIC=https://github.com/imperorrp/brodie-starter-brodiegroove.git
root="$(cd "$(dirname "$0")/../.." && pwd)"
if [ -n "$(git -C "$root" status --porcelain -- team facetime)" ]; then echo "Commit (and push to the private repo) your team/ and facetime changes first." >&2; exit 1; fi
git -C "$root" fetch "$PUBLIC" main
wt="$(dirname "$root")/brodie-submit"
[ -d "$wt" ] && git -C "$root" worktree remove --force "$wt"
git -C "$root" worktree add --detach "$wt" FETCH_HEAD
git -C "$wt" checkout main -- team facetime requirements.txt .gitignore
rm -rf "$wt/team/docs"
git -C "$wt" add -A
git -C "$wt" commit -m "Add the Brodie Groove skill: dance engine, backend, web, worker fixes" -m "Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git -C "$wt" push "$PUBLIC" HEAD:main
git -C "$root" worktree remove --force "$wt"
echo "Published to $PUBLIC (main). team/docs stayed private. Check it on GitHub, then open the submission issue."
