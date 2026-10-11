# Publish team/ (WITHOUT docs/) plus the facetime edits to the PUBLIC fork as one clean commit on top of its main.
# Re-runnable: each run adds one commit with the latest state. Uses URLs, not remote names, so it works from any clone.
# See team/docs/private-repo.md.
$PUBLIC = 'https://github.com/imperorrp/ivyhacks-brodiegroove.git'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
function run { & git @args; if ($LASTEXITCODE) { throw "git $($args -join ' ') failed" } }
if (git -C $root status --porcelain -- team facetime) { throw 'Commit (and push to the private repo) your team/ and facetime changes first.' }
run -C $root fetch $PUBLIC main
$wt = Join-Path (Split-Path $root -Parent) 'brodie-submit'
if (Test-Path $wt) { run -C $root worktree remove --force $wt }
run -C $root worktree add --detach $wt FETCH_HEAD
run -C $wt rm -r -q --ignore-unmatch team facetime     # so files removed from main (e.g. research-licensed keys) leave the public tree too
run -C $wt checkout main -- team facetime requirements.txt .gitignore vercel.json README.md
Remove-Item -Recurse -Force (Join-Path $wt 'team\docs')
run -C $wt add -A
run -C $wt commit -m 'Add the Brodie Groove skill: dance engine, backend, web, worker fixes' -m 'Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>'
run -C $wt push $PUBLIC HEAD:main
run -C $root worktree remove --force $wt
Write-Host "Published to $PUBLIC (main). team/docs stayed private. Check it on GitHub, then open the submission issue."
