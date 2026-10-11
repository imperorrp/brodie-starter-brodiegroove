# Publish team/web/groove as the gh-pages branch of the PUBLIC fork (GitHub Pages). Force-pushes a generated branch; safe to re-run.
$PUBLIC = 'https://github.com/imperorrp/brodie-starter-brodiegroove.git'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$site = Join-Path $root 'team\web\site'
$tmp = Join-Path ([IO.Path]::GetTempPath()) ("brodie-site-" + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
Copy-Item (Join-Path $site '*') $tmp -Recurse
New-Item -ItemType File -Path (Join-Path $tmp '.nojekyll') | Out-Null
function run { & git @args; if ($LASTEXITCODE) { throw "git $($args -join ' ') failed" } }
run -C $tmp init -q -b gh-pages
run -C $tmp add -A
run -C $tmp commit -q -m ("Publish site " + (Get-Date -Format 'yyyy-MM-dd_HH:mm'))
run -C $tmp push -f $PUBLIC gh-pages:gh-pages
Remove-Item -Recurse -Force $tmp
Write-Host "Published. Enable once: GitHub repo Settings > Pages > Deploy from a branch > gh-pages / root. URL: https://brodiegroove.vercel.app/ (Vercel; the gh-pages branch is the fallback)"
