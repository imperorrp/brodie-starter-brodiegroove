# The Chert FaceTime CLI on this Windows laptop (the CLI refuses win32; this runs it through a preload that reports 'linux' and
# patches the two Windows-only file issues, see chert_win_shim.js).
#   .\team\scripts\chert.ps1 --version
#   .\team\scripts\chert.ps1 login --device        # prints a URL and a code; approve in a browser, pick the assigned project
#   .\team\scripts\chert.ps1 whoami
#   .\team\scripts\chert.ps1 doctor                # read-only; reads LIVEKIT_* from .env.local in the repo root
#   .\team\scripts\chert.ps1 dev --agent $env:AGENT_NAME     # claims the line (only during a booked slot); Ctrl-C releases it
# The worker must be running under the SAME dispatch name:  .\team\scripts\dance.ps1 -AgentName $env:AGENT_NAME
# Node: the CLI wants 22.12+, 24.x or 26.x; nvm-windows switches with `nvm use 24.18.0` (admin prompt). The CLI package is
# looked up under the active Node's global folder first, then the per-user npm folder.
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
. "$PSScriptRoot\env.ps1"
$rel = 'node_modules\@trychert\facetime\dist\src\cli\main.js'
$candidates = @()
try { $g = (& npm root -g 2>$null | Select-Object -First 1); if ($g) { $candidates += (Join-Path (Split-Path $g -Parent) $rel) } } catch {}
$candidates += (Join-Path $env:APPDATA "npm\$rel")
$candidates += (Join-Path 'C:\Program Files\nodejs' $rel)
$main = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $main) { Write-Error "Chert CLI not installed for this Node ($(& node --version)): npm install -g @trychert/facetime@0.1.0-beta.4"; return }
& node -r "$PSScriptRoot\chert_win_shim.js" $main @args
