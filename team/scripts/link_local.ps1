# A phone-page link for ANY browser (Chrome on Android or a laptop) via our own https page.  .\team\scripts\link_local.ps1 [-Room name] [-Ip <laptop LAN ip>] [-Port 8443] [-Facing environment|user]
# Needs team\scripts\serve_web.py running (Terminal D) and the dance worker running (Terminal B). Prints the local URL (carries a token: do not post it)
# and, as a zero-setup fallback, LiveKit's hosted Meet page which also shows Brodie's video.
param([string]$Room = ('dance-' + (Get-Date -Format 'MMdd-HHmmss')), [string]$Ip = '', [int]$Port = 8443, [string]$Facing = 'environment')
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
if (-not $Ip) { $Ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Sort-Object InterfaceMetric | Select-Object -First 1).IPAddress }
$token = (& (Join-Path $root '.venv\Scripts\python.exe') -m facetime.mint_token --room $Room | Select-Object -First 1).Trim()
$enc = [uri]::EscapeDataString
"room: $Room"
"local page (phone or laptop Chrome; accept the certificate warning once):"
"  https://$Ip`:$Port/phone.html?url=$($enc.Invoke($env:LIVEKIT_URL))&token=$token&facing=$Facing"
"LiveKit Meet fallback (no local server needed; shows every video track):"
"  https://meet.livekit.io/custom?liveKitUrl=$($enc.Invoke($env:LIVEKIT_URL))&token=$token"
