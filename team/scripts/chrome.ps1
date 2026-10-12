# One command to test Brodie dance from Chrome: on this laptop (opens Chrome) or on an Android phone (prints a short link).
#   .\team\scripts\chrome.ps1                                # laptop: starts what is missing, opens Chrome on the page, front camera
#   .\team\scripts\chrome.ps1 -Android                       # phone on the same Wi-Fi: type the printed short link, back camera
#   .\team\scripts\chrome.ps1 -Restart -Key cc_huntrix -Mode battle   # restart the worker with another key / mode
#   .\team\scripts\chrome.ps1 -Restart -Mode class -Code 1771         # join a class (or a duel / chain) by code
# Options: -Key <library key> -Mode learn|battle|telephone|class -Facing user|environment -MinVisible 0.4 -Code 1234 -NoOpen -Restart -Record (save what Brodie shows, opt-in)
# Guide: team\CHROME_TEST_GUIDE.md
param([string]$Key = '', [string]$Mode = 'learn', [string]$Facing = '', [string]$MinVisible = '0.4', [string]$Code = '',
      [switch]$Android, [switch]$NoOpen, [switch]$Restart, [switch]$Record, [switch]$Demo, [int]$Port = 8443, [string]$Rtt = '0.3')
#   -Demo : the room skips the chat, offer and ready gates (one routine straight after the camera check); laptop rooms are arms-only
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { Write-Error "no venv at $py"; return }
if (-not $env:LIVEKIT_URL) { Write-Error '.env.local has no LIVEKIT_URL'; return }
if (-not $Facing) { $Facing = if ($Android) { 'environment' } else { 'user' } }
New-Item -ItemType Directory -Force (Join-Path $root 'data\tmp') | Out-Null

# 1. the https server for the page (self-signed certificate; the phone needs https for its camera)
$serve = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*serve_web.py*' }
if ($serve) { "https server: running (pid $($serve[0].ProcessId))" }
else {
  Start-Process -FilePath $py -ArgumentList @('team\scripts\serve_web.py', '--port', "$Port") -WorkingDirectory $root -WindowStyle Minimized
  Start-Sleep -Seconds 2; 'https server: started (minimised window)'
}

# 2. the dance worker (Brodie). A running one keeps its key and mode; -Restart replaces it.
$worker = Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and $_.CommandLine -like '*team.dance.run_worker*' }
if ($worker -and $Restart) { $worker | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }; $worker = $null; Start-Sleep -Seconds 1; 'worker: stopped the old one' }
if ($worker) { "worker: running (pid $($worker[0].ProcessId)) with the key and mode it was started with; use -Restart -Key ... -Mode ... to change them" }
else {
  $env:DANCE_KEY = $Key; $env:DANCE_MODE = $Mode; $env:DANCE_MIN_VISIBLE = $MinVisible; $env:DANCE_RTT = $Rtt
  if (-not $env:DANCE_MAX_POSES) { $env:DANCE_MAX_POSES = '1' }     # solo: 4 poses at 384 px cost 237 ms a frame on this laptop (4 fps)
  if (-not $env:DANCE_POSE_PX) { $env:DANCE_POSE_PX = '320' }
  if ($Record) { $env:DANCE_DEBUG_MP4 = 'data\tmp\dance_chrome.mp4' } else { Remove-Item env:DANCE_DEBUG_MP4 -ErrorAction SilentlyContinue }   # recording is opt-in
  $env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
  if (-not $env:BACKEND_URL) { $env:BACKEND_URL = 'fake' }
  if ($Code) { $env:DANCE_CODE = $Code } else { Remove-Item env:DANCE_CODE -ErrorAction SilentlyContinue }
  # Same dispatch name mint_link.py puts in the token (AGENT_NAME from .env.local; empty = automatic), or the room never reaches this worker.
  $args = @('-u', '-m', 'team.dance.run_worker', '--perception', 'off', '--voice', 'gemini', '--persona', 'team\dance\persona.md', '--handlers', 'team\dance\live.py',
            '--plan-file', 'team\dance\plan.txt', '--log-caller', '--audio-queue-ms', '300', "--agent-name=$($env:AGENT_NAME)", 'dev')
  Start-Process -FilePath $py -ArgumentList $args -WorkingDirectory $root
  "worker: started in a new window (key $Key, mode $Mode, min visible $MinVisible, dispatch $(if ($env:AGENT_NAME) { $env:AGENT_NAME } else { 'automatic' })$(if ($Code) { ", code $Code" })). Wait for 'registered worker' there."
  Start-Sleep -Seconds 12
}

# 3. a room token and the links (mint_link.py adds the named dispatch when AGENT_NAME is set, e.g. while the Chert line is up).
#    Room-name prefixes steer the session: -Demo = 'demo-' (one routine, no chat gates), laptop = 'desk-' (arms-only camera check).
$prefix = $(if ($Demo) { 'demo-' } else { '' }) + $(if ($Android) { 'dance' } else { 'desk' })
$mintArgs = @('team\scripts\mint_link.py', '--prefix', $prefix, '--facing', $Facing, '--port', "$Port", '--json')
if (-not $Android) { $mintArgs += '--localhost' }
$link = (& $py @mintArgs | Select-Object -Last 1) | ConvertFrom-Json
if (-not $link.page) { Write-Error 'could not mint a token (LIVEKIT_API_KEY / LIVEKIT_API_SECRET in .env.local?)'; return }
$url = $link.page; $meet = $link.meet; $room = $link.room
$ip = (Get-NetIPAddress -AddressFamily IPv4 | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.*' -and $_.PrefixOrigin -ne 'WellKnown' } | Sort-Object InterfaceMetric | Select-Object -First 1).IPAddress
''
"room: $room   dispatch: $(if ($link.agent) { $link.agent } else { 'automatic' })   (the links carry a one-hour token for this room only; do not post them)"
"short link: $($link.short)"
if ($Android) {
  ''
  'On the phone: open the short link in Chrome, tap Advanced then Proceed on the certificate warning, allow camera and mic, tap Join.'
  'Prop the phone at knee height, two to three metres away, whole body in frame, volume up.'
  "Fallback without our page (also shows Brodie's video):  $meet"
} else {
  if ($NoOpen) { "laptop link:  $url" }
  else {
    $chrome = @('C:\Program Files\Google\Chrome\Application\chrome.exe', 'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe') | Where-Object { Test-Path $_ } | Select-Object -First 1
    if ($chrome) { Start-Process -FilePath $chrome -ArgumentList "`"$url`"" } else { Start-Process $url }
    'Chrome opened: tap Advanced then Proceed on the certificate warning, allow camera and mic, tap Join. Step back so your head and hands are in frame.'
  }
}
''
'Then: say your name ("I am Maya"), watch the ghost twice, dance on the count-in. "again", "slower", "faster", "next", "stop" between phrases.'
'What you saw is saved to data\tmp\dance_chrome_<time>.mp4 when the call ends; the worker window shows every scored phrase as a [dance] line.'
