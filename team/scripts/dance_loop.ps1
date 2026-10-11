# The dance worker under a restart loop, for the Chert line. The LiveKit native library on Windows can panic when a room is
# deleted while another is connecting ("timed out waiting for ReadyForRoomEventRequest"); the worker then exits and the line
# would decline calls. This loop starts it again within 3 s. Stop it with Ctrl-C (or create data\tmp\worker_stop).
#   .\team\scripts\dance_loop.ps1 -AgentName dance-battle-agent-ivyhacks [-Mode learn]
#   -MaxPoses 4 enables squad mode (2-4 dancers in one camera); the default 1 keeps pose at ~60 ms a frame on this laptop
#   (4 poses at 384 px measured 237 ms a frame = 4 fps and a 64% tracking rate, which breaks timing scores).
param([string]$AgentName = $env:AGENT_NAME, [string]$Mode = 'learn', [int]$MaxPoses = 1, [int]$PosePx = 320)
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
$py = Join-Path $root '.venv\Scripts\python.exe'
$env:DANCE_MODE = $Mode; $env:PYTHONUTF8 = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:DANCE_MAX_POSES = "$MaxPoses"; $env:DANCE_POSE_PX = "$PosePx"
if (-not $env:BACKEND_URL) { $env:BACKEND_URL = 'fake' }
$stop = Join-Path $root 'data\tmp\worker_stop'
if (Test-Path $stop) { Remove-Item $stop }
$n = 0
while (-not (Test-Path $stop) -and $n -lt 50) {
  $n++
  "=== worker start #$n $(Get-Date -Format HH:mm:ss) (agent '$AgentName')"
  & $py -u -m team.dance.run_worker --perception off --voice gemini --persona team\dance\persona.md --handlers team\dance\live.py --plan-file team\dance\plan.txt --log-caller --audio-queue-ms 300 "--agent-name=$AgentName" --leave-grace-s 5 dev
  "=== worker exited (code $LASTEXITCODE) $(Get-Date -Format HH:mm:ss); restarting in 3 s"
  Start-Sleep -Seconds 3
}
