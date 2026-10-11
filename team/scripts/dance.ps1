# Terminal B in dance mode: the Brodie worker with the dance skill.  .\team\scripts\dance.ps1 [-AgentName <chert dispatch name>] [-Key test_squats] [-Mode learn|battle|telephone|class] [-NoVoice] [-Debug data\tmp\dance_live.mp4]
# No -AgentName = automatic dispatch (the phone page / fake bridge). With a name = the Chert line (then run chert dev --agent <name>).
param([string]$AgentName = '', [string]$Key = '', [string]$Mode = 'learn', [switch]$NoVoice, [string]$Debug = '', [string]$Tempo = '1.0', [string]$Rtt = '0.3')
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$py = Join-Path $root '.venv\Scripts\python.exe'
if ($Key) { $env:DANCE_KEY = $Key }
$env:DANCE_MODE = $Mode; $env:DANCE_TEMPO = $Tempo; $env:DANCE_RTT = $Rtt
if ($Debug) { $env:DANCE_DEBUG_MP4 = $Debug }
$pyArgs = @('-m', 'team.dance.run_worker', '--perception', 'off', '--persona', "$root\team\dance\persona.md", '--handlers', "$root\team\dance\live.py", '--plan-file', "$root\team\dance\plan.txt", '--log-caller', '--audio-queue-ms', '300', "--agent-name=$AgentName")
if (-not $NoVoice) { $pyArgs += @('--voice', 'gemini') }
$pyArgs += 'dev'
Set-Location $root
& $py @pyArgs
