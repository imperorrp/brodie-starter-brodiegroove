# Terminal B: the Brodie coach with our skill.  .\team\scripts\worker.ps1 [-AgentName <chert dispatch name>] [-NoVoice] [-Extra '--plan "squats:5"']
# No -AgentName = automatic dispatch (the phone page). With a name = the Chert line (then run chert dev --agent <name>).
param([string]$AgentName = '', [switch]$NoVoice, [string]$Extra = '')
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
$skill = Join-Path $root 'team\skill'
$py = Join-Path $root '.venv\Scripts\python.exe'
$pyArgs = @('-m', 'facetime.worker', '--persona', "$skill\persona.md", '--plan-file', "$skill\plan.txt", '--handlers', "$skill\handlers.py", "--agent-name=$AgentName")
if (-not $NoVoice) { $pyArgs += @('--voice', 'gemini') }
if ($Extra) { $pyArgs += ($Extra -split ' ') }
$pyArgs += 'dev'
Set-Location $root
& $py @pyArgs
