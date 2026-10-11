# Terminal A: the team backend on http://127.0.0.1:8787  (.\team\scripts\backend.ps1)
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
& (Join-Path $root '.venv\Scripts\python.exe') -m uvicorn app:app --app-dir team/backend --port 8787 --reload
