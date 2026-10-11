# Terminal C: a phone link for a fresh room (start the worker first).  .\team\scripts\link.ps1 [-Room name]
# Prints three lines; open the THIRD on the phone. The link carries a token: don't post it anywhere.
param([string]$Room = ('team-' + (Get-Date -Format 'MMdd-HHmmss')))
. "$PSScriptRoot\env.ps1"
$root = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $root
& (Join-Path $root '.venv\Scripts\python.exe') -m facetime.mint_token --room $Room --url
