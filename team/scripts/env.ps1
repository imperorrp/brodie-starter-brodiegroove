# Load .env.local (repo root) into THIS PowerShell session. Dot-source it:  . .\team\scripts\env.ps1
# Note: PowerShell can't hold an empty variable, so a line like AGENT_NAME= simply removes it. worker.ps1 passes --agent-name= instead.
$envFile = Join-Path $PSScriptRoot '..\..\.env.local'
if (-not (Test-Path $envFile)) { Write-Warning ".env.local not found at $envFile (copy .env.example and fill it in)"; return }
Get-Content $envFile | Where-Object { $_ -match '^\s*([^#=\s]+)\s*=\s*(.*)$' } | ForEach-Object {
  $name = $matches[1]; $value = $matches[2].Trim().Trim('"').Trim("'")
  if ($value -eq '') { Remove-Item "env:$name" -ErrorAction SilentlyContinue } else { Set-Item "env:$name" $value }
}
$env:PYTHONUTF8 = '1'   # the starter opens text files without an encoding; Windows would default to cp1252
