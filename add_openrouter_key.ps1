# Adds OPENROUTER_API_KEY / OPENROUTER_LLM_MODEL to .env, reading the key from
# the User-scope environment variable so it is never typed or echoed here.
# Idempotent: removes any existing OPENROUTER_* lines first, so re-running is
# safe and will not create duplicates.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$key = [Environment]::GetEnvironmentVariable("OPENROUTER_API_KEY", "User")
if (-not $key) {
    Write-Host "No key found at User scope. Set it first:"
    Write-Host '  [Environment]::SetEnvironmentVariable("OPENROUTER_API_KEY", "sk-or-v1-...", "User")'
    exit 1
}
if (-not (Test-Path .env)) {
    Write-Host "ERROR: .env not found in $PSScriptRoot"
    exit 1
}

# Strip any prior OPENROUTER lines so this is safe to run repeatedly.
$kept = Get-Content .env | Where-Object { $_ -notmatch '^\s*OPENROUTER_' }
$new  = $kept + "OPENROUTER_API_KEY=$key" + "OPENROUTER_LLM_MODEL=cohere/north-mini-code:free"
Set-Content -Path .env -Value $new

Write-Host "Wrote OPENROUTER_API_KEY (length $($key.Length)) and OPENROUTER_LLM_MODEL to .env"
Write-Host "Verify with:"
Write-Host '  Select-String -Path .env -Pattern ''OPENROUTER'''
