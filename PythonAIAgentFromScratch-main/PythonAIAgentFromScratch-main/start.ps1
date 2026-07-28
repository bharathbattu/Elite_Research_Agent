$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    Write-Host "Creating project environment..."
    python -m venv .venv
}

& $python -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing project dependencies..."
    & $python -m pip install --disable-pip-version-check -e .
}

Write-Host "Starting Elite Research Assistant at http://localhost:8501"
& $python -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
