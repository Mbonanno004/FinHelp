# Starts FinHelp for development: the Python API (in its own window) and the website.
# Usage, from the project folder:   .\start.ps1
# If PowerShell blocks scripts:     powershell -ExecutionPolicy Bypass -File start.ps1

$root = $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
$web = Join-Path $root "web"

if (-not (Test-Path $python)) {
    Write-Host "No .venv found. Run the setup steps in README.md first." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path (Join-Path $root "data\generated_transactions.csv"))) {
    Write-Host "Creating the training data (a few seconds)..." -ForegroundColor Cyan
    & $python (Join-Path $root "src\generate_data.py")
}
if (-not (Test-Path (Join-Path $web "node_modules"))) {
    Write-Host "Installing website packages (first run only)..." -ForegroundColor Cyan
    npm --prefix $web install
}

Write-Host "Starting the Python API on http://localhost:8000 (new window)..." -ForegroundColor Cyan
Start-Process -FilePath $python -WorkingDirectory $root -ArgumentList @(
    "-m", "uvicorn", "api.main:app", "--port", "8000"
)

Write-Host "Starting the website on http://localhost:5173 ..." -ForegroundColor Cyan
Start-Process "http://localhost:5173"
npm --prefix $web run dev
