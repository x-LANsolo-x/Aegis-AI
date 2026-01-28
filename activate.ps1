# Quick activation script for the Python 3.10 virtual environment
# Usage: .\activate.ps1

Write-Host "🚀 Activating Aegis-AI Python 3.10 environment..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1

Write-Host ""
Write-Host "✅ Environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "Quick commands:" -ForegroundColor Yellow
Write-Host "  pytest services/api/tests/ -v     # Run all tests"
Write-Host "  cd services/api && uvicorn app.main:app --reload     # Start API"
Write-Host "  deactivate                         # Exit environment"
Write-Host ""
