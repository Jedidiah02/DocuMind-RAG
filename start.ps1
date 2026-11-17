Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host "         Starting DocuMind RAG Assistant               " -ForegroundColor Cyan
Write-Host "=======================================================" -ForegroundColor Cyan
Write-Host ""

Set-Location -Path $PSScriptRoot

Write-Host "Opening http://localhost:8080 in default browser..." -ForegroundColor Green
Start-Process "http://localhost:8080"

python app.py 8080
