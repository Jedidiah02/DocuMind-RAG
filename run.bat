@echo off
title DocuMind RAG Assistant
echo =======================================================
echo          Starting DocuMind RAG Assistant
echo =======================================================
echo.
cd /d "%~dp0"

echo Launching DocuMind on http://localhost:8080...
start "" "http://localhost:8080"
python app.py 8080

pause
