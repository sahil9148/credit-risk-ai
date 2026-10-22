@echo off
title Credit Risk AI Server
echo ===================================
echo   Credit Risk AI - Starting Server
echo ===================================
echo.
cd /d "%~dp0"
python -c "import uvicorn; uvicorn.run('app.main:app', host='127.0.0.1', port=8000, log_level='info')"
pause
