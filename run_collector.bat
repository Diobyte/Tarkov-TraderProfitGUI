@echo off
cd /d "%~dp0"
PowerShell -NoProfile -ExecutionPolicy Bypass -File "run_collector.ps1"
pause
