@echo off
cd /d "%~dp0"
PowerShell -NoProfile -ExecutionPolicy Bypass -File "run.ps1"
pause
