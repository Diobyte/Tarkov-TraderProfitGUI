@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"

echo ==========================================
echo Tarkov Trader Profit - Collector Launcher
echo ==========================================
echo.

where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is not found in your PATH.
    pause
    exit /b 1
)

PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_collector.ps1"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Collector exited with error code: %errorlevel%
    pause
)
