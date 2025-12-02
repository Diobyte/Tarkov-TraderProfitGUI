@echo off
cd /d "%~dp0"
echo Starting Tarkov Trader Profit...
echo.

:: Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is not found in your PATH.
    echo Please install PowerShell or fix your PATH environment variable.
    pause
    exit /b 1
)

:: Run the PowerShell script with Bypass policy
PowerShell -NoProfile -ExecutionPolicy Bypass -File "run.ps1"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application exited with an error.
    pause
)

