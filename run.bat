@echo off
setlocal EnableExtensions DisableDelayedExpansion

:: Ensure we are in the script's directory
cd /d "%~dp0"

echo ==========================================
echo Tarkov Trader Profit - Launcher
echo ==========================================
echo.
echo Working Directory: %CD%
echo.

:: Check if PowerShell is available
where powershell >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] PowerShell is not found in your PATH.
    echo Please install PowerShell or fix your PATH environment variable.
    pause
    exit /b 1
)

:: Run the PowerShell script
:: We use -ExecutionPolicy Bypass to allow running scripts
:: We pass the full path to the script to avoid path issues
echo Launching PowerShell script...
PowerShell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] The application exited with error code: %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Application finished.
pause

