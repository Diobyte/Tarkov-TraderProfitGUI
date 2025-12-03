# run_collector.ps1
$ErrorActionPreference = "Stop"
$ScriptPath = $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Tarkov Trader Profit - Data Collector Only" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- Data Directory Setup ---
function Initialize-DataDirectory {
    <#
    .SYNOPSIS
    Creates the data directory structure in Documents if it doesn't exist.
    #>
    
    # Determine data directory (check for custom env var)
    $DataDir = $env:TARKOV_DATA_DIR
    if (-not $DataDir) {
        $DataDir = Join-Path $env:USERPROFILE "Documents\TarkovTraderProfit"
    }
    
    # Create directory structure
    $Directories = @(
        $DataDir,
        (Join-Path $DataDir "exports"),
        (Join-Path $DataDir "logs")
    )
    
    foreach ($dir in $Directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    return $DataDir
}

# Initialize data directory
$DataDir = Initialize-DataDirectory
Write-Host "[INFO] Data directory: $DataDir" -ForegroundColor Cyan

# --- Helper: Find Python ---
function Get-PythonPath {
    param([string]$Version)
    
    # 1. Try 'py' launcher
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        try {
            $path = py -$Version -c "import sys; print(sys.executable)" 2>$null
            if ($path -and (Test-Path $path)) { return $path }
        } catch {}
    }

    # 2. Try 'python' in PATH
    if (Get-Command "python" -ErrorAction SilentlyContinue) {
        try {
            $verStr = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($verStr -eq $Version) {
                return (Get-Command "python").Source
            }
        } catch {}
    }
    return $null
}

# --- Step 1: Check for Compatible Python (3.11 - 3.13) ---
$PreferredVersion = "3.12"
$CompatibleVersions = @("3.13", "3.12", "3.11", "3.10")
$PythonPath = $null

# 0. Check specific hardcoded paths first (common for Winget/Store installs)
$CommonPaths = @(
    "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe",
    "$env:LOCALAPPDATA\Programs\Python\Python310\python.exe"
)
foreach ($p in $CommonPaths) {
    if (Test-Path $p) {
        $PythonPath = $p
        Write-Host "[INFO] Found Python at known path: $PythonPath" -ForegroundColor Green
        break
    }
}

# 1. Check Venv
$VenvPath = Join-Path $ScriptPath ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "[ERROR] Virtual environment not found. Please run 'run.ps1' first to set up the project." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Step 2: Launch Collector ---
Write-Host "[SUCCESS] Starting Collector in Standalone Mode..." -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray

# Run collector with --standalone flag
& $VenvPython "collector.py" --standalone
