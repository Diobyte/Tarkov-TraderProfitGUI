# run.ps1
$ErrorActionPreference = "Stop"
$ScriptPath = $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Tarkov Trader Profit - Auto-Setup & Launch" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

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

if (-not $PythonPath) {
    foreach ($ver in $CompatibleVersions) {
        $path = Get-PythonPath -Version $ver
        if ($path) {
            $PythonPath = $path
            Write-Host "[INFO] Found compatible Python $ver at: $PythonPath" -ForegroundColor Green
            break
        }
    }
}

# --- Step 2: Install Python if Missing ---
if (-not $PythonPath) {
    Write-Host "[WARN] No compatible Python version found." -ForegroundColor Yellow
    Write-Host "[INFO] Attempting to install Python $PreferredVersion via Winget..." -ForegroundColor Yellow
    
    if (Get-Command "winget" -ErrorAction SilentlyContinue) {
        try {
            # Install Python 3.12
            winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
            
            # Refresh Environment Variables (User & System) correctly
            $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
            $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
            $NewPath = "$MachinePath;$UserPath"
            [Environment]::SetEnvironmentVariable("Path", $NewPath, "Process")
            
            # Re-check via standard detection
            $PythonPath = Get-PythonPath -Version $PreferredVersion

            # Fallback: Check default install location for Python 3.12
            if (-not $PythonPath) {
                $DefaultInstall = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
                if (Test-Path $DefaultInstall) {
                    $PythonPath = $DefaultInstall
                }
            }
        } catch {
            Write-Host "[ERROR] Winget installation failed." -ForegroundColor Red
        }
    } else {
        Write-Host "[ERROR] Winget package manager not found." -ForegroundColor Red
    }
}

if (-not $PythonPath) {
    Write-Host "[CRITICAL] Could not find or install Python." -ForegroundColor Red
    Write-Host "Please manually install Python 3.12 from https://www.python.org/downloads/"
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Step 3: Setup Virtual Environment ---
$VenvPath = Join-Path $ScriptPath ".venv"

# Check if venv exists and is valid
if (Test-Path $VenvPath) {
    $VenvPython = Join-Path $VenvPath "Scripts\python.exe"
    $VenvInvalid = $true
    
    if (Test-Path $VenvPython) {
        try {
            # Check version of venv python
            $VenvVer = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($CompatibleVersions -contains $VenvVer) {
                $VenvInvalid = $false
                Write-Host "[INFO] Existing .venv is valid (Python $VenvVer)." -ForegroundColor Green
            } else {
                Write-Host "[WARN] Existing .venv uses incompatible Python $VenvVer. Recreating..." -ForegroundColor Yellow
            }
        } catch {
            Write-Host "[WARN] Existing .venv is broken. Recreating..." -ForegroundColor Yellow
        }
    }

    if ($VenvInvalid) {
        Remove-Item -Recurse -Force $VenvPath -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Host "[INFO] Creating virtual environment using $PythonPath..." -ForegroundColor Cyan
    & $PythonPath -m venv $VenvPath
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# --- Step 4: Install Dependencies ---
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "[INFO] Installing/Updating dependencies..." -ForegroundColor Cyan
# Upgrade pip & build tools
& $VenvPython -m pip install --upgrade pip wheel setuptools cmake

# Install PyArrow binary (Critical fix)
& $VenvPython -m pip install --only-binary=:all: pyarrow

# Install requirements
& $VenvPython -m pip install -r (Join-Path $ScriptPath "requirements.txt")

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Dependency installation failed." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# --- Step 5: Launch Dashboard ---
Write-Host "[SUCCESS] Starting Dashboard..." -ForegroundColor Green
Write-Host "Logs are being saved to 'streamlit_server.log'" -ForegroundColor Gray
Write-Host "You can close this window to stop the application." -ForegroundColor Gray

# Clean up previous session logs
$FilesToClean = @("app.log", "collector.log", "collector.pid", "streamlit_server.log")
foreach ($file in $FilesToClean) {
    $fullPath = Join-Path $ScriptPath $file
    if (Test-Path $fullPath) {
        try {
            Remove-Item $fullPath -Force -ErrorAction Stop
        } catch {
            Write-Host "[WARN] Could not delete $file (locked?). Clearing content instead." -ForegroundColor Yellow
            try { Clear-Content $fullPath -ErrorAction SilentlyContinue } catch {}
        }
    }
}

$StreamlitLog = Join-Path $ScriptPath "streamlit_server.log"

# Use Start-Process to run python so we can redirect stdout/stderr reliably without PowerShell pipe issues
# We use -Wait to keep the script running until Streamlit exits
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = $VenvPython
$ProcessInfo.Arguments = "-u -m streamlit run `"$($ScriptPath)\app.py`" --server.headless=false"
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardError = $true
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $false
$ProcessInfo.WorkingDirectory = $ScriptPath

$Process = New-Object System.Diagnostics.Process
$Process.StartInfo = $ProcessInfo

# Set up event handlers to write to file and console
$LogHandler = {
    param($sender, $e)
    if ($e.Data) {
        $e.Data | Out-File -FilePath $StreamlitLog -Append -Encoding utf8
        Write-Host $e.Data
    }
}

Register-ObjectEvent -InputObject $Process -EventName OutputDataReceived -Action $LogHandler | Out-Null
Register-ObjectEvent -InputObject $Process -EventName ErrorDataReceived -Action $LogHandler | Out-Null

# Clear old log
"" | Out-File -FilePath $StreamlitLog -Encoding utf8

try {
    $Process.Start() | Out-Null
    $Process.BeginOutputReadLine()
    $Process.BeginErrorReadLine()

    $Process.WaitForExit()
} finally {
    if (-not $Process.HasExited) {
        $Process.Kill()
    }

    # --- Cleanup Background Collector ---
    $PidFile = Join-Path $ScriptPath "collector.pid"
    if (Test-Path $PidFile) {
        try {
            $CollectorPid = Get-Content $PidFile -ErrorAction SilentlyContinue
            if ($CollectorPid -and (Get-Process -Id $CollectorPid -ErrorAction SilentlyContinue)) {
                Stop-Process -Id $CollectorPid -Force -ErrorAction SilentlyContinue
                Write-Host "Stopped background collector (PID: $CollectorPid)" -ForegroundColor Yellow
            }
            Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
        } catch {
            # Ignore errors during cleanup
        }
    }
}

if ($Process.ExitCode -ne 0) {
    Write-Host "[ERROR] Dashboard crashed with exit code $($Process.ExitCode)." -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
