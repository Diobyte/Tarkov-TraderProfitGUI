# run.ps1
$ErrorActionPreference = "Stop"
$ScriptPath = $PSScriptRoot

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Tarkov Trader Profit - Auto-Setup & Launch" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# --- Debug: Check Default Browser Configuration ---
function Test-BrowserConfiguration {
    <#
    .SYNOPSIS
    Checks if Windows has a properly configured default browser for HTTP URLs.
    #>
    Write-Host "[DEBUG] Checking browser configuration..." -ForegroundColor Gray
    
    try {
        # Check HTTP URL handler in registry
        $httpHandler = Get-ItemProperty -Path "HKCU:\Software\Microsoft\Windows\Shell\Associations\UrlAssociations\http\UserChoice" -ErrorAction SilentlyContinue
        if ($httpHandler -and $httpHandler.ProgId) {
            Write-Host "[DEBUG] Default HTTP handler: $($httpHandler.ProgId)" -ForegroundColor Gray
        } else {
            Write-Host "[WARN] No default HTTP browser handler found. Browser may not open automatically." -ForegroundColor Yellow
            Write-Host "[INFO] To fix: Windows Settings > Apps > Default apps > Set defaults by link type > HTTP" -ForegroundColor Cyan
        }
    } catch {
        Write-Host "[DEBUG] Could not check browser registry settings" -ForegroundColor Gray
    }
}

Test-BrowserConfiguration

# --- Data Directory Setup & Migration ---
function Initialize-DataDirectory {
    <#
    .SYNOPSIS
    Creates the data directory structure in Documents and migrates existing files.
    #>
    
    # Determine data directory (check for custom env var)
    $DataDir = $env:TARKOV_DATA_DIR
    if (-not $DataDir) {
        $DataDir = Join-Path $env:USERPROFILE "Documents\TarkovTraderProfit"
    }
    
    Write-Host "[INFO] Data directory: $DataDir" -ForegroundColor Cyan
    
    # Create directory structure
    $Directories = @(
        $DataDir,
        (Join-Path $DataDir "exports"),
        (Join-Path $DataDir "logs")
    )
    
    foreach ($dir in $Directories) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "[INFO] Created directory: $dir" -ForegroundColor Green
        }
    }
    
    # Files to migrate from project directory to data directory
    $FilesToMigrate = @(
        @{ Source = "tarkov_data.db"; Dest = "tarkov_data.db" },
        @{ Source = "ml_model_state.pkl"; Dest = "ml_model_state.pkl" },
        @{ Source = "ml_learned_history.json"; Dest = "ml_learned_history.json" },
        @{ Source = "collector.pid"; Dest = "collector.pid" },
        @{ Source = "collector_standalone.pid"; Dest = "collector_standalone.pid" }
    )
    
    # Log files to migrate
    $LogFilesToMigrate = @(
        "app.log",
        "collector.log", 
        "collector_startup.log",
        "streamlit_server.log"
    )
    
    $MigratedCount = 0
    
    # Migrate main files
    foreach ($file in $FilesToMigrate) {
        $SourcePath = Join-Path $ScriptPath $file.Source
        $DestPath = Join-Path $DataDir $file.Dest
        
        if ((Test-Path $SourcePath) -and (-not (Test-Path $DestPath))) {
            try {
                Move-Item -Path $SourcePath -Destination $DestPath -Force
                Write-Host "[MIGRATE] Moved $($file.Source) to Documents" -ForegroundColor Yellow
                $MigratedCount++
            } catch {
                Write-Host "[WARN] Could not migrate $($file.Source): $_" -ForegroundColor Yellow
            }
        } elseif ((Test-Path $SourcePath) -and (Test-Path $DestPath)) {
            # Both exist - keep the newer one, delete the old project file
            $SourceInfo = Get-Item $SourcePath
            $DestInfo = Get-Item $DestPath
            
            if ($SourceInfo.LastWriteTime -gt $DestInfo.LastWriteTime) {
                # Source is newer, replace destination
                try {
                    Move-Item -Path $SourcePath -Destination $DestPath -Force
                    Write-Host "[MIGRATE] Updated $($file.Dest) (source was newer)" -ForegroundColor Yellow
                    $MigratedCount++
                } catch {
                    # If can't move (locked), just delete old project file
                    Remove-Item $SourcePath -Force -ErrorAction SilentlyContinue
                }
            } else {
                # Destination is newer or same, just delete old project file
                Remove-Item $SourcePath -Force -ErrorAction SilentlyContinue
                Write-Host "[CLEANUP] Removed old $($file.Source) from project folder" -ForegroundColor Gray
            }
        }
    }
    
    # Migrate log files
    $LogsDir = Join-Path $DataDir "logs"
    foreach ($logFile in $LogFilesToMigrate) {
        $SourcePath = Join-Path $ScriptPath $logFile
        $DestPath = Join-Path $LogsDir $logFile
        
        if (Test-Path $SourcePath) {
            try {
                if (Test-Path $DestPath) {
                    # Append old logs to new location, then delete
                    Get-Content $SourcePath | Add-Content $DestPath
                    Remove-Item $SourcePath -Force
                } else {
                    Move-Item -Path $SourcePath -Destination $DestPath -Force
                }
                Write-Host "[MIGRATE] Moved $logFile to logs folder" -ForegroundColor Yellow
                $MigratedCount++
            } catch {
                # Log files might be locked, just try to delete
                Remove-Item $SourcePath -Force -ErrorAction SilentlyContinue
            }
        }
    }
    
    if ($MigratedCount -gt 0) {
        Write-Host "[INFO] Migrated $MigratedCount file(s) to new data directory" -ForegroundColor Green
    }
    
    return $DataDir
}

# Initialize data directory and migrate files
$DataDir = Initialize-DataDirectory

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
$PreferredVersion = "3.13"
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
            # Install Python 3.13
            winget install -e --id Python.Python.3.13 --scope user --accept-source-agreements --accept-package-agreements
            
            # Refresh Environment Variables (User & System) correctly
            $MachinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
            $UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
            $NewPath = "$MachinePath;$UserPath"
            [Environment]::SetEnvironmentVariable("Path", $NewPath, "Process")
            
            # Re-check via standard detection
            $PythonPath = Get-PythonPath -Version $PreferredVersion

            # Fallback: Check default install location for Python 3.13
            if (-not $PythonPath) {
                $DefaultInstall = "$env:LOCALAPPDATA\Programs\Python\Python313\python.exe"
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
    Write-Host "Please manually install Python 3.13 from https://www.python.org/downloads/"
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

# --- Step 4.5: Pre-Launch Cleanup ---
Write-Host "[INFO] Cleaning up previous sessions..." -ForegroundColor Cyan

# Helper to kill process by PID file
function Kill-ProcessByPidFile {
    param([string]$PidFilePath)
    if (Test-Path $PidFilePath) {
        try {
            $PidVal = Get-Content $PidFilePath -ErrorAction SilentlyContinue
            if ($PidVal -and (Get-Process -Id $PidVal -ErrorAction SilentlyContinue)) {
                Stop-Process -Id $PidVal -Force -ErrorAction SilentlyContinue
                Write-Host "[INFO] Stopped previous process (PID: $PidVal)" -ForegroundColor Yellow
            }
            Remove-Item $PidFilePath -Force -ErrorAction SilentlyContinue
        } catch {
            Write-Host "[WARN] Failed to clean up PID file" -ForegroundColor Yellow
        }
    }
}

# Kill processes using new data directory paths
Kill-ProcessByPidFile (Join-Path $DataDir "collector.pid")
Kill-ProcessByPidFile (Join-Path $DataDir "collector_standalone.pid")

# Also check old locations in project folder (for backward compatibility)
Kill-ProcessByPidFile (Join-Path $ScriptPath "collector.pid")
Kill-ProcessByPidFile (Join-Path $ScriptPath "collector_standalone.pid")

# Give the OS a moment to release file locks
Start-Sleep -Milliseconds 500

# --- Step 5: Launch Dashboard ---
Write-Host "[SUCCESS] Starting Dashboard..." -ForegroundColor Green
Write-Host "Data stored in: $DataDir" -ForegroundColor Gray
Write-Host "You can close this window to stop the application." -ForegroundColor Gray

# Log file location (now in data directory)
$LogsDir = Join-Path $DataDir "logs"
$StreamlitLog = Join-Path $LogsDir "streamlit_server.log"

# Clear old streamlit log
if (Test-Path $StreamlitLog) {
    try { Clear-Content $StreamlitLog -ErrorAction SilentlyContinue } catch {}
}

# Use Start-Process to run python so we can redirect stdout/stderr reliably without PowerShell pipe issues
# We use -Wait to keep the script running until Streamlit exits
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = $VenvPython
$ProcessInfo.Arguments = "-u -m streamlit run `"$($ScriptPath)\app.py`" --server.headless=false --server.address=localhost"
$ProcessInfo.RedirectStandardOutput = $true
$ProcessInfo.RedirectStandardError = $true
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $false
$ProcessInfo.WorkingDirectory = $ScriptPath

$Process = New-Object System.Diagnostics.Process
$Process.StartInfo = $ProcessInfo

# Set up event handlers to write to file and console
$Script:BrowserOpened = $false
$LogHandler = {
    param($sender, $e)
    if ($e.Data) {
        $e.Data | Out-File -FilePath $StreamlitLog -Append -Encoding utf8
        Write-Host $e.Data

        # Auto-open browser when URL is detected
        if (-not $Script:BrowserOpened -and $e.Data -match "Local URL:\s+(http://\S+)") {
            $Script:BrowserOpened = $true
            $url = $matches[1]
            Write-Host "[INFO] Opening browser at $url..." -ForegroundColor Green
            try {
                # Try multiple methods to open browser for better compatibility
                # Method 1: Start-Process with explicit shell execute (most reliable)
                $browserOpened = $false
                try {
                    Start-Process -FilePath $url -ErrorAction Stop
                    $browserOpened = $true
                } catch {
                    Write-Host "[DEBUG] Start-Process direct URL failed, trying alternative methods..." -ForegroundColor Yellow
                }
                
                # Method 2: Use cmd /c start (fallback for Windows URL association issues)
                if (-not $browserOpened) {
                    try {
                        Start-Process cmd -ArgumentList "/c", "start", $url -WindowStyle Hidden -ErrorAction Stop
                        $browserOpened = $true
                    } catch {
                        Write-Host "[DEBUG] cmd /c start failed, trying explorer..." -ForegroundColor Yellow
                    }
                }
                
                # Method 3: Use explorer.exe (another Windows fallback)
                if (-not $browserOpened) {
                    try {
                        Start-Process explorer.exe -ArgumentList $url -ErrorAction Stop
                        $browserOpened = $true
                    } catch {
                        Write-Host "[DEBUG] explorer.exe method also failed." -ForegroundColor Yellow
                    }
                }
                
                if (-not $browserOpened) {
                    Write-Host "[WARN] Could not open browser automatically. Please open manually: $url" -ForegroundColor Yellow
                    Write-Host "[INFO] To fix this, set a default browser in Windows Settings > Apps > Default apps" -ForegroundColor Cyan
                }
            } catch {
                Write-Host "[WARN] Could not open browser automatically. Please open manually: $url" -ForegroundColor Yellow
            }
        }
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
    $PidFile = Join-Path $DataDir "collector.pid"
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
