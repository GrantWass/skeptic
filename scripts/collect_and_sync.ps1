<#
.SYNOPSIS
    Collect Polymarket price data all day and sync to GitHub every 30 minutes.

.DESCRIPTION
    1. Runs collect_prices.py continuously (auto-restarts on crash).
    2. Every 30 minutes commits and pushes data/prices/ + data/sessions.db
       to GitHub via Git LFS.
    3. Keeps your PC awake using the Windows SetThreadExecutionState API.
    4. On Ctrl+C a final sync is always performed before exit.

    PREREQUISITES
    -------------
    • Git for Windows   https://git-scm.com/download/win
    • Git LFS           https://git-lfs.com  (or: winget install GitHub.GitLFS)
    • Python 3 with the project's .venv created, OR Python 3 on PATH
    • Git remote configured and authenticated (HTTPS credential manager or SSH key)

    FIRST RUN
    ---------
    1. Open PowerShell in the repo root.
    2. Allow the script to run for this session:
           Set-ExecutionPolicy -Scope Process Bypass
    3. Start collecting:
           .\scripts\collect_and_sync.ps1
       Or with specific assets:
           .\scripts\collect_and_sync.ps1 -CollectArgs "--assets BTC ETH SOL"

    HOW TO STOP
    -----------
    Press Ctrl+C once. The script will stop the collector, do a final git push,
    then exit cleanly.

.PARAMETER SyncIntervalMinutes
    How often (minutes) to commit and push new data. Default: 30.

.PARAMETER CollectArgs
    Extra arguments forwarded to collect_prices.py, e.g. "--assets BTC ETH --interval 2".

.PARAMETER Branch
    Git branch/ref to push to. Default: HEAD (the currently checked-out branch).
#>
param(
    [int]   $SyncIntervalMinutes = 30,
    [string]$CollectArgs         = "",
    [string]$Branch              = "HEAD"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
$ROOT    = Split-Path -Parent $PSScriptRoot
$COLLECT = Join-Path $PSScriptRoot "collect_prices.py"
$LOG     = Join-Path $ROOT "data\collect_and_sync.log"

New-Item -ItemType Directory -Force -Path "$ROOT\data" | Out-Null
Set-Content $LOG ""   # reset log on each run

# ─────────────────────────────────────────────────────────────────────────────
# Logging  (console + append to log file; the log file is git-ignored)
# ─────────────────────────────────────────────────────────────────────────────
function Log([string]$msg) {
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $msg"
    Write-Host $line
    Add-Content $LOG $line
}

# ─────────────────────────────────────────────────────────────────────────────
# Prevent Windows from sleeping while the script runs
# ─────────────────────────────────────────────────────────────────────────────
Add-Type -TypeDefinition @"
using System.Runtime.InteropServices;
public static class WakePC {
    [DllImport("kernel32.dll")]
    public static extern uint SetThreadExecutionState(uint f);
    public const uint ES_CONTINUOUS      = 0x80000000u;
    public const uint ES_SYSTEM_REQUIRED = 0x00000001u;
}
"@ -ErrorAction SilentlyContinue   # silently skip if already compiled in session

[WakePC]::SetThreadExecutionState([WakePC]::ES_CONTINUOUS -bor [WakePC]::ES_SYSTEM_REQUIRED) | Out-Null
Log "Sleep prevention active (SetThreadExecutionState)."

# ─────────────────────────────────────────────────────────────────────────────
# Find Python 3  (prefers .venv)
# ─────────────────────────────────────────────────────────────────────────────
function Find-Python {
    $venv = Join-Path $ROOT ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    foreach ($cmd in "python", "python3") {
        try {
            if ((& $cmd --version 2>&1) -match "Python 3") { return $cmd }
        } catch {}
    }
    return $null
}

$PY = Find-Python
if (-not $PY) {
    Log "ERROR: Python 3 not found. Install Python 3 or run: python -m venv .venv"
    exit 1
}
Log "Python: $PY"

# ─────────────────────────────────────────────────────────────────────────────
# One-time Git LFS initialisation (runs fast on subsequent calls)
# ─────────────────────────────────────────────────────────────────────────────
function Initialize-GitLFS {
    $lfs = Get-Command git-lfs -ErrorAction SilentlyContinue
    if (-not $lfs) {
        Log "WARNING: git-lfs not found. Install from https://git-lfs.com and re-run."
        Log "         Data will still be pushed but WITHOUT LFS (CSV files inline in git)."
        return
    }

    Push-Location $ROOT
    try {
        git lfs install --local 2>&1 | Out-Null
        git lfs track "data/prices/*.csv" 2>&1 | Out-Null
        git lfs track "data/sessions.db"  2>&1 | Out-Null

        # Commit .gitattributes if it changed
        $dirty = (git status --porcelain ".gitattributes" 2>&1) -ne ""
        if ($dirty) {
            git add ".gitattributes" | Out-Null
            git commit -m "chore: track price data via Git LFS" | Out-Null
            git push -u origin $Branch 2>&1 | ForEach-Object { Log "lfs-init: $_" }
            Log "Git LFS .gitattributes committed and pushed."
        } else {
            Log "Git LFS already configured."
        }
    } finally {
        Pop-Location
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Commit and push new data
# ─────────────────────────────────────────────────────────────────────────────
function Sync-Data([string]$label) {
    Push-Location $ROOT
    try {
        # Force-add even though data/ is in .gitignore; LFS filter intercepts large files.
        if (Test-Path "data\prices") {
            git add -f "data/prices/" 2>&1 | Out-Null
        }
        if (Test-Path "data\sessions.db") {
            git add -f "data/sessions.db" 2>&1 | Out-Null
        }

        $changed = git diff --cached --name-only 2>&1
        if ($changed) {
            $ts  = Get-Date -Format "yyyy-MM-dd HH:mm"
            $msg = "data: $label $ts"
            git commit -m $msg 2>&1 | ForEach-Object { Log "git: $_" }
            git push -u origin $Branch 2>&1 | ForEach-Object { Log "git: $_" }
            Log "Pushed $(@($changed).Count) file(s) to GitHub."
        } else {
            Log "Sync: no new data to commit."
        }
    } catch {
        Log "Sync error: $_"
    } finally {
        Pop-Location
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Launch collect_prices.py as a child process
# Output goes directly to this console window (no extra disk writes).
# PYTHONUNBUFFERED=1 ensures you see log lines in real time.
# ─────────────────────────────────────────────────────────────────────────────
function Start-Collector {
    $env:PYTHONUNBUFFERED = "1"
    $argList = @($COLLECT)
    if ($CollectArgs) { $argList += $CollectArgs -split "\s+" | Where-Object { $_ } }

    return Start-Process `
        -FilePath         $PY `
        -ArgumentList     $argList `
        -WorkingDirectory $ROOT `
        -NoNewWindow `
        -PassThru
}

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
Log "=== collect_and_sync starting ==="
Log "Sync interval : every $SyncIntervalMinutes minute(s)"
Log "Push target   : $Branch"
Log "Press Ctrl+C to stop (a final sync will run automatically)."

Initialize-GitLFS

$proc        = $null
$restarts    = 0
$maxRestarts = 100
$lastSync    = Get-Date   # start the clock; first push after $SyncIntervalMinutes

try {
    while ($restarts -lt $maxRestarts) {
        # ── (Re)start the Python collector if it isn't running ────────────────
        if ($null -eq $proc -or $proc.HasExited) {
            if ($null -ne $proc) {
                $exitCode = $proc.ExitCode
                $restarts++
                Log "Collector exited (code $exitCode). Restart $restarts/$maxRestarts in 5 s..."
                Start-Sleep -Seconds 5
                if ($restarts -ge $maxRestarts) {
                    Log "ERROR: hit max restarts ($maxRestarts). Check the log and fix the issue."
                    break
                }
            } else {
                Log "Launching collector..."
            }
            $proc = Start-Collector
            Log "Collector running (PID $($proc.Id))."
        }

        # ── Periodic git sync ─────────────────────────────────────────────────
        if (((Get-Date) - $lastSync).TotalMinutes -ge $SyncIntervalMinutes) {
            Log "=== Scheduled sync ==="
            Sync-Data "auto-sync"
            $lastSync = Get-Date
        }

        Start-Sleep -Seconds 15
    }
} finally {
    # ── Cleanup: stop the collector then do one last push ─────────────────────
    if ($null -ne $proc -and -not $proc.HasExited) {
        Log "Stopping collector (PID $($proc.Id))..."
        try { $proc.Kill(); $proc.WaitForExit(5000) | Out-Null } catch {}
    }

    Log "=== Final sync ==="
    Sync-Data "final-sync"

    # Allow Windows to sleep again
    [WakePC]::SetThreadExecutionState([WakePC]::ES_CONTINUOUS) | Out-Null
    Log "=== collect_and_sync stopped ==="
}
