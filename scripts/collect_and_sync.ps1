<#
.SYNOPSIS
    Collect Polymarket price data all day and sync to AWS S3 every 30 minutes.

.DESCRIPTION
    1. Runs collect_prices.py continuously (auto-restarts on crash).
    2. Runs collect_orderbook.py continuously (auto-restarts on crash).
    3. Every 30 minutes uploads today's data/prices/ and data/orderbook/ to S3
       via scripts/sync_to_s3.py (--today-only for speed).
    4. On Ctrl+C collectors are stopped immediately (no final S3 sync).
    5. Keeps your PC awake using the Windows SetThreadExecutionState API.

    PREREQUISITES
    -------------
    • Python 3 with the project's .venv created, OR Python 3 on PATH
    • boto3 installed:  pip install boto3
    • S3_BUCKET (and optionally S3_PREFIX) set in your .env file

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
    Press Ctrl+C once. The script will stop the collectors and exit cleanly.

.PARAMETER SyncIntervalMinutes
    How often (minutes) to upload new data to S3. Default: 30.

.PARAMETER CollectArgs
    Extra arguments forwarded to collect_prices.py, e.g. "--assets BTC ETH --interval 2".
#>
param(
    [int]   $SyncIntervalMinutes = 30,
    [string]$CollectArgs         = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
$ROOT      = Split-Path -Parent $PSScriptRoot
$COLLECT   = Join-Path $PSScriptRoot "collect_prices.py"
$ORDERBOOK = Join-Path $PSScriptRoot "collect_orderbook.py"
$SYNC_S3   = Join-Path $PSScriptRoot "sync_to_s3.py"
$LOG       = Join-Path $ROOT "data\collect_and_sync.log"

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
    foreach ($cmd in "python", "python3") {
        try {
            if ((& $cmd --version 2>&1) -match "Python 3") { return $cmd }
        } catch {}
    }
    return $null
}

$venvPython = Join-Path $ROOT ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $PY = $venvPython
} else {
    $PY = Find-Python
}

if (-not $PY) {
    Log "ERROR: Python 3 not found. Install Python 3 or run: python -m venv .venv"
    exit 1
}
Log "Python: $PY"

# ─────────────────────────────────────────────────────────────────────────────
# S3 sync — uploads data/prices/ and data/orderbook/ to S3
# --today-only on periodic syncs (fast)
# ─────────────────────────────────────────────────────────────────────────────
function Sync-ToS3([string]$label, [switch]$TodayOnly) {
    Log "=== S3 sync ($label) ==="
    try {
        $syncArgs = @($SYNC_S3, "--dirs", "prices", "orderbook")
        if ($TodayOnly) { $syncArgs += "--today-only" }

        $p = Start-Process `
            -FilePath         $PY `
            -ArgumentList     $syncArgs `
            -WorkingDirectory $ROOT `
            -NoNewWindow `
            -PassThru `
            -Wait

        if ($p.ExitCode -eq 0) {
            Log "S3 sync complete."
        } else {
            Log "WARNING: sync_to_s3.py exited with code $($p.ExitCode)."
        }
    } catch {
        Log "ERROR during S3 sync: $_"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# Launch collect_prices.py as a child process
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

function Start-OrderbookCollector {
    $env:PYTHONUNBUFFERED = "1"
    $argList = @($ORDERBOOK, "--interval", "0.25")

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
Log "Press Ctrl+C to stop."

$proc        = $null
$obProc      = $null
$restarts    = 0
$maxRestarts = 100
$lastSync    = Get-Date   # start the clock; first push after $SyncIntervalMinutes

try {
    while ($restarts -lt $maxRestarts) {
        # ── (Re)start the price collector if it isn't running ─────────────────
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

        # ── (Re)start the orderbook collector if it isn't running ─────────────
        if ($null -eq $obProc -or $obProc.HasExited) {
            if ($null -ne $obProc) {
                Log "Orderbook collector exited (code $($obProc.ExitCode)). Restarting in 5 s..."
                Start-Sleep -Seconds 5
            } else {
                Log "Launching orderbook collector..."
            }
            $obProc = Start-OrderbookCollector
            Log "Orderbook collector running (PID $($obProc.Id))."
        }

        # ── Periodic S3 sync (today's files only — fast) ─────────────────────
        if (((Get-Date) - $lastSync).TotalMinutes -ge $SyncIntervalMinutes) {
            Sync-ToS3 "auto-sync" -TodayOnly
            $lastSync = Get-Date
        }

        Start-Sleep -Seconds 15
    }
} finally {
    # ── Cleanup: stop both collectors and exit ─────────────────────────────────
    if ($null -ne $proc -and -not $proc.HasExited) {
        Log "Stopping collector (PID $($proc.Id))..."
        try { $proc.Kill(); $proc.WaitForExit(5000) | Out-Null } catch {}
    }
    if ($null -ne $obProc -and -not $obProc.HasExited) {
        Log "Stopping orderbook collector (PID $($obProc.Id))..."
        try { $obProc.Kill(); $obProc.WaitForExit(5000) | Out-Null } catch {}
    }

    # Allow Windows to sleep again
    [WakePC]::SetThreadExecutionState([WakePC]::ES_CONTINUOUS) | Out-Null
    Log "=== collect_and_sync stopped ==="
}
