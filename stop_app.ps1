<#
.SYNOPSIS
    Stop the API and dashboard that start_app.ps1 launched.

.DESCRIPTION
    Reads the process ids recorded in .run/ and stops exactly those processes.

    Before stopping anything it checks that the recorded id still belongs to the
    process that was started, by comparing the recorded start time with the
    live one. Windows reuses process ids, so an id alone is not proof of
    identity -- and stopping a stranger's process would be much worse than
    leaving a stale file behind.

    Nothing here searches by image name. `taskkill /F /IM python.exe` would end
    every Python process on the machine, including other projects' work.

    Safe to run when nothing is running: it says so and exits cleanly.

.EXAMPLE
    .\stop_app.ps1
#>
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$RunDir = Join-Path $ProjectRoot '.run'

function Write-Ok { param([string]$Message) Write-Host "  [ok] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "  [--] $Message" -ForegroundColor DarkGray }
function Write-Warn { param([string]$Message) Write-Host "  [!] $Message" -ForegroundColor Yellow }

function Get-Descendant {
    <#  Every descendant of a process id, deepest first.

        Walked through Win32_Process parent links rather than by image name, so
        only processes genuinely spawned by ours are returned. #>
    param([int]$ProcessId)

    $found = New-Object System.Collections.Generic.List[int]
    $queue = New-Object System.Collections.Generic.Queue[int]
    $queue.Enqueue($ProcessId)

    while ($queue.Count -gt 0) {
        $current = $queue.Dequeue()
        $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $current" -ErrorAction SilentlyContinue
        foreach ($child in $children) {
            if (-not $found.Contains([int]$child.ProcessId)) {
                $found.Add([int]$child.ProcessId)
                $queue.Enqueue([int]$child.ProcessId)
            }
        }
    }

    # Deepest first, so a child is never orphaned by its parent dying first.
    $found.Reverse()
    return $found
}

function Wait-PortFree {
    <#  Give a listening socket a moment to be released.

        A force-stopped listener can leave its socket attributed to the dead
        process for a few seconds. Reporting that honestly beats claiming the
        port is free when the next start would fail to bind. #>
    param([int]$Port, [int]$TimeoutSeconds = 10)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        if (-not $listening) { return $true }
        Start-Sleep -Milliseconds 500
    }

    Write-Warn "Port $Port is still held moments after stopping; it should clear shortly."
    return $false
}

function Stop-RecordedService {
    <#  Stop one recorded process, or explain why it was left alone.
        Returns $true only if something was actually stopped. #>
    param([string]$StateFile, [string]$Label)

    if (-not (Test-Path $StateFile)) {
        Write-Info "$Label was not running"
        return $false
    }

    try {
        $state = Get-Content $StateFile -Raw | ConvertFrom-Json
    } catch {
        Write-Warn "$Label state file was unreadable; removing it"
        Remove-Item $StateFile -Force
        return $false
    }

    $processId = $state.Pid
    if (-not $processId) {
        Remove-Item $StateFile -Force
        Write-Info "$Label state file held no process id; removed"
        return $false
    }

    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        Remove-Item $StateFile -Force
        Write-Info "$Label (pid $processId) had already exited; cleaned up"
        return $false
    }

    # Identity check: same id AND same start time, or we do not touch it.
    if ($state.StartTime) {
        $recorded = [DateTime]::Parse($state.StartTime)
        if ([Math]::Abs(($process.StartTime - $recorded).TotalSeconds) -gt 2) {
            Write-Warn "pid $processId now belongs to another process ($($process.ProcessName)); not stopping it"
            Remove-Item $StateFile -Force
            return $false
        }
    }

    # Descendants first, parent last. uvicorn --reload runs the application in
    # a child process, and that child is what actually holds the port: stopping
    # only the parent leaves it orphaned and the port occupied. The tree has to
    # be walked *before* the parent dies, because that is what makes the
    # parent/child links discoverable.
    $descendants = Get-Descendant -ProcessId $processId
    $childCount = 0

    foreach ($child in $descendants) {
        try {
            Stop-Process -Id $child -Force -ErrorAction Stop
            $childCount++
        } catch {
            # Already gone is fine: a child often exits with its siblings.
            if (Get-Process -Id $child -ErrorAction SilentlyContinue) {
                Write-Warn "  child process $child could not be stopped"
            }
        }
    }

    # A parent commonly exits once its worker is gone -- uvicorn's reloader
    # does exactly that. Finding it already dead here is success, not failure.
    $stillAlive = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($stillAlive) {
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
        } catch {
            # It may have exited in the moment between the check above and this
            # call -- a parent very often follows its worker down. Gone is the
            # outcome we wanted, however it happened.
            if (Get-Process -Id $processId -ErrorAction SilentlyContinue) {
                Write-Warn "$Label (pid $processId) could not be stopped: $($_.Exception.Message)"
                Remove-Item $StateFile -Force -ErrorAction SilentlyContinue
                return $false
            }
        }
    }

    Remove-Item $StateFile -Force -ErrorAction SilentlyContinue

    $detail = "pid $processId"
    if ($childCount -gt 0) { $detail += ", $childCount child process(es)" }
    if ($state.Port) { $detail += ", port $($state.Port)" }
    Write-Ok "$Label stopped ($detail)"

    if ($state.Port) { Wait-PortFree -Port $state.Port | Out-Null }

    return $true
}

# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "Resume Screening AI - stopping" -ForegroundColor Cyan
Write-Host "------------------------------"

if (-not (Test-Path $RunDir)) {
    Write-Info "Nothing is running (no .run directory)"
    Write-Host ""
    exit 0
}

$stopped = 0
if (Stop-RecordedService -StateFile (Join-Path $RunDir 'ui.json') -Label 'Dashboard') { $stopped++ }
if (Stop-RecordedService -StateFile (Join-Path $RunDir 'api.json') -Label 'API') { $stopped++ }

Write-Host ""
if ($stopped -eq 0) {
    Write-Host "  Nothing was running." -ForegroundColor DarkGray
} else {
    Write-Host "  Stopped $stopped service(s)." -ForegroundColor Green
}

# Logs are kept deliberately: if a service died on its own, the reason is in
# them, and this script is often the first thing run after noticing.
$remaining = Get-ChildItem $RunDir -Filter '*.json' -ErrorAction SilentlyContinue
if (-not $remaining) {
    Write-Host "  Logs kept in $RunDir" -ForegroundColor DarkGray
}
Write-Host ""
