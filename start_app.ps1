<#
.SYNOPSIS
    Start the screening API and the recruiter dashboard.

.DESCRIPTION
    Starts two background processes from the project root:

        FastAPI    python -m uvicorn app.api.main:app --reload
        Streamlit  python -m streamlit run app/ui/dashboard.py

    Each process id is recorded in .run/ together with the time it started, so
    stop_app.ps1 can identify exactly the processes this script launched. It
    never searches for processes by name, because that would match Python and
    Streamlit belonging to other projects.

    Running it twice is safe: a service that is already up is reported and left
    alone.

.PARAMETER ApiPort
    Port for the API. Default 8000.

.PARAMETER UiPort
    Port for the dashboard. Default 8501.

.PARAMETER Python
    Python executable to use. Defaults to $env:PYTHON, then the active virtual
    environment, then .venv in the project, then python on PATH.

.EXAMPLE
    .\start_app.ps1

.EXAMPLE
    .\start_app.ps1 -ApiPort 8100 -UiPort 8600
#>
[CmdletBinding()]
param(
    [int]$ApiPort = 8000,
    [int]$UiPort = 8501,
    [string]$Python
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = $PSScriptRoot
$RunDir = Join-Path $ProjectRoot '.run'
$ApiState = Join-Path $RunDir 'api.json'
$UiState = Join-Path $RunDir 'ui.json'

function Write-Step { param([string]$Message) Write-Host "  $Message" }
function Write-Ok { param([string]$Message) Write-Host "  [ok] $Message" -ForegroundColor Green }
function Write-Info { param([string]$Message) Write-Host "  [--] $Message" -ForegroundColor DarkGray }
function Write-Warn { param([string]$Message) Write-Host "  [!] $Message" -ForegroundColor Yellow }
function Write-Fail {
    param([string]$Message, [string]$Fix)
    Write-Host ""
    Write-Host "  [!!] $Message" -ForegroundColor Red
    if ($Fix) { Write-Host "       $Fix" -ForegroundColor Yellow }
    Write-Host ""
}

function Resolve-Python {
    <#  First match wins: explicit argument, PYTHON, the active venv, a .venv in
        the project, then whatever is on PATH. No path is hardcoded, so this
        works on any machine. #>
    param([string]$Preferred)

    $candidates = @()
    if ($Preferred) { $candidates += $Preferred }
    if ($env:PYTHON) { $candidates += $env:PYTHON }
    if ($env:VIRTUAL_ENV) { $candidates += (Join-Path $env:VIRTUAL_ENV 'Scripts\python.exe') }
    $candidates += (Join-Path $ProjectRoot '.venv\Scripts\python.exe')

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return (Resolve-Path $candidate).Path }
    }

    $onPath = Get-Command python -ErrorAction SilentlyContinue
    if ($onPath) { return $onPath.Source }
    return $null
}

function Get-RunningService {
    <#  Return the recorded process only if it is still alive AND is the same
        process we started. The start time is checked because Windows reuses
        process ids, and killing a stranger would be far worse than a stale
        state file. #>
    param([string]$StateFile)

    if (-not (Test-Path $StateFile)) { return $null }

    try { $state = Get-Content $StateFile -Raw | ConvertFrom-Json } catch { return $null }
    if (-not $state.Pid) { return $null }

    $process = Get-Process -Id $state.Pid -ErrorAction SilentlyContinue
    if (-not $process) { return $null }

    if ($state.StartTime) {
        $recorded = [DateTime]::Parse($state.StartTime)
        if ([Math]::Abs(($process.StartTime - $recorded).TotalSeconds) -gt 2) {
            # Same id, different process: the one we started is gone.
            return $null
        }
    }

    return $process
}

function Test-PortServed {
    <#  Is something already listening on this port, and is it ours?

        A port held by another project is a reason to stop, not to start a
        second process that will fail to bind. Ownership is decided by the
        command line referencing this project root -- never by image name. #>
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if (-not $connection) { return $null }

    $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $($connection.OwningProcess)" -ErrorAction SilentlyContinue
    $commandLine = if ($owner) { $owner.CommandLine } else { $null }

    return [pscustomobject]@{
        Pid   = $connection.OwningProcess
        IsOurs = ($commandLine -and ($commandLine -match 'app\.api\.main:app' -or $commandLine -match 'app/ui/dashboard\.py'))
        Command = $commandLine
    }
}

function Start-Service {
    param(
        [string]$Name,
        [string]$Exe,
        [string[]]$ArgumentList,
        [string]$StateFile,
        [string]$LogFile,
        [int]$Port,
        [string]$PortParameter
    )

    $existing = Get-RunningService -StateFile $StateFile
    if ($existing) {
        Write-Info "$Name already running (pid $($existing.Id)); leaving it alone"
        return $existing
    }

    # The recorded process may be gone while its port is still served -- a
    # reloader child outliving its parent, for instance. Starting another one
    # would just fail to bind.
    $served = Test-PortServed -Port $Port
    if ($served) {
        if ($served.IsOurs) {
            Write-Info "$Name already serving port $Port (pid $($served.Pid)); leaving it alone"
        } else {
            Write-Warn "Port $Port is in use by something else (pid $($served.Pid))."
            Write-Warn "Free it, or choose another port: .\start_app.ps1 -$PortParameter <n>"
        }
        return $null
    }

    $process = Start-Process -FilePath $Exe -ArgumentList $ArgumentList `
        -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $LogFile -RedirectStandardError "$LogFile.err"

    # Recorded together so stop_app.ps1 can prove the pid is still ours.
    [pscustomobject]@{
        Name      = $Name
        Pid       = $process.Id
        StartTime = $process.StartTime.ToString('o')
        Port      = $Port
        Command   = "$Exe $($ArgumentList -join ' ')"
        Log       = $LogFile
    } | ConvertTo-Json | Set-Content -Path $StateFile -Encoding utf8

    Write-Ok "$Name started (pid $($process.Id))"
    return $process
}

function Wait-ForUrl {
    param([string]$Url, [int]$TimeoutSeconds = 90)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Url -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) { return $true }
        } catch {
            Start-Sleep -Milliseconds 700
        }
    }
    return $false
}

# ---------------------------------------------------------------------------

Write-Host ""
Write-Host "Resume Screening AI" -ForegroundColor Cyan
Write-Host "-------------------"

# 1. Environment ------------------------------------------------------------
$PythonExe = Resolve-Python -Preferred $Python
if (-not $PythonExe) {
    Write-Fail "No Python interpreter found." "Install Python 3.11+, or set `$env:PYTHON to its full path."
    exit 1
}
Write-Step "Python: $PythonExe"

if (-not (Test-Path (Join-Path $ProjectRoot 'app\api\main.py'))) {
    Write-Fail "This does not look like the project root." "Run the script from the repository root: .\start_app.ps1"
    exit 1
}

# 2. Dependencies -----------------------------------------------------------
$check = & $PythonExe -c "import fastapi, uvicorn, streamlit, httpx" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Fail "Required packages are missing from $PythonExe." "Install them with: $PythonExe -m pip install -r requirements.txt"
    Write-Host "  $check" -ForegroundColor DarkGray
    exit 1
}
Write-Ok "Dependencies present"

if (-not (Test-Path $RunDir)) { New-Item -ItemType Directory -Path $RunDir | Out-Null }

# 3. Start ------------------------------------------------------------------
$apiProcess = Start-Service -Name 'API' -Exe $PythonExe `
    -ArgumentList @('-m', 'uvicorn', 'app.api.main:app', '--reload', '--port', "$ApiPort") `
    -StateFile $ApiState -LogFile (Join-Path $RunDir 'api.log') -Port $ApiPort `
    -PortParameter 'ApiPort'

$env:API_BASE_URL = "http://127.0.0.1:$ApiPort"
$uiProcess = Start-Service -Name 'Dashboard' -Exe $PythonExe `
    -ArgumentList @('-m', 'streamlit', 'run', 'app/ui/dashboard.py',
                    '--server.port', "$UiPort", '--server.headless', 'true') `
    -StateFile $UiState -LogFile (Join-Path $RunDir 'ui.log') -Port $UiPort `
    -PortParameter 'UiPort'

# 4. Wait -------------------------------------------------------------------
Write-Host ""
Write-Step "Waiting for services (the API loads an embedding model on first use)..."

$apiUp = Wait-ForUrl -Url "http://127.0.0.1:$ApiPort/health"
if ($apiUp) { Write-Ok "API healthy" }
else { Write-Info "API not answering yet; check $RunDir\api.log" }

$uiUp = Wait-ForUrl -Url "http://127.0.0.1:$UiPort/_stcore/health" -TimeoutSeconds 60
if ($uiUp) { Write-Ok "Dashboard serving" }
else { Write-Info "Dashboard not answering yet; check $RunDir\ui.log" }

# 5. Report -----------------------------------------------------------------
Write-Host ""
Write-Host "  Dashboard  http://localhost:$UiPort" -ForegroundColor Cyan
Write-Host "  API        http://127.0.0.1:$ApiPort"
Write-Host "  API docs   http://127.0.0.1:$ApiPort/docs"
Write-Host ""
Write-Host "  Logs       $RunDir"
Write-Host "  Stop with  .\stop_app.ps1"
Write-Host ""

if (-not ($apiUp -and $uiUp)) { exit 1 }
