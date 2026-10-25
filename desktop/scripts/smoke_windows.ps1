param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [Parameter(Mandatory=$true)][string]$ProfileRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Each stage names itself and reports its duration, so a timeout says which
# stage stalled (install, first launch or uninstall) instead of a bare PID.
# On a timeout, record what the stalled process tree was doing before it is
# killed: processes and window titles (a modal loader/error dialog shows here),
# their command lines, and the EasyICU data folder with its logs. The folder is
# an ASCII path under RUNNER_TEMP so the diagnostics upload can always find it.
function Save-StallDiagnostics([int]$RootId, [string]$Stage) {
    $base = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { [IO.Path]::GetTempPath() }
    $out = Join-Path $base 'easyicu-smoke-diag'
    New-Item -ItemType Directory -Force $out | Out-Null
    try {
        $all = @(Get-CimInstance Win32_Process)
        $ids = @($RootId)
        for ($round = 0; $round -lt 6; $round++) {
            $ids = @($ids + @($all | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { $_.ProcessId })) | Select-Object -Unique
        }
        $all | Where-Object { $ids -contains $_.ProcessId } |
            Select-Object ProcessId, ParentProcessId, Name, CreationDate, CommandLine |
            Format-List | Out-File (Join-Path $out 'process-tree.txt') -Encoding utf8
        Get-Process | Where-Object { $ids -contains $_.Id -or $_.MainWindowTitle } |
            Select-Object Id, ProcessName, MainWindowTitle, Responding |
            Format-Table -AutoSize | Out-File (Join-Path $out 'windows.txt') -Encoding utf8
        $root = Join-Path $env:LOCALAPPDATA 'EasyICU'
        if (Test-Path -LiteralPath $root) {
            Get-ChildItem -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue |
                Select-Object FullName, Length, LastWriteTime |
                Format-Table -AutoSize | Out-File (Join-Path $out 'data-root.txt') -Encoding utf8 -Width 400
            $logs = Join-Path $root 'logs'
            if (Test-Path -LiteralPath $logs) { Copy-Item -Path (Join-Path $logs '*') -Destination $out -Force -ErrorAction SilentlyContinue }
        } else {
            "No EasyICU data folder at $root" | Out-File (Join-Path $out 'data-root.txt') -Encoding utf8
        }
        "$Stage stalled; root process $RootId" | Out-File (Join-Path $out 'stage.txt') -Encoding utf8
    } catch {
        "diagnostics failed: $_" | Out-File (Join-Path $out 'diagnostics-error.txt') -Encoding utf8
    }
}

function Wait-CheckedProcess($Process, [int]$Timeout, [string]$Stage) {
    $clock = [Diagnostics.Stopwatch]::StartNew()
    if (-not $Process.WaitForExit($Timeout)) {
        Save-StallDiagnostics $Process.Id $Stage
        & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
        throw "$Stage timed out after $([int]$clock.Elapsed.TotalSeconds) s (process $($Process.Id))"
    }
    if ($Process.ExitCode -ne 0) {
        throw "$Stage failed with exit code $($Process.ExitCode) after $([int]$clock.Elapsed.TotalSeconds) s"
    }
    Write-Host "$Stage finished in $([int]$clock.Elapsed.TotalSeconds) s"
}

# The hosted runner remains the OS user. Its temporary application-data profile
# exercises paths with spaces and Unicode without touching its normal data.
$saved = @{}
foreach ($name in @('USERPROFILE', 'LOCALAPPDATA', 'APPDATA', 'PATH', 'PYTHONPATH', 'PYTHONHOME')) {
    $saved[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$app = $null
$installed = $false
try {
    $env:USERPROFILE = [IO.Path]::GetFullPath($ProfileRoot)
    $env:LOCALAPPDATA = Join-Path $env:USERPROFILE 'AppData\Local'
    $env:APPDATA = Join-Path $env:USERPROFILE 'AppData\Roaming'
    $installDir = Join-Path $env:LOCALAPPDATA 'Programs\EasyICU Desktop'
    New-Item -ItemType Directory -Force $env:LOCALAPPDATA, $env:APPDATA | Out-Null
    # NSIS requires /D last and unquoted, including when the value has spaces.
    $install = Start-Process -FilePath $Installer -ArgumentList "/S /D=$installDir" -PassThru
    # A per-user NSIS install writes thousands of runtime files; on a scanned
    # Windows disk this takes minutes, not seconds.
    Wait-CheckedProcess $install 600000 'Install'
    $installed = $true
    $executable = Join-Path $installDir 'easyicu-desktop.exe'
    if (-not (Test-Path -LiteralPath $executable)) { throw "Missing installed app: $executable" }

    # Prove launch uses bundled Python/Node and does not require Docker or Git.
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    Remove-Item Env:PYTHONPATH, Env:PYTHONHOME -ErrorAction SilentlyContinue
    $app = Start-Process -FilePath $executable -ArgumentList '--smoke-test' -PassThru
    # The first launch also unpacks nothing but is scanned file by file.
    Wait-CheckedProcess $app 300000 'First launch smoke test'
    $dataRoot = Join-Path $env:LOCALAPPDATA 'EasyICU'
    $result = Get-Content -LiteralPath (Join-Path $dataRoot 'runtime\desktop-smoke-result.json') -Raw | ConvertFrom-Json
    if (-not $result.healthy -or -not $result.unauthenticated_rejected) {
        throw 'Backend readiness/session checks did not pass'
    }

    # No clinical data: two tiny sentinels prove default uninstall retention.
    $stateSentinel = Join-Path $dataRoot 'state\synthetic-retention.txt'
    $exportDir = Join-Path $dataRoot 'exports'
    New-Item -ItemType Directory -Force $exportDir | Out-Null
    $exportSentinel = Join-Path $exportDir 'synthetic-retention.txt'
    Set-Content -LiteralPath $stateSentinel -Value 'synthetic state'
    Set-Content -LiteralPath $exportSentinel -Value 'synthetic export'
    # _?= keeps the uninstaller in place so WaitForExit tracks real completion.
    $uninstall = Start-Process -FilePath (Join-Path $installDir 'uninstall.exe') -ArgumentList "/S _?=$installDir" -PassThru
    Wait-CheckedProcess $uninstall 300000 'Uninstall'
    $installed = $false
    if ((Test-Path -LiteralPath $executable) -or
        -not (Test-Path -LiteralPath $stateSentinel) -or
        -not (Test-Path -LiteralPath $exportSentinel)) {
        throw 'Uninstall did not remove the app while preserving user state and exports'
    }
} finally {
    if ($null -ne $app -and -not $app.HasExited) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $app.Id /T /F | Out-Null
    }
    if ($installed) {
        $uninstaller = Join-Path $installDir 'uninstall.exe'
        if (Test-Path -LiteralPath $uninstaller) {
            $cleanup = Start-Process -FilePath $uninstaller -ArgumentList "/S _?=$installDir" -PassThru
            Wait-CheckedProcess $cleanup 300000 'Cleanup uninstall'
        }
    }
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
