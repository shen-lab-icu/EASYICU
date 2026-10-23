param(
    [Parameter(Mandatory=$true)][string]$Installer,
    [Parameter(Mandatory=$true)][string]$ProfileRoot
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Wait-CheckedProcess($Process, [int]$Timeout) {
    if (-not $Process.WaitForExit($Timeout)) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
        throw "Process timed out: $($Process.Id)"
    }
    if ($Process.ExitCode -ne 0) {
        throw "Process failed with exit code $($Process.ExitCode)"
    }
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
    Wait-CheckedProcess $install 180000
    $installed = $true
    $executable = Join-Path $installDir 'easyicu-desktop.exe'
    if (-not (Test-Path -LiteralPath $executable)) { throw "Missing installed app: $executable" }

    # Prove launch uses bundled Python/Node and does not require Docker or Git.
    $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
    Remove-Item Env:PYTHONPATH, Env:PYTHONHOME -ErrorAction SilentlyContinue
    $app = Start-Process -FilePath $executable -ArgumentList '--smoke-test' -PassThru
    Wait-CheckedProcess $app 150000
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
    Wait-CheckedProcess $uninstall 120000
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
            Wait-CheckedProcess $cleanup 120000
        }
    }
    foreach ($name in $saved.Keys) {
        [Environment]::SetEnvironmentVariable($name, $saved[$name], 'Process')
    }
}
