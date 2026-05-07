@echo off
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "EASYICU_VERBOSE=0"
chcp 65001 >nul 2>nul

set "PY_CMD="
set "PY_ARGS="
set "EXIT_CODE=1"

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY_CMD (
      call :try_python py "-%%V"
    )
  )
)

if not defined PY_CMD call :try_python python ""
if not defined PY_CMD call :try_python python3 ""

if not defined PY_CMD (
  if defined CONDA_PREFIX (
    call :try_python "%CONDA_PREFIX%\python.exe" ""
  )
)

if not defined PY_CMD (
  where conda >nul 2>nul
  if !ERRORLEVEL!==0 (
    for /f "usebackq delims=" %%B in (`conda info --base 2^>nul`) do (
      if not defined PY_CMD call :try_python "%%B\python.exe" ""
    )
  )
)

if not defined PY_CMD (
  for %%P in (
    "%USERPROFILE%\miniconda3\python.exe"
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\mambaforge\python.exe"
    "%USERPROFILE%\miniforge3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe"
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "%ProgramData%\miniconda3\python.exe"
    "%ProgramData%\anaconda3\python.exe"
  ) do (
    if not defined PY_CMD call :try_python "%%~P" ""
  )
)

if defined PY_CMD (
  "%PY_CMD%" %PY_ARGS% -X utf8 "%SCRIPT_DIR%scripts\launch_easyicu.py" start %*
  set "EXIT_CODE=!ERRORLEVEL!"
  goto :done
)

echo Python 3.10+ was not found. Checked PATH, the Python launcher, and common Conda/Anaconda locations.
echo If you use Conda, install Python 3.10+ in base or activate/set CONDA_PREFIX before launching.
set "EXIT_CODE=1"
goto :done

:try_python
set "TRY_CMD=%~1"
set "TRY_ARGS=%~2"
if not defined TRY_CMD exit /b 0
if not "%TRY_CMD%"=="py" (
  if not exist "%TRY_CMD%" (
    where "%TRY_CMD%" >nul 2>nul
    if errorlevel 1 exit /b 0
  )
)
"%TRY_CMD%" %TRY_ARGS% -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
if errorlevel 1 exit /b 0
set "PY_CMD=%TRY_CMD%"
set "PY_ARGS=%TRY_ARGS%"
exit /b 0

:done
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
