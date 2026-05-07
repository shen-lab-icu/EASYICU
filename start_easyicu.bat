@echo off
setlocal EnableDelayedExpansion
set "SCRIPT_DIR=%~dp0"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "EASYICU_VERBOSE=0"
chcp 65001 >nul 2>nul

set "PY_CMD="
where py >nul 2>nul
if %ERRORLEVEL%==0 (
  for %%V in (3.13 3.12 3.11 3.10) do (
    if not defined PY_CMD (
      py -%%V -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
      if !ERRORLEVEL!==0 (
        set "PY_CMD=py -%%V"
      )
    )
  )
  if defined PY_CMD (
    !PY_CMD! -X utf8 "%SCRIPT_DIR%scripts\\launch_easyicu.py" start %*
    set "EXIT_CODE=!ERRORLEVEL!"
    goto :done
  )
)

where python >nul 2>nul
if %ERRORLEVEL%==0 (
  python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if !ERRORLEVEL!==0 (
    python -X utf8 "%SCRIPT_DIR%scripts\\launch_easyicu.py" start %*
    set "EXIT_CODE=!ERRORLEVEL!"
    goto :done
  )
)

where python3 >nul 2>nul
if %ERRORLEVEL%==0 (
  python3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
  if !ERRORLEVEL!==0 (
    python3 -X utf8 "%SCRIPT_DIR%scripts\\launch_easyicu.py" start %*
    set "EXIT_CODE=!ERRORLEVEL!"
    goto :done
  )
)

echo Python 3.10+ was not found. Please install Python from https://www.python.org/downloads/
set "EXIT_CODE=1"

:done
if not "%EXIT_CODE%"=="0" pause
exit /b %EXIT_CODE%
