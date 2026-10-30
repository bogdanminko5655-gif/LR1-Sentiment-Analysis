@echo off
setlocal
pushd "%~dp0"
if errorlevel 1 (
    echo ERROR: Cannot open the laboratory project folder.
    pause
    exit /b 1
)

rem Prefer the Windows Python launcher when available.
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set "PY=py -3"
) else (
    python --version >nul 2>&1
    if errorlevel 1 (
        echo ERROR: Python 3 not found. Install Python 3.10 or newer.
        echo https://www.python.org/downloads/windows/
        pause
        exit /b 1
    )
    set "PY=python"
)

%PY% -m pip install -r requirements.txt
if errorlevel 1 goto pip_error

%PY% main.py
if errorlevel 1 goto run_error

%PY% make_report.py
if errorlevel 1 goto run_error

echo.
echo DONE. See the results folder and Lab1_Report.docx.
popd
pause
exit /b 0

:pip_error
echo ERROR: Could not install required libraries. Check Internet access.
popd
pause
exit /b 1

:run_error
echo ERROR: The program failed. See the error message above.
popd
pause
exit /b 1
