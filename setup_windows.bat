@echo off
REM One-time setup: creates .venv, installs core dependencies, generates SIMULATED data.
cd /d "%~dp0"
py -3 -m venv .venv || goto :fail
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :fail
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :fail
".venv\Scripts\python.exe" generate_data.py || goto :fail
echo.
echo Setup complete. Start the dashboard with run_app.bat
pause
exit /b 0
:fail
echo Setup failed. See the messages above.
pause
exit /b 1
