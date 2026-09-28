@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe live.py doctor
) else (
  python live.py doctor
)
if exist outputs\v9_doctor\doctor.md start "" outputs\v9_doctor\doctor.md
pause
