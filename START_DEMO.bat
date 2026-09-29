@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe live.py demo
) else (
  python live.py demo
)
if errorlevel 1 (
  echo Please install dependencies following README.md first.
) else (
  explorer outputs\v8_demo
)
pause
