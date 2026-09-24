@echo off
cd /d "%~dp0"
"venv\Scripts\python.exe" -m sistema_final.main %*
if errorlevel 1 pause
