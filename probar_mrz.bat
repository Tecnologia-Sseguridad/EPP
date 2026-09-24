@echo off
cd /d "%~dp0"
"venv\Scripts\python.exe" probar_mrz.py
if errorlevel 1 pause
