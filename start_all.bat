@echo off
echo ========================================
echo Starting HCI Coach - Backend and Frontend
echo ========================================
echo.

echo Starting Backend Server...
start "HCI Coach Backend" cmd /k "cd /d %~dp0backend && call venv\Scripts\activate.bat && python main.py"

echo Waiting 3 seconds for backend to start...
timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "HCI Coach Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ========================================
echo Both servers are starting!
echo ========================================
echo.
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Two new windows will open - one for each server.
echo Close those windows to stop the servers.
echo.
pause



