@echo off
echo Finding processes using port 8000...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do (
    echo Killing process %%a...
    taskkill /F /PID %%a >nul 2>&1
)
echo Port 8000 should now be free.
timeout /t 2 /nobreak >nul
echo You can now start the backend.

