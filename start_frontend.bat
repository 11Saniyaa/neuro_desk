@echo off
echo Starting HCI Coach Frontend...
cd /d %~dp0frontend
if not exist "node_modules" (
    echo Installing dependencies...
    call npm install
)
echo Frontend starting on http://localhost:3000
npm run dev
pause

