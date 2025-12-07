@echo off
echo Starting HCI Coach Backend...
cd /d %~dp0backend
if not exist "venv\Scripts\activate.bat" (
    echo Error: Virtual environment not found!
    echo Please run: python -m venv venv
    pause
    exit /b 1
)
call venv\Scripts\activate.bat
if not exist "main.py" (
    echo Error: main.py not found!
    pause
    exit /b 1
)
echo Backend starting on http://localhost:8000
python main.py
pause

