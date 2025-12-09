@echo off
echo Starting HCI Coach Backend Server...
echo.
cd /d %~dp0
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
    python main.py
) else (
    echo Virtual environment not found. Please run: python -m venv venv
    echo Then install dependencies: pip install -r requirements.txt
    pause
)

