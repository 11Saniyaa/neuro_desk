# Neuro Desk - AI Human-Computer Interaction Coach

Real-time wellness monitoring system that analyzes your workspace behavior using computer vision.

## Features

- 📊 **Productivity Score** - Real-time productivity tracking
- 💺 **Posture Detection** - Monitors slouching and sitting position
- 👁️ **Eye Strain Analysis** - Tracks eye strain risk levels
- 🧠 **Engagement Monitoring** - Measures concentration levels
- 😌 **Stress Detection** - Analyzes stress indicators
- 💡 **Smart Recommendations** - Personalized wellness suggestions

## Tech Stack

- **Frontend**: React + Vite
- **Backend**: FastAPI + Python
- **Computer Vision**: OpenCV, MediaPipe (when available)
- **Real-time**: WebSocket + HTTP POST

## Quick Start

### Backend
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python main.py
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Or use the provided batch files:
- `start_backend.bat` - Start backend server
- `start_frontend.bat` - Start frontend dev server
- `start_all.bat` - Start both services

## Requirements

- Python 3.8+
- Node.js 16+
- Webcam access

## License

MIT
