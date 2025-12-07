# 🤖 AI Human-Computer Interaction Coach

Real-time wellness monitoring system that tracks posture, eye strain, engagement, and stress levels using AI and computer vision.

## Features

- Real-time posture detection
- Eye strain monitoring
- Engagement tracking
- Stress analysis
- Productivity scoring
- Wellness recommendations

## Tech Stack

**Backend:** FastAPI, OpenCV, WebSocket  
**Frontend:** React, Vite

## Quick Start

```bash
# Start both servers
start_all.bat

# Or separately
start_backend.bat
start_frontend.bat
```

Open `http://localhost:3000` in your browser.

## Installation

**Backend:**
```bash
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Start servers → Open `http://localhost:3000`
2. Click "Start Camera & Analysis"
3. Allow camera permissions
4. View real-time metrics

## Metrics

- **Productivity Score** (0-100): Overall wellness
- **Posture**: Slouching detection
- **Eye Strain**: Risk level
- **Engagement**: Concentration
- **Stress**: Stress indicators

## Privacy

All processing is local. No data leaves your device.

---

**Made with ❤️ for better workplace wellness**
