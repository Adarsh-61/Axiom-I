# Axiom-I

Axiom-I is a complete deepfake image detection project.
It includes:

- a backend API for analysis
- a frontend web app for user interaction
- a feedback and calibration flow to improve results over time

The project is designed to run locally without global Python package installation.

## 1. Important Rules

Please follow these rules for every setup:

- Use a virtual environment for Python.
- Do not install Python packages globally.
- Always use uv pip for Python dependencies.
- Use npm only for frontend dependencies.

## 2. Project Structure

- Coding/Image/backend: FastAPI backend, ML pipeline, feedback and calibration logic.
- Coding/Image/frontend: Next.js frontend UI.
- Coding/Image/test: Real and Fake image dataset for testing and training.
- Documents/Image/Diagrams: project diagrams for architecture and flow.
- Documents/Image/Axiom-I.odp: presentation file.

Detailed backend API guide:

- Coding/Image/backend/README.md

## 3. Requirements

- Python 3.11+
- uv
- Node.js 20+
- npm

## 4. Full Setup Guide

### Step 1: Go to project root

```bash
cd /path/to/Axiom-I
```

### Step 2: Create virtual environment

Linux/macOS:

```bash
uv venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
uv venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install backend dependencies

Do not use pip install directly.
Use uv pip:

```bash
uv pip install -r Coding/Image/backend/requirements.txt
```

### Step 4: Install frontend dependencies

```bash
cd Coding/Image/frontend
npm install
cd ../../..
```

### Step 5: Create backend environment file

```bash
cp Coding/Image/backend/.env.example Coding/Image/backend/.env
```

## 5. Run in Development Mode

Open two terminals.

Terminal 1 (backend):

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):

```bash
cd /path/to/Axiom-I/Coding/Image/frontend
npm run dev
```

Open these URLs:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs
- Backend health: http://localhost:8000/api/v1/health

## 6. Backend Environment Variables

Edit Coding/Image/backend/.env as needed.

- AXIOM_DEVICE: cpu or cuda.
- AXIOM_DEBUG: true or false.
- AXIOM_API_HOST: backend host.
- AXIOM_API_PORT: backend port.
- AXIOM_ALLOWED_ORIGINS: allowed frontend URLs.
- AXIOM_ALLOW_MODEL_DOWNLOAD: true or false.
- AXIOM_VIT_MODEL_NAME: model name for image classifier.

Note:

- AXIOM_ALLOW_MODEL_DOWNLOAD=false keeps startup predictable.
- Set it to true if you want first-time model download automatically.

## 7. API Endpoints

Main endpoints:

- GET / : basic service info.
- GET /api/v1/health : health check.
- POST /api/v1/analyze : analyze one image.
- POST /api/v1/analyze/video : analyze one video (physics-based).
- POST /api/v1/feedback : submit user image feedback.
- POST /api/v1/feedback/video : submit user video feedback.
- GET /api/v1/feedback/metrics : confusion matrix and calibration metrics.


## 8. Training and Calibration

You can rebuild seed training data from the local test folder.

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
python train_from_testset.py --mode fallback --dataset-root ../test
```

Modes:

- fallback: faster, lower compute
- full: slower, more detailed processing

## 9. Frontend Scripts

Run from Coding/Image/frontend:

- npm run dev: start local frontend
- npm run build: production build
- npm run start: run production server
- npm run lint: TypeScript check

## 10. Production Run

Backend:

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Frontend:

```bash
cd /path/to/Axiom-I/Coding/Image/frontend
npm run build
npm run start
```

## 11. Common Issues and Fixes

Issue: frontend cannot connect to backend.

- Check backend is running on port 8000.
- Check AXIOM_ALLOWED_ORIGINS includes http://localhost:3000.

Issue: model warning about local cache.

- Set AXIOM_ALLOW_MODEL_DOWNLOAD=true in backend .env.
- Restart backend.

Issue: command not found for uv.

- Install uv first, then recreate virtual environment.

## 12. Final Notes

- Keep virtual environment active while working.
- Use uv pip for every Python dependency installation.
- Do not install Python dependencies globally.
- This keeps your machine clean and keeps project setup stable.
