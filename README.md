# Axiom-I

Axiom-I is an image checking system.
It takes an image and gives a result: Real or Fake.

## Important Rules

- This project uses a virtual environment.
- Do not install Python packages globally.
- Always use uv pip for Python package install.
- Use npm only for frontend packages.

## Main Folders

- Coding/Image/backend: FastAPI backend and image analysis logic.
- Coding/Image/frontend: Next.js web app.
- Coding/Image/test: Real and Fake test images.
- Documents/Image/Diagrams: project diagrams.

## What You Need

- Python 3.11 or newer
- Node.js 20 or newer
- npm
- uv

## Setup (Step by Step)

### 1. Go to project root

```bash
cd /path/to/Axiom-I
```

### 2. Create and activate virtual environment

Linux or macOS:

```bash
uv venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
uv venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install backend dependencies (use uv pip only)

```bash
uv pip install -r Coding/Image/backend/requirements.txt
```

### 4. Install frontend dependencies

```bash
cd Coding/Image/frontend
npm install
cd ../../..
```

### 5. Create backend env file

```bash
cp Coding/Image/backend/.env.example Coding/Image/backend/.env
```

## Run the Project

Open two terminals.

Terminal 1 (backend):

```bash
cd Coding/Image/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Terminal 2 (frontend):

```bash
cd Coding/Image/frontend
npm run dev
```

Now open:

- Frontend: http://localhost:3000
- Backend docs: http://localhost:8000/docs

## Optional: Rebuild Training Seed Data

Use this when you want to rebuild seed features from the test set.

```bash
cd Coding/Image/backend
python train_from_testset.py --mode fallback --dataset-root ../test
```

Modes:

- fallback: faster
- full: slower, more detailed analysis

## Production Run

Frontend:

```bash
cd Coding/Image/frontend
npm run build
npm run start
```

Backend:

```bash
cd Coding/Image/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Notes

- Keep the virtual environment active while working.
- Use uv pip every time for Python package install.
- Do not use global pip install for this project.
