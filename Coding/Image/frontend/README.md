# Axiom-I Frontend

This folder contains the web interface for Axiom-I.
Users upload an image, run analysis, and submit feedback from this UI.

## 1. Important Rules

- The full project uses a virtual environment.
- Do not install Python packages globally.
- Always use uv pip for Python dependencies.
- Use npm for frontend dependencies.

## 2. Frontend Tech Stack

- Next.js 16
- React 19
- TypeScript

## 3. Setup (Recommended Full Flow)

### Step 1: From project root, create and activate virtual environment

```bash
cd /path/to/Axiom-I
uv venv .venv
source .venv/bin/activate
```

### Step 2: Install backend Python dependencies with uv pip

```bash
uv pip install -r Coding/Image/backend/requirements.txt
```

### Step 3: Install frontend dependencies

```bash
cd Coding/Image/frontend
npm install
```

## 4. Run Frontend (Development)

```bash
cd /path/to/Axiom-I/Coding/Image/frontend
npm run dev
```

Open: http://localhost:3000

## 5. Run Backend (Required for API Calls)

In a second terminal:

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend docs: http://localhost:8000/docs

## 6. Frontend Environment (Optional)

If backend URL is different, create a local frontend env file:

```bash
cd /path/to/Axiom-I/Coding/Image/frontend
echo "NEXT_PUBLIC_API_BASE=http://localhost:8000" > .env.local
```

Then restart frontend.

## 7. Available Scripts

Run in this folder:

- npm run dev: start development server
- npm run build: create production build
- npm run start: run production server
- npm run lint: run TypeScript check

## 8. Production Run

```bash
cd /path/to/Axiom-I/Coding/Image/frontend
npm run build
npm run start
```

## 9. Common Problems

Problem: page loads but analysis fails.

- Check backend is running.
- Check NEXT_PUBLIC_API_BASE points to correct backend URL.

Problem: CORS error in browser.

- Update AXIOM_ALLOWED_ORIGINS in backend .env.
- Restart backend.

Problem: command not found for uv.

- Install uv first, then run setup again.

## 10. Final Reminder

- Keep virtual environment active while working.
- Use uv pip for Python installs every time.
- Do not install Python dependencies globally.
