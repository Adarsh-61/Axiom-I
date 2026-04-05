# Axiom-I Frontend

This folder has the web app for Axiom-I.

## Important Rules

- The full project uses a virtual environment.
- Do not install Python packages globally.
- Always use uv pip for Python dependencies.
- For this frontend folder, use npm for JavaScript dependencies.

## Full Project Setup (Quick)

Run these from the project root first:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r Coding/Image/backend/requirements.txt
```

Then install frontend packages:

```bash
cd Coding/Image/frontend
npm install
```

## Run Frontend

```bash
npm run dev
```

Open: http://localhost:3000

## If Backend Is Not Running

In another terminal:

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend docs: http://localhost:8000/docs

## Production Commands

```bash
npm run build
npm run start
```

## Reminder

- Keep virtual environment active while working.
- Use uv pip for Python package installation every time.
