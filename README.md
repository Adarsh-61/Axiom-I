# Axiom-I

Axiom-I is a hardware-agnostic image forensics framework focused on practical deepfake detection using engineered visual physics signals and lightweight calibration.

## Repository Structure

- `Coding/Image/backend/` FastAPI backend and forensic pipeline modules.
- `Coding/Image/frontend/` Next.js frontend for analysis, diagnostics, and feedback.
- `Coding/Image/test/` Real/Fake image dataset used for local calibration and testing.
- `Documents/Image/Diagrams/` Presentation-ready architecture and flow diagrams.
- `Documents/Image/Axiom-I.odp` Project presentation source.

## Backend Setup

```bash
cd Coding/Image/backend
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

```bash
cd Coding/Image/frontend
npm install
npm run dev
```

## Environment Configuration

1. Copy `Coding/Image/backend/.env.example` to `Coding/Image/backend/.env`.
2. Update values if needed for your local machine.

## Local Calibration / Training

```bash
cd Coding/Image/backend
python train_from_testset.py --mode fallback --dataset-dir ../test
```

Use `--mode full` for full-physics feature extraction (slower, higher compute).

## Notes

- Build outputs, caches, runtime logs, and local virtual environments are excluded via `.gitignore`.
- External reference research PDFs are intentionally excluded from source control.
- The repository includes code, dataset, and diagrams needed to run and present Axiom-I locally.
