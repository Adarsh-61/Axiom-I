# Axiom-I

Axiom-I is a hardware-agnostic image forensics system for deepfake detection. It combines physics-driven image signals, learned calibration, and a feedback loop to improve reliability over time.

## Key Capabilities

- FastAPI backend for image analysis, health monitoring, and feedback ingestion.
- Multi-signal forensic pipeline with full-physics and fallback modes.
- Calibration engine trained from seed samples and trust-weighted user feedback.
- Next.js frontend for upload, analysis visualization, metrics, and feedback workflows.
- Ready-to-use diagrams and presentation material for project demonstration.

## Project Layout

- `Coding/Image/backend/`: Backend API, pipeline modules, calibration logic, training utility.
- `Coding/Image/frontend/`: Frontend application (Next.js + TypeScript).
- `Coding/Image/test/`: Real/Fake dataset used for local calibration and validation.
- `Documents/Image/Diagrams/`: Exported architecture and flow diagrams.
- `Documents/Image/Axiom-I.odp`: Presentation source.

## Prerequisites

- Python 3.11+ (project currently tested with virtual environment setup).
- Node.js 20+ and npm.
- Linux/macOS/Windows environment with OpenCV-compatible dependencies.

## Backend Setup

```bash
cd Coding/Image/backend
python -m pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend base URL: `http://localhost:8000`

## Frontend Setup

```bash
cd Coding/Image/frontend
npm install
npm run dev
```

Frontend URL: `http://localhost:3000`

## Environment Variables

Backend variables are loaded from `Coding/Image/backend/.env` using `AXIOM_` prefix.

- `AXIOM_DEVICE`: runtime device (`cpu`, `cuda`, etc.).
- `AXIOM_DEBUG`: enable/disable debug mode.
- `AXIOM_API_HOST`: backend host.
- `AXIOM_API_PORT`: backend port.
- `AXIOM_ALLOWED_ORIGINS`: JSON array of allowed CORS origins.
- `AXIOM_ALLOW_MODEL_DOWNLOAD`: if `true`, allows first-run model download.
- `AXIOM_VIT_MODEL_NAME`: Hugging Face model identifier for ViT classifier.

## Calibration and Training

The backend uses seed features plus accepted feedback records for calibration.

Run seed regeneration and model rebuild:

```bash
cd Coding/Image/backend
python train_from_testset.py --mode fallback --dataset-root ../test
```

Modes:

- `fallback`: quicker, full-image signal path.
- `full`: full-physics extraction, slower but richer signal set.

## API Summary

- `GET /`: service metadata and links.
- `GET /api/v1/health`: service health status.
- `POST /api/v1/analyze`: image forensic analysis.
- `POST /api/v1/feedback`: user feedback submission.
- `GET /api/v1/feedback/metrics`: confusion matrix and calibration diagnostics.

## Security and Reliability Notes

- Rate limiting is enabled for analysis and feedback endpoints.
- Security headers and CORS controls are enabled in backend and frontend layers.
- JSON persistence for calibration artifacts uses atomic writes for safer updates.
- By default, model auto-download is disabled for predictable offline behavior.

## Repository Hygiene

The repository is configured for clean source control:

- Excludes local virtual environments, caches, build outputs, and logs.
- Excludes runtime-only feedback logs and temporary calibration artifacts.
- Excludes external reference research PDFs not required to run the system.

Included assets are production-relevant: source code, dataset, diagrams, and presentation material.
