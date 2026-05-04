# Axiom-I: Image Forensics Analysis System

## Overview

Axiom-I is an image forensics system designed to detect synthetic imagery (deepfakes, GAN-generated images, and manipulated media) using physics-based analysis. Instead of relying on a single neural network, Axiom-I uses multiple independent analysis methods and combines their results to make a final decision.

The system analyzes images through the following pipeline:

1. **Face Detection** (MTCNN) - Finds and crops the face from the image
2. **Surface Normals** - Estimates the 3D shape of the face
3. **Retinex Texture Extraction** - Separates skin texture from lighting
4. **Spherical Harmonics** - Models the lighting on the face
5. **Specular Residual** - Extracts shiny reflections and checks if they match the texture
6. **Frequency Analysis (FFT)** - Detects unnatural frequency patterns left by GANs
7. **Patch Noise Consistency** - Checks if camera noise is uniform across the face
8. **Topological Complexity** - Measures how fragmented the specular reflections are
9. **Wavelet Decomposition** - Detects directional artifacts from image generation
10. **Vision Transformer (ViT)** - Neural network trained on deepfake detection
11. **Noisy-OR Fusion** - Combines all 6 anomaly scores probabilistically
12. **Calibration** - Blends physics score with a learned model (trust-based weighting)
13. **Final Verdict** - Compares the final score to 0.50 threshold

If no face is detected, the system switches to a fallback mode that uses only FFT, Wavelet, and ViT signals.

---

## Technology Stack

### Frontend
- **Framework:** Next.js (App Router)
- **Language:** TypeScript
- **Styling:** Vanilla CSS with CSS variables
- **Math Rendering:** KaTeX
- **Pages:**
  - `/` - Main analysis workspace (upload, results, pipeline flow, decision summary)
  - `/math` - Deep mathematics page (complete step-by-step derivations with actual values)

### Backend
- **Framework:** FastAPI (Python)
- **ML Libraries:** NumPy, OpenCV, PyWavelets, PyTorch, Transformers
- **Endpoints:**
  - `POST /api/analyze` - Accepts an image and returns the full analysis
  - `POST /api/feedback` - Accepts user feedback for calibration model training
  - `GET /api/feedback/diagnostics` - Returns confusion matrix and calibration metrics
  - `GET /api/health` - Health check

---

## Project Structure

```
Coding/Image/
  backend/
    app/
      api/
        routes.py          - FastAPI endpoints
        schemas.py         - Pydantic response models
        feedback.py        - Feedback submission and diagnostics
      ml/
        pipeline.py        - Main orchestrator (calls all modules)
        face_detector.py   - MTCNN face detection with bounding box padding
        face_alignment.py  - Bilateral filter + Sobel normal estimation
        retinex.py         - Multi-Scale Retinex (3 scales: 15, 80, 120)
        illumination.py    - 9-term Spherical Harmonics least squares
        specular.py        - Lambertian subtraction for specular residual
        sri_net.py         - NCC specular anomaly + Noisy-OR fusion
        frequency.py       - FFT radial power spectrum + HFER
        patch_analysis.py  - Laplacian PRNU with 4x4 grid
        topology.py        - Connected components at 3 thresholds
        wavelet.py         - Daubechies-2 DWT with HH weighting
        vit_classifier.py  - Pre-trained Vision Transformer
        evolution.py       - Trust-based calibration with feedback learning
      config.py            - Application settings
      security/
        rate_limiter.py    - Rate limiting middleware
  frontend/
    src/app/
      page.tsx             - Main analysis workspace UI
      math/page.tsx        - Deep mathematics derivation page
      helpers.ts           - Types, API wrappers, pipeline step definitions
      Tex.tsx              - KaTeX rendering component
      globals.css          - All styles (CSS variables, components, math page)
      layout.tsx           - Root layout with meta tags
```

---

## Key Formulas (Backend to Frontend Verification)

All formulas on the frontend have been verified against the backend source code:

| Signal | Backend File | Formula |
|--------|-------------|---------|
| Specular | sri_net.py | sigma(15 * (NCC - 0.30)) |
| Frequency | frequency.py | 1/(1+e^(3*(log10(HFER)+4.5))) |
| Patch | patch_analysis.py | 1/(1+e^(20*(CV-0.25))) |
| Topology | topology.py | sigma(0.11*(C-18)) |
| Wavelet | wavelet.py | sigma(3*(ln(1+E)-4.5)) |
| ViT | vit_classifier.py | softmax(logits)[fake_id] |
| Fusion | sri_net.py | 1-prod(1-w*s), then sigma(8*(ens-0.40)) |

---

## Calibration: Trust-Based Model Weighting

The system uses a "trust must be earned" approach for blending the physics-based heuristic score with the learned model score:

```
if feedback_count < 15:
    model_weight = 0.0      # Pure physics, model has no say
else:
    model_weight = 0.10 + 0.02 * (feedback_count - 15)   # Gradual ramp-up
    model_weight = clamp(model_weight, 0.0, 0.80)

final_score = model_weight * learned_score + (1 - model_weight) * heuristic_score
```

| Feedback count | Model weight | Who decides? |
|---|---|---|
| 0 to 14 | 0.00 | Pure physics heuristic |
| 15 | 0.10 | 10% model, 90% physics |
| 25 | 0.30 | 30% model, 70% physics |
| 35 | 0.50 | Equal blend |
| 50+ | 0.80 | Model dominates (earned trust) |

### Poison Guard (Anti-Tampering)

Before accepting user feedback for training, a poison guard checks whether the feedback is suspicious:
- Features more than 4 sigma from the claimed class mean (max 5 outlier features allowed)
- If the image is statistically closer to the opposite class (with 0.50 margin), it is rejected

---

## Color Palette

The UI uses a consistent, purpose-driven color palette:

| Element | Color | Usage |
|---------|-------|-------|
| Main title | Blue (#1a73e8) | Primary brand accent |
| Step numbers | Blue circles | Numbered indicators |
| Python file badges | Green (#1b6e2d on #e6f4ec) | Code file indicators |
| "View full mathematics" button | Red (#c5221f) with white text | Primary action |
| Back button | Dark gray with left arrow | Navigation |
| All body text | Black (#1a1a1a) | Content |
| Real verdict | Green (#0d7a3e) | Positive outcome |
| Fake verdict | Red (#c62828) | Negative outcome |

---

## Configuration

### Environment Variables
- `NEXT_PUBLIC_API_BASE` - Backend URL (default: `http://localhost:8000`)

### Backend Settings (app/config.py)
- `MAX_UPLOAD_SIZE` - Maximum file size for uploads
- `MAX_IMAGE_PIXELS` - Maximum image resolution
- `MIN_FACE_SIZE` - Minimum face dimension (64px)
- `MIN_FACE_CONFIDENCE` - Minimum MTCNN confidence (0.9)

---

## Running Locally

### Backend
```bash
cd Coding/Image/backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd Coding/Image/frontend
npm install
npm run dev
```

The frontend runs on `http://localhost:3000` and proxies API calls to the backend at `http://localhost:8000`.

---

## Deep Mathematics Page

The `/math` page provides a complete walkthrough of every calculation. It reads the actual analysis results from sessionStorage (passed from the main page when the user clicks "View full mathematics"). For each step, it shows:

- **What the file does** in simple, clear English
- **Why it is needed** for deepfake detection
- **The complete formula** with KaTeX rendering
- **Actual computed values** from the specific image that was analyzed

This includes the full Noisy-OR fusion table showing all 6 signal weights, individual scores, and the step-by-step multiplication that produces the ensemble probability.

All explanations are written in simple, beginner-friendly English with no jargon, no emojis, and no em dashes.
