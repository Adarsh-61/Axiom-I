---
title: Axiom Backend
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Axiom-I Backend API Guide


This file is a backend-only guide for Axiom-I.
It explains setup, run steps, API usage, response formats, rate limits, and calibration behavior.

## 1. Rules You Must Follow

- Use the project virtual environment.
- Do not install Python packages globally.
- Always use uv pip for Python dependency installation.

## 2. Backend Folder Overview

Main path: Coding/Image/backend

Important files:

- app/main.py: FastAPI app entry point.
- app/api/routes.py: analysis and health endpoints.
- app/api/feedback.py: feedback and metrics endpoints.
- app/api/schemas.py: request and response schemas.
- app/ml/pipeline.py: analysis pipeline and fusion logic.
- app/ml/evolution.py: calibration, feedback storage, trust weighting.
- .env.example: backend environment template.
- requirements.txt: Python dependencies.

## 3. Setup

From project root:

```bash
cd /path/to/Axiom-I
uv venv .venv
source .venv/bin/activate
uv pip install -r Coding/Image/backend/requirements.txt
cp Coding/Image/backend/.env.example Coding/Image/backend/.env
```

Windows PowerShell:

```powershell
cd /path/to/Axiom-I
uv venv .venv
.venv\Scripts\Activate.ps1
uv pip install -r Coding/Image/backend/requirements.txt
Copy-Item Coding/Image/backend/.env.example Coding/Image/backend/.env
```

## 4. Run Backend

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
cd Coding/Image/backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

OpenAPI docs:

- http://localhost:8000/docs

Health endpoint:

- http://localhost:8000/api/v1/health

## 5. Base API Information

- API prefix: /api/v1
- Content type for image analysis: multipart/form-data
- Main image field name: file

## 6. Rate Limits

Current limits configured in backend:

- POST /api/v1/analyze: 10 requests per minute
- POST /api/v1/analyze/video: 5 requests per minute
- GET /api/v1/health: 60 requests per minute
- POST /api/v1/feedback: 30 requests per minute
- POST /api/v1/feedback/video: 15 requests per minute
- GET /api/v1/feedback/metrics: 60 requests per minute


## 7. Endpoint Details

### 7.1 GET /

Purpose:

- Returns basic service info.

Example response:

```json
{
  "service": "Axiom-I Image Forensics",
  "health": "/api/v1/health",
  "docs": "/docs"
}
```

### 7.2 GET /api/v1/health

Purpose:

- Quick health check for backend service.

Example response:

```json
{
  "status": "ok",
  "service": "Axiom-I Image Forensics"
}
```

### 7.3 POST /api/v1/analyze

Purpose:

- Runs full image analysis and returns model output and diagnostics.

Request:

- Method: POST
- Body: multipart/form-data
- Field: file

Accepted file types:

- image/jpeg
- image/png
- image/webp
- image/*
- application/octet-stream

Validation rules:

- Empty file is rejected.
- File size limit: 50 MB.
- Max resolution: 20,000,000 pixels.

Example request:

```bash
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/absolute/path/to/image.jpg"
```

Example success response (shape):

```json
{
  "verdict": "Real",
  "confidence": 0.701,
  "faces_detected": 1,
  "faces": [
    {
      "bbox": [94, 35, 232, 198],
      "confidence": 0.99,
      "verdict": "Real",
      "score": 0.1495,
      "signal_breakdown": {
        "specular": 0.981,
        "frequency": 0.783,
        "topology": 0.236,
        "patch_consistency": 0.849,
        "wavelet_score": 0.918,
        "vit_score": 0.5,
        "physics_ensemble": 0.72,
        "raw_fusion": 0.518,
        "calibrated": 0.1495
      }
    }
  ],
  "steps": [],
  "full_image_score": 0.1495,
  "analysis_mode": "full_physics",
  "fallback_breakdown": null,
  "feature_vector": [0.981, 0.783, 0.236, 0.849, 0.918, 0.5, 1.0, 0.11, 0.44, 0.29, 0.42],
  "calibration_breakdown": {
    "calibrated_score": 0.1495,
    "heuristic_score": 0.7201,
    "learned_score": 0.0352,
    "model_weight": 0.833,
    "training_samples": 13290.0,
    "feedback_samples": 0.0,
    "feedback_weight_mean": 0.0
  },
  "quality_metrics": {
    "jpeg_blockiness": 0.11,
    "sharpness": 0.44,
    "colorfulness": 0.29,
    "resolution": 0.42
  },
  "process_inputs": {
    "image_shape": [512, 512, 3],
    "analysis_mode": "full_physics",
    "components": ["face_detector", "geometry_normals", "retinex", "spherical_harmonics", "specular_residual", "frequency", "topology", "patch_consistency", "wavelet", "vit", "calibration"],
    "feature_names": ["specular", "frequency", "topology", "patch_consistency", "wavelet_score", "vit_score", "face_present", "jpeg_blockiness", "sharpness", "colorfulness", "resolution"]
  },
  "decision_factors": {
    "heuristic_score": 0.7201,
    "learned_score": 0.0352,
    "model_weight": 0.833,
    "final_score": 0.1495
  },
  "explanation": [
    "Face region is analyzed with geometry, illumination, residual, and signal modules."
  ],
  "error": null
}
```

Possible error codes:

- 400: invalid file type, empty payload, decode failure
- 413: file too large or image resolution too large
- 429: rate limit reached
- 500: unexpected internal error

### 7.4 POST /api/v1/feedback

Purpose:

- Stores user correction data.
- Updates confusion matrix and calibration diagnostics.

Request body fields:

- full_image_score: number in range [0, 1]
- original_prediction: Real or Fake
- user_truth: Real or Fake
- feature_vector: optional list of numeric values
- user_id: optional identifier for trust weighting

Example request:

```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{
    "full_image_score": 0.1495,
    "original_prediction": "Real",
    "user_truth": "Real",
    "feature_vector": [0.981, 0.783, 0.236, 0.849, 0.918, 0.5, 1.0, 0.11, 0.44, 0.29, 0.42],
    "user_id": "demo_user_01"
  }'
```

Example success response:

```json
{
  "status": "success",
  "message": "Feedback recorded. Ground truth: Real.",
  "confusion_matrix": {
    "TP": 5700,
    "TN": 3566,
    "FP": 2724,
    "FN": 1300,
    "total": 13290
  },
  "training_eligible": false,
  "training_exclusion_reason": "poison_suspected_sigma_outlier",
  "calibration_metrics": {
    "total_samples": 13290,
    "brier_score": 0.1867,
    "log_loss": 0.5398,
    "ece": 0.09,
    "mce": 0.22,
    "mean_confidence": 0.62,
    "mean_accuracy": 0.70,
    "overconfidence_gap": -0.08,
    "bins": []
  },
  "user_trust_score": 0.5,
  "user_sample_weight": 1.0
}
```

Possible error codes:

- 400: invalid label or invalid score range
- 429: rate limit reached
- 500: internal error

### 7.5 GET /api/v1/feedback/metrics

Purpose:

- Returns confusion matrix, calibration metrics, calibration history, and feedback summary.

Example request:

```bash
curl -X GET "http://localhost:8000/api/v1/feedback/metrics" -H "accept: application/json"
```

Example response (shape):

```json
{
  "confusion_matrix": {
    "TP": 0,
    "TN": 0,
    "FP": 0,
    "FN": 0,
    "total": 0
  },
  "calibration_metrics": {
    "total_samples": 0,
    "brier_score": 0.0,
    "log_loss": 0.0,
    "ece": 0.0,
    "mce": 0.0,
    "mean_confidence": 0.0,
    "mean_accuracy": 0.0,
    "overconfidence_gap": 0.0,
    "bins": []
  },
  "calibration_history": [],
  "feedback_summary": {
    "total_feedback_records": 0,
    "training_eligible_records": 0,
    "training_excluded_records": 0,
    "training_exclusion_reasons": {},
    "trust_summary": {
      "tracked_users": 0,
      "mean_trust_score": null,
      "min_trust_score": null,
      "max_trust_score": null,
      "mean_sample_weight": null
    }
  }
}
```

### 7.6 POST /api/v1/analyze/video

Purpose:

- Saves video upload to a temporary path, runs the physics-based deepfake video detection pipeline (optical flow, rPPG, lighting, etc.), and returns the calibrated forensic verdict.

Request:

- Method: POST
- Body: multipart/form-data
- Field: file

Accepted video formats:

- .mp4, .webm, .mov, .avi

Validation rules:

- Reject mismatching MIME content types.
- Header magic bytes check.
- Upload size limit: 50 MB.

Example response:

```json
{
  "verdict": "Real",
  "confidence": 0.658,
  "faces_detected": 1,
  "analysis_mode": "video_full",
  "feature_vector": [0.0483, 0.9548, 1.0, 0.94, 0.07, 0.0794, 0.0286, 0.502, 1.0, 0.12, 0.85],
  "quality_metrics": {
    "video_quality": 0.85
  },
  "heuristic_score": 0.171,
  "calibrated_score": 0.171,
  "calibration_breakdown": {
    "model_weight": 0.0,
    "learned_score": 0.5,
    "heuristic_score": 0.171,
    "calibrated_score": 0.171
  },
  "decision_factors": {
    "final_score": 0.171,
    "heuristic_score": 0.171,
    "learned_score": 0.5,
    "model_weight": 0.0,
    "physics_ensemble": 0.452
  },
  "full_image_score": 0.171,
  "contributions": [
    {
      "signal": "optical_boundary",
      "raw_score": 0.0483,
      "weight": 0.18,
      "contribution": 0.0087
    }
  ],
  "explanation": [
    "Video frames were adaptively sampled for efficiency.",
    "Face-local physics (Optical Flow, rPPG) were extracted if a face was present."
  ]
}
```

### 7.7 POST /api/v1/feedback/video

Purpose:

- Submits ground-truth correction feedback for a specific analyzed video.
- Integrates Poison Guard checks: ignores/rejects feedback if the user's label contradicts the physics ensemble results drastically.

Request:

- Method: POST
- Content-Type: application/json
- Fields:
  - video_id: string
  - is_correct: boolean (whether model verdict matches user's ground-truth)
  - feature_vector: list of 11 floats
  - user_rating: float (1.0 for Fake, 0.0 for Real)

Example request:

```json
{
  "video_id": "8cf552a8-12cd-4da0-90eb-14b3dc04c552",
  "is_correct": false,
  "feature_vector": [0.0483, 0.9548, 1.0, 0.94, 0.07, 0.0794, 0.0286, 0.502, 1.0, 0.12, 0.85],
  "user_rating": 1.0
}
```

Example success response:

```json
{
  "status": "success",
  "message": "Video feedback recorded."
}
```

## 8. Calibration Notes


### 8.1 Feature Vector

- Expected feature length: 11.
- If feature vector is missing or invalid, the record can still be stored but training eligibility may become false.

### 8.2 Training Eligibility Rules

A feedback record must pass all checks below to be used for training:

- confidence margin check: abs(full_image_score - 0.5) >= 0.05
- resolution feature check: feature_vector[10] >= 0.08
- sharpness feature check: feature_vector[8] >= 0.05
- poison guard checks must pass

Common exclusion reasons:

- missing_feature_vector
- low_confidence_margin
- low_resolution
- low_sharpness
- poison_suspected_sigma_outlier
- poison_suspected_label_mismatch

### 8.3 Trust Weighting

- User trust score is converted into sample weight.
- Current sample weight range: 0.40 to 1.60.
- Seed samples are given base weight 1.15.

### 8.4 Files Used for Feedback and Calibration

Stored in app/ml/feedback_data:

- feedback_log.json
- confusion_matrix.json
- calibration_metrics.json
- calibration_history.json
- quarantine_log.json
- seed_features.json

## 9. Security and Reliability Notes

- Security headers are applied by middleware.
- CORS is enabled for configured frontend origins.
- Feedback and metrics endpoints are rate-limited.
- JSON write operations for feedback and calibration use atomic replace flow.
- ViT model loading is offline-safe by default unless AXIOM_ALLOW_MODEL_DOWNLOAD=true.

## 10. Quick Verification Commands

Backend syntax check:

```bash
cd /path/to/Axiom-I
source .venv/bin/activate
python -m compileall -q Coding/Image/backend
```

Backend smoke call:

```bash
curl -X GET "http://localhost:8000/api/v1/health"
```

This confirms the backend is reachable.
