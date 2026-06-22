---
title: Axiom-I Backend
emoji: 🤩
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Axiom-I Backend Engine and API Guide

Welcome to the backend engine documentation for Axiom-I. This service acts as the analytical core of the platform, processing uploaded image and video assets to compute physical, signal, and deep learning anomalies before fusing them into a final forensic verdict.

Official project links:
* GitHub Repository: https://github.com/Adarsh-61/Axiom-I
* Backend Space: https://huggingface.co/spaces/Adarsh-61/axiom-backend

---

## 1. Setup and Installation

The backend is written in Python 3.11+ using the FastAPI framework. To isolate dependencies, run the service inside a dedicated Python virtual environment.

### Installation Steps

1. Navigate to the project root:
   ```bash
   cd /path/to/Axiom-I
   ```
2. Create and activate a virtual environment:
   ```bash
   uv venv .venv
   source .venv/bin/activate
   ```
   On Windows PowerShell:
   ```powershell
   uv venv .venv
   .venv\Scripts\Activate.ps1
   ```
3. Install backend dependencies using uv:
   ```bash
   uv pip install -r Coding/backend/requirements.txt
   ```
4. Copy the environment configuration template:
   ```bash
   cp Coding/backend/.env.example Coding/backend/.env
   ```

---

## 2. Configuration Settings

The application loads environment variables prefixed with `AXIOM_` using Pydantic Settings. Edit the `Coding/backend/.env` file to customize settings:

| Variable Name | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| **AXIOM_DEVICE** | str | cpu | Targets hardware for neural network execution (cpu or cuda). |
| **AXIOM_DEBUG** | bool | False | Activates verbose logs and API exception traces. |
| **AXIOM_API_HOST** | str | 0.0.0.0 | Bind host for the FastAPI server application. |
| **AXIOM_API_PORT** | int | 8000 | Bind port for the FastAPI server application. |
| **AXIOM_ALLOW_MODEL_DOWNLOAD** | bool | False | Allows automatic download of pre-trained ViT weights on startup. |
| **AXIOM_VIT_MODEL_NAME** | str | prithivMLmods/Deep-Fake-Detector-v2-Model | Hugging Face path for model weight ingestion. |
| **AXIOM_ALLOWED_ORIGINS** | list | ["http://localhost:3000"] | Allowed CORS origins for browser security. |
| **AXIOM_MAX_UPLOAD_SIZE** | int | 52428800 | Maximum payload size in bytes (default is 50 MB). |
| **AXIOM_MODE** | str | local | Storage mode (local or host). host syncs telemetry logs to Hugging Face. |
| **AXIOM_HF_TOKEN** | str | "" | Write token for Hugging Face private repository commits. |
| **AXIOM_HF_DATASET_PATH** | str | "" | Private Hugging Face dataset identifier (username/dataset). |

---

## 3. Running the Server

Start the FastAPI application using Uvicorn:

```bash
cd Coding/backend
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Once running, the following endpoints are available:
* Swagger UI documentation: http://localhost:8000/docs
* ReDoc documentation: http://localhost:8000/redoc
* Health check: http://localhost:8000/api/v1/health

---

## 4. API Endpoint Reference

All endpoints are prefixed with `/api/v1` and implement request rate limits.

### 4.1 GET /api/v1/health
* **Description**: Verifies that the service is running and checks whether ML model cache singletons (ViT classifier, MTCNN detector) are active.
* **Rate Limit**: 60 requests per minute.
* **Response Example (200 OK)**:
  ```json
  {
    "status": "ok",
    "service": "Axiom-I Media Forensics",
    "models": {
      "vit_classifier": "ready",
      "face_detector": "ready"
    }
  }
  ```

### 4.2 POST /api/v1/analyze
* **Description**: Accepts an image file payload, detects faces, extracts anomalies, and runs the Noisy-OR fusion and calibration logic.
* **Rate Limit**: 10 requests per minute.
* **Request Header**: `Content-Type: multipart/form-data`
* **Request Body**:
  * `file`: Binary image file (JPEG, PNG, WebP). Max 50 MB.
* **Response Example (200 OK)**:
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
    "steps": [
      { "step": 1, "label": "Input Image", "data": "data:image/jpeg;base64,..." }
    ],
    "full_image_score": 0.1495,
    "analysis_mode": "full_physics",
    "feature_vector": [0.981, 0.783, 0.236, 0.849, 0.918, 0.5, 1.0, 0.11, 0.44, 0.29, 0.42],
    "calibration_breakdown": {
      "calibrated_score": 0.1495,
      "heuristic_score": 0.7201,
      "learned_score": 0.0352,
      "model_weight": 0.833,
      "training_samples": 13290.0,
      "feedback_samples": 0.0
    }
  }
  ```

### 4.3 POST /api/v1/analyze/video
* **Description**: Uploads a video clip, samples keyframes, tracks facial regions, and evaluates temporal anomalies (rPPG, optical flow divergence, SH light drift).
* **Rate Limit**: 5 requests per minute.
* **Request Header**: `Content-Type: multipart/form-data`
* **Request Body**:
  * `file`: Binary video file (MP4, WebM, MOV, AVI). Max 50 MB.
* **Response Example (200 OK)**:
  ```json
  {
    "verdict": "Real",
    "confidence": 0.658,
    "faces_detected": 1,
    "analysis_mode": "video_full",
    "feature_vector": [0.0483, 0.9548, 1.0, 0.94, 0.07, 0.0794, 0.0286, 0.502, 1.0, 0.12, 0.85],
    "heuristic_score": 0.171,
    "calibrated_score": 0.171,
    "video_id": "8cf552a8-12cd-4da0-90eb-14b3dc04c552"
  }
  ```

### 4.4 POST /api/v1/feedback
* **Description**: Records user classification feedback to update the image calibration dataset.
* **Rate Limit**: 30 requests per minute.
* **Request Header**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "full_image_score": 0.1495,
    "original_prediction": "Real",
    "user_truth": "Real",
    "feature_vector": [0.981, 0.783, 0.236, 0.849, 0.918, 0.5, 1.0, 0.11, 0.44, 0.29, 0.42],
    "user_id": "client_abc123"
  }
  ```
* **Response Example (200 OK)**:
  ```json
  {
    "status": "success",
    "message": "Feedback recorded. Ground truth: Real.",
    "confusion_matrix": { "TP": 5700, "TN": 3566, "FP": 2724, "FN": 1300, "total": 13290 },
    "training_eligible": false,
    "training_exclusion_reason": "poison_suspected_sigma_outlier"
  }
  ```

### 4.5 POST /api/v1/feedback/video
* **Description**: Submits user feedback for a video analysis result.
* **Rate Limit**: 15 requests per minute.
* **Request Header**: `Content-Type: application/json`
* **Request Body**:
  ```json
  {
    "video_id": "8cf552a8-12cd-4da0-90eb-14b3dc04c552",
    "is_correct": true,
    "feature_vector": [0.0483, 0.9548, 1.0, 0.94, 0.07, 0.0794, 0.0286, 0.502, 1.0, 0.12, 0.85],
    "user_rating": 0.0
  }
  ```
* **Response Example (200 OK)**:
  ```json
  {
    "status": "success",
    "message": "Video feedback recorded."
  }
  ```

---

## 5. Security Protocols

* **CSP Headers**: The middleware enforces content isolation. In host mode (Hugging Face Spaces), standard headers are dynamically relaxed to allow embedding in cross-origin iframe interfaces.
* **Rate Limiting**: Custom limits prevent denial of service (DoS) attacks on heavy physics engines and PyTorch inference pipelines.
* **Adversarial Poison Guard**: Prior to training on user feedback, a multivariate distance validation check is run. If the telemetry features deviate from the seed dataset distribution by more than 4 standard deviations, the submission is rejected as suspected adversarial poisoning.

---

## 6. Project Architecture Details

The system employs a client-server microservices pattern:
* **Storage Provider**: A unified filesystem layer (`app/ml/storage_provider.py`) handles atomic file replacing. When running in local mode, logs are stored in `app/ml/feedback_data/`. When running in host mode, logs are synced to a private Hugging Face dataset.
* **Bifurcated pipelines**: Detailed image and video pipelines implement graceful fallback routing. If face detection fails, the system switches from face-local analysis to global scene/frequency checks.

---

## 7. Known Limitations

* **rPPG Neck ROI**: Face cropping padding (10%) restricts the neck area inside the cropped bounding box. Thus, the bottom section of the crop often reads the jawline or collar rather than the neck.
* **Untrained VideoMamba-Proxy**: The temporal backbone model uses an untrained proxy CNN architecture. Its Noisy-OR fusion weight is disabled (0.00) pending model training.

---

## 8. License

This module is distributed under the MIT License. See the public [LICENSE](https://github.com/Adarsh-61/Axiom-I/blob/main/LICENSE) file for details.
