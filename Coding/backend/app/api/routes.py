import numpy as np
import cv2
import asyncio
from fastapi import APIRouter, File, UploadFile, HTTPException, Request
from app.api.schemas import AnalysisResponse, VisualizationStep, FaceResult, SignalBreakdown
from app.config import settings
from app.ml.pipeline import analyze
from app.security.rate_limiter import limiter
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/analyze", response_model=AnalysisResponse)
@limiter.limit("10/minute")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    allowed_exact = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}
    if content_type and (content_type not in allowed_exact and not content_type.startswith("image/")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only image uploads are allowed.")

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload payload.")
        if len(contents) > settings.MAX_UPLOAD_SIZE:
            limit_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
            raise HTTPException(status_code=413, detail=f"File exceeds {limit_mb}MB limit.")

        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to decode image data.")

        h, w = img_bgr.shape[:2]
        if h * w > settings.MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image resolution is too large.")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        logger.info("Analysis request received.")

        result_dict = await asyncio.to_thread(analyze, img_rgb)
        face_results = []
        for f in result_dict.get("faces", []):
            sb = f.get("signal_breakdown")
            sb_model = SignalBreakdown(**sb) if sb else None
            face_results.append(FaceResult(
                bbox=f.get("bbox", [0, 0, 0, 0]),
                confidence=f.get("confidence", 0.0),
                verdict=f.get("verdict", "Unknown"),
                score=f.get("score", 0.0),
                signal_breakdown=sb_model,
            ))

        response = AnalysisResponse(
            verdict=result_dict["verdict"],
            confidence=result_dict["confidence"],
            faces_detected=result_dict["faces_detected"],
            faces=face_results,
            steps=[VisualizationStep(**s) for s in result_dict.get("steps", [])],
            full_image_score=result_dict.get("full_image_score"),
            analysis_mode=result_dict.get("analysis_mode"),
            fallback_breakdown=result_dict.get("fallback_breakdown"),
            feature_vector=result_dict.get("feature_vector"),
            calibration_breakdown=result_dict.get("calibration_breakdown"),
            quality_metrics=result_dict.get("quality_metrics"),
            process_inputs=result_dict.get("process_inputs"),
            decision_factors=result_dict.get("decision_factors"),
            explanation=result_dict.get("explanation"),
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during analysis.")


@router.get("/health")
@limiter.limit("60/minute")
async def health_check(request: Request):
    from app.ml.vit_classifier import warmup_classifier
    from app.ml.face_detector import warmup_detector
    
    vit_ok = warmup_classifier()
    detector_ok = warmup_detector()
    
    status = "ok" if (vit_ok and detector_ok) else "degraded"
    
    return {
        "status": status,
        "service": settings.PROJECT_NAME,
        "models": {
            "vit_classifier": "ready" if vit_ok else "unavailable",
            "face_detector": "ready" if detector_ok else "unavailable"
        }
    }
