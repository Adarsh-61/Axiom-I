import os
import uuid
import tempfile
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, Request
from pydantic import BaseModel

from app.ml.video_pipeline import analyze_video
from app.ml.video_evolution import add_video_feedback
from app.security.rate_limiter import limiter
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Video Forensics"])

class VideoFeedbackRequest(BaseModel):
    video_id: str
    is_correct: bool
    feature_vector: list[float]
    user_rating: float

def validate_video_header(header_bytes: bytes) -> bool:
    """
    Validates file headers (magic bytes) for common video containers:
    - WebM: starts with \x1a\x45\xdf\xa3 (EBML)
    - AVI: starts with RIFF, has AVI at offset 8
    - MP4/MOV: has an ftyp box near the file start
    """
    if header_bytes.startswith(b'\x1a\x45\xdf\xa3'):
        return True
    if header_bytes.startswith(b'RIFF') and len(header_bytes) >= 12 and header_bytes[8:12] == b'AVI ':
        return True
    if len(header_bytes) >= 12 and header_bytes[4:8] == b'ftyp':
        return True
    return False

@router.post("/analyze/video")
@limiter.limit("5/minute")
async def analyze_video_endpoint(request: Request, file: UploadFile = File(...)):
    # Basic extension validation
    filename_lower = (file.filename or "").lower()
    if not filename_lower.endswith((".mp4", ".webm", ".mov", ".avi")):
        raise HTTPException(status_code=400, detail="Invalid video format extension.")

    # MIME type validation if present
    content_type = (file.content_type or "").split(";", 1)[0].strip().lower()
    if content_type and not (content_type.startswith("video/") or content_type == "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Invalid MIME type. Expected video.")

    # Save uploaded video to a temporary file in chunks to prevent memory exhaustion
    temp_dir = tempfile.gettempdir()
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename or "")[1]
    temp_path = os.path.join(temp_dir, f"{video_id}{ext}")

    total_size = 0
    chunk_size = 1024 * 1024  # 1MB chunk size
    header_verified = False

    try:
        with open(temp_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                total_size += len(chunk)
                
                # Dynamic size check
                if total_size > settings.MAX_UPLOAD_SIZE:
                    limit_mb = settings.MAX_UPLOAD_SIZE // (1024 * 1024)
                    raise HTTPException(
                        status_code=413, 
                        detail=f"Video exceeds the maximum size limit of {limit_mb}MB."
                    )
                
                # Magic bytes check on the first chunk
                if not header_verified:
                    if not validate_video_header(chunk):
                        raise HTTPException(
                            status_code=400, 
                            detail="File content verification failed: not a valid video structure."
                        )
                    header_verified = True
                
                f.write(chunk)

        if total_size == 0:
            raise HTTPException(status_code=400, detail="Empty upload payload.")

        logger.info(f"Analyzing video: {file.filename} (Size: {total_size} bytes)")

        # Run pipeline
        result = analyze_video(temp_path)

        # Inject video_id for feedback tracking
        result["video_id"] = video_id
        return result

    except HTTPException:
        # Re-raise standard FastAPI HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error analyzing video {file.filename}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Video analysis failed")
    finally:
        # Cleanup temp file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")

@router.post("/feedback/video")
@limiter.limit("15/minute")
async def video_feedback_endpoint(request: Request, req: VideoFeedbackRequest):
    try:
        add_video_feedback(
            video_id=req.video_id,
            is_correct=req.is_correct,
            feature_vector=req.feature_vector,
            user_rating=req.user_rating
        )
        return {"status": "success", "message": "Video feedback recorded."}
    except ValueError as e:
        # ValidationError / Poison Guard triggers
        logger.warning(f"Invalid video feedback: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error recording video feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to record video feedback.")
