from fastapi import APIRouter, HTTPException, Request
import logging

from app.api.schemas import FeedbackRequest, FeedbackResponse, FeedbackDiagnosticsResponse
from app.ml.evolution import record_feedback, get_feedback_diagnostics, normalize_label
from app.security.rate_limiter import limiter

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit("30/minute")
async def submit_feedback(request: Request, payload: FeedbackRequest):
    try:
        predicted_label = normalize_label(payload.original_prediction)
        truth_label = normalize_label(payload.user_truth)

        if predicted_label is None or truth_label is None:
            raise HTTPException(status_code=400, detail="Prediction and truth must each be 'Real' or 'Fake'.")

        logger.info(
            f"Received feedback: predicted={predicted_label}, "
            f"truth={truth_label}"
        )

        updated_matrix = record_feedback(
            full_image_score=payload.full_image_score,
            original_prediction=predicted_label,
            user_truth=truth_label,
            feature_vector=payload.feature_vector,
            user_id=payload.user_id,
        )

        return FeedbackResponse(
            status="success",
            message=f"Feedback recorded. Ground truth: {truth_label}.",
            confusion_matrix=updated_matrix["confusion_matrix"],
            training_eligible=updated_matrix["training_eligible"],
            training_exclusion_reason=updated_matrix["training_exclusion_reason"],
            calibration_metrics=updated_matrix["calibration_metrics"],
            user_trust_score=updated_matrix.get("user_trust_score"),
            user_sample_weight=updated_matrix.get("user_sample_weight"),
        )

    except HTTPException:
        raise

    except ValueError as e:
        logger.warning(f"Invalid feedback payload: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Failed to process feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/feedback/metrics", response_model=FeedbackDiagnosticsResponse)
@limiter.limit("60/minute")
async def get_feedback_metrics(request: Request):
    try:
        return FeedbackDiagnosticsResponse(**get_feedback_diagnostics())
    except Exception as e:
        logger.error(f"Failed to load feedback diagnostics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

