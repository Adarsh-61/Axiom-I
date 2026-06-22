from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class VisualizationStep(BaseModel):
    step: int
    label: str
    data: str


class SignalBreakdown(BaseModel):
    specular: float = 0.0
    frequency: float = 0.0
    topology: float = 0.0
    patch_consistency: float = 0.0
    wavelet_score: float = 0.0
    vit_score: float = 0.5
    physics_ensemble: float = 0.0
    raw_fusion: float = 0.0
    calibrated: float = 0.0


class FaceResult(BaseModel):
    bbox: List[int]
    confidence: float
    verdict: str
    score: float
    signal_breakdown: Optional[SignalBreakdown] = None


class AnalysisResponse(BaseModel):
    verdict: str
    confidence: float
    faces_detected: int
    faces: List[FaceResult] = Field(default_factory=list)
    steps: List[VisualizationStep] = Field(default_factory=list)
    full_image_score: Optional[float] = None
    analysis_mode: Optional[str] = None
    fallback_breakdown: Optional[Dict[str, float]] = None
    feature_vector: Optional[List[float]] = None
    calibration_breakdown: Optional[Dict[str, float]] = None
    quality_metrics: Optional[Dict[str, float]] = None
    process_inputs: Optional[Dict[str, Any]] = None
    decision_factors: Optional[Dict[str, float]] = None
    explanation: Optional[List[str]] = None
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    full_image_score: float = 0.0
    original_prediction: str
    user_truth: str
    feature_vector: Optional[List[float]] = None
    user_id: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    message: str
    confusion_matrix: Dict[str, int]
    training_eligible: bool
    training_exclusion_reason: Optional[str] = None
    calibration_metrics: Optional[Dict[str, Any]] = None
    user_trust_score: Optional[float] = None
    user_sample_weight: Optional[float] = None


class FeedbackDiagnosticsResponse(BaseModel):
    confusion_matrix: Dict[str, int]
    calibration_metrics: Dict[str, Any]
    calibration_history: List[Dict[str, Any]] = Field(default_factory=list)
    feedback_summary: Dict[str, Any]
