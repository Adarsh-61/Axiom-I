import json
import logging
import os
import tempfile
from threading import RLock

import numpy as np

logger = logging.getLogger(__name__)

# Video-specific calibration persistence
DB_FILE = os.path.join(os.path.dirname(__file__), "video_feedback_db.json")
CALIBRATION_FILE = os.path.join(os.path.dirname(__file__), "video_calibration_db.json")
db_lock = RLock()

# Same philosophy as Image Module
REQUIRED_FEEDBACKS_FOR_EVOLUTION = 15
MAX_ML_WEIGHT = 0.80

from app.ml.storage_provider import load_json_data, save_json_data

def _load_db(filepath: str, default: dict) -> dict:
    return load_json_data(os.path.basename(filepath), default)

def _save_db(filepath: str, data: dict):
    save_json_data(os.path.basename(filepath), data)

def calibrate_video_score(heuristic_score: float) -> float:
    """
    Calibrates the heuristic score using historical feedback trust.
    Matches Image Module exactly: No weight until 15 feedbacks, then linear ramp.
    """
    stats = _load_db(CALIBRATION_FILE, {"total_feedback": 0, "trust_score": 0.5})
    
    count = stats.get("total_feedback", 0)
    if count < REQUIRED_FEEDBACKS_FOR_EVOLUTION:
        return heuristic_score
        
    trust = stats.get("trust_score", 0.5)
    
    # Ramp up weight
    weight = 0.10 + (0.02 * min(count - REQUIRED_FEEDBACKS_FOR_EVOLUTION, 35))
    weight = min(MAX_ML_WEIGHT, max(0.0, weight))
    
    # Simple linear interpolation with the historical trust
    # If trust is high (>0.5), it nudges the score slightly.
    # We will refine this if we want full model retraining later.
    calibrated = heuristic_score * (1.0 - weight) + trust * weight
    return float(np.clip(calibrated, 0.0, 1.0))

def add_video_feedback(video_id: str, is_correct: bool, feature_vector: list[float], user_rating: float):
    """
    Registers user feedback. 
    Implements Poison Guard: rejects feedback if the user_rating drastically contradicts physics.
    """
    if not isinstance(video_id, str) or not video_id.strip():
        raise ValueError("video_id must be a non-empty string.")

    if len(feature_vector) != 11:
        raise ValueError("Feature vector must have exactly 11 elements.")

    if not np.isfinite(user_rating) or user_rating < 0.0 or user_rating > 1.0:
        raise ValueError("user_rating must be a finite float in [0, 1].")

    cleaned_features = []
    for value in feature_vector:
        scalar = float(value)
        if not np.isfinite(scalar):
            scalar = 0.0
        cleaned_features.append(float(np.clip(scalar, 0.0, 1.0)))
        
    with db_lock:
        db = _load_db(DB_FILE, {"feedbacks": []})
        
        # Poison Guard: Z-Score Outlier Check (simplified)
        # If the physics strongly say Fake (>0.8) and user says Real (0.0), flag it.
        physics_score = sum(cleaned_features[0:8]) / 8.0 # Rough estimate
        
        if physics_score > 0.8 and user_rating < 0.2:
            logger.warning("Poison Guard triggered: Ignoring highly contradictory feedback.")
            raise ValueError("Poison Guard triggered: Contradictory feedback rejected.")
            
        if physics_score < 0.2 and user_rating > 0.8:
            logger.warning("Poison Guard triggered: Ignoring highly contradictory feedback.")
            raise ValueError("Poison Guard triggered: Contradictory feedback rejected.")
            
        db["feedbacks"].append({
            "video_id": video_id,
            "is_correct": is_correct,
            "features": cleaned_features,
            "user_rating": float(user_rating)
        })
        
        _save_db(DB_FILE, db)
        _recalculate_video_trust()

def _recalculate_video_trust():
    """Bayesian Beta-Binomial updating for trust score."""
    with db_lock:
        db = _load_db(DB_FILE, {"feedbacks": []})
        feedbacks = db["feedbacks"]
        
        alpha = 1.0
        beta = 1.0
        
        for fb in feedbacks:
            if fb["is_correct"]:
                alpha += 1.0
            else:
                beta += 1.0
                
        trust = alpha / (alpha + beta)
        
        cal_db = {
            "total_feedback": len(feedbacks),
            "trust_score": trust,
            "alpha": alpha,
            "beta": beta
        }
        _save_db(CALIBRATION_FILE, cal_db)
