import logging
import numpy as np
from typing import Any

from app.ml.video_ingest import ingest_video
from app.ml.face_tracker import track_main_face
from app.ml.optical_boundary import compute_optical_boundary_anomaly
from app.ml.rppg_signal import compute_rppg_anomaly
from app.ml.temporal_lighting import _fit_sh_for_video, compute_temporal_lighting_anomaly
from app.ml.temporal_fft import compute_temporal_fft_anomaly
from app.ml.wavelet_temporal import compute_temporal_wavelet_anomaly
from app.ml.compression_residual import compute_compression_residual_anomaly
from app.ml.fullframe_temporal import compute_global_scene_anomaly
from app.ml.temporal_backbone import get_temporal_backbone_score
from app.ml.video_sri_net import evaluate_video_signals
from app.ml.video_evolution import calibrate_video_score
from app.ml.face_alignment import get_3d_shape
from app.ml.retinex import extract_msr

logger = logging.getLogger(__name__)

def analyze_video(file_path: str) -> dict[str, Any]:
    """
    Master pipeline for Axiom-I Video Deepfake Detection.
    Implements Graceful Degradation (Fallback Mode) and 11-D Feature Fusion.
    """
    try:
        # 1. Ingestion
        frames, fps, quality_metrics = ingest_video(file_path, max_frames=30)
        
        # 2. Face Gateway
        face_crops, avg_face_size = track_main_face(frames)
        face_present = len(face_crops) > 0
        
        # Initialize features
        optical_score = 0.5
        rppg_score = 0.5
        lighting_score = 0.5
        
        # 3. Branch A: Face-Local Physics
        if face_present:
            optical_score = compute_optical_boundary_anomaly(face_crops)
            rppg_score = compute_rppg_anomaly(face_crops, fps)
            
            # Temporal Lighting requires 3D normals and MSR texture
            gamma_seq = []
            for crop in face_crops:
                try:
                    _, normals = get_3d_shape(crop)
                    texture = extract_msr(crop, sigmas=[15, 80, 120])
                    gamma = _fit_sh_for_video(crop, normals, texture)
                    gamma_seq.append(gamma)
                except Exception:
                    gamma_seq.append(np.zeros((9, 3), dtype=np.float32))
            lighting_score = compute_temporal_lighting_anomaly(gamma_seq)
            
            # Run these on the face crops for higher precision
            fft_score = compute_temporal_fft_anomaly(face_crops)
            wavelet_score = compute_temporal_wavelet_anomaly(face_crops)
            compression_score = compute_compression_residual_anomaly(face_crops)
        else:
            # Fallback Route: Run on full frames instead of face crops
            fft_score = compute_temporal_fft_anomaly(frames)
            wavelet_score = compute_temporal_wavelet_anomaly(frames)
            compression_score = compute_compression_residual_anomaly(frames)
            
        # 4. Branch B: Universal Scene & Temporal Foundation
        global_scene_score = compute_global_scene_anomaly(frames)
        mamba_score = get_temporal_backbone_score(frames)
        
        # 5. Construct 11-Dimensional Feature Vector
        feature_vector = [
            optical_score,        # 0
            rppg_score,           # 1
            lighting_score,       # 2
            fft_score,            # 3
            wavelet_score,        # 4
            compression_score,    # 5
            global_scene_score,   # 6
            mamba_score,          # 7
            1.0 if face_present else 0.0, # 8
            avg_face_size,        # 9
            quality_metrics["video_quality"] # 10
        ]
        
        # 6. Noisy-OR Fusion
        fusion_result = evaluate_video_signals(feature_vector)
        
        # 7. Calibration Layer
        calibrated_score = calibrate_video_score(fusion_result["heuristic_score"])
        
        # 8. Final Verdict
        verdict = "Fake" if calibrated_score >= 0.50 else "Real"
        confidence = abs(calibrated_score - 0.50) * 2.0
        
        calibration_breakdown = {
            "heuristic_score": round(fusion_result["heuristic_score"], 4),
            "calibrated_score": round(calibrated_score, 4)
        }
        
        decision_factors = {
            "final_score": round(calibrated_score, 4),
            "heuristic_score": round(fusion_result["heuristic_score"], 4),
            "physics_ensemble": round(fusion_result["physics_ensemble"], 4)
        }
        
        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "faces_detected": 1 if face_present else 0,
            "analysis_mode": "video_full" if face_present else "video_fallback",
            "feature_vector": [round(v, 4) for v in feature_vector],
            "quality_metrics": {k: round(v, 4) for k, v in quality_metrics.items()},
            "heuristic_score": round(fusion_result["heuristic_score"], 4),
            "calibrated_score": round(calibrated_score, 4),
            "calibration_breakdown": calibration_breakdown,
            "decision_factors": decision_factors,
            "full_image_score": round(calibrated_score, 4),
            "contributions": fusion_result["contributions"],
            "explanation": [
                "Video frames were adaptively sampled for efficiency.",
                "Face-local physics (Optical Flow, rPPG) were extracted if a face was present.",
                "Universal temporal anomalies (Mamba, ∇v variance) were calculated globally.",
                "A calibrated Noisy-OR Bayesian network generated the final verdict."
            ]
        }
        
    except Exception as e:
        logger.error(f"Video Pipeline failed: {e}")
        raise e
