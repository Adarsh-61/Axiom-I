import numpy as np
import logging
from app.ml.illumination import get_sh_basis

logger = logging.getLogger(__name__)

def _fit_sh_for_video(image_crop: np.ndarray, normals: np.ndarray, texture: np.ndarray) -> np.ndarray:
    """
    Fits Spherical Harmonics directly for a video frame to get gamma coefficients.
    Reuses logic from image pipeline but focuses only on gamma.
    """
    try:
        image = image_crop.astype(np.float64) / 255.0
        tex_log = np.clip(texture.astype(np.float64), -5.0, 5.0)
        tex_linear = np.exp(tex_log)
        tex_linear = np.clip(tex_linear / max(float(tex_linear.max()), 1e-8), 1e-6, 1.0)

        basis = get_sh_basis(normals.astype(np.float64))
        a = basis.reshape(-1, 9)
        tex_flat = tex_linear.reshape(-1, 3)
        image_flat = image.reshape(-1, 3)

        reg = 1e-3
        ata = (a.T @ a) + (reg * np.eye(9, dtype=np.float64))
        gamma = np.zeros((9, 3), dtype=np.float64)

        for channel in range(3):
            y = image_flat[:, channel] / np.clip(tex_flat[:, channel], 1e-6, None)
            y = np.clip(y, 0.0, 3.0)
            gamma[:, channel] = np.linalg.solve(ata, a.T @ y)

        return gamma.astype(np.float32)
    except Exception:
        return np.zeros((9, 3), dtype=np.float32)

def compute_temporal_lighting_anomaly(gamma_sequence: list[np.ndarray]) -> float:
    """
    Computes temporal lighting anomaly based on the drift of Spherical Harmonics coefficients.
    
    In a real video, the lighting environment (represented by the 9 SH coefficients per channel)
    should vary smoothly across frames unless there is a sudden scene change. 
    In Deepfakes, independent frame generation causes unnatural lighting jitter, 
    especially in the higher-order (directional) harmonics.
    """
    if len(gamma_sequence) < 2:
        return 0.5
        
    try:
        # gamma_sequence is a list of (9, 3) arrays
        # We want to measure the temporal derivative
        gammas = np.stack(gamma_sequence, axis=0) # shape: (T, 9, 3)
        
        # Calculate frame-to-frame differences (temporal derivative)
        diffs = np.diff(gammas, axis=0) # shape: (T-1, 9, 3)
        
        # We weight the higher order coefficients more, as ambient (L0) can flicker naturally 
        # (e.g. auto-exposure), but directional light (L1, L2) shouldn't jitter randomly.
        # Index 0: Ambient (L0)
        # Index 1-3: Linear (L1)
        # Index 4-8: Quadratic (L2)
        
        weights = np.array([0.1, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0])
        
        # Apply weights along the 9-coefficient axis
        weighted_diffs = diffs * weights[None, :, None]
        
        # Magnitude of the jitter per frame
        jitter_per_frame = np.linalg.norm(weighted_diffs, axis=(1, 2))
        
        # Average jitter
        avg_jitter = float(np.mean(jitter_per_frame))
        
        # Sigmoid Normalization
        # Based on Axiom physics-first calibration:
        # Real videos have smooth SH transitions (jitter < 0.2)
        # Fake videos jitter heavily (jitter > 0.5)
        # Formula: σ(8.0 * (avg_jitter - 0.35))
        logit = np.clip(8.0 * (avg_jitter - 0.35), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Temporal lighting analysis failed: {e}")
        return 0.5
