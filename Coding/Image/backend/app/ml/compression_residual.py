import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def compute_compression_residual_anomaly(frames: list[np.ndarray]) -> float:
    """
    Computes compression residual anomaly.
    
    Separates Real Compression (e.g. H.264 macroblocks) from Generative Artifacts.
    Real video compression creates blocky artifacts that remain relatively 
    static across P-frames and B-frames. 
    Generative AI models create pixel-level noise that is independently 
    sampled per frame, leading to high temporal variance in the noise residual.
    
    We extract the noise residual using a bilateral filter (which preserves edges)
    and then measure the temporal variance of this residual.
    """
    if len(frames) < 2:
        return 0.5
        
    try:
        target_size = 256
        residuals = []
        
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            else:
                gray = f
                
            if gray.shape[:2] != (target_size, target_size):
                gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
            # Extract noise residual
            # Smooth the image while keeping edges sharp
            smoothed = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
            
            # The residual is the high-frequency noise/texture
            residual = cv2.absdiff(gray, smoothed)
            residuals.append(residual)
            
        residual_cube = np.stack(residuals, axis=0).astype(np.float32)
        
        # Calculate temporal variance of the noise residual for each pixel
        # shape: (H, W)
        temporal_variance = np.var(residual_cube, axis=0)
        
        # Mean variance across the frame
        mean_var = float(np.mean(temporal_variance))
        
        # Sigmoid Normalization
        # Real compression: noise residual is static across P-frames (low variance < 5.0)
        # Fake generation: noise residual flickers wildly per frame (high variance > 15.0)
        # Formula: σ(0.5 * (mean_var - 10.0))
        logit = np.clip(0.5 * (mean_var - 10.0), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Compression residual analysis failed: {e}")
        return 0.5
