import numpy as np
import pywt
import cv2
import logging

logger = logging.getLogger(__name__)

def compute_temporal_wavelet_anomaly(frames: list[np.ndarray]) -> float:
    """
    Computes temporal wavelet anomaly.
    
    Generative models often leave high-frequency checkerboard or grid artifacts 
    due to upsampling layers (e.g. Conv2DTranspose) or patch-based transformers.
    These artifacts are most visible in the Diagonal (HH) band of the Haar wavelet transform.
    
    In a real video, the HH band varies smoothly with the subject's texture.
    In a fake video, the HH grid artifacts jitter rapidly across frames.
    We extract the HH band for each frame and measure its temporal variance.
    """
    if len(frames) < 2:
        return 0.5
        
    try:
        target_size = 256
        hh_bands = []
        
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            else:
                gray = f
                
            if gray.shape[:2] != (target_size, target_size):
                gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
            # Perform 2D Discrete Wavelet Transform
            # 'haar' is fast and good for grid-like artifacts
            coeffs = pywt.dwt2(gray, 'haar')
            cA, (cH, cV, cD) = coeffs
            
            # cD is the diagonal detail (HH band)
            hh_bands.append(np.abs(cD))
            
        hh_cube = np.stack(hh_bands, axis=0) # Shape: (T, H/2, W/2)
        
        # Calculate frame-to-frame absolute difference of the HH band
        temporal_diff = np.diff(hh_cube, axis=0)
        
        # Mean absolute temporal difference
        # High value means the grid artifacts are jumping around (temporal instability)
        mean_diff = float(np.mean(np.abs(temporal_diff)))
        
        # Sigmoid Normalization
        # Real videos: mean_diff < 1.0 (smooth texture movement)
        # Fake videos: mean_diff > 2.5 (jittering grid artifacts)
        # Formula: σ(3.0 * (mean_diff - 1.5))
        logit = np.clip(3.0 * (mean_diff - 1.5), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Temporal wavelet analysis failed: {e}")
        return 0.5
