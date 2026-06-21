import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_global_scene_anomaly(frames: list[np.ndarray]) -> float:
    """
    Computes global scene coherence anomaly (Branch B).
    
    Detects fully AI-generated videos (e.g., Sora, Kling, Veo) which often 
    contain subtle structural hallucinations and physics-defying morphing 
    in the background.
    
    In a real video with camera motion, optical flow vectors should follow 
    rigid epipolar geometry (smooth gradients). In AI generated video, 
    background objects "warp" or "melt" slightly, creating chaotic, non-rigid 
    optical flow fields.
    
    We compute the global variance of the velocity gradient ∇v.
    """
    if len(frames) < 2:
        return 0.5
        
    try:
        # Resize to a medium resolution. We don't want it too small 
        # because we need to see structural details melting.
        target_size = 384
        grays = []
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            else:
                gray = f
            if gray.shape[:2] != (target_size, target_size):
                gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            grays.append(gray)
            
        global_chaos = []
        
        for i in range(len(grays) - 1):
            prev = grays[i]
            next_f = grays[i+1]
            
            # Compute Dense Optical Flow
            flow = cv2.calcOpticalFlowFarneback(
                prev, next_f, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            vx = flow[..., 0]
            vy = flow[..., 1]
            
            # Compute spatial gradients of the velocity field
            dvx_dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=3)
            dvy_dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=3)
            
            # The divergence of the flow field
            # div(v) = ∂vx/∂x + ∂vy/∂y
            # In rigid scenes, divergence is usually low except near occlusions.
            # In melting/morphing AI videos, divergence is highly erratic globally.
            divergence = dvx_dx + dvy_dy
            
            # Measure the spatial variance of the divergence.
            # High variance = chaotic morphing
            chaos_score = np.var(divergence)
            global_chaos.append(chaos_score)
            
        if not global_chaos:
            return 0.5
            
        avg_chaos = float(np.mean(global_chaos))
        
        # Sigmoid Normalization
        # Real videos: avg_chaos < 500 (rigid motion)
        # Fake videos: avg_chaos > 2000 (Sora melting artifacts)
        # Formula: σ(0.003 * (avg_chaos - 1200))
        logit = np.clip(0.003 * (avg_chaos - 1200.0), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Global scene analysis failed: {e}")
        return 0.5
