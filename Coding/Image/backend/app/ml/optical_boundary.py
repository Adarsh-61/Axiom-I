import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def compute_optical_boundary_anomaly(frames: list[np.ndarray]) -> float:
    """
    Computes optical flow boundary anomaly across a sequence of face crops.
    
    In face-swapped deepfakes, the inner face is spliced onto the target head.
    This creates microscopic motion discontinuities at the jawline, hairline, 
    and cheek boundaries. By computing dense optical flow and analyzing the 
    spatial gradient of the velocity field (∇v), we can detect these "slips".
    
    Formula:
        v = (v_x, v_y) from Farneback Optical Flow
        ∇v_magnitude = sqrt((∂v_x/∂x)² + (∂v_y/∂y)²)
        
    We compare the variance of ∇v in the boundary ring vs the inner face.
    """
    if len(frames) < 2:
        return 0.5
        
    try:
        # We need a standard size for consistent gradient scales
        target_size = 256
        grays = []
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            else:
                gray = f
            if gray.shape[:2] != (target_size, target_size):
                gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            grays.append(gray)
            
        # Create masks for inner face and boundary ring
        y, x = np.ogrid[:target_size, :target_size]
        cx, cy = target_size // 2, target_size // 2
        
        # Elliptical mask approximations (since we only have bounding box crops)
        # Inner face: 50% radius
        # Boundary ring: 50% to 90% radius
        dist_sq = ((x - cx) / (target_size * 0.4))**2 + ((y - cy) / (target_size * 0.45))**2
        
        inner_mask = (dist_sq <= 1.0)
        boundary_mask = (dist_sq > 1.0) & (dist_sq <= 2.25)  # 1.5^2
        
        boundary_discontinuities = []
        
        # Process frame pairs
        for i in range(len(grays) - 1):
            prev = grays[i]
            next_f = grays[i+1]
            
            # Compute Dense Optical Flow (Farneback is CPU-friendly and smooth)
            flow = cv2.calcOpticalFlowFarneback(
                prev, next_f, None, 
                pyr_scale=0.5, levels=3, winsize=15, 
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            vx = flow[..., 0]
            vy = flow[..., 1]
            
            # Compute spatial gradients of the velocity field
            # Sobel derivatives with ksize=3
            dvx_dx = cv2.Sobel(vx, cv2.CV_64F, 1, 0, ksize=3)
            dvy_dy = cv2.Sobel(vy, cv2.CV_64F, 0, 1, ksize=3)
            
            # Magnitude of the gradient vector field ∇v
            grad_v_mag = np.sqrt(dvx_dx**2 + dvy_dy**2)
            
            # Analyze discontinuities
            # In a real face, the skin moves cohesively. 
            # In a fake, the boundary mask slips, causing high gradient spikes in the boundary ring.
            
            inner_mean = np.mean(grad_v_mag[inner_mask])
            boundary_mean = np.mean(grad_v_mag[boundary_mask])
            
            # To avoid division by zero
            inner_mean = max(inner_mean, 1e-6)
            
            # Ratio of boundary motion discontinuity to inner face motion
            ratio = boundary_mean / inner_mean
            boundary_discontinuities.append(ratio)
            
        if not boundary_discontinuities:
            return 0.5
            
        # Average ratio across the temporal window
        avg_ratio = float(np.mean(boundary_discontinuities))
        
        # Sigmoid normalization
        # Typical real face ratio is ~1.0 to 1.5 (natural expression stretch)
        # Deepfake mask slips cause ratio > 2.0 or 3.0
        # Formula: σ(4.0 * (avg_ratio - 1.8))
        logit = np.clip(4.0 * (avg_ratio - 1.8), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Optical boundary analysis failed: {e}")
        return 0.5
