import cv2
import numpy as np
import logging
from scipy.signal import butter, filtfilt, welch

logger = logging.getLogger(__name__)

def _get_pos_signal(rgb_signals: np.ndarray, fps: float) -> np.ndarray:
    """
    Implements the Plane-Orthogonal-to-Skin (POS) algorithm for rPPG.
    rgb_signals: shape (frames, 3) - Mean RGB values over time
    """
    if len(rgb_signals) < 10:
        return np.zeros(len(rgb_signals))
        
    # POS algorithm requires a sliding window approach. 
    # For a short clip, we can process the whole trace.
    # 1. Temporal normalization
    mean_color = np.mean(rgb_signals, axis=0)
    normalized = rgb_signals / mean_color
    
    # 2. Projection
    X = 3 * normalized[:, 0] - 2 * normalized[:, 1]
    Y = 1.5 * normalized[:, 0] + normalized[:, 1] - 1.5 * normalized[:, 2]
    
    # 3. Alpha tuning
    std_X = np.std(X)
    std_Y = np.std(Y)
    alpha = std_X / (std_Y + 1e-6)
    
    # 4. Extract BVP (Blood Volume Pulse)
    bvp = X - alpha * Y
    
    # 5. Bandpass filter (Heart rate is typically 0.7 to 3 Hz / 40-180 BPM)
    nyquist = fps / 2.0
    low = 0.7 / nyquist
    high = 3.0 / nyquist
    
    if low >= 1.0 or high >= 1.0 or low <= 0:
        logger.warning(
            f"rPPG bandpass filter bypassed: low={low:.3f}, high={high:.3f} "
            f"due to low or invalid FPS: {fps}"
        )
        return bvp
        
    b, a = butter(3, [low, high], btype='bandpass')
    
    try:
        filtered_bvp = filtfilt(b, a, bvp)
        return filtered_bvp
    except ValueError:
        return bvp

def compute_rppg_anomaly(frames: list[np.ndarray], fps: float = 30.0) -> float:
    """
    Computes rPPG (Remote Photoplethysmography) phase mismatch.
    
    In face swaps, the face and neck come from different source videos.
    By extracting the Blood Volume Pulse (BVP) using POS from the face and neck
    separately, we can measure their phase alignment. A strong mismatch indicates a fake.
    """
    if len(frames) < 15: # Need enough frames for frequency analysis
        return 0.5
        
    try:
        face_rgb = []
        neck_rgb = []
        
        target_size = 256
        
        for f in frames:
            if len(f.shape) == 2:
                # rPPG requires color, fallback if grayscale
                return 0.5
                
            if f.shape[:2] != (target_size, target_size):
                f = cv2.resize(f, (target_size, target_size), interpolation=cv2.INTER_AREA)
                
            # Define ROIs for an aligned 256x256 face crop.
            # Face/Cheeks: center region capturing cheeks and forehead.
            # Neck: bottom portion; widened window to reliably capture the neck
            # area rather than just the chin/jaw edge.
            h, w = f.shape[:2]

            face_roi = f[int(h*0.4):int(h*0.7), int(w*0.3):int(w*0.7)]
            neck_roi = f[int(h*0.80):int(h*0.97), int(w*0.25):int(w*0.75)]

            
            face_rgb.append(np.mean(face_roi, axis=(0, 1)))
            neck_rgb.append(np.mean(neck_roi, axis=(0, 1)))
            
        face_rgb = np.array(face_rgb)
        neck_rgb = np.array(neck_rgb)
        
        # Extract BVP
        face_bvp = _get_pos_signal(face_rgb, fps)
        neck_bvp = _get_pos_signal(neck_rgb, fps)
        
        # Guard against constant signals
        if np.std(face_bvp) < 1e-6 or np.std(neck_bvp) < 1e-6:
            return 0.5
            
        # Compute Pearson correlation between the two BVP signals
        # Highly correlated = same biological pulse (Real)
        # Uncorrelated or negatively correlated = phase mismatch (Fake)
        correlation = np.corrcoef(face_bvp, neck_bvp)[0, 1]
        
        if np.isnan(correlation):
            return 0.5
            
        # Sigmoid Normalization
        # A real video should have correlation > 0.4
        # A face swap will have correlation ~ 0.0 or negative
        # Formula: σ(-8.0 * (correlation - 0.2))
        # High correlation -> low anomaly (Real)
        # Low correlation -> high anomaly (Fake)
        logit = np.clip(-8.0 * (correlation - 0.2), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"rPPG analysis failed: {e}")
        return 0.5
