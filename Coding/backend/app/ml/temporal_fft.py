import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)

def compute_temporal_fft_anomaly(frames: list[np.ndarray]) -> float:
    """
    Computes temporal frequency anomaly using 1D FFT along the time axis.
    
    Generative models (GANs, Diffusion) often struggle with temporal consistency
    at the pixel level, leading to high-frequency "flicker" that is invisible 
    spatially but obvious temporally.
    
    We take the 1D FFT of each pixel over time, and measure the ratio of 
    high-frequency energy to total energy.
    """
    if len(frames) < 10:
        return 0.5
        
    try:
        # Resize to a smaller resolution to save CPU and focus on structural flicker 
        # rather than pure camera noise.
        target_size = 64
        grays = []
        for f in frames:
            if len(f.shape) == 3:
                gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            else:
                gray = f
            if gray.shape[:2] != (target_size, target_size):
                gray = cv2.resize(gray, (target_size, target_size), interpolation=cv2.INTER_AREA)
            grays.append(gray)
            
        # Stack into shape (T, H, W)
        video_cube = np.stack(grays, axis=0).astype(np.float32)
        T, H, W = video_cube.shape
        
        # Compute 1D FFT along the time axis (axis 0)
        # fft_cube shape: (T, H, W)
        fft_cube = np.fft.fft(video_cube, axis=0)
        
        # Get power spectrum
        power_spectrum = np.abs(fft_cube) ** 2
        
        # We only care about the positive frequencies (first half)
        half_T = T // 2
        power_spectrum = power_spectrum[1:half_T] # exclude DC component (index 0)
        
        if len(power_spectrum) == 0:
            return 0.5
            
        # Total energy (excluding DC)
        total_energy = np.sum(power_spectrum, axis=0)
        
        # High-frequency energy (upper half of the available spectrum)
        # Nyquist frequency is the max. We take the top 50% of the frequency bins.
        hf_start = len(power_spectrum) // 2
        hf_energy = np.sum(power_spectrum[hf_start:], axis=0)
        
        # Avoid division by zero
        total_energy = np.clip(total_energy, 1e-6, None)
        
        # Ratio of high-frequency energy to total energy per pixel
        hf_ratio_map = hf_energy / total_energy
        
        # Average HF ratio across the frame
        avg_hf_ratio = float(np.mean(hf_ratio_map))
        
        # Sigmoid Normalization
        # Real videos have smooth motion (mostly low frequency energy, HF ratio < 0.1)
        # Fake videos flicker (HF ratio > 0.25)
        # Formula: σ(15.0 * (avg_hf_ratio - 0.15))
        logit = np.clip(15.0 * (avg_hf_ratio - 0.15), -20.0, 20.0)
        anomaly_score = 1.0 / (1.0 + np.exp(-logit))
        
        return float(anomaly_score)
        
    except Exception as e:
        logger.error(f"Temporal FFT analysis failed: {e}")
        return 0.5
