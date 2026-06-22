import cv2
import numpy as np
import functools
import logging

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=8)
def _get_radial_geometry(shape: tuple) -> np.ndarray:
    y, x = np.indices(shape)
    center = np.array([shape[0] // 2, shape[1] // 2])
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    return r.astype(np.int32).ravel()

def compute_radial_profile(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    
                    
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)**2
    
                                                                  
    r_ravel = _get_radial_geometry(gray.shape)
    
                                
    tbin = np.bincount(r_ravel, magnitude_spectrum.ravel())
    nr = np.bincount(r_ravel)
    radial_profile = tbin / np.maximum(nr, 1)
    
    return radial_profile

def calculate_frequency_anomaly(img: np.ndarray) -> float:
    try:
        profile = compute_radial_profile(img)
        
                                       
                                                                                  
                                                                                      
        
        total_len = len(profile)
        if total_len < 10:
            return 0.0
            
                                                       
                                                                               
        start_idx = max(2, int(total_len * 0.10))
        end_idx = int(total_len * 0.70)
        
                                       
                                                                                              
                                                                     
        low_band = int(total_len * 0.15)
        high_band_start = int(total_len * 0.70)
        
        low_energy = np.sum(profile[1:low_band]) + 1e-6
        high_energy = np.sum(profile[high_band_start:])
        
                                                                      
                                              
        hfer = high_energy / low_energy
        log_hfer = np.log10(hfer + 1e-12)
        
                                   
                                                                    
                                                                  
                                                                         
        anomaly = float(1.0 / (1.0 + np.exp(-3.0 * (log_hfer + 4.5))))
        
        logger.info(f"Frequency HFER: Log-Ratio={log_hfer:.3f} → Anomaly={anomaly:.2f}")
        return anomaly
        
    except Exception as e:
        logger.error(f"FFT detection failed: {e}")
        return 0.5
