
import numpy as np
import pywt
import cv2
import logging

logger = logging.getLogger(__name__)

def calculate_wavelet_anomaly(image: np.ndarray, wavelet='db2', level=2) -> float:
    try:
        if image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        else:
            gray = image if image.ndim == 2 else image

                                    
        coeffs = pywt.wavedec2(gray, wavelet=wavelet, level=level)
        
                                                                                        
                                                                       
        
        total_high_energy = 0.0
        details_count = 0
        
        for details in coeffs[1:]:
            LH, HL, HH = details
                                                             
            energy_lh = np.sum(LH ** 2) / LH.size
            energy_hl = np.sum(HL ** 2) / HL.size
            energy_hh = np.sum(HH ** 2) / HH.size
            
                                                                                       
            level_energy = (energy_lh + energy_hl + energy_hh * 1.5) / 3.0
            total_high_energy += level_energy
            details_count += 1
            
        avg_energy = total_high_energy / (details_count + 1e-6)
        
                                                                
                                                                  
                                                        
        log_e = np.log1p(avg_energy)
        
                                   
                                                                
                                                                                
                                                                              
        anomaly = float(1.0 / (1.0 + np.exp(-3.0 * (log_e - 4.5))))
        
        logger.info(f"Wavelet Energy log={log_e:.2f} -> Anomaly={anomaly:.3f}")
        return float(anomaly)

    except Exception as e:
        logger.error(f"Wavelet analysis failed: {e}")
        return 0.5
