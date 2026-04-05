import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

def extract_msr(image: np.ndarray, sigmas=[15, 80, 120]):
    try:
                                                
        img_float = np.float32(image)
        msr = np.zeros_like(img_float)
        
                                                                           
        base_log = np.log1p(img_float)
        
        for sigma in sigmas:
                                                 
            blur = cv2.GaussianBlur(img_float, (0, 0), sigma)
                                       
            msr += (base_log - np.log1p(blur))
            
        msr = msr / len(sigmas)
        
        logger.debug("MSR Texture extraction completed.")
        return msr
        
    except Exception as e:
        logger.error(f"Failed to extract MSR texture: {e}")
        return np.zeros_like(image, dtype=np.float32)

def normalize_for_visualization(msr_map: np.ndarray):
                           
    min_val = np.min(msr_map)
    max_val = np.max(msr_map)
    
    if max_val - min_val == 0:
        return np.zeros_like(msr_map, dtype=np.uint8)
        
    norm = (msr_map - min_val) / (max_val - min_val)
    norm = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    return norm
