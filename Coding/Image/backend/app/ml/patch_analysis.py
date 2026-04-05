import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_patch_anomaly(img: np.ndarray, grid_size: int = 4) -> float:
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        
                                           
                                                                                 
                                                                                            
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_residual = np.abs(laplacian)
        
        h, w = noise_residual.shape
        block_h, block_w = h // grid_size, w // grid_size
        
        if block_h == 0 or block_w == 0:
            return 0.5
            
                                                             
                                                                           
        valid_h, valid_w = block_h * grid_size, block_w * grid_size
        valid_res = noise_residual[:valid_h, :valid_w]
        
                                                                                     
        blocks = valid_res.reshape(grid_size, block_h, grid_size, block_w).transpose(0, 2, 1, 3)
        blocks = blocks.reshape(grid_size * grid_size, -1)
        
                                     
        p_means = np.mean(blocks, axis=1) + 1e-6
        p_stds = np.std(blocks, axis=1)
        
                             
        snr_patches = p_stds / p_means
        
                                
                                                                                                  
                                                                                      
        cv = np.std(snr_patches) / (np.mean(snr_patches) + 1e-6)
        
                                   
                                                                  
                                                              
                                                                            
        anomaly = float(1.0 / (1.0 + np.exp(20.0 * (cv - 0.25))))
        
        logger.info(f"Physics Laplacian PRNU: CV={cv:.3f} → Anomaly={anomaly:.3f}")
        return anomaly
        
    except Exception as e:
        logger.error(f"Patch analysis failed: {e}")
        return 0.5
