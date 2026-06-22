
import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)


def extract(image: np.ndarray, ambient: np.ndarray, direct: np.ndarray, texture: np.ndarray):
    try:
                                            
        I = np.float32(image) / 255.0

                                                                        
                                                           
                                         
        T_log = np.clip(np.float32(texture), -5.0, 5.0)
        T_linear = np.exp(T_log)
                                            
        T_linear = np.clip(T_linear / (T_linear.max() + 1e-8), 1e-6, 1.0)

                                                     
        H_gamma = np.float32(ambient) + np.float32(direct)

                                       
                                                                           
        if np.abs(H_gamma).max() < 1e-8:
            logger.warning("SPR extraction: illumination is all-zero. Returning zero SPR.")
            return np.zeros_like(image, dtype=np.float32)

                                                                    
        if np.std(T_linear) < 1e-6:
            logger.warning("SPR extraction: texture is constant. Returning zero SPR.")
            return np.zeros_like(image, dtype=np.float32)

                                        
        lambertian = H_gamma * T_linear
        spr = I - lambertian

                                                                             
                                                          
                                                                
        spr = np.maximum(spr, 0.0)
        
                                                                          
        spr = np.clip(spr, 0.0, 3.0)

                                              
        for ch, name in enumerate(['R', 'G', 'B']):
            ch_data = spr[:, :, ch]
            logger.debug(
                f"SPR [{name}]: range=[{ch_data.min():.4f}, {ch_data.max():.4f}], "
                f"mean={ch_data.mean():.4f}, std={ch_data.std():.4f}"
            )

        return spr

    except Exception as e:
        logger.error(f"SPR extraction failed: {e}")
        return np.zeros_like(image, dtype=np.float32)


def heatmap(spr: np.ndarray):
    try:
                                                
        if len(spr.shape) == 3 and spr.shape[2] == 3:
            magnitude = np.mean(np.abs(spr), axis=2)
        else:
            magnitude = np.abs(spr)

                               
        min_val = magnitude.min()
        max_val = magnitude.max()

        if max_val - min_val < 1e-8:
            return np.zeros((*magnitude.shape, 3), dtype=np.uint8)

        normalized = ((magnitude - min_val) / (max_val - min_val) * 255.0).astype(np.uint8)

                                                                 
        heatmap_img = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

                                             
        heatmap_img = cv2.cvtColor(heatmap_img, cv2.COLOR_BGR2RGB)

        return heatmap_img

    except Exception as e:
        logger.error(f"SPR heatmap generation failed: {e}")
        return np.zeros_like(spr, dtype=np.uint8)
