import numpy as np
import logging

logger = logging.getLogger(__name__)


def compute_specular_anomaly(tex: np.ndarray, spr: np.ndarray) -> float:
    try:
        t_flat = tex.flatten().astype(np.float64)
        s_flat = spr.flatten().astype(np.float64)

        if np.std(t_flat) < 1e-5 or np.std(s_flat) < 1e-5:
            return 0.5

        ncc = np.corrcoef(t_flat, s_flat)[0, 1]
        if not np.isfinite(ncc):
            return 0.5

                                                                         
                                                       
                                                                        
                                                   
        anomaly = float(1.0 / (1.0 + np.exp(-15.0 * (ncc - 0.30))))
        logger.info(f"Specular NCC={ncc:.4f} → Anomaly={anomaly:.3f}")
        return anomaly

    except Exception as e:
        logger.error(f"SPR analysis failed: {e}")
        return 0.5


def evaluate_multi_signal(
    specular_anomaly: float,
    freq_power: float,
    topo_score: float,
    patch_score: float,
    wavelet_score: float = 0.0,
    vit_score: float = 0.5,
) -> dict:
    try:
                                                                 
                                                          
                            
                                                                 
        w_topology   = 0.22                                      
        w_patch      = 0.22                                      
        w_freq       = 0.18                                    
        w_wavelet    = 0.15                                           
        w_specular   = 0.10                                      
        w_vit        = 0.13                                           

                                     
        physics_spec = max(0.0, min(1.0, specular_anomaly))
        freq_s       = max(0.0, min(1.0, freq_power))
        topo_s       = max(0.0, min(1.0, topo_score))
        patch_s      = max(0.0, min(1.0, patch_score))
        wavelet_s    = max(0.0, min(1.0, wavelet_score))
        vit_s        = max(0.0, min(1.0, vit_score))

                                                         
                                                                          
        p_real_spec    = 1.0 - (w_specular * physics_spec)
        p_real_topo    = 1.0 - (w_topology * topo_s)
        p_real_freq    = 1.0 - (w_freq * freq_s)
        p_real_patch   = 1.0 - (w_patch * patch_s)
        p_real_wavelet = 1.0 - (w_wavelet * wavelet_s)
        p_real_vit     = 1.0 - (w_vit * vit_s)

        p_all_real = (p_real_spec * p_real_topo * p_real_freq * 
                      p_real_patch * p_real_wavelet * p_real_vit)

        physics_ensemble = 1.0 - p_all_real

                                                                                    
                                                                 
        final_score = float(1.0 / (1.0 + np.exp(-8.0 * (physics_ensemble - 0.40))))

        breakdown = {
            "specular": round(physics_spec, 4),
            "frequency": round(freq_s, 4),
            "topology": round(topo_s, 4),
            "patch_consistency": round(patch_s, 4),
            "wavelet_score": round(wavelet_s, 4),
            "vit_score": round(vit_s, 4),
            "physics_ensemble": round(physics_ensemble, 4),
            "raw_fusion": round(physics_ensemble, 4),
            "calibrated": round(final_score, 4),
        }

        logger.info(
            f"6-Pillar Fusion: spec={physics_spec:.3f}, freq={freq_s:.3f}, "
            f"topo={topo_s:.3f}, patch={patch_s:.3f}, "
            f"wavelet={wavelet_s:.3f}, vit={vit_s:.3f} "
            f"→ Noisy-OR={physics_ensemble:.3f} → final={final_score:.3f}"
        )

        return breakdown

    except Exception as e:
        logger.error(f"Fusion calculation failed: {e}")
        return {
            "specular": 0.0, "frequency": 0.0, "topology": 0.0, 
            "patch_consistency": 0.0, "wavelet_score": 0.0, "vit_score": 0.5,
            "physics_ensemble": 0.0, "raw_fusion": 0.0, "calibrated": 0.0
        }
