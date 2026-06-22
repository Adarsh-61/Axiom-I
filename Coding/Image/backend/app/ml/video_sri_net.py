import numpy as np

def _noisy_or(probabilities: list[float], weights: list[float]) -> float:
    # P(Fake) = 1 - ∏(1 - w_i * f_i)
    prob_real = 1.0
    for p, w in zip(probabilities, weights):
        prob_real *= (1.0 - (w * p))
    return 1.0 - prob_real

def evaluate_video_signals(feature_vector: list[float]) -> dict:
    """
    Evaluates the 11-dimensional video feature vector using a calibrated 
    Noisy-OR Bayesian network.
    
    Feature indices:
    0: optical_boundary
    1: rppg
    2: lighting
    3: temporal_fft
    4: wavelet_temporal
    5: compression_residual
    6: global_scene
    7: temporal_backbone
    8: face_present (1.0 or 0.0)
    9: avg_face_size [0.0, 1.0]
    10: video_quality [0.0, 1.0]
    """
    if len(feature_vector) != 11:
        raise ValueError(f"Expected 11 features, got {len(feature_vector)}")
        
    face_present = feature_vector[8] > 0.5
    avg_face_size = feature_vector[9]
    video_quality = feature_vector[10]
    
    # Base weights
    weights = [
        0.18, # 0: optical_boundary
        0.18, # 1: rppg
        0.15, # 2: lighting
        0.10, # 3: temporal_fft
        0.10, # 4: wavelet_temporal
        0.05, # 5: compression_residual
        0.24, # 6: global_scene (boosted from 0.12 to absorb temporal_backbone)
        0.00  # 7: temporal_backbone (disabled because proxy uses untrained random weights)
    ]
    
    signals = feature_vector[0:8]
    
    if not face_present:
        # Fallback routing
        # Zero out Face-local physics weights
        weights[0] = 0.0
        weights[1] = 0.0
        weights[2] = 0.0
        
        # Boost universal weights
        weights[3] = 0.15
        weights[4] = 0.15
        weights[5] = 0.10
        weights[6] = 0.60 # Global scene is crucial (boosted from 0.30 to absorb temporal_backbone)
        weights[7] = 0.00 # Backbone is disabled
    else:
        # Adjust weights dynamically based on face size
        # Small faces have less reliable boundary and rPPG signals
        if avg_face_size < 0.05: # Tiny face
            scale = 0.5
            weights[0] *= scale
            weights[1] *= scale
            weights[2] *= scale
            
            # Rebalance
            diff = (0.18 + 0.18 + 0.15) * (1.0 - scale)
            weights[6] += diff
            
    # Apply Noisy-OR Fusion
    physics_ensemble = _noisy_or(signals, weights)
    
    # Apply a gentle penalty if video quality is extremely poor
    # meaning the signals are less trustworthy
    if video_quality < 0.3:
        physics_ensemble *= (0.7 + video_quality)
        
    # Sigmoid scaling consistent with the image module
    # Maps the raw ensemble probability to a calibrated heuristic score in [0, 1]
    heuristic_score = float(1.0 / (1.0 + np.exp(-8.0 * (physics_ensemble - 0.40))))
        
    # Return transparent breakdown
    contributions = []
    names = [
        "optical_boundary", "rppg", "temporal_lighting", "temporal_fft",
        "wavelet_temporal", "compression_residual", "global_scene", "temporal_backbone"
    ]
    
    for name, p, w in zip(names, signals, weights):
        contribution = p * w
        contributions.append({
            "signal": name,
            "raw_score": p,
            "weight": w,
            "contribution": contribution
        })
        
    return {
        "physics_ensemble": float(physics_ensemble),
        "heuristic_score": float(heuristic_score), # Calibrated via sigmoid
        "contributions": contributions,
        "is_fallback": not face_present
    }
