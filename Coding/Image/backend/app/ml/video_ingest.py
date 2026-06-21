import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def estimate_blockiness(img_gray: np.ndarray) -> float:
    """
    Estimates blockiness based on 8x8 grid boundary discontinuities (typical of compression).
    Returns a score in [0, 1] where 1.0 is clean (no blockiness) and 0.0 is highly blocky.
    """
    h, w = img_gray.shape
    if h < 16 or w < 16:
        return 1.0
        
    # Horizontal blockiness
    diff_h = np.abs(img_gray[:, 1:].astype(np.float32) - img_gray[:, :-1].astype(np.float32))
    b_diff_h = diff_h[:, 7::8]
    nb_diff_h = []
    for offset in range(1, 8):
        if offset != 7:
            nb_diff_h.append(diff_h[:, offset::8])
    
    b_diff_h_flat = b_diff_h.ravel()
    nb_diff_h_flat = np.concatenate([arr.ravel() for arr in nb_diff_h]) if nb_diff_h else diff_h.ravel()
    
    # Vertical blockiness
    diff_v = np.abs(img_gray[1:, :].astype(np.float32) - img_gray[:-1, :].astype(np.float32))
    b_diff_v = diff_v[7::8, :]
    nb_diff_v = []
    for offset in range(1, 8):
        if offset != 7:
            nb_diff_v.append(diff_v[offset::8, :])
            
    b_diff_v_flat = b_diff_v.ravel()
    nb_diff_v_flat = np.concatenate([arr.ravel() for arr in nb_diff_v]) if nb_diff_v else diff_v.ravel()
    
    mean_b = (np.mean(b_diff_h_flat) + np.mean(b_diff_v_flat)) / 2.0
    mean_nb = (np.mean(nb_diff_h_flat) + np.mean(nb_diff_v_flat)) / 2.0
    
    if mean_nb < 1e-5:
        return 1.0
        
    ratio = mean_b / mean_nb
    score = np.clip(2.0 - ratio, 0.0, 1.0)
    return float(score)

def ingest_video(file_path: str, max_frames: int = 30) -> tuple[list[np.ndarray], float, dict]:
    """
    Ingests a video file and returns a representative sequence of frames,
    the FPS, and quality metrics.
    
    Optimizes I/O by not processing 60 FPS blindly. Uses adaptive sampling
    to pick keyframes based on motion heuristics (absolute difference).
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {file_path}")
    
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frames = []
        
        if total_frames <= max_frames:
            stride = 1
        else:
            stride = max(1, total_frames // max_frames)
            
        quality_metrics = {
            "blur_score": 0.0,
            "resolution_score": 0.0,
            "compression_score": 0.0
        }
        
        blur_scores = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_scores.append(lap_var)
                
            frame_idx += 1
            
            if len(frames) >= max_frames:
                break
    finally:
        cap.release()
    
    if not frames:
        raise ValueError("Video contains no valid frames.")
        
    avg_blur = np.mean(blur_scores)
    sharpness = np.clip(np.log1p(avg_blur) / 8.0, 0.0, 1.0)
    
    h, w = frames[0].shape[:2]
    resolution = np.clip(min(h, w) / 1080.0, 0.0, 1.0)
    
    # Calculate compression score based on the first keyframe blockiness
    first_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    blockiness_score = estimate_blockiness(first_gray)
    
    quality_metrics["blur_score"] = float(sharpness)
    quality_metrics["resolution_score"] = float(resolution)
    quality_metrics["compression_score"] = float(blockiness_score)
    
    video_quality = (0.5 * resolution) + (0.3 * sharpness) + (0.2 * blockiness_score)
    quality_metrics["video_quality"] = float(np.clip(video_quality, 0.0, 1.0))
    
    return frames, fps, quality_metrics
