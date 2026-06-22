import logging
import numpy as np
from app.ml.face_detector import detect

logger = logging.getLogger(__name__)

def _calculate_iou(box1: list[int], box2: list[int]) -> float:
    x1, y1, x2, y2 = box1
    x1_b, y1_b, x2_b, y2_b = box2
    
    xi1 = max(x1, x1_b)
    yi1 = max(y1, y1_b)
    xi2 = min(x2, x2_b)
    yi2 = min(y2, y2_b)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    if inter_area == 0:
        return 0.0
        
    box1_area = max(0, x2 - x1) * max(0, y2 - y1)
    box2_area = max(0, x2_b - x1_b) * max(0, y2_b - y1_b)
    
    denominator = float(box1_area + box2_area - inter_area)
    if denominator <= 0:
        return 0.0
        
    iou = inter_area / denominator
    return iou

def track_main_face(frames: list[np.ndarray]) -> tuple[list[np.ndarray], float]:
    """
    Detects and tracks the main face across a sequence of frames.
    
    Returns:
        tuple: (list of face crops, average_face_size_ratio)
               If no face is consistently found, returns ([], 0.0)
    """
    if not frames:
        return [], 0.0
        
    face_crops = []
    avg_size_ratios = []
    
    # Find the first frame (up to index 9) that has a face
    init_idx = -1
    initial_faces = []
    max_scan_frames = min(10, len(frames))
    for i in range(max_scan_frames):
        faces = detect(frames[i], min_confidence=0.85, min_size=64)
        if faces:
            init_idx = i
            initial_faces = faces
            break
            
    if init_idx == -1:
        return [], 0.0
        
    # Pick the largest face
    main_face = max(initial_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    prev_bbox = main_face.bbox
    
    # Backfill early frames before init_idx
    for j in range(init_idx):
        x1 = max(0, prev_bbox[0])
        y1 = max(0, prev_bbox[1])
        x2 = min(frames[j].shape[1], prev_bbox[2])
        y2 = min(frames[j].shape[0], prev_bbox[3])
        crop = frames[j][y1:y2, x1:x2]
        face_crops.append(crop)
        
    # Append the main face crop for the init_idx frame
    face_crops.append(main_face.crop)
    
    h_img, w_img = frames[init_idx].shape[:2]
    img_area = h_img * w_img
    face_area = (main_face.bbox[2] - main_face.bbox[0]) * (main_face.bbox[3] - main_face.bbox[1])
    avg_size_ratios.append(face_area / img_area)
    
    # Track through remaining frames (from init_idx + 1 onwards)
    for i in range(init_idx + 1, len(frames)):
        faces = detect(frames[i], min_confidence=0.85, min_size=64)
        if not faces:
            # Face lost in this frame, use previous bbox to crop anyway (assume still there but MTCNN failed)
            x1 = max(0, prev_bbox[0])
            y1 = max(0, prev_bbox[1])
            x2 = min(frames[i].shape[1], prev_bbox[2])
            y2 = min(frames[i].shape[0], prev_bbox[3])
            crop = frames[i][y1:y2, x1:x2]
            face_crops.append(crop)
            continue
            
        # Match with previous bbox via highest IoU
        best_match = None
        best_iou = 0.0
        for f in faces:
            iou = _calculate_iou(prev_bbox, f.bbox)
            if iou > best_iou:
                best_iou = iou
                best_match = f
                
        if best_match and best_iou > 0.3: # Threshold for tracking
            face_crops.append(best_match.crop)
            prev_bbox = best_match.bbox
            
            f_area = (best_match.bbox[2] - best_match.bbox[0]) * (best_match.bbox[3] - best_match.bbox[1])
            h_curr, w_curr = frames[i].shape[:2]
            avg_size_ratios.append(f_area / (h_curr * w_curr))
        else:
            # Assume stationary if lost
            x1 = max(0, prev_bbox[0])
            y1 = max(0, prev_bbox[1])
            x2 = min(frames[i].shape[1], prev_bbox[2])
            y2 = min(frames[i].shape[0], prev_bbox[3])
            crop = frames[i][y1:y2, x1:x2]
            face_crops.append(crop)
            
    final_avg_size = float(np.mean(avg_size_ratios)) if avg_size_ratios else 0.0
    
    return face_crops, final_avg_size
