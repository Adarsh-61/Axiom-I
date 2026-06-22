
import numpy as np
import cv2
import logging

from app.config import settings

logger = logging.getLogger(__name__)

                                                                          

from threading import Lock

_detector = None
_detector_failed = False
_detector_lock = Lock()


def _ensure_detector():
    global _detector, _detector_failed

    if _detector_failed:
        return False
    if _detector is not None:
        return True

    with _detector_lock:
        if _detector is not None:
            return True
        if _detector_failed:
            return False

        try:
            import torch
            from facenet_pytorch import MTCNN

                                           
                                                                                          
                                                                                               
            requested_device = (settings.DEVICE or "cpu").strip().lower()
            if requested_device.startswith("cuda") and not torch.cuda.is_available():
                logger.warning("AXIOM_DEVICE=%s requested but CUDA is unavailable; using cpu.", requested_device)
                requested_device = "cpu"

            device = torch.device(requested_device)
            _detector = MTCNN(keep_all=True, device=device)
            logger.info(f"MTCNN Face Detector loaded on {device}.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize MTCNN: {e}")
            _detector_failed = True
            return False


                                                                          

class FaceDetectionResult:
    __slots__ = ('bbox', 'landmarks', 'confidence', 'crop')

    def __init__(self, bbox, landmarks, confidence, crop):
        self.bbox = bbox                            
        self.landmarks = landmarks
        self.confidence = confidence
        self.crop = crop                           


def _extract_crop(image: np.ndarray, bbox: list, pad_frac: float = 0.1) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]

    pad_x = int((x2 - x1) * pad_frac)
    pad_y = int((y2 - y1) * pad_frac)

    nx1 = max(0, x1 - pad_x)
    ny1 = max(0, y1 - pad_y)
    nx2 = min(w, x2 + pad_x)
    ny2 = min(h, y2 + pad_y)

    return image[ny1:ny2, nx1:nx2]


                                                                          

def detect(image: np.ndarray, min_confidence: float = 0.9, min_size: int = 64):
    if not _ensure_detector():
        logger.error("MTCNN detector is not available.")
        return []

    try:
        boxes, probs, landmarks = _detector.detect(image, landmarks=True)

        results = []
        if boxes is not None:
            for i in range(len(boxes)):
                score = probs[i]
                if score is None or score < min_confidence:
                    continue

                box = [int(v) for v in boxes[i]]
                x1, y1, x2, y2 = box

                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue

                lms = landmarks[i] if landmarks is not None else []
                lm_dict = {
                    "right_eye": lms[0] if len(lms) > 0 else [],
                    "left_eye": lms[1] if len(lms) > 1 else [],
                    "nose": lms[2] if len(lms) > 2 else [],
                    "mouth_right": lms[3] if len(lms) > 3 else [],
                    "mouth_left": lms[4] if len(lms) > 4 else [],
                }

                crop = _extract_crop(image, box)
                if crop.size == 0:
                    continue
                results.append(FaceDetectionResult(box, lm_dict, score, crop))

        return results

    except Exception as e:
        logger.error(f"Face detection failed: {e}")
        return []


def warmup_detector() -> bool:
    return _ensure_detector()
