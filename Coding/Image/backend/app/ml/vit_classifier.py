
import torch
import logging
from PIL import Image
import numpy as np
import warnings

                                
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)


class ViTClassifier:

    def __init__(self, model_name="prithivMLmods/Deep-Fake-Detector-v2-Model"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.fake_label_id = None

        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            logger.info(f"Loading ViT classifier: {model_name}")

            self.processor = AutoImageProcessor.from_pretrained(model_name)
            self.model = AutoModelForImageClassification.from_pretrained(model_name)
            self.model.eval()
            self.model.to(self.device)

                                                                                         
            self.fake_label_id = 0           
            for idx, label in self.model.config.id2label.items():
                label_lower = str(label).lower()
                if 'fake' in label_lower or 'synthetic' in label_lower or 'deepfake' in label_lower:
                    self.fake_label_id = int(idx)
                    break

            logger.info(
                f"ViT loaded on {self.device}. "
                f"Labels: {self.model.config.id2label}, fake_id={self.fake_label_id}"
            )

        except ImportError:
            logger.error("transformers library not installed!")
        except Exception as e:
            logger.error(f"Failed to load ViT model: {e}")

    def classify(self, image: np.ndarray) -> float:
        if self.model is None or self.processor is None:
            logger.warning("ViT model not available, returning neutral 0.5")
            return 0.5

        try:
                                           
            if image.ndim == 2:
                image = np.stack((image,) * 3, axis=-1)
            elif image.shape[2] == 4:
                image = image[:, :, :3]

            pil_image = Image.fromarray(image.astype(np.uint8))
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=-1)

            fake_prob = probs[0][self.fake_label_id].item()
            return float(fake_prob)

        except Exception as e:
            logger.error(f"ViT classification failed: {e}")
            return 0.5


                                                                             
_classifier = None


def _ensure_classifier():
    global _classifier
    if _classifier is None:
        _classifier = ViTClassifier()
    return _classifier


def get_full_image_score(image: np.ndarray) -> float:
    clf = _ensure_classifier()
    score = clf.classify(image)
    logger.info(f"Full-image ViT score: {score:.4f}")
    return score


def get_vit_score(face_crop: np.ndarray) -> float:
    clf = _ensure_classifier()
    score = clf.classify(face_crop)
    logger.info(f"Face-crop ViT score: {score:.4f}")
    return score
