import torch
import logging
import warnings
from PIL import Image
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ViTClassifier:

    def __init__(self, model_name: str | None = None):
        resolved_model_name = model_name or settings.VIT_MODEL_NAME
        allow_download = bool(settings.ALLOW_MODEL_DOWNLOAD)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.fake_label_id = None
        self._warned_unavailable = False

        # Suppress noisy informational output from transformers and tokenizer
        # only during model loading. Using a scoped context manager avoids
        # accidentally suppressing unrelated warnings from the rest of the codebase.
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logging.getLogger("transformers").setLevel(logging.ERROR)

                from transformers import AutoImageProcessor, AutoModelForImageClassification
                logger.info(
                    "Loading ViT classifier: %s (allow_download=%s)",
                    resolved_model_name,
                    allow_download,
                )

                self.processor = AutoImageProcessor.from_pretrained(
                    resolved_model_name,
                    local_files_only=not allow_download,
                )
                self.model = AutoModelForImageClassification.from_pretrained(
                    resolved_model_name,
                    local_files_only=not allow_download,
                )
                self.model.eval()
                self.model.to(self.device)

            # Determine which output label index corresponds to "Fake".
            # Default to index 0; update if the model exposes a recognisable label.
            self.fake_label_id = 0
            for idx, label in self.model.config.id2label.items():
                label_lower = str(label).lower()
                target_words = [
                    'fake', 'synthetic', 'deepfake', 'spoof',
                    'altered', 'manipulated', 'forged', 'tampered', 'generated',
                ]
                if any(word in label_lower for word in target_words):
                    self.fake_label_id = int(idx)
                    break

            logger.info(
                "ViT loaded on %s. Labels: %s, fake_id=%s",
                self.device,
                self.model.config.id2label,
                self.fake_label_id,
            )

        except ImportError:
            logger.error("transformers library not installed.")
        except Exception as e:
            if allow_download:
                logger.error("Failed to load ViT model: %s", e)
            else:
                logger.warning(
                    "ViT model unavailable in local cache (%s). "
                    "Set AXIOM_ALLOW_MODEL_DOWNLOAD=true to allow auto-download. Error: %s",
                    resolved_model_name,
                    e,
                )

    def classify(self, image: np.ndarray) -> float:
        if self.model is None or self.processor is None:
            if not self._warned_unavailable:
                logger.warning("ViT model not available, returning neutral 0.5")
                self._warned_unavailable = True
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


                                                                             
from threading import Lock

_classifier = None
_classifier_lock = Lock()


def _ensure_classifier():
    global _classifier
    if _classifier is None:
        with _classifier_lock:
            if _classifier is None:
                _classifier = ViTClassifier()
    return _classifier


def warmup_classifier() -> bool:
    clf = _ensure_classifier()
    return bool(clf.model is not None and clf.processor is not None)


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
