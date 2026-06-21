import os
import json
import logging
import tempfile
from threading import RLock
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

# Shared locks to prevent thread conflicts during local filesystem operations
_db_lock = RLock()

# Global reference for HF file system to support lazy loading
_hf_fs = None

# Local directory where JSON logs are stored when running in "local" mode
_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "feedback_data")


def _get_hf_fs():
    """
    Lazy loads HfFileSystem and returns the instance.
    Raises ImportError if huggingface_hub is missing, or ValueError if token is empty.
    """
    global _hf_fs
    if _hf_fs is None:
        if not settings.HF_TOKEN or not settings.HF_TOKEN.strip():
            raise ValueError("AXIOM_HF_TOKEN is not configured or is empty. Cannot read/write in host mode.")
            
        try:
            from huggingface_hub import HfFileSystem
            _hf_fs = HfFileSystem(token=settings.HF_TOKEN)
            logger.info("HfFileSystem successfully initialized.")
        except ImportError as e:
            logger.error("huggingface_hub library is not installed. Failed to import HfFileSystem.")
            raise e
    return _hf_fs


def _load_local_json(filename: str, default_val: Any) -> Any:
    """
    Reads a JSON file from the local feedback_data directory in a thread-safe manner.
    """
    path = os.path.join(_LOCAL_DIR, filename)
    with _db_lock:
        if not os.path.exists(path):
            return default_val
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load local JSON from {path}: {e}")
            return default_val


def _save_local_json(filename: str, payload: Any):
    """
    Saves payload to a local JSON file atomically using a temporary file and replace.
    """
    os.makedirs(_LOCAL_DIR, exist_ok=True)
    path = os.path.join(_LOCAL_DIR, filename)
    
    with _db_lock:
        tmp_path = ""
        try:
            with tempfile.NamedTemporaryFile(mode="w", dir=_LOCAL_DIR, delete=False) as tmp:
                json.dump(payload, tmp, indent=2)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = tmp.name

            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Failed to save local JSON to {path}: {e}")
            raise e
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass


def load_json_data(filename: str, default_val: Any) -> Any:
    """
    Loads JSON data either from the local filesystem or Hugging Face private dataset,
    depending on the configured AXIOM_MODE.
    """
    mode = (settings.MODE or "local").strip().lower()
    
    if mode == "host":
        if not settings.HF_DATASET_PATH or not settings.HF_DATASET_PATH.strip():
            logger.warning("AXIOM_HF_DATASET_PATH is not configured. Falling back to local storage.")
            return _load_local_json(filename, default_val)
            
        dataset_file_path = f"datasets/{settings.HF_DATASET_PATH}/{filename}"
        
        try:
            fs = _get_hf_fs()
            if fs.exists(dataset_file_path):
                with fs.open(dataset_file_path, "r") as f:
                    return json.load(f)
            else:
                logger.info(f"File {dataset_file_path} does not exist in HF dataset. Returning default.")
                return default_val
        except Exception as e:
            logger.error(f"Failed to load JSON from Hugging Face dataset {dataset_file_path}: {e}")
            return default_val
    else:
        return _load_local_json(filename, default_val)


def save_json_data(filename: str, payload: Any):
    """
    Saves JSON data either to the local filesystem or Hugging Face private dataset,
    depending on the configured AXIOM_MODE.
    """
    mode = (settings.MODE or "local").strip().lower()
    
    if mode == "host":
        if not settings.HF_DATASET_PATH or not settings.HF_DATASET_PATH.strip():
            logger.warning("AXIOM_HF_DATASET_PATH is not configured. Falling back to local storage.")
            _save_local_json(filename, payload)
            return
            
        dataset_file_path = f"datasets/{settings.HF_DATASET_PATH}/{filename}"
        
        try:
            fs = _get_hf_fs()
            with fs.open(dataset_file_path, "w") as f:
                json.dump(payload, f, indent=2)
            logger.info(f"Successfully saved and committed {filename} to Hugging Face dataset.")
        except Exception as e:
            logger.error(f"Failed to save and commit JSON to Hugging Face dataset {dataset_file_path}: {e}")
            raise e
    else:
        _save_local_json(filename, payload)
