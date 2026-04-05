
import json
import logging
import os
import tempfile
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

_FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), "feedback_data")
_FEEDBACK_FILE = os.path.join(_FEEDBACK_DIR, "feedback_log.json")
_METRICS_FILE = os.path.join(_FEEDBACK_DIR, "confusion_matrix.json")
_CALIBRATION_FILE = os.path.join(_FEEDBACK_DIR, "calibration_metrics.json")
_CALIBRATION_HISTORY_FILE = os.path.join(_FEEDBACK_DIR, "calibration_history.json")
_QUARANTINE_FILE = os.path.join(_FEEDBACK_DIR, "quarantine_log.json")
_EXPECTED_FEATURE_DIM = 11
_ECE_BINS = 10
_MIN_CONFIDENCE_MARGIN = 0.05
_MIN_RESOLUTION_FEATURE = 0.08
_MIN_SHARPNESS_FEATURE = 0.05
_MAX_CALIBRATION_HISTORY = 500
_MAX_QUARANTINE_RECORDS = 2000
_MIN_NONLINEAR_SAMPLES = 24
_MIN_NONLINEAR_CLASS_COUNT = 6
_POISON_SIGMA_THRESHOLD = 3.0
_POISON_MAX_FEATURE_OUTLIERS = 2
_POISON_CLASS_DISTANCE_MARGIN = 0.15
_TRUST_DEFAULT_USER_ID = "anonymous"
_TRUST_PRIOR_ALPHA = 2.0
_TRUST_PRIOR_BETA = 2.0
_TRUST_DECAY_HALF_LIFE_DAYS = 45.0
_TRUST_WARMUP_EVENTS = 8.0
_TRUST_WEIGHT_MIN = 0.40
_TRUST_WEIGHT_MAX = 1.60
_SEED_SAMPLE_WEIGHT = 1.15
_MODEL_CACHE: dict[str, Any] = {
    "feedback_mtime": None,
    "seed_mtime": None,
    "model": None,
    "samples": 0,
    "feedback_samples": 0,
    "model_family": "none",
    "cv_log_loss": None,
    "feedback_weight_mean": 0.0,
}
_MODEL_CACHE_LOCK = threading.Lock()
_MODEL_REFRESH_THREAD: threading.Thread | None = None
_SEED_STATS_CACHE: dict[str, Any] = {
    "seed_mtime": None,
    "stats": None,
}


def _ensure_dir():
    os.makedirs(_FEEDBACK_DIR, exist_ok=True)


def _load_json(path: str, default):
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON {path}: {e}")
    return default


def _save_json(path: str, payload):
    _ensure_dir()
    target_dir = os.path.dirname(path) or _FEEDBACK_DIR
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target_dir, delete=False) as tmp:
            json.dump(payload, tmp, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name

        os.replace(tmp_path, path)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _append_quarantine_record(record: dict, reason: str):
    try:
        quarantine = _load_json(_QUARANTINE_FILE, [])
        if not isinstance(quarantine, list):
            quarantine = []

        quarantine.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "reason": reason,
                "full_image_score": float(record.get("full_image_score", 0.5)),
                "original_prediction": record.get("original_prediction"),
                "user_truth": record.get("user_truth"),
                "feature_vector": record.get("feature_vector"),
            }
        )

        if len(quarantine) > _MAX_QUARANTINE_RECORDS:
            quarantine = quarantine[-_MAX_QUARANTINE_RECORDS:]

        _save_json(_QUARANTINE_FILE, quarantine)
    except Exception as e:
        logger.warning(f"Failed to append quarantine record: {e}")


def _load_metrics() -> dict:
    return _load_json(_METRICS_FILE, {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0})


def _save_metrics(metrics: dict):
    _save_json(_METRICS_FILE, metrics)


def _load_calibration_metrics() -> dict:
    return _load_json(
        _CALIBRATION_FILE,
        {
            "total_samples": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "mean_confidence": 0.0,
            "mean_accuracy": 0.0,
            "overconfidence_gap": 0.0,
            "bins": [],
        },
    )


def _save_calibration_metrics(metrics: dict):
    _save_json(_CALIBRATION_FILE, metrics)


def _load_calibration_history() -> list[dict]:
    payload = _load_json(_CALIBRATION_HISTORY_FILE, [])
    return payload if isinstance(payload, list) else []


def _save_calibration_history(history: list[dict]):
    _save_json(_CALIBRATION_HISTORY_FILE, history)


def _append_calibration_history(
    calibration: dict,
    confusion: dict,
    total_feedback_records: int,
) -> None:
    history = _load_calibration_history()
    snapshot = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_feedback_records": int(total_feedback_records),
        "total_samples": int(calibration.get("total_samples", 0)),
        "brier_score": float(calibration.get("brier_score", 0.0)),
        "log_loss": float(calibration.get("log_loss", 0.0)),
        "ece": float(calibration.get("ece", 0.0)),
        "mce": float(calibration.get("mce", 0.0)),
        "overconfidence_gap": float(calibration.get("overconfidence_gap", 0.0)),
        "confusion_total": int(confusion.get("total", 0)),
        "fp": int(confusion.get("FP", 0)),
        "fn": int(confusion.get("FN", 0)),
    }

    history.append(snapshot)
    if len(history) > _MAX_CALIBRATION_HISTORY:
        history = history[-_MAX_CALIBRATION_HISTORY:]
    _save_calibration_history(history)


def _round_feature_vector(feature_vector: list[float]) -> tuple[float, ...]:
    return tuple(round(float(v), 6) for v in feature_vector)


def _normalize_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    value = label.strip().lower()
    if value == "real":
        return "Real"
    if value == "fake":
        return "Fake"
    return None


def _normalize_user_id(user_id: Any) -> str:
    if not isinstance(user_id, str):
        return _TRUST_DEFAULT_USER_ID

    raw = user_id.strip().lower()
    if not raw:
        return _TRUST_DEFAULT_USER_ID

    sanitized = "".join(ch for ch in raw if ch.isalnum() or ch in {"-", "_", "."})
    if not sanitized:
        return _TRUST_DEFAULT_USER_ID
    return sanitized[:64]


def _safe_parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _coerce_feature_vector(feature_vector: list[float] | None) -> list[float] | None:
    if not feature_vector or len(feature_vector) != _EXPECTED_FEATURE_DIM:
        return None
    cleaned = []
    for value in feature_vector:
        try:
            scalar = float(value)
        except Exception:
            return None
        if not np.isfinite(scalar):
            scalar = 0.0
        cleaned.append(scalar)
    return cleaned


def _feedback_training_eligibility(
    full_image_score: float,
    feature_vector: list[float] | None,
    user_truth: str | None = None,
) -> tuple[bool, str | None]:
    if feature_vector is None:
        return False, "missing_feature_vector"

    margin = abs(float(full_image_score) - 0.5)
    if margin < _MIN_CONFIDENCE_MARGIN:
        return False, "low_confidence_margin"

    resolution_feature = float(feature_vector[10])
    if resolution_feature < _MIN_RESOLUTION_FEATURE:
        return False, "low_resolution"

    sharpness_feature = float(feature_vector[8])
    if sharpness_feature < _MIN_SHARPNESS_FEATURE:
        return False, "low_sharpness"

    truth_label = _normalize_label(user_truth) if user_truth is not None else None
    if truth_label is not None:
        is_safe, poison_reason = _poison_guard(feature_vector, truth_label)
        if not is_safe:
            return False, poison_reason

    return True, None


def _feedback_records() -> list[dict]:
    return _load_json(_FEEDBACK_FILE, [])


def _dedup_feedback_rows(records: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen: set[tuple[str, str, str, float, tuple[float, ...] | None]] = set()

    for item in records:
        predicted = _normalize_label(item.get("original_prediction"))
        truth = _normalize_label(item.get("user_truth"))
        if predicted is None or truth is None:
            continue

        try:
            score = float(item.get("full_image_score", 0.5))
        except Exception:
            continue
        if not np.isfinite(score):
            continue
        score = float(np.clip(score, 0.0, 1.0))

        feature_vector = _coerce_feature_vector(item.get("feature_vector"))
        feature_key = _round_feature_vector(feature_vector) if feature_vector is not None else None
        user_id = _normalize_user_id(item.get("user_id"))

        key = (predicted, truth, user_id, round(score, 6), feature_key)
        if key in seen:
            continue
        seen.add(key)

        deduped.append(
            {
                "original_prediction": predicted,
                "user_truth": truth,
                "user_id": user_id,
                "full_image_score": score,
                "feature_vector": feature_vector,
            }
        )

    return deduped


def _compute_calibration_metrics(deduped_rows: list[dict]) -> dict:
    if not deduped_rows:
        return {
            "total_samples": 0,
            "brier_score": 0.0,
            "log_loss": 0.0,
            "ece": 0.0,
            "mce": 0.0,
            "mean_confidence": 0.0,
            "mean_accuracy": 0.0,
            "overconfidence_gap": 0.0,
            "bins": [],
        }

    y_true = np.asarray(
        [1.0 if row["user_truth"] == "Fake" else 0.0 for row in deduped_rows],
        dtype=np.float64,
    )
    y_prob = np.asarray(
        [float(np.clip(row["full_image_score"], 0.0, 1.0)) for row in deduped_rows],
        dtype=np.float64,
    )

    y_pred = (y_prob >= 0.5).astype(np.float64)
    brier = float(np.mean((y_prob - y_true) ** 2))

    eps = 1e-8
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    log_loss = float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))

    bins = np.linspace(0.0, 1.0, _ECE_BINS + 1)
    ece = 0.0
    mce = 0.0
    bin_rows: list[dict[str, float | int]] = []

    for i in range(_ECE_BINS):
        lo = float(bins[i])
        hi = float(bins[i + 1])
        if i == _ECE_BINS - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)

        count = int(np.sum(mask))
        if count == 0:
            continue

        avg_conf = float(np.mean(y_prob[mask]))
        avg_acc = float(np.mean(y_true[mask]))
        gap = abs(avg_acc - avg_conf)

        ece += gap * (count / len(deduped_rows))
        mce = max(mce, gap)

        bin_rows.append(
            {
                "bin_start": round(lo, 3),
                "bin_end": round(hi, 3),
                "count": count,
                "accuracy": round(avg_acc, 4),
                "confidence": round(avg_conf, 4),
                "gap": round(gap, 4),
            }
        )

    mean_conf = float(np.mean(y_prob))
    mean_acc = float(np.mean(y_pred == y_true))

    return {
        "total_samples": int(len(deduped_rows)),
        "brier_score": round(brier, 6),
        "log_loss": round(log_loss, 6),
        "ece": round(float(ece), 6),
        "mce": round(float(mce), 6),
        "mean_confidence": round(mean_conf, 6),
        "mean_accuracy": round(mean_acc, 6),
        "overconfidence_gap": round(mean_conf - mean_acc, 6),
        "bins": bin_rows,
    }


def _compute_confusion_metrics(deduped_rows: list[dict]) -> dict[str, int]:
    metrics = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "total": 0}
    for row in deduped_rows:
        predicted = row["original_prediction"]
        truth = row["user_truth"]
        y_true = 1 if truth == "Fake" else 0
        y_pred = 1 if predicted == "Fake" else 0
        if y_true == 1 and y_pred == 1:
            metrics["TP"] += 1
        elif y_true == 0 and y_pred == 0:
            metrics["TN"] += 1
        elif y_true == 1 and y_pred == 0:
            metrics["FN"] += 1
        else:
            metrics["FP"] += 1
        metrics["total"] += 1
    return metrics


def _seed_dataset_root() -> Path:
    return Path(__file__).resolve().parents[3] / "test"


_SEED_CACHE_FILE = os.path.join(_FEEDBACK_DIR, "seed_features.json")
_SEED_CACHE_VERSION = 2

def _seed_samples() -> list[tuple[list[float], int]]:
                                                                              
                                                                          
    cached = _load_json(_SEED_CACHE_FILE, None)
    if isinstance(cached, dict):
        if cached.get("version") == _SEED_CACHE_VERSION and isinstance(cached.get("samples"), list):
            return cached["samples"]
        logger.warning("Ignoring stale seed feature cache due to version mismatch.")
    elif isinstance(cached, list):
        logger.warning("Ignoring legacy seed feature cache format; rebuilding.")

    seed_root = _seed_dataset_root()
    if not seed_root.exists():
        return []

    from app.ml.pipeline import extract_feature_vector_from_image

    samples: list[tuple[list[float], int]] = []
    for label_name, y in (("Real", 0), ("Fake", 1)):
        class_dir = seed_root / label_name
        if not class_dir.exists():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue
            img_bgr = cv2.imread(str(image_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            try:
                result = extract_feature_vector_from_image(img_rgb, include_steps=False)
                feature_vector = _coerce_feature_vector(result.get("feature_vector"))
                if feature_vector is not None:
                    samples.append((feature_vector, y))
            except Exception as e:
                logger.warning(f"Skipping seed sample {image_path.name}: {e}")
                
                                                 
    _save_json(
        _SEED_CACHE_FILE,
        {
            "version": _SEED_CACHE_VERSION,
            "samples": samples,
        },
    )
        
    return samples


def _seed_feature_stats() -> dict[str, dict[str, np.ndarray]]:
    seed_mtime = os.path.getmtime(_SEED_CACHE_FILE) if os.path.exists(_SEED_CACHE_FILE) else 0.0
    cached_stats = _SEED_STATS_CACHE.get("stats")
    if _SEED_STATS_CACHE.get("seed_mtime") == seed_mtime and isinstance(cached_stats, dict):
        return cached_stats

    grouped: dict[str, list[np.ndarray]] = {"Real": [], "Fake": []}
    for feature_vector, y in _seed_samples():
        label = "Fake" if int(y) == 1 else "Real"
        grouped[label].append(np.asarray(feature_vector, dtype=np.float64))

    stats: dict[str, dict[str, np.ndarray]] = {}
    for label, rows in grouped.items():
        if not rows:
            continue
        arr = np.vstack(rows)
        std = np.clip(arr.std(axis=0), 1e-3, None)
        stats[label] = {
            "mean": arr.mean(axis=0),
            "std": std,
        }

    _SEED_STATS_CACHE["seed_mtime"] = seed_mtime
    _SEED_STATS_CACHE["stats"] = stats
    return stats


def _poison_guard(feature_vector: list[float], user_truth: str) -> tuple[bool, str | None]:
    truth_label = _normalize_label(user_truth)
    if truth_label is None:
        return False, "invalid_user_truth"

    stats = _seed_feature_stats()
    claimed = stats.get(truth_label)
    if claimed is None:
        return True, None

    vec = np.asarray(feature_vector, dtype=np.float64)
    claimed_z = np.abs((vec - claimed["mean"]) / claimed["std"])
    outlier_count = int(np.sum(claimed_z > _POISON_SIGMA_THRESHOLD))

    if outlier_count <= _POISON_MAX_FEATURE_OUTLIERS:
        return True, None

    other_label = "Fake" if truth_label == "Real" else "Real"
    other = stats.get(other_label)
    if other is None:
        return False, "poison_suspected_sigma_outlier"

    claimed_dist = float(np.mean(claimed_z))
    other_dist = float(np.mean(np.abs((vec - other["mean"]) / other["std"])))
    if other_dist + _POISON_CLASS_DISTANCE_MARGIN < claimed_dist:
        return False, "poison_suspected_label_mismatch"

    return False, "poison_suspected_sigma_outlier"


def _cross_validated_log_loss(estimator, X: np.ndarray, y: np.ndarray) -> float | None:
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2:
        return None

    min_class = int(np.min(counts))
    n_splits = min(5, min_class)
    if n_splits < 2:
        return None

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    try:
        probas = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
        return float(log_loss(y, np.clip(probas[:, 1], 1e-6, 1.0 - 1e-6), labels=[0, 1]))
    except Exception as e:
        logger.warning(f"CV scoring failed for {type(estimator).__name__}: {e}")
        return None


def _fit_estimator_with_weights(estimator, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None):
    if sample_weights is None:
        estimator.fit(X, y)
        return

    try:
        if isinstance(estimator, Pipeline):
            estimator.fit(X, y, logreg__sample_weight=sample_weights)
        else:
            estimator.fit(X, y, sample_weight=sample_weights)
    except TypeError:
        estimator.fit(X, y)


def _select_calibration_estimator(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray | None = None,
) -> tuple[Any, str, float | None]:
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.35)),
        ]
    )

    candidates: list[tuple[str, Any]] = [("logistic", logistic)]

    _, counts = np.unique(y, return_counts=True)
    min_class = int(np.min(counts)) if counts.size else 0
    if len(X) >= _MIN_NONLINEAR_SAMPLES and min_class >= _MIN_NONLINEAR_CLASS_COUNT:
        candidates.extend(
            [
                (
                    "random_forest",
                    RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
                (
                    "hist_gradient_boosting",
                    HistGradientBoostingClassifier(
                        max_depth=5,
                        max_iter=120,
                        learning_rate=0.08,
                        random_state=42,
                    ),
                ),
            ]
        )

    best_name = "logistic"
    best_estimator: Any = logistic
    best_loss: float | None = None
    for name, estimator in candidates:
        loss = _cross_validated_log_loss(estimator, X, y)
        if loss is None:
            continue
        if best_loss is None or loss < best_loss:
            best_name = name
            best_estimator = estimator
            best_loss = loss

    _fit_estimator_with_weights(best_estimator, X, y, sample_weights)
    return best_estimator, best_name, best_loss


def _trust_weight_from_score(score: float) -> float:
    clipped = float(np.clip(score, 0.0, 1.0))
    return float(_TRUST_WEIGHT_MIN + ((_TRUST_WEIGHT_MAX - _TRUST_WEIGHT_MIN) * clipped))


def _compute_user_trust_profiles(records: list[dict]) -> dict[str, dict[str, float]]:
    now = datetime.now(timezone.utc)
    aggregate: dict[str, dict[str, float]] = {}

    for item in records:
        predicted = _normalize_label(item.get("original_prediction"))
        truth = _normalize_label(item.get("user_truth"))
        if predicted is None or truth is None:
            continue

        user_id = _normalize_user_id(item.get("user_id"))
        parsed_ts = _safe_parse_timestamp(item.get("timestamp"))
        age_days = 0.0
        if parsed_ts is not None:
            age_days = max(0.0, (now - parsed_ts).total_seconds() / 86400.0)

        decay = 0.5 ** (age_days / _TRUST_DECAY_HALF_LIFE_DAYS)
        bucket = aggregate.setdefault(user_id, {"success": 0.0, "failure": 0.0, "events": 0.0})

        if predicted == truth:
            bucket["success"] += decay
        else:
            bucket["failure"] += decay

        reason = str(item.get("training_exclusion_reason", "") or "")
        if reason.startswith("poison_suspected"):
            bucket["failure"] += decay

        bucket["events"] += decay

    profiles: dict[str, dict[str, float]] = {}
    for user_id, vals in aggregate.items():
        success = float(vals["success"])
        failure = float(vals["failure"])
        support = success + failure

        posterior = (success + _TRUST_PRIOR_ALPHA) / (success + failure + _TRUST_PRIOR_ALPHA + _TRUST_PRIOR_BETA)
        warmup = min(1.0, support / _TRUST_WARMUP_EVENTS)
        trust_score = (warmup * posterior) + ((1.0 - warmup) * 0.5)
        weight = _trust_weight_from_score(trust_score)

        profiles[user_id] = {
            "trust_score": round(float(np.clip(trust_score, 0.0, 1.0)), 6),
            "sample_weight": round(float(np.clip(weight, _TRUST_WEIGHT_MIN, _TRUST_WEIGHT_MAX)), 6),
            "support": round(support, 6),
            "posterior_accuracy": round(float(np.clip(posterior, 0.0, 1.0)), 6),
        }

    if _TRUST_DEFAULT_USER_ID not in profiles:
        neutral_score = 0.5
        profiles[_TRUST_DEFAULT_USER_ID] = {
            "trust_score": neutral_score,
            "sample_weight": round(_trust_weight_from_score(neutral_score), 6),
            "support": 0.0,
            "posterior_accuracy": 0.5,
        }

    return profiles


def _feedback_samples() -> list[tuple[list[float], int, float]]:
    weighted_samples: dict[tuple[int, tuple[float, ...]], dict[str, Any]] = {}
    records = _feedback_records()
    trust_profiles = _compute_user_trust_profiles(records)

    for record in records:
        truth_label = _normalize_label(record.get("user_truth"))
        if truth_label is None:
            continue

        feature_vector = _coerce_feature_vector(record.get("feature_vector"))
        if feature_vector is None:
            continue

        try:
            score = float(record.get("full_image_score", 0.5))
        except Exception:
            score = 0.5
        score = float(np.clip(score, 0.0, 1.0))

        is_eligible, _ = _feedback_training_eligibility(score, feature_vector, truth_label)
        if not is_eligible:
            continue

        y = 1 if truth_label == "Fake" else 0
        key = (y, _round_feature_vector(feature_vector))
        user_id = _normalize_user_id(record.get("user_id"))
        profile = trust_profiles.get(user_id, trust_profiles[_TRUST_DEFAULT_USER_ID])
        sample_weight = float(profile.get("sample_weight", 1.0))

        if key not in weighted_samples:
            weighted_samples[key] = {
                "feature_vector": feature_vector,
                "y": y,
                "weight": sample_weight,
            }
        else:
                                                                             
                                                                              
            weighted_samples[key]["weight"] = min(
                4.0,
                float(weighted_samples[key]["weight"]) + sample_weight,
            )

    return [
        (entry["feature_vector"], int(entry["y"]), float(entry["weight"]))
        for entry in weighted_samples.values()
    ]


def _build_calibration_model() -> tuple[Any | None, dict[str, Any]]:
    seed = _seed_samples()
    feedback = _feedback_samples()
    all_samples: list[tuple[list[float], int, float]] = [
        (x, y, _SEED_SAMPLE_WEIGHT) for x, y in seed
    ]
    all_samples.extend(feedback)

    y_values = [y for _, y, _ in all_samples]
    if len(all_samples) < 8 or len(set(y_values)) < 2:
        return None, {
            "samples": len(all_samples),
            "feedback_samples": len(feedback),
            "model_family": "none",
            "cv_log_loss": None,
            "feedback_weight_mean": 0.0,
        }

    X = np.asarray([x for x, _, _ in all_samples], dtype=np.float64)
    y = np.asarray(y_values, dtype=np.int32)
    sample_weights = np.asarray([w for _, _, w in all_samples], dtype=np.float64)

    model, model_family, cv_loss = _select_calibration_estimator(X, y, sample_weights)

    feedback_weight_mean = 0.0
    if feedback:
        feedback_weight_mean = float(np.mean([w for _, _, w in feedback]))

    return model, {
        "samples": len(all_samples),
        "feedback_samples": len(feedback),
        "model_family": model_family,
        "cv_log_loss": round(cv_loss, 6) if cv_loss is not None else None,
        "feedback_weight_mean": round(feedback_weight_mean, 6),
    }


def _cached_model() -> tuple[Any | None, dict[str, Any]]:
    with _MODEL_CACHE_LOCK:
        feedback_mtime = os.path.getmtime(_FEEDBACK_FILE) if os.path.exists(_FEEDBACK_FILE) else 0.0
        seed_mtime = os.path.getmtime(_SEED_CACHE_FILE) if os.path.exists(_SEED_CACHE_FILE) else 0.0
        if _MODEL_CACHE["feedback_mtime"] == feedback_mtime and _MODEL_CACHE["seed_mtime"] == seed_mtime:
            return _MODEL_CACHE["model"], {
                "samples": _MODEL_CACHE["samples"],
                "feedback_samples": _MODEL_CACHE["feedback_samples"],
                "model_family": _MODEL_CACHE.get("model_family", "none"),
                "cv_log_loss": _MODEL_CACHE.get("cv_log_loss"),
                "feedback_weight_mean": _MODEL_CACHE.get("feedback_weight_mean", 0.0),
            }

        model, meta = _build_calibration_model()
        _MODEL_CACHE["feedback_mtime"] = feedback_mtime
        _MODEL_CACHE["seed_mtime"] = os.path.getmtime(_SEED_CACHE_FILE) if os.path.exists(_SEED_CACHE_FILE) else 0.0
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["samples"] = meta["samples"]
        _MODEL_CACHE["feedback_samples"] = meta["feedback_samples"]
        _MODEL_CACHE["model_family"] = meta.get("model_family", "none")
        _MODEL_CACHE["cv_log_loss"] = meta.get("cv_log_loss")
        _MODEL_CACHE["feedback_weight_mean"] = meta.get("feedback_weight_mean", 0.0)
        return model, meta


def _refresh_model_worker():
    try:
        _cached_model()
    except Exception as e:
        logger.warning(f"Background model refresh failed: {e}")


def _refresh_model_async():
    global _MODEL_REFRESH_THREAD
    if _MODEL_REFRESH_THREAD is not None and _MODEL_REFRESH_THREAD.is_alive():
        return

    _MODEL_REFRESH_THREAD = threading.Thread(
        target=_refresh_model_worker,
        name="axiom-calibration-refresh",
        daemon=True,
    )
    _MODEL_REFRESH_THREAD.start()


def calibrate_prediction(feature_vector: list[float] | None, heuristic_score: float) -> dict[str, float]:
    heuristic = float(np.clip(heuristic_score, 0.0, 1.0))
    coerced = _coerce_feature_vector(feature_vector)
    model, meta = _cached_model()

    if model is None or coerced is None:
        return {
            "calibrated_score": round(heuristic, 4),
            "heuristic_score": round(heuristic, 4),
            "learned_score": round(heuristic, 4),
            "model_weight": 0.0,
            "training_samples": float(meta["samples"]),
            "feedback_samples": float(meta["feedback_samples"]),
        }

    learned = float(model.predict_proba(np.asarray([coerced], dtype=np.float64))[0, 1])
    feedback_count = int(meta["feedback_samples"])
    training_samples = int(meta["samples"])

                                                                         
                                    
    model_weight = 0.80 + (0.002 * min(feedback_count, 50))

                                                        
    if training_samples < 12:
        model_weight = min(model_weight, 0.70)

    model_weight = min(0.90, max(0.60, model_weight))
    calibrated = (model_weight * learned) + ((1.0 - model_weight) * heuristic)

    return {
        "calibrated_score": round(float(np.clip(calibrated, 0.0, 1.0)), 4),
        "heuristic_score": round(heuristic, 4),
        "learned_score": round(learned, 4),
        "model_weight": round(model_weight, 4),
        "training_samples": float(meta["samples"]),
        "feedback_samples": float(meta["feedback_samples"]),
        "feedback_weight_mean": float(meta.get("feedback_weight_mean", 0.0)),
    }


def get_feedback_diagnostics() -> dict[str, Any]:
    records = _feedback_records()
    deduped_rows = _dedup_feedback_rows(records)

    confusion = _compute_confusion_metrics(deduped_rows)
    calibration = _compute_calibration_metrics(deduped_rows)

    _save_metrics(confusion)
    _save_calibration_metrics(calibration)

    training_eligible_count = 0
    reason_counts: Counter[str] = Counter()
    trust_profiles = _compute_user_trust_profiles(records)
    trust_scores = [
        p["trust_score"]
        for user, p in trust_profiles.items()
        if user != _TRUST_DEFAULT_USER_ID
    ]
    sample_weights = [
        p["sample_weight"]
        for user, p in trust_profiles.items()
        if user != _TRUST_DEFAULT_USER_ID
    ]

    for item in records:
        truth_label = _normalize_label(item.get("user_truth"))
        feature_vector = _coerce_feature_vector(item.get("feature_vector"))
        try:
            score = float(item.get("full_image_score", 0.5))
        except Exception:
            score = 0.5
        score = float(np.clip(score, 0.0, 1.0))

        if truth_label is None:
            is_eligible = False
            reason = "invalid_user_truth"
        else:
            is_eligible, reason = _feedback_training_eligibility(score, feature_vector, truth_label)

        if is_eligible:
            training_eligible_count += 1
        elif reason:
            reason_counts[str(reason)] += 1
        else:
            reason_counts["unspecified_exclusion"] += 1

    tracked_users_count = int(max(0, len(trust_profiles) - 1))

    summary = {
        "total_feedback_records": len(records),
        "training_eligible_records": int(training_eligible_count),
        "training_excluded_records": int(len(records) - training_eligible_count),
        "training_exclusion_reasons": dict(reason_counts),
        "trust_summary": {
            "tracked_users": tracked_users_count,
            "mean_trust_score": round(float(np.mean(trust_scores)), 4) if tracked_users_count > 0 and trust_scores else None,
            "min_trust_score": round(float(np.min(trust_scores)), 4) if tracked_users_count > 0 and trust_scores else None,
            "max_trust_score": round(float(np.max(trust_scores)), 4) if tracked_users_count > 0 and trust_scores else None,
            "mean_sample_weight": round(float(np.mean(sample_weights)), 4) if tracked_users_count > 0 and sample_weights else None,
        },
    }

                                                                     
    calibration_history = _load_calibration_history()
    if not calibration_history and int(calibration.get("total_samples", 0)) > 0:
                                                                               
        _append_calibration_history(
            calibration=calibration,
            confusion=confusion,
            total_feedback_records=len(records),
        )
        calibration_history = _load_calibration_history()
    calibration_history = calibration_history[-30:]

    return {
        "confusion_matrix": confusion,
        "calibration_metrics": calibration,
        "calibration_history": calibration_history,
        "feedback_summary": summary,
    }


def record_feedback(
    full_image_score: float,
    original_prediction: str,
    user_truth: str,
    feature_vector: list | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    _ensure_dir()

    predicted_label = _normalize_label(original_prediction)
    truth_label = _normalize_label(user_truth)
    if predicted_label is None or truth_label is None:
        raise ValueError("original_prediction and user_truth must each be 'Real' or 'Fake'.")

    try:
        score = float(full_image_score)
    except Exception as e:
        raise ValueError("full_image_score must be a finite float in [0, 1].") from e
    if not np.isfinite(score) or score < 0.0 or score > 1.0:
        raise ValueError("full_image_score must be in [0, 1].")

    feature_vector = _coerce_feature_vector(feature_vector)
    normalized_user_id = _normalize_user_id(user_id)
    training_eligible, exclusion_reason = _feedback_training_eligibility(
        score,
        feature_vector,
        truth_label,
    )

    if not training_eligible:
        logger.warning(
            "Feedback excluded from calibration training: reason=%s",
            exclusion_reason,
        )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "full_image_score": score,
        "original_prediction": predicted_label,
        "user_truth": truth_label,
        "user_id": normalized_user_id,
        "was_correct": predicted_label == truth_label,
        "feature_vector": feature_vector,
        "training_eligible": training_eligible,
        "training_exclusion_reason": exclusion_reason,
    }
    if exclusion_reason and exclusion_reason.startswith("poison_suspected"):
        record["quarantined"] = True
        _append_quarantine_record(record, exclusion_reason)
    else:
        record["quarantined"] = False

    existing = []
    try:
        existing = _feedback_records()
        duplicate_key = (
            predicted_label,
            truth_label,
            normalized_user_id,
            round(score, 6),
            _round_feature_vector(feature_vector) if feature_vector is not None else None,
        )
        existing_keys: set[
            tuple[str | None, str | None, str, float, tuple[float, ...] | None]
        ] = set()
        for item in existing:
            pred = _normalize_label(item.get("original_prediction"))
            truth = _normalize_label(item.get("user_truth"))
            existing_user_id = _normalize_user_id(item.get("user_id"))

            try:
                existing_score = float(item.get("full_image_score", 0.0))
            except Exception:
                existing_score = 0.0
            existing_score = round(float(np.clip(existing_score, 0.0, 1.0)), 6)

            existing_vec = _coerce_feature_vector(item.get("feature_vector"))
            vec_key = _round_feature_vector(existing_vec) if existing_vec is not None else None

            existing_keys.add((pred, truth, existing_user_id, existing_score, vec_key))

        if duplicate_key not in existing_keys:
            existing.append(record)
            _save_json(_FEEDBACK_FILE, existing)
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE["feedback_mtime"] = None
                _MODEL_CACHE["seed_mtime"] = None
            _refresh_model_async()
    except Exception as e:
        logger.error(f"Failed to save feedback record: {e}")

    deduped_rows = _dedup_feedback_rows(existing)
    metrics = _compute_confusion_metrics(deduped_rows)
    _save_metrics(metrics)

    calibration = _compute_calibration_metrics(deduped_rows)
    _save_calibration_metrics(calibration)
    _append_calibration_history(
        calibration=calibration,
        confusion=metrics,
        total_feedback_records=len(existing),
    )

    logger.info(
        f"Feedback recorded: predicted={predicted_label}, truth={truth_label}, "
        f"correct={record['was_correct']}, training_eligible={training_eligible}, "
        f"feature_vector={'yes' if feature_vector else 'no'}"
    )

    trust_profile = _compute_user_trust_profiles(existing).get(normalized_user_id, {})
    return {
        "confusion_matrix": metrics,
        "training_eligible": training_eligible,
        "training_exclusion_reason": exclusion_reason,
        "calibration_metrics": calibration,
        "user_trust_score": trust_profile.get("trust_score", 0.5),
        "user_sample_weight": trust_profile.get("sample_weight", 1.0),
    }
