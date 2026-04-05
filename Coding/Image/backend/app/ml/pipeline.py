import base64
import math
import logging
from typing import Any

import cv2
import numpy as np

from app.ml.vit_classifier import get_full_image_score, get_vit_score

logger = logging.getLogger(__name__)

_TARGET_SIZE = 256
_FEATURE_NAMES = [
    "specular",
    "frequency",
    "topology",
    "patch_consistency",
    "wavelet_score",
    "vit_score",
    "face_present",
    "jpeg_blockiness",
    "sharpness",
    "colorfulness",
    "resolution",
]


def _to_base64(img_np: np.ndarray) -> str:
    try:
        if img_np.ndim == 2 or (img_np.ndim == 3 and img_np.shape[2] == 1):
            img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif img_np.ndim == 3 and img_np.shape[2] == 3:
            img_bgr = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np.astype(np.uint8)
        ok, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if not ok:
            return ""
        b64 = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return ""


def _normalize_map(arr: np.ndarray) -> np.ndarray:
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    if max_v - min_v < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - min_v) / (max_v - min_v)
    return np.clip(norm * 255.0, 0, 255).astype(np.uint8)


def _resize_crop(crop: np.ndarray, target: int = _TARGET_SIZE) -> np.ndarray:
    h, w = crop.shape[:2]
    if h == target and w == target:
        return crop
    return cv2.resize(crop, (target, target), interpolation=cv2.INTER_AREA)


def _compute_quality_metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]

    xs = np.arange(8, w, 8)
    ys = np.arange(8, h, 8)
    block_values = []
    if xs.size > 0:
        block_values.append(np.mean(np.abs(gray[:, xs - 1].astype(np.float32) - gray[:, xs].astype(np.float32))))
    if ys.size > 0:
        block_values.append(np.mean(np.abs(gray[ys - 1, :].astype(np.float32) - gray[ys, :].astype(np.float32))))
    blockiness = float(np.mean(block_values) / 255.0) if block_values else 0.0

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness = float(np.clip(np.log1p(lap_var) / 8.0, 0.0, 1.0))

    if image.ndim == 3 and image.shape[2] == 3:
        rgb = image.astype(np.float32) / 255.0
        rg = rgb[:, :, 0] - rgb[:, :, 1]
        yb = 0.5 * (rgb[:, :, 0] + rgb[:, :, 1]) - rgb[:, :, 2]
        colorfulness = float(
            np.sqrt(np.var(rg) + np.var(yb)) + 0.3 * np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
        )
        colorfulness = float(np.clip(colorfulness, 0.0, 1.0))
    else:
        colorfulness = 0.0

    resolution = float(np.clip(min(h, w) / 1024.0, 0.0, 1.0))

    return {
        "jpeg_blockiness": round(blockiness, 4),
        "sharpness": round(sharpness, 4),
        "colorfulness": round(colorfulness, 4),
        "resolution": round(resolution, 4),
    }


def _build_feature_vector(
    specular: float,
    frequency: float,
    topology: float,
    patch: float,
    wavelet: float,
    vit_score: float,
    face_present: float,
    quality: dict[str, float],
) -> list[float]:
    return [
        float(specular),
        float(frequency),
        float(topology),
        float(patch),
        float(wavelet),
        float(vit_score),
        float(face_present),
        float(quality["jpeg_blockiness"]),
        float(quality["sharpness"]),
        float(quality["colorfulness"]),
        float(quality["resolution"]),
    ]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _build_fft_preview(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    norm = _normalize_map(magnitude)
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)


def _build_wavelet_preview(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Laplacian(gray, cv2.CV_64F)
    norm = _normalize_map(np.abs(edges))
    return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)


def _build_process_inputs(image: np.ndarray, mode: str, components: list[str]) -> dict[str, Any]:
    h, w = image.shape[:2]
    channels = int(image.shape[2]) if image.ndim == 3 else 1
    return {
        "image_shape": [int(h), int(w), channels],
        "analysis_mode": mode,
        "components": components,
        "feature_names": _FEATURE_NAMES,
    }


def _build_explanation(mode: str) -> list[str]:
    common = [
        "Each signal is converted to a value between 0 and 1.",
        "Signals are fused into a heuristic probability.",
        "A calibration model adjusts the heuristic score using verified feedback.",
        "Final verdict uses threshold 0.50 and confidence is |score - 0.5| * 2.",
    ]
    if mode == "fallback":
        return [
            "No face was detected, so the system uses full-image fallback analysis.",
            "Fallback uses frequency, wavelet, and ViT signals.",
            *common,
        ]
    return [
        "Face region is analyzed with geometry, illumination, residual, and signal modules.",
        "Main signals are specular, frequency, topology, patch, wavelet, and ViT.",
        *common,
    ]


def _run_no_face_fallback(image: np.ndarray, include_steps: bool = True) -> dict[str, Any]:
    from app.ml.frequency import calculate_frequency_anomaly
    from app.ml.wavelet import calculate_wavelet_anomaly

    resized = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    quality = _compute_quality_metrics(image)

    freq_score = calculate_frequency_anomaly(resized)
    wavelet_score = calculate_wavelet_anomaly(resized)
    vit_score = get_full_image_score(image)

    combined = (0.30 * freq_score) + (0.30 * wavelet_score) + (0.40 * vit_score)
    calibrated = _sigmoid(8.0 * (combined - 0.45))

    verdict = "Fake" if calibrated >= 0.50 else "Real"
    confidence = abs(calibrated - 0.50) * 2.0

    steps = []
    if include_steps:
        thumb = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        steps = [
            {"step": 1, "label": "Input Image", "data": _to_base64(thumb)},
            {"step": 2, "label": "Frequency Spectrum", "data": _to_base64(_build_fft_preview(thumb))},
            {"step": 3, "label": "Wavelet Detail", "data": _to_base64(_build_wavelet_preview(thumb))},
            {"step": 4, "label": "Final Decision", "data": _to_base64(thumb)},
        ]

    feature_vector = _build_feature_vector(
        specular=0.5,
        frequency=freq_score,
        topology=0.5,
        patch=0.5,
        wavelet=wavelet_score,
        vit_score=vit_score,
        face_present=0.0,
        quality=quality,
    )

    decision_factors = {
        "frequency": round(freq_score, 4),
        "wavelet_score": round(wavelet_score, 4),
        "vit_score": round(vit_score, 4),
        "combined": round(combined, 4),
        "heuristic_score": round(calibrated, 4),
    }

    return {
        "verdict": verdict,
        "confidence": round(confidence, 4),
        "faces_detected": 0,
        "faces": [],
        "steps": steps,
        "full_image_score": round(vit_score, 4),
        "analysis_mode": "fallback",
        "feature_vector": [round(v, 4) for v in feature_vector],
        "quality_metrics": quality,
        "heuristic_score": round(calibrated, 4),
        "fallback_breakdown": {
            "frequency": round(freq_score, 4),
            "wavelet": round(wavelet_score, 4),
            "vit_score": round(vit_score, 4),
            "combined": round(combined, 4),
            "calibrated": round(calibrated, 4),
        },
        "process_inputs": _build_process_inputs(
            image,
            "fallback",
            ["frequency", "wavelet", "vit", "calibration"],
        ),
        "decision_factors": decision_factors,
        "explanation": _build_explanation("fallback"),
    }


def extract_feature_vector_from_image(image: np.ndarray, include_steps: bool = True) -> dict[str, Any]:
    from app.ml.face_detector import detect
    from app.ml.face_alignment import get_3d_shape
    from app.ml.retinex import extract_msr, normalize_for_visualization
    from app.ml.illumination import fit_sh_gpu
    from app.ml.specular import extract, heatmap
    from app.ml.frequency import calculate_frequency_anomaly
    from app.ml.patch_analysis import calculate_patch_anomaly
    from app.ml.topology import compute_topological_anomaly
    from app.ml.sri_net import evaluate_multi_signal, compute_specular_anomaly
    from app.ml.wavelet import calculate_wavelet_anomaly

    faces = detect(image, min_confidence=0.9, min_size=64)
    if not faces:
        return _run_no_face_fallback(image, include_steps=include_steps)

    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    crop = _resize_crop(face.crop)
    quality = _compute_quality_metrics(image)

    steps = []
    if include_steps:
        steps.append({"step": 1, "label": "Input Image", "data": _to_base64(cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA))})
        steps.append({"step": 2, "label": "Detected Face", "data": _to_base64(crop)})

    _, normals = get_3d_shape(crop)
    texture = extract_msr(crop, sigmas=[15, 80, 120])
    tex_vis = normalize_for_visualization(texture)
    if include_steps:
        norm_vis = np.clip((normals + 1.0) * 127.5, 0, 255).astype(np.uint8)
        steps.append({"step": 3, "label": "Surface Normals", "data": _to_base64(norm_vis)})
        steps.append({"step": 4, "label": "Retinex Texture", "data": _to_base64(tex_vis)})

    ambient, direct, _ = fit_sh_gpu(crop, normals, texture)
    if include_steps:
        steps.append({"step": 5, "label": "Ambient Light", "data": _to_base64(_normalize_map(ambient))})
        steps.append({"step": 6, "label": "Direct Light", "data": _to_base64(_normalize_map(direct))})

    specular = extract(crop, ambient, direct, texture)
    specular_heat = heatmap(specular)
    if include_steps:
        steps.append({"step": 7, "label": "Specular Residual", "data": _to_base64(specular_heat)})

    spec_score = compute_specular_anomaly(texture, specular)
    freq_score = calculate_frequency_anomaly(crop)
    patch_score = calculate_patch_anomaly(crop)
    topo_score = compute_topological_anomaly(specular)
    wavelet_score = calculate_wavelet_anomaly(crop)
    vit_face_score = get_vit_score(crop)

    breakdown = evaluate_multi_signal(
        specular_anomaly=spec_score,
        freq_power=freq_score,
        topo_score=topo_score,
        patch_score=patch_score,
        wavelet_score=wavelet_score,
        vit_score=vit_face_score,
    )

    feature_vector = _build_feature_vector(
        specular=spec_score,
        frequency=freq_score,
        topology=topo_score,
        patch=patch_score,
        wavelet=wavelet_score,
        vit_score=vit_face_score,
        face_present=1.0,
        quality=quality,
    )

    if include_steps:
        steps.append({"step": 8, "label": "Final Decision", "data": _to_base64(crop)})

    face_data = {
        "bbox": face.bbox,
        "confidence": float(face.confidence),
        "verdict": "Pending",
        "score": round(breakdown["calibrated"], 4),
        "signal_breakdown": breakdown,
    }

    decision_factors = {
        "specular": round(spec_score, 4),
        "frequency": round(freq_score, 4),
        "topology": round(topo_score, 4),
        "patch_consistency": round(patch_score, 4),
        "wavelet_score": round(wavelet_score, 4),
        "vit_score": round(vit_face_score, 4),
        "physics_ensemble": round(breakdown["physics_ensemble"], 4),
        "heuristic_score": round(breakdown["calibrated"], 4),
    }

    return {
        "verdict": "Pending",
        "confidence": 0.0,
        "faces_detected": len(faces),
        "faces": [face_data],
        "steps": steps,
        "full_image_score": round(vit_face_score, 4),
        "analysis_mode": "full_physics",
        "feature_vector": [round(v, 4) for v in feature_vector],
        "quality_metrics": quality,
        "heuristic_score": round(breakdown["calibrated"], 4),
        "process_inputs": _build_process_inputs(
            image,
            "full_physics",
            [
                "face_detector",
                "geometry_normals",
                "retinex",
                "spherical_harmonics",
                "specular_residual",
                "frequency",
                "topology",
                "patch_consistency",
                "wavelet",
                "vit",
                "calibration",
            ],
        ),
        "decision_factors": decision_factors,
        "explanation": _build_explanation("full_physics"),
    }


def analyze(image: np.ndarray) -> dict[str, Any]:
    from app.ml.evolution import calibrate_prediction

    result = extract_feature_vector_from_image(image, include_steps=True)
    calibration = calibrate_prediction(
        feature_vector=result.get("feature_vector"),
        heuristic_score=float(result.get("heuristic_score", 0.5)),
    )

    final_score = float(calibration["calibrated_score"])
    verdict = "Fake" if final_score >= 0.50 else "Real"
    confidence = abs(final_score - 0.50) * 2.0

    if result.get("faces"):
        result["faces"][0]["verdict"] = verdict
        result["faces"][0]["score"] = round(final_score, 4)
        if result["faces"][0].get("signal_breakdown"):
            result["faces"][0]["signal_breakdown"]["calibrated"] = round(final_score, 4)

    if result.get("steps"):
        result["steps"][-1]["label"] = f"Final Decision: {verdict}"

    decision_factors = dict(result.get("decision_factors") or {})
    decision_factors["learned_score"] = round(float(calibration.get("learned_score", final_score)), 4)
    decision_factors["model_weight"] = round(float(calibration.get("model_weight", 0.0)), 4)
    decision_factors["final_score"] = round(final_score, 4)

    result["decision_factors"] = decision_factors
    result["verdict"] = verdict
    result["confidence"] = round(confidence, 4)
    result["full_image_score"] = round(final_score, 4)
    result["calibration_breakdown"] = calibration

    logger.info(
        "Analysis complete: verdict=%s confidence=%.4f final_score=%.4f",
        verdict,
        confidence,
        final_score,
    )
    return result
