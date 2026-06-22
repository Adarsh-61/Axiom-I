import cv2
import numpy as np


def _component_complexity(binary: np.ndarray) -> tuple[int, int]:
    num_labels, labels = cv2.connectedComponents(binary)
    components = max(0, int(num_labels - 1))

    holes = 0
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for idx in range(len(contours)):
            parent = hierarchy[0][idx][3]
            if parent != -1:
                holes += 1

    return components, holes


def compute_topological_anomaly(spr_map: np.ndarray) -> float:
    try:
        if spr_map.ndim == 3 and spr_map.shape[2] == 3:
            magnitude = np.mean(np.abs(spr_map), axis=2)
        else:
            magnitude = np.abs(spr_map)

        max_val = float(magnitude.max())
        if max_val < 1e-8:
            return 0.5

        field = magnitude / max_val
        field = cv2.resize(field, (96, 96), interpolation=cv2.INTER_AREA)
        field_u8 = np.clip(field * 255.0, 0, 255).astype(np.uint8)

        _, bin_low = cv2.threshold(field_u8, 64, 255, cv2.THRESH_BINARY)
        _, bin_mid = cv2.threshold(field_u8, 128, 255, cv2.THRESH_BINARY)
        _, bin_high = cv2.threshold(field_u8, 192, 255, cv2.THRESH_BINARY)

        c_low, h_low = _component_complexity(bin_low)
        c_mid, h_mid = _component_complexity(bin_mid)
        c_high, h_high = _component_complexity(bin_high)

        complexity = (
            0.20 * c_low
            + 0.35 * c_mid
            + 0.45 * c_high
            + 0.50 * h_low
            + 0.80 * h_mid
            + 1.10 * h_high
        )

        logit = float(np.clip(0.11 * (complexity - 18.0), -40.0, 40.0))
        anomaly = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(anomaly, 0.0, 1.0))
    except Exception:
        return 0.5
