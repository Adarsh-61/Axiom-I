import cv2
import numpy as np


def _estimate_depth(image_crop: np.ndarray) -> np.ndarray:
    if image_crop.ndim == 3:
        gray = cv2.cvtColor(image_crop, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_crop

    smoothed = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
    depth = smoothed.astype(np.float64) / 255.0
    depth = cv2.GaussianBlur(depth, (11, 11), sigmaX=0)

    h, w = depth.shape
    y, x = np.ogrid[:h, :w]
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    ny = (y - cy) / max(cy, 1.0)
    nx = (x - cx) / max(cx, 1.0)
    radius = np.sqrt(nx * nx + ny * ny)
    prior = np.clip(1.0 - radius * radius, 0.0, 1.0)

    return (0.75 * depth) + (0.25 * prior)


def _compute_normals(depth: np.ndarray) -> np.ndarray:
    dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)

    normals = np.zeros((*depth.shape, 3), dtype=np.float64)
    normals[:, :, 0] = -2.0 * dx
    normals[:, :, 1] = -2.0 * dy
    normals[:, :, 2] = 1.0

    norms = np.linalg.norm(normals, axis=2, keepdims=True)
    return normals / np.clip(norms, 1e-8, None)


def get_3d_shape(image_crop: np.ndarray):
    try:
        depth = _estimate_depth(image_crop)
        normals = _compute_normals(depth)
        return None, normals
    except Exception:
        h, w = image_crop.shape[:2]
        fallback = np.zeros((h, w, 3), dtype=np.float64)
        fallback[:, :, 2] = 1.0
        return None, fallback
