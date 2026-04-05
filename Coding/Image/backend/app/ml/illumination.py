import numpy as np


def get_sh_basis(normals: np.ndarray) -> np.ndarray:
    nx = normals[..., 0]
    ny = normals[..., 1]
    nz = normals[..., 2]

    c1 = 0.28209479177387814
    c2 = 0.4886025119029199
    c3 = 1.0925484305920792
    c4 = 0.31539156525252005
    c5 = 0.5462742152960396

    y0 = np.ones_like(nx) * c1
    y1 = c2 * ny
    y2 = c2 * nz
    y3 = c2 * nx
    y4 = c3 * nx * ny
    y5 = c3 * ny * nz
    y6 = c4 * (3.0 * nz * nz - 1.0)
    y7 = c3 * nx * nz
    y8 = c5 * (nx * nx - ny * ny)

    return np.stack([y0, y1, y2, y3, y4, y5, y6, y7, y8], axis=-1)


def fit_sh_gpu(
    image_crop: np.ndarray,
    normals: np.ndarray,
    texture: np.ndarray,
    iterations: int = 60,
    lr: float = 0.10,
):
    try:
        image = image_crop.astype(np.float64) / 255.0
        tex_log = np.clip(texture.astype(np.float64), -5.0, 5.0)
        tex_linear = np.exp(tex_log)
        tex_linear = np.clip(tex_linear / max(float(tex_linear.max()), 1e-8), 1e-6, 1.0)

        basis = get_sh_basis(normals.astype(np.float64))
        a = basis.reshape(-1, 9)
        tex_flat = tex_linear.reshape(-1, 3)
        image_flat = image.reshape(-1, 3)

        reg = 1e-3
        ata = (a.T @ a) + (reg * np.eye(9, dtype=np.float64))
        gamma = np.zeros((9, 3), dtype=np.float64)

        for channel in range(3):
            y = image_flat[:, channel] / np.clip(tex_flat[:, channel], 1e-6, None)
            y = np.clip(y, 0.0, 3.0)
            gamma[:, channel] = np.linalg.solve(ata, a.T @ y)

        illumination = (a @ gamma).reshape(image.shape)
        illumination = np.clip(illumination, 0.0, None)

        ambient = basis[..., [0]] * gamma[[0], :]
        direct = illumination - ambient

        return ambient.astype(np.float32), direct.astype(np.float32), gamma.astype(np.float32)
    except Exception:
        zero_img = np.zeros_like(image_crop, dtype=np.float32)
        zero_gamma = np.zeros((9, 3), dtype=np.float32)
        return zero_img, zero_img, zero_gamma
