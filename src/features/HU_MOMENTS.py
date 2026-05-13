from time import time
import numpy as np
from skimage import color, util
from skimage.measure import moments, moments_central, moments_hu

from src.utils.log import _log_timed

def HU_MOMENTS(image: np.ndarray | None, **kwargs) -> np.ndarray:
    """Compute Hu moments (7 scale, rotation, translation invariant moments)."""
    inicio = time()
    if image is None:
        return np.zeros(7, dtype=np.float32)

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    # Ensure image is in [0, 1] range
    gray = np.clip(gray, 0, 1)

    # Compute Hu moments
    hu = moments_hu(gray)
    hu = hu.astype(np.float32)

    extra = " ".join([f"hu{idx}={val:.6f}" for idx, val in enumerate(hu, 1)])
    _log_timed(time() - inicio, "HU_MOMENTS", extra)
    return hu
