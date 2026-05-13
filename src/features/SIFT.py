from time import time
import numpy as np
from skimage import color
from skimage.feature import SIFT as SkimageSIFT


from src.utils.log import _log_timed

def SIFT(image: np.ndarray | None, n_octaves: int = 4, n_bins: int = 24, n_hist: int = 3, **kwargs) -> np.ndarray:
    """Compute a fixed-length SIFT feature vector for an image."""
    inicio = time()
    if image is None:
        return np.zeros(128, dtype=np.float32)

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    sift = SkimageSIFT(n_octaves=n_octaves, n_bins=n_bins, n_hist=n_hist)
    try:
        sift.detect_and_extract(gray)
    except Exception:
        return np.zeros(72, dtype=np.float32)
    descriptors = sift.descriptors
    if descriptors is None or descriptors.size == 0:
        return np.zeros(72, dtype=np.float32)

    _log_timed(time() - inicio, "SIFT", f"{descriptors.shape[0]} keypoints")
    return descriptors.mean(axis=0).astype(np.float32)


