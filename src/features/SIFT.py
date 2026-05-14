from time import time
import numpy as np
from skimage import color
from skimage.feature import SIFT as SkimageSIFT


from src.utils.log import _log_timed

def SIFT(image: np.ndarray | None, n_octaves: int = 4, n_bins: int = 24, n_hist: int = 3, **kwargs) -> np.ndarray:
    """Compute SIFT descriptors for an image. Returns all descriptors (not averaged).
    
    Returns:
        np.ndarray: Array of descriptors with shape (n_descriptors, 128), or empty array if no descriptors found.
    """
    inicio = time()
    descriptor_size = 128
    if image is None:
        return np.array([], dtype=np.float32).reshape(0, descriptor_size)

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    sift = SkimageSIFT(n_octaves=n_octaves, n_bins=n_bins, n_hist=n_hist)
    try:
        sift.detect_and_extract(gray)
    except Exception:
        return np.array([], dtype=np.float32).reshape(0, descriptor_size)
    descriptors = sift.descriptors
    if descriptors is None or descriptors.size == 0:
        return np.array([], dtype=np.float32).reshape(0, descriptor_size)

    _log_timed(time() - inicio, "SIFT", f"{descriptors.shape[0]} keypoints")
    return descriptors.astype(np.float32)



