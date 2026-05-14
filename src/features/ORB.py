from time import time
import numpy as np
from skimage import color
import cv2

from src.utils.log import _log_timed

def ORB(image: np.ndarray | None, n_features: int = 500, scale_factor: float = 1.2, n_levels: int = 8, **kwargs) -> np.ndarray:
    """Compute a fixed-length ORB (Oriented FAST and Rotated BRIEF) feature vector for an image."""
    inicio = time()
    descriptor_size = 32
    if image is None:
        return np.zeros(descriptor_size, dtype=np.float32)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        if image.dtype == np.float32 or image.dtype == np.float64:
            gray = (image * 255).astype(np.uint8)
        else:
            gray = image.astype(np.uint8)

    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=n_features, scaleFactor=scale_factor, nlevels=n_levels)
    
    try:
        # Detect keypoints and compute descriptors
        keypoints, descriptors = orb.detectAndCompute(gray, None)
    except Exception:
        return np.zeros(descriptor_size, dtype=np.float32)
    
    if descriptors is None or descriptors.size == 0:
        return np.zeros(descriptor_size, dtype=np.float32)

    # Convert descriptors to float and compute mean
    descriptors = descriptors.astype(np.float32)
    mean_descriptor = descriptors.mean(axis=0).astype(np.float32)
    
    _log_timed(time() - inicio, "ORB", f"{len(keypoints)} keypoints")
    assert len(mean_descriptor) == descriptor_size, 'ORB() created a vector of wrong size'
    return mean_descriptor
