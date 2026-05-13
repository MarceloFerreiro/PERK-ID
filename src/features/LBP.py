from time import time
import numpy as np
from skimage import color, util
from skimage.feature import local_binary_pattern

from src.utils.log import _log_timed

def LBP(image: np.ndarray | None, radius: int = 3, n_points: int = 24, bins: int = 59, method: str = "uniform", **kwargs) -> np.ndarray:
    """Compute Local Binary Pattern (LBP) histogram for texture analysis."""
    inicio = time()
    if image is None:
        return np.zeros(bins, dtype=np.float32)

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    # Convert to uint8 for LBP (required for proper results)
    gray = util.img_as_ubyte(gray)
    
    # Compute LBP
    lbp = local_binary_pattern(gray, n_points, radius, method=method)
    
    # Compute histogram
    hist, _ = np.histogram(lbp, bins=bins, range=(0, bins), density=True)
    hist = hist.astype(np.float32)

    extra = f"radius={radius} n_points={n_points} entropy={-(hist[hist > 0] * np.log2(hist[hist > 0])).sum():.4f}"
    _log_timed(time() - inicio, "LBP", extra)
    return hist
