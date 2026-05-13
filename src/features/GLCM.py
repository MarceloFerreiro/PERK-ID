from time import time
import numpy as np
from skimage import color, util
from skimage.feature import graycomatrix, graycoprops

from src.utils.log import _log_timed

def GLCM(image: np.ndarray | None) -> np.ndarray:
    """Compute GLCM texture descriptors: energy, entropy, contrast, correlation, homogeneity."""
    inicio = time()
    if image is None:
        return np.zeros(5, dtype=np.float32)

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image.astype(np.float32) / 255.0

    levels = 32
    quant = util.img_as_ubyte(gray)
    quant = (quant // (256 // levels)).astype(np.uint8)

    glcm = graycomatrix(
        quant,
        distances=[1],
        angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        levels=levels,
        symmetric=True,
        normed=True,
    )

    energy = graycoprops(glcm, "energy").mean()
    contrast = graycoprops(glcm, "contrast").mean()
    correlation = graycoprops(glcm, "correlation").mean()
    homogeneity = graycoprops(glcm, "homogeneity").mean()
    eps = 1e-12
    entropy = -(glcm * np.log2(glcm + eps)).sum(axis=(0, 1)).mean()

    extra = (
        f"energy={energy:.4f} contrast={contrast:.4f} correlation={correlation:.4f} "
        f"homogeneity={homogeneity:.4f} entropy={entropy:.4f}"
    )
    _log_timed(time() - inicio, "GLCM", extra)
    return np.array([energy, entropy, contrast, correlation, homogeneity], dtype=np.float32)
