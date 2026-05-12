from time import time
import numpy as np
from skimage import util


def _log_timed(_elapsed: float, _feature: str, _extra: str) -> None:
    return

def HIST(image: np.ndarray | None, bins: int = 32) -> np.ndarray:
    """Compute concatenated per-channel histograms plus per-channel means."""
    inicio = time()
    if image is None:
        return np.zeros(bins * 3 + 3, dtype=np.float32)

    img = util.img_as_float(image)
    if img.ndim == 2:
        img = img[:, :, None]

    channels = img.shape[2]
    hist_list = []
    means = []
    for c in range(channels):
        chan = img[:, :, c]
        means.append(float(chan.mean()))
        hist, _ = np.histogram(chan, bins=bins, range=(0.0, 1.0), density=True)
        hist_list.append(hist.astype(np.float32))

    while len(hist_list) < 3:
        hist_list.append(np.zeros(bins, dtype=np.float32))
        means.append(0.0)

    extra = " ".join([f"m{idx}={val:.4f}" for idx, val in enumerate(means[:3], 1)])
    _log_timed(time() - inicio, "HIST", extra)
    return np.concatenate([*hist_list[:3], np.array(means[:3], dtype=np.float32)])
