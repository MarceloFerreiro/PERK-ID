from pathlib import Path
import numpy as np
from skimage import io
from skimage.transform import resize
from PIL import ImageFile

def _to_uint8(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.floating):
        max_val = float(image.max()) if image.size else 0.0
        if max_val <= 1.0:
            image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    if image.dtype != np.uint8:
        info = np.iinfo(image.dtype)
        if info.max != 255:
            image = (image.astype(np.float32) / info.max) * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def read_image(path: Path, target_size: int = 256) -> np.ndarray | None:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        image = io.imread(str(path))
        image = np.asarray(image)

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 4:
                image = image[:, :, :3]
            elif channels == 1:
                image = image[:, :, 0]

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return None

        scale = target_size / min(h, w)
        new_h = max(target_size, int(round(h * scale)))
        new_w = max(target_size, int(round(w * scale)))

        image = resize(
            image,
            (new_h, new_w),
            preserve_range=True,
            anti_aliasing=True,
        )

        image = _to_uint8(image)

        crop_y = (new_h - target_size) // 2
        crop_x = (new_w - target_size) // 2
        image = image[crop_y:crop_y + target_size, crop_x:crop_x + target_size]
        return image
    except (OSError, ValueError, RuntimeError):
        return None
