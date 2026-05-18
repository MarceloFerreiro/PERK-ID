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


def read_image(
    source: Path | str | np.ndarray,
    target_size: int = 256,
    return_meta: bool = False,
) -> np.ndarray | None | tuple[np.ndarray | None, dict | None]:
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        if isinstance(source, (str, Path)):
            image = io.imread(str(source))
        else:
            image = np.asarray(source)

        if image.ndim == 3:
            channels = image.shape[2]
            if channels == 4:
                image = image[:, :, :3]
            elif channels != 3:
                # For any other channel count (1, 2, etc.), convert to RGB
                if channels == 1:
                    image = np.stack([image[:, :, 0]] * 3, axis=2)
                else:
                    # For channels > 3, take first 3; for channels == 2, replicate
                    image = image[:, :, :3] if channels > 3 else np.concatenate([image, image[:, :, :1]], axis=2)
        elif image.ndim == 2:
            # Grayscale 2D image: convert to 3-channel RGB
            image = np.stack([image] * 3, axis=2)

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return (None, None) if return_meta else None

        # recorta conservando el aspect ratio a 256
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
        if return_meta:
            meta = {
                "frame_h": h,
                "frame_w": w,
                "new_h": new_h,
                "new_w": new_w,
                "crop_x": crop_x,
                "crop_y": crop_y,
                "scale_x": new_w / w,
                "scale_y": new_h / h,
            }
            return image, meta
        return image
    except (OSError, ValueError, RuntimeError):
        return (None, None) if return_meta else None
