from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from skimage.util import random_noise
from skimage.transform import AffineTransform, warp
import numpy as np
import random
import argparse
from pathlib import Path
from collections.abc import Callable

from .read_image import read_image
from src.utils.config import get_config

def transform(img):
    cfg = get_config().get("transform", {})
    arr = np.array(img)
    arr_f = arr.astype(np.float32) / 255.0

    # --- Salt & pepper noise ---
    if random.random() < cfg.get("noise_prob", 0.3):
        arr_f = random_noise(
            arr_f,
            mode="s&p",
            amount=random.uniform(cfg.get("noise_amount_min", 0.001), cfg.get("noise_amount_max", 0.008))
        )

    # --- Affine transform ---
    rot = cfg.get("rotation_deg", 15.0)
    shear = cfg.get("shear_deg", 5.0)
    scale_min = cfg.get("scale_min", 0.85)
    scale_max = cfg.get("scale_max", 1.15)
    trans = cfg.get("translation_px", 5.0)
    at = AffineTransform(
        scale=(random.uniform(scale_min, scale_max), random.uniform(scale_min, scale_max)),
        rotation=random.uniform(-rot, rot) * np.pi / 180.0,
        shear=random.uniform(-shear, shear) * np.pi / 180.0,
        translation=(random.uniform(-trans, trans), random.uniform(-trans, trans)),
    )
    arr_f = warp(arr_f, inverse_map=at, mode="edge", preserve_range=True)
    arr = np.clip(arr_f * 255.0, 0, 255).astype(np.uint8)

    # back to PIL for PIL-only ops
    img = Image.fromarray(arr)

    # --- Blur ---
    if random.random() < cfg.get("blur_prob", 0.3):
        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(cfg.get("blur_radius_min", 0.5), cfg.get("blur_radius_max", 1.5))
            )
        )

    # --- Color jitter ---
    if random.random() < cfg.get("jitter_prob", 0.6):
        jmin = cfg.get("jitter_min", 0.85)
        jmax = cfg.get("jitter_max", 1.15)
        img = ImageEnhance.Brightness(img).enhance(random.uniform(jmin, jmax))
        img = ImageEnhance.Contrast(img).enhance(  random.uniform(jmin, jmax))
        img = ImageEnhance.Color(img).enhance(     random.uniform(jmin, jmax))
    return np.array(img)


class TransformPipeline:
    def __init__(
        self,
        transforms: list[Callable[[np.ndarray, np.ndarray | None], tuple[np.ndarray, np.ndarray | None]]],
    ) -> None:
        self.transforms = list(transforms)

    def __call__(
        self,
        image: np.ndarray,
        bbox: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        for transform_fn in self.transforms:
            image, bbox = transform_fn(image, bbox)
        return image, bbox


class RandomAffineXYWH:
    def __init__(
        self,
        rotation_deg: float = 10.0,
        shear_deg: float = 5.0,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        p: float = 1.0,
        min_box_size: float = 1.0,
        mode: str = "edge",
        order: int = 1,
    ) -> None:
        if scale_min <= 0 or scale_max <= 0 or scale_max < scale_min:
            raise ValueError("Invalid scale range")

        self.rotation_deg = float(rotation_deg)
        self.shear_deg = float(shear_deg)
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.p = float(p)
        self.min_box_size = float(min_box_size)
        self.mode = mode
        self.order = int(order)

    def _sample_transform(self, image_shape: tuple[int, int]) -> AffineTransform:
        h, w = image_shape
        scale_x = random.uniform(self.scale_min, self.scale_max)
        scale_y = random.uniform(self.scale_min, self.scale_max)
        rotation = random.uniform(-self.rotation_deg, self.rotation_deg) * np.pi / 180.0
        shear = random.uniform(-self.shear_deg, self.shear_deg) * np.pi / 180.0
        max_tx = max(0.0, 1.0 - scale_x) * float(w)
        max_ty = max(0.0, 1.0 - scale_y) * float(h)
        trans_x = random.uniform(-max_tx, max_tx)
        trans_y = random.uniform(-max_ty, max_ty)
        return AffineTransform(
            scale=(scale_x, scale_y),
            rotation=rotation,
            shear=shear,
            translation=(trans_x, trans_y),
        )

    def _warp_image(self, image: np.ndarray, tform: AffineTransform) -> np.ndarray:
        image_f = image.astype(np.float32, copy=False)
        warped = warp(
            image_f,
            inverse_map=tform.inverse,
            mode=self.mode,
            order=self.order,
            preserve_range=True,
        )
        if image.dtype == np.uint8:
            return np.clip(warped, 0, 255).astype(np.uint8)
        return warped.astype(image.dtype, copy=False)

    def __call__(
        self,
        image: np.ndarray,
        bbox: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if self.p < 1.0 and random.random() > self.p:
            return image, bbox

        h, w = image.shape[:2]
        tform = self._sample_transform((h, w))
        image_out = self._warp_image(image, tform)

        if bbox is None:
            return image_out, None

        bbox_arr = np.asarray(bbox, dtype=np.float32)
        if bbox_arr.shape[-1] != 4:
            raise ValueError("bbox must be [x, y, width, height]")

        x, y, bw, bh = bbox_arr.tolist()
        corners = np.array(
            [
                [x, y],
                [x + bw, y],
                [x + bw, y + bh],
                [x, y + bh],
            ],
            dtype=np.float32,
        )

        # Apply the same forward transform used for the image.
        warped_corners = tform(corners)
        x_min = float(np.min(warped_corners[:, 0]))
        y_min = float(np.min(warped_corners[:, 1]))
        x_max = float(np.max(warped_corners[:, 0]))
        y_max = float(np.max(warped_corners[:, 1]))

        x_min = float(np.clip(x_min, 0.0, w - 1.0))
        y_min = float(np.clip(y_min, 0.0, h - 1.0))
        x_max = float(np.clip(x_max, 0.0, w - 1.0))
        y_max = float(np.clip(y_max, 0.0, h - 1.0))

        new_w = x_max - x_min
        new_h = y_max - y_min
        if new_w < self.min_box_size or new_h < self.min_box_size:
            return image_out, None

        new_bbox = np.array([x_min, y_min, new_w, new_h], dtype=np.float32)
        return image_out, new_bbox


def _load_random_images(image_dir: Path, count: int, seed: int = 42) -> list[Image.Image]:
    paths = sorted(image_dir.glob("*.jp*g")) + sorted(image_dir.glob("*.png"))
    if not paths:
        raise RuntimeError(f"No images found in {image_dir}")
    random.seed(seed)
    picks = random.sample(paths, k=count)
    images: list[Image.Image] = []
    for path in picks:
        arr = read_image(path)
        if arr is None:
            continue
        images.append(Image.fromarray(arr))

    return images


def _make_grid(pairs: list[tuple[Image.Image, Image.Image]], h: int, w: int) -> Image.Image:
    if not pairs:
        raise RuntimeError("No image pairs to plot")
    tile_w, tile_h = pairs[0][0].size
    gutter = 12
    outer = 16
    cell_w = tile_w * 2 + gutter
    cell_h = tile_h
    grid_w = outer * 2 + w * cell_w - gutter
    grid_h = outer * 2 + h * cell_h + (h - 1) * gutter
    grid = Image.new("RGB", (grid_w, grid_h), (245, 245, 245))
    draw = ImageDraw.Draw(grid)
    idx = 0
    for row in range(h):
        for col in range(w):
            if idx >= len(pairs):
                break
            orig, aug = pairs[idx]
            x = outer + col * cell_w
            y = outer + row * (cell_h + gutter)
            frame = (x - 1, y - 1, x + tile_w * 2 + 1, y + tile_h + 1)
            draw.rectangle(frame, outline=(210, 210, 210), width=1)
            grid.paste(orig, (x, y))
            grid.paste(aug, (x + tile_w, y))
            idx += 1
    return grid


# Example
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preview data augmentation grid")
    parser.add_argument("--dir", type=Path, default=Path("data/imagenes"))
    parser.add_argument("--h", type=int, default=3, help="Grid rows")
    parser.add_argument("--w", type=int, default=4, help="Grid columns")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("tmp/transform_grid.png"))
    args = parser.parse_args()

    total = max(args.h * args.w, 1)
    originals = _load_random_images(args.dir, total, args.seed)
    augmented = [Image.fromarray(transform(img)) for img in originals]
    pairs = list(zip(originals, augmented))
    grid = _make_grid(pairs, args.h, args.w)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    grid.save(args.out)
