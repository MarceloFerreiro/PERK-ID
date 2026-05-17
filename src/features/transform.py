from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
from skimage.util import random_noise
from skimage.transform import AffineTransform, warp
import numpy as np
import random
import argparse
from pathlib import Path

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
