"""
Binary search for the maximum global scale factor α ∈ [0,1] such that
all augmentation parameters scaled by α keep Top-K >= target.

Each parameter is scaled from 0 (identity) to its individual threshold value
(found by sweep.py). α=1 means all params at their individual threshold.

Usage:
    python -m src.combo_search
    python -m src.combo_search --features data/indices/f_sn.npz --bow data/indices/bow.pkl
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

import src.utils.config as _cfg_mod
from src.Ranker import Ranker
from src.features import read_image
from src.features.transform import transform
from src.features.BoW import BoWExtractor
from src.utils.config import get_config

# Individual thresholds from sweep (α=1 values)
_T = dict(
    scale_dev      = 0.2833,
    rotation_deg   = 45.0,
    shear_deg      = 32.5,
    translation_px = 80.0,   # sweep showed >50 is fine, using 80 as ceiling
    noise_amount   = 0.01977,
    blur_radius    = 0.8316,
    jitter_range   = 0.3818,
)


def _build_cfg(alpha: float) -> dict:
    a = max(alpha, 0.0)
    return dict(
        scale_min        = 1 - _T["scale_dev"]    * a,
        scale_max        = 1 + _T["scale_dev"]    * a,
        rotation_deg     = _T["rotation_deg"]     * a,
        shear_deg        = _T["shear_deg"]        * a,
        translation_px   = _T["translation_px"]   * a,
        noise_prob       = a,                          # scales with α, not binary
        noise_amount_min = _T["noise_amount"] / 2 * a,
        noise_amount_max = _T["noise_amount"]     * a,
        blur_prob        = a,                          # scales with α, not binary
        blur_radius_min  = max(0.1, _T["blur_radius"] / 2 * a),
        blur_radius_max  = max(0.1, _T["blur_radius"]     * a),
        jitter_prob      = a,                          # scales with α, not binary
        jitter_min       = 1 - _T["jitter_range"] * a,
        jitter_max       = 1 + _T["jitter_range"] * a,
    )


def _eval(alpha, ranker, paths, eval_indices, topk):
    _cfg_mod._cache["transform"] = _build_cfg(alpha)
    acc1, acck = [], []
    for idx in eval_indices:
        img = read_image(Path(paths[idx]))
        if img is None:
            continue
        aug = transform(img)
        indices, _, _ = ranker.rank(aug)
        acc1.append(1 if indices[0] == idx else 0)
        acck.append(1 if idx in indices else 0)
    return float(np.mean(acc1)), float(np.mean(acck))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   type=Path, default=Path("data/indices/f_sn.npz"))
    parser.add_argument("--bow",        type=Path, default=Path("data/indices/bow.pkl"))
    parser.add_argument("--eval-size",  type=int,  default=100)
    parser.add_argument("--topk",       type=int,  default=10)
    parser.add_argument("--target",     type=float, default=0.9)
    parser.add_argument("--iterations", type=int,  default=8)
    parser.add_argument("--seed",       type=int,  default=42)
    args = parser.parse_args()

    data   = np.load(args.features, allow_pickle=False)
    matrix = data["matrix"].astype(np.float32)
    paths  = data["paths"]
    n      = len(paths)

    rng          = np.random.default_rng(args.seed)
    eval_indices = rng.choice(n, size=min(args.eval_size, n), replace=False)

    get_config()

    bow_extractor = None
    if args.bow.exists():
        n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 250)
        bow_extractor = BoWExtractor.load(args.bow, n_clusters=n_clusters)

    ranker = Ranker(matrix, paths, n_neighbors=args.topk, bow_extractor=bow_extractor)
    original_transform = get_config().get("transform", {}).copy()

    print(f"\nBinary search: α ∈ [0, 1]  |  target Top-{args.topk} >= {args.target}  |  {args.iterations} iterations\n")

    lo, hi = 0.0, 1.0
    best_alpha, best_top1, best_topk = 0.0, 1.0, 1.0

    try:
        for i in range(args.iterations):
            mid = (lo + hi) / 2
            t1, tk = _eval(mid, ranker, paths, eval_indices, args.topk)
            status = ">= target ✓" if tk >= args.target else "< target  ✗"
            print(f"  iter {i+1:2d}  α={mid:.4f}  top1={t1:.3f}  top{args.topk}={tk:.3f}  {status}")

            if tk >= args.target:
                best_alpha, best_top1, best_topk = mid, t1, tk
                lo = mid
            else:
                hi = mid

    finally:
        _cfg_mod._cache["transform"] = original_transform

    print(f"\n{'='*52}")
    print(f"  Best α = {best_alpha:.4f}")
    print(f"  Top-1  = {best_top1:.3f}  |  Top-{args.topk} = {best_topk:.3f}")
    print(f"\n  Combined profile at α={best_alpha:.4f}:")
    cfg = _build_cfg(best_alpha)
    for k, v in cfg.items():
        print(f"    {k:<22} = {v:.4f}")
    print(f"{'='*52}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
