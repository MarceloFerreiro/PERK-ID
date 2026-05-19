"""
Sweep each augmentation parameter independently (all others at identity)
and find the maximum value where Top-10 stays >= 0.9.

Usage:
    python -m src.sweep
    python -m src.sweep --eval-size 100 --n-steps 7 --target 0.9 --out-dir tmp/sweep
"""
import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import src.utils.config as _cfg_mod
from src.Ranker import Ranker
from src.features import read_image
from src.features.transform import transform
from src.features.BoW import BoWExtractor
from src.utils.config import get_config

# ---------------------------------------------------------------------------
# Identity base config (no augmentation)
# ---------------------------------------------------------------------------
_BASE = dict(
    scale_min=1.0, scale_max=1.0,
    rotation_deg=0.0, shear_deg=0.0, translation_px=0.0,
    noise_prob=0.0, noise_amount_min=0.001, noise_amount_max=0.001,
    blur_prob=0.0,  blur_radius_min=0.1,   blur_radius_max=0.1,
    jitter_prob=0.0, jitter_min=1.0, jitter_max=1.0,
)


def _cfg(overrides: dict) -> dict:
    return {**_BASE, **overrides}


# ---------------------------------------------------------------------------
# Parameter sweeps
# ---------------------------------------------------------------------------
SWEEPS = [
    dict(
        name="rotation", xlabel="rotation (deg)",
        values=[0, 5, 10, 20, 30, 45, 60, 90],
        build=lambda v: _cfg({"rotation_deg": v}),
    ),
    dict(
        name="scale", xlabel="scale deviation ±d  (min=1-d, max=1+d)",
        values=[0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
        build=lambda v: _cfg({"scale_min": 1 - v, "scale_max": 1 + v}),
    ),
    dict(
        name="shear", xlabel="shear (deg)",
        values=[0, 3, 8, 15, 25, 35, 45],
        build=lambda v: _cfg({"shear_deg": v}),
    ),
    dict(
        name="translation", xlabel="translation (px)",
        values=[0, 3, 7, 12, 18, 25, 35, 50],
        build=lambda v: _cfg({"translation_px": v}),
    ),
    dict(
        name="noise", xlabel="noise amount (max, prob=1)",
        values=[0.0, 0.003, 0.007, 0.015, 0.03, 0.06, 0.10, 0.15],
        build=lambda v: _cfg({
            "noise_prob": 1.0 if v > 0 else 0.0,
            "noise_amount_min": v / 2,
            "noise_amount_max": v,
        }),
    ),
    dict(
        name="blur", xlabel="blur radius (max, prob=1)",
        values=[0.0, 0.3, 0.7, 1.2, 2.0, 3.0, 4.5, 6.0],
        build=lambda v: _cfg({
            "blur_prob": 1.0 if v > 0 else 0.0,
            "blur_radius_min": max(0.1, v / 2),
            "blur_radius_max": max(0.1, v),
        }),
    ),
    dict(
        name="jitter", xlabel="jitter range ±r  (min=1-r, max=1+r, prob=1)",
        values=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        build=lambda v: _cfg({
            "jitter_prob": 1.0 if v > 0 else 0.0,
            "jitter_min": 1 - v,
            "jitter_max": 1 + v,
        }),
    ),
]


# ---------------------------------------------------------------------------
# Eval helpers
# ---------------------------------------------------------------------------
def _eval_config(cfg_override, ranker, paths, eval_indices, topk):
    _cfg_mod._cache["transform"] = cfg_override
    acc1, acck = [], []
    for idx in eval_indices:
        img_arr = read_image(Path(paths[idx]))
        if img_arr is None:
            continue
        aug = transform(img_arr)
        indices, _, _ = ranker.rank(aug)
        acc1.append(1 if indices[0] == idx else 0)
        acck.append(1 if idx in indices else 0)
    return float(np.mean(acc1)), float(np.mean(acck))


def _find_threshold(values, scores, target):
    """Linear interpolation of the last crossing point from above."""
    for i in range(len(scores) - 1):
        if scores[i] >= target > scores[i + 1]:
            t = (target - scores[i]) / (scores[i + 1] - scores[i])
            return values[i] + t * (values[i + 1] - values[i])
    if all(s >= target for s in scores):
        return values[-1]   # never drops below in tested range
    return None             # drops below immediately


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep each aug parameter and find Top-10=0.9 threshold")
    parser.add_argument("--features",  type=Path, default=Path("data/features.npz"))
    parser.add_argument("--bow",       type=Path, default=Path("data/bow.pkl"))
    parser.add_argument("--eval-size", type=int,  default=100)
    parser.add_argument("--topk",      type=int,  default=10)
    parser.add_argument("--target",    type=float, default=0.9,
                        help="Top-K accuracy threshold to find (default 0.9)")
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--out-dir",   type=Path, default=Path("tmp/sweep"))
    parser.add_argument("--only",      nargs="+", default=None,
                        help="Run only these parameter names")
    args = parser.parse_args()

    if not args.features.exists():
        print(f"[ERROR] features not found: {args.features}")
        return 1

    data   = np.load(args.features, allow_pickle=False)
    matrix = data["matrix"].astype(np.float32)
    paths  = data["paths"]
    n      = matrix.shape[0]

    rng          = np.random.default_rng(args.seed)
    eval_size    = min(args.eval_size, n)
    eval_indices = rng.choice(n, size=eval_size, replace=False)

    get_config()  # warm cache before patching

    bow_extractor = None
    if args.bow.exists():
        n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 100)
        bow_extractor = BoWExtractor.load(args.bow, n_clusters=n_clusters)

    ranker = Ranker(matrix, paths, n_neighbors=args.topk, bow_extractor=bow_extractor)

    sweeps = [s for s in SWEEPS if args.only is None or s["name"] in args.only]
    original_transform = get_config().get("transform", {}).copy()

    results = {}  # name -> (values, top1_scores, topk_scores)

    total_runs = sum(len(s["values"]) for s in sweeps)
    print(f"\nSweeping {len(sweeps)} parameters | {total_runs} eval runs | "
          f"eval_size={eval_size} | target Top-{args.topk}={args.target}\n")

    try:
        for sweep in sweeps:
            name   = sweep["name"]
            values = sweep["values"]
            top1_scores, topk_scores = [], []

            print(f"  [{name}]")
            for v in tqdm(values, desc=f"    {name}", unit="val", leave=False):
                cfg = sweep["build"](v)
                t1, tk = _eval_config(cfg, ranker, paths, eval_indices, args.topk)
                top1_scores.append(t1)
                topk_scores.append(tk)
                tqdm.write(f"    {name}={v:<8.3g}  top1={t1:.3f}  top{args.topk}={tk:.3f}")

            results[name] = (values, top1_scores, topk_scores)

    finally:
        _cfg_mod._cache["transform"] = original_transform

    # ── thresholds ────────────────────────────────────────────────────────────
    print(f"\n{'='*52}")
    print(f"  Threshold (max value where Top-{args.topk} >= {args.target})")
    print(f"{'─'*52}")
    thresholds = {}
    for name, (values, _, topk_scores) in results.items():
        thresh = _find_threshold(values, topk_scores, args.target)
        thresholds[name] = thresh
        if thresh is None:
            print(f"  {name:<14}  drops below {args.target} immediately")
        elif thresh == values[-1]:
            print(f"  {name:<14}  > {values[-1]}  (stays above {args.target} in tested range)")
        else:
            print(f"  {name:<14}  {thresh:.4g}")
    print(f"{'='*52}\n")

    # ── plot ─────────────────────────────────────────────────────────────────
    ncols = 4
    nrows = int(np.ceil(len(sweeps) / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 4.5, nrows * 3.5),
                             squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for ax, sweep in zip(axes.flat, sweeps):
        ax.set_visible(True)
        name = sweep["name"]
        values, top1_scores, topk_scores = results[name]

        ax.plot(values, top1_scores,  "o-", color="#4c9be8", label="Top-1",        linewidth=1.8)
        ax.plot(values, topk_scores,  "s-", color="#e8994c", label=f"Top-{args.topk}", linewidth=1.8)
        ax.axhline(args.target, color="#555", linewidth=1.2,
                   linestyle="--", label=f"target {args.target}")

        thresh = thresholds[name]
        if thresh is not None and thresh != values[-1]:
            ax.axvline(thresh, color="#e05c5c", linewidth=1.2, linestyle=":")
            ax.text(thresh, args.target + 0.02, f" {thresh:.3g}",
                    color="#e05c5c", fontsize=8, va="bottom")

        ax.set_title(name, fontweight="bold")
        ax.set_xlabel(sweep["xlabel"], fontsize=7.5)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.suptitle(f"Augmentation sweep — Top-{args.topk} = {args.target} threshold",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "sweep.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
