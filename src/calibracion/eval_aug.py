"""
Evaluate multiple augmentation profiles against the same 100 random samples.

Profiles are defined in a TOML file (default: profiles.toml).
Each top-level table in that file becomes one profile.

Usage:
    python -m src.eval_aug
    python -m src.eval_aug --profiles profiles.toml --eval-size 100 --topk 10 --seed 42
    python -m src.eval_aug --profiles profiles.toml --only none easy current
"""
import argparse
from pathlib import Path
from time import time

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import src.utils.config as _cfg_mod
from src.Ranker import Ranker
from src.features import features, read_image, extract_sift_descriptors
from src.features.transform import transform
from src.features.BoW import BoWExtractor
from src.utils.config import get_config
from src.utils.cummulative_rank import cummulative_rank
from src.utils.grid_rank import grid_rank


def _load_profiles(path: Path) -> dict[str, dict]:
    with open(path, "rb") as f:
        return tomllib.load(f)


def _run_profile(
    name: str,
    profile_cfg: dict,
    ranker: Ranker,
    paths: np.ndarray,
    eval_indices: np.ndarray,
    topk: int,
    n_grid: int,
    seed: int,
) -> tuple[dict, list, list, list, list]:
    # patch the cached transform config in-place
    _cfg_mod._cache["transform"] = profile_cfg

    acc1, acck, dist_list, rank_list, secs = [], [], [], [], []
    pills_images, rank_images, dists_grid, true_ranks = [], [], [], []

    for idx in tqdm(eval_indices, desc=f"  {name}", unit="img", leave=False):
        img_arr = read_image(Path(paths[idx]))
        if img_arr is None:
            continue

        aug = transform(img_arr)
        t0 = time()
        indices, distances, ranked_paths = ranker.rank(aug)
        secs.append(time() - t0)

        true_pos = int(np.where(indices == idx)[0][0]) if idx in indices else -1
        acc1.append(1 if indices[0] == idx else 0)
        acck.append(1 if idx in indices else 0)
        dist_list.append(distances[0])
        rank_list.append(true_pos)

        if len(pills_images) < n_grid:
            pills_images.append(aug)
            rank_images.append([read_image(p) for p in ranked_paths[:topk]])
            dists_grid.append(distances[:topk])
            true_ranks.append(true_pos)

    missed = sum(1 for r in rank_list if r == -1)
    valid_ranks = [r for r in rank_list if r != -1]
    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else float("nan")

    metrics = {
        "profile":    name,
        "n":          len(acc1),
        "top1":       float(np.mean(acc1)),
        f"top{topk}": float(np.mean(acck)),
        "mean_rank":  mean_rank,
        "missed":     missed,
        "mean_dist":  float(np.mean(dist_list)),
        "mean_ms":    float(np.mean(secs)) * 1000,
    }
    return metrics, pills_images, rank_images, dists_grid, true_ranks


def _group_profiles(rows: list[dict]) -> tuple[dict | None, list]:
    """
    Split rows into:
      - baseline_row : the 'baseline' row (or None)
      - slots        : list of (label, low_row, high_row)
                       for paired profiles (sharing a prefix before the last _)
                       or (label, row, None) for standalone ones
    """
    from collections import defaultdict, OrderedDict

    baseline_row = None
    ordered: dict[str, list] = OrderedDict()

    for row in rows:
        name = row["profile"]
        if name == "baseline":
            baseline_row = row
            continue
        # group key = everything before the last underscore (if present)
        key = name.rsplit("_", 1)[0] if "_" in name else name
        ordered.setdefault(key, []).append(row)

    slots = []
    for key, group_rows in ordered.items():
        if len(group_rows) == 2:
            slots.append((key, group_rows[0], group_rows[1]))
        else:
            for r in group_rows:
                slots.append((r["profile"], r, None))

    return baseline_row, slots


def _plot_summary(rows: list[dict], topk: int, save_path: Path) -> None:
    baseline_row, slots = _group_profiles(rows)

    # reference line: baseline if present, else first row
    ref = baseline_row or rows[0]
    ref_top1 = ref["top1"]
    ref_topk = ref[f"top{topk}"]
    ref_label = f"{ref['profile']} = {{:.3f}}"

    bar_w   = 0.35
    n       = len(slots)
    centers = np.arange(n, dtype=float)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(10, n * 1.1), 5))

    for ax, key, ref_val in (
        (ax1, "top1",       ref_top1),
        (ax2, f"top{topk}", ref_topk),
    ):
        for i, (label, low_row, high_row) in enumerate(slots):
            if high_row is not None:
                ax.bar(centers[i] - bar_w / 2, low_row[key],  width=bar_w,
                       color="#4c9be8", zorder=2)
                ax.bar(centers[i] + bar_w / 2, high_row[key], width=bar_w,
                       color="#e05c5c", zorder=2)
                for x_off, row in ((-bar_w / 2, low_row), (bar_w / 2, high_row)):
                    ax.text(centers[i] + x_off, row[key] + 0.01,
                            f"{row[key]:.2f}", ha="center", va="bottom", fontsize=7.5)
            else:
                ax.bar(centers[i], low_row[key], width=bar_w * 1.4,
                       color="#888888", zorder=2)
                ax.text(centers[i], low_row[key] + 0.01,
                        f"{low_row[key]:.2f}", ha="center", va="bottom", fontsize=7.5)

        ax.axhline(ref_val, color="#333333", linewidth=1.4, linestyle="--",
                   label=ref_label.format(ref_val), zorder=3)
        ax.set_xticks(centers)
        ax.set_xticklabels([s[0] for s in slots], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel("Accuracy")
        ax.set_title("Top-1" if key == "top1" else f"Top-{topk}")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3, zorder=0)

    # shared legend for colours
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color="#4c9be8", label="low"),
                        Patch(color="#e05c5c", label="high")],
               loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Augmentation sensitivity", fontsize=12)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _print_table(rows: list[dict], topk: int) -> None:
    header = (
        f"{'Profile':<10}  {'N':>4}  {'Top-1':>6}  {f'Top-{topk}':>7}  "
        f"{'MnRank':>7}  {'Missed':>6}  {'MnDist':>7}  {'ms/q':>6}"
    )
    sep = "=" * len(header)
    print(f"\n{sep}\n{header}\n{'-' * len(header)}")
    for r in rows:
        print(
            f"{r['profile']:<10}  {r['n']:>4}  {r['top1']:>6.3f}  "
            f"{r[f'top{topk}']:>7.3f}  {r['mean_rank']:>7.2f}  "
            f"{r['missed']:>6}  {r['mean_dist']:>7.4f}  {r['mean_ms']:>6.1f}"
        )
    print(sep)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare augmentation profiles")
    parser.add_argument("--profiles",   type=Path, default=Path("profiles.toml"))
    parser.add_argument("--features",   type=Path, default=Path("data/features.npz"))
    parser.add_argument("--bow",        type=Path, default=Path("data/bow.pkl"))
    parser.add_argument("--eval-size",  type=int,  default=100)
    parser.add_argument("--topk",       type=int,  default=10)
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("-H",           type=int,  default=5,  dest="grid_h",
                        help="Grid rows: number of query examples shown")
    parser.add_argument("-W",           type=int,  default=10, dest="grid_w",
                        help="Grid columns: number of ranked results shown per row (max=topk)")
    parser.add_argument("--out-dir",    type=Path, default=Path("tmp/aug_eval"))
    parser.add_argument("--only",       nargs="+", default=None,
                        help="Run only these profile names (default: all)")
    args = parser.parse_args()

    if not args.features.exists():
        print(f"[ERROR] features not found: {args.features}")
        return 1
    if not args.profiles.exists():
        print(f"[ERROR] profiles file not found: {args.profiles}")
        return 1

    data   = np.load(args.features, allow_pickle=False)
    matrix = data["matrix"].astype(np.float32)
    paths  = data["paths"]
    n      = matrix.shape[0]

    rng          = np.random.default_rng(args.seed)
    eval_size    = min(args.eval_size, n)
    eval_indices = rng.choice(n, size=eval_size, replace=False)

    # ensure config cache is loaded before we start patching it
    get_config()

    bow_extractor = None
    if args.bow.exists():
        n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 100)
        bow_extractor = BoWExtractor.load(args.bow, n_clusters=n_clusters)

    ranker = Ranker(matrix, paths, n_neighbors=args.topk, bow_extractor=bow_extractor)

    all_profiles = _load_profiles(args.profiles)
    names = args.only if args.only else list(all_profiles.keys())
    unknown = [n for n in names if n not in all_profiles]
    if unknown:
        print(f"[ERROR] unknown profiles: {unknown}  (available: {list(all_profiles.keys())})")
        return 1

    grid_w = min(args.grid_w, args.topk)

    print(f"\nEvaluating {eval_size} samples | top-k={args.topk} | seed={args.seed}")
    print(f"Grid: {args.grid_h}x{grid_w}  |  Profiles: {names}\n")

    results = []
    original_transform = get_config().get("transform", {}).copy()

    try:
        for name in names:
            metrics, pills, ranks, dists, true_ranks = _run_profile(
                name, all_profiles[name],
                ranker, paths, eval_indices,
                args.topk, args.grid_h, args.seed,
            )
            results.append(metrics)
            print(
                f"  {name:<10}  top1={metrics['top1']:.3f}  "
                f"top{args.topk}={metrics[f'top{args.topk}']:.3f}  "
                f"mean_rank={metrics['mean_rank']:.2f}"
            )
            # trim ranked columns to grid_w
            ranks_trimmed = [r[:grid_w] for r in ranks]
            dists_trimmed = [d[:grid_w] for d in dists]
            save_path = args.out_dir / f"{name}_grid.png"
            grid_rank(pills, ranks_trimmed, dists_trimmed,
                      true_rank_pos=true_ranks,
                      save_path=save_path)
            print(f"  -> grid saved to {save_path}")
    finally:
        # restore original transform config
        _cfg_mod._cache["transform"] = original_transform

    _print_table(results, args.topk)

    summary_path = args.out_dir / "summary.png"
    _plot_summary(results, args.topk, summary_path)
    print(f"\nSummary plot saved to {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
