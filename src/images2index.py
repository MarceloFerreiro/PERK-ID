import argparse
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time

from tqdm import tqdm


from src.features import GLCM, HIST, SIFT
from skimage import io
# --- Placeholder feature functions ---

_WORKERS = 0
_VERBOSE = False


def _fmt_seconds(seconds: float) -> str:
    return f"{seconds:.2f}".replace(".", ",")


def _log_timed(elapsed: float, feature: str, extra: str) -> None:
    if not _VERBOSE:
        return
    print(f"{_fmt_seconds(elapsed)}s [*] w{_WORKERS} [*] {feature}: {extra}")


def _init_worker(verbose: bool, workers: int) -> None:
    global _VERBOSE, _WORKERS
    _VERBOSE = verbose
    _WORKERS = workers



def compute_features(path: Path) -> np.ndarray:
    image = io.imread(str(path))
    features = [
        SIFT(image),
        GLCM(image),
        HIST(image),
    ]
    return np.concatenate([f.ravel() for f in features])


def _save_partial(out_path: Path, paths: list[Path], results: dict[Path, np.ndarray]) -> None:
    completed_paths = [p for p in paths if p in results]
    if not completed_paths:
        return
    partial_matrix = np.stack([results[p] for p in completed_paths])
    np.savez(out_path, matrix=partial_matrix, paths=[str(p) for p in completed_paths])


def build_feature_matrix(
    image_dir: Path,
    workers: int = 8,
    save_every: int = 0,
    out_path: Path | None = None,
) -> tuple[np.ndarray, list[Path]]:
    global _WORKERS
    _WORKERS = workers
    paths = sorted(image_dir.glob("*.jp*g"))  # matches .jpg and .jpeg

    results = {}
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(_VERBOSE, workers),
    ) as pool:
        futures = {pool.submit(compute_features, p): p for p in paths}
        with tqdm(
            total=len(paths),
            unit="imagenes",
            postfix="Extraccion",
            #disable=not _VERBOSE,
        ) as pbar:
            for future in as_completed(futures):
                path = futures[future]
                results[path] = future.result()
                pbar.update(1)
                if save_every > 0 and out_path and len(results) % save_every == 0:
                    _save_partial(out_path, paths, results)

    ordered = [results[p] for p in paths]
    return np.stack(ordered), paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("data/images"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=Path("data/features.npz"))
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    global _VERBOSE
    _VERBOSE = args.verbose
    start = time()
    matrix, paths = build_feature_matrix(
        args.dir,
        args.workers,
        save_every=args.save_every,
        out_path=args.out,
    )
    np.savez(args.out, matrix=matrix, paths=[str(p) for p in paths])
    _log_timed(time() - start, "MATRIX", f"{matrix.shape} ({len(paths)} images)")
    _log_timed(0.0, "SAVED", str(args.out))


if __name__ == "__main__":
    main()
