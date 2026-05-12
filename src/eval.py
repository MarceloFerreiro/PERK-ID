import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.images2index import compute_features


def _resolve_image_path(raw_path: str, images_dir: Path | None) -> Path | None:
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate
    if images_dir is None:
        return None
    alt = images_dir / candidate.name
    return alt if alt.exists() else None


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / (norms + 1e-10)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evalua Top-1/Top-K sobre un indice de features")
    parser.add_argument("--features", type=Path, default=Path("data/features.npz"),
                        help="Ruta a features.npz (matrix + paths)")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Directorio con imagenes si las rutas del npz no existen")
    parser.add_argument("--eval-size", type=int, default=200,
                        help="Numero de imagenes para evaluar (muestreo aleatorio)")
    parser.add_argument("--topk", type=int, default=5,
                        help="K para Top-K accuracy")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.features.exists():
        print(f"[ERROR] No se encuentra {args.features}")
        return 1

    data = np.load(args.features, allow_pickle=False)
    if "matrix" not in data or "paths" not in data:
        print("[ERROR] El NPZ debe contener 'matrix' y 'paths'")
        return 1

    matrix = data["matrix"].astype(np.float32)
    paths = data["paths"]

    if matrix.ndim != 2:
        print("[ERROR] 'matrix' debe ser 2D (N x D)")
        return 1

    n = matrix.shape[0]
    if n == 0:
        print("[ERROR] 'matrix' esta vacia")
        return 1

    eval_size = min(max(args.eval_size, 1), n)
    rng = np.random.default_rng(args.seed)
    eval_indices = rng.choice(n, size=eval_size, replace=False)

    db_norm = _normalize_rows(matrix)

    acc1 = []
    acck = []
    skipped = 0

    for idx in tqdm(eval_indices, total=len(eval_indices), unit="img", desc="Evaluando"):
        img_path = _resolve_image_path(str(paths[idx]), args.images_dir)
        if img_path is None:
            skipped += 1
            continue

        q = compute_features(img_path).astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-10)
        sims = db_norm @ q_norm

        ranked = np.argsort(-sims)
        acc1.append(1 if ranked[0] == idx else 0)
        acck.append(1 if idx in ranked[: args.topk] else 0)

    if not acc1:
        print("[ERROR] No se pudo evaluar ninguna imagen (rutas invalidas)")
        return 1

    print(f"Evaluadas: {len(acc1)}  Omitidas: {skipped}")
    print(f"Top-1 Accuracy: {np.mean(acc1):.4f}")
    print(f"Top-{args.topk} Accuracy: {np.mean(acck):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())