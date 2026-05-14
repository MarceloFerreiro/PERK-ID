import argparse
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm


from src.Ranker import Ranker
from src.features import features, read_image
from src.utils.cummulative_rank import cummulative_rank

def main() -> int:
    parser = argparse.ArgumentParser(description="Evalua Top-1/Top-K sobre un indice de features")
    parser.add_argument("--features", type=Path, default=Path("data/features.npz"),
                        help="Ruta al índice .npz")
    parser.add_argument("--images-dir", type=Path, default=None,
                        help="Directorio con imagenes")
    parser.add_argument("--eval-size", type=int, default=200,
                        help="Numero de imagenes para evaluar (muestreo aleatorio)")
    parser.add_argument("--topk", type=int, default=10,
                        help="K para Top-K accuracy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-transform", action="store_true", help="Evaluar sin data augmentation")
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

    n = matrix.shape[0]

    eval_size = min(max(args.eval_size, 1), n)
    rng = np.random.default_rng(args.seed)
    eval_indices = rng.choice(n, size=eval_size, replace=False)

    ranker = Ranker(matrix, paths, n_neighbors=args.topk)

    acc1 = []
    acck = []
    secs = []
    dist = []
    rank = []

    skipped = 0

    for idx in tqdm(eval_indices, total=len(eval_indices), unit="img", desc="Evaluando"):
        inicio  = time()
        img_path = Path(paths[idx])
        if img_path is None:
            skipped += 1
            continue

        indices, distances, _ = ranker.rank(read_image(img_path), transformed=not args.no_transform)

        acc1.append(1 if indices[0] == idx else 0)
        acck.append(1 if idx in indices else 0)
        dist.append(distances[0])
        secs.append(time() - inicio)
        rank.append(np.where(indices == idx)[0][0] if idx in indices else -1)

    if not acc1:
        print("[ERROR] No se pudo evaluar ninguna imagen (rutas invalidas)")
        return 1

    print(f"Evaluadas: {len(acc1)}  Omitidas: {skipped}")
    print(f"Top-1 Accuracy: {np.mean(acc1):.4f}")
    print(f"Top-{args.topk} Accuracy: {np.mean(acck):.4f}")
    print(f"Distancia media a la pastilla más cercana: {np.mean(dist):.4f}")
    print(f"Duración media inferencia: {np.mean(secs):.4f}s")
    print(f"Ranking medio: {np.mean(rank):.4f}")
    cummulative_rank(rank)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
