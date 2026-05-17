import argparse
from pathlib import Path
from time import time

import numpy as np
from tqdm import tqdm

from src.Ranker import Ranker
from src.features import features, read_image, transform
from src.features.BoW import BoWExtractor
from src.utils import cummulative_rank, grid_rank
from src.utils.config import get_config


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
    parser.add_argument("--bow", type=Path, default=Path("data/bow.pkl"), help="Ruta al modelo BoW")
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

    n = matrix.shape[0]

    eval_size = min(max(args.eval_size, 1), n)
    rng = np.random.default_rng(args.seed)
    eval_indices = rng.choice(n, size=eval_size, replace=False)


    bow_extractor=None
    if args.bow.exists():
        n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 100)
        bow_extractor = BoWExtractor.load(args.bow, n_clusters=n_clusters)

    ranker = Ranker(matrix, paths, n_neighbors=args.topk, bow_extractor=bow_extractor)

    acc1 = []
    acck = []
    secs = []
    dist = []
    rank = []

    num_pills_grid = 5
    pills_images = []
    num_rank_grid = 10
    rank_images = []
    dists = []

    skipped = 0

    for idx in tqdm(eval_indices, total=len(eval_indices), unit="img", desc="Evaluando"):
        inicio  = time()
        img_path = Path(paths[idx])
        if img_path is None:
            skipped += 1
            continue

        img_tranformed = transform(read_image(img_path))

        indices, distances, ranked_paths = ranker.rank(img_tranformed)

        if num_pills_grid > len(pills_images):
            pills_images.append(img_tranformed)
            rank_images.append([read_image(p) for p in ranked_paths[:num_rank_grid]])
            dists.append(distances[:num_rank_grid])
        
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
    print(f"La pastilla quedo fuera del top-{args.topk} en: {sum([1 for r in rank if r == -1])}/{eval_size} casos")
    cummulative_rank(rank)
    grid_rank(pills_images, rank_images, dists)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
