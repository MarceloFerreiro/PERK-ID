from sklearn.neighbors import NearestNeighbors
import argparse
from pathlib import Path
import numpy as np
import csv

from src.features import features, read_image

class Ranker:
    def __init__(self, features_matrix: np.ndarray, paths: list[str], n_neighbors: int = 5):
        self.features_matrix = features_matrix
        self.paths = paths
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.features_matrix)

    def rank(self, image: np.array, transformed: bool = False) -> list[tuple[str, float]]:
        q = features(image, transformed=transformed).astype(np.float32)
        distances, indices = self.nn.kneighbors(q.reshape(1, -1), return_distance=True)
        paths = [self.paths[i] for i in indices[0]]
        return indices[0], distances[0], paths

def main() -> int:
    parser = argparse.ArgumentParser(description="Ranking")
    parser.add_argument("image_path")
    parser.add_argument("--features", type=Path, default=Path("data/features.npz"), help="Ruta al índice .npz")
    parser.add_argument("--output_dir", type=Path, default=Path("results"), help="directorio donde se guardan los resultados.")
    parser.add_argument("--images-dir", type=Path, default=None, help="Directorio con imagenes")
    parser.add_argument("--topk", type=int, default=10, help="K para Top-K accuracy")
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

    ranker = Ranker(matrix, paths, n_neighbors=args.topk)
    indices, distances, paths = ranker.rank(read_image(args.image_path))
    
    #-
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_name = Path(args.image_path).stem
    output_file = output_dir / f"{image_name}.csv"

    with open(output_file, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Path", "Distance"])
        writer.writerows(zip(paths, distances))

    print(f"Se guardaron los resultados en {output_file}")
    return 0 

if __name__ == '__main__':
    main()
