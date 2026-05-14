from sklearn.neighbors import NearestNeighbors
import numpy as np

from src.features import features

class Ranker:
    def __init__(self, features_matrix: np.ndarray, paths: list[str], n_neighbors: int = 5):
        self.features_matrix = features_matrix
        self.paths = paths
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(self.features_matrix)

    def rank(self, image: np.array, transformed: bool = True) -> list[tuple[str, float]]:
        q = features(image, transformed=transformed).astype(np.float32)
        distances, indices = self.nn.kneighbors(q.reshape(1, -1), return_distance=True)
        paths = [self.paths[i] for i in indices[0]]
        return indices[0], distances[0], paths
