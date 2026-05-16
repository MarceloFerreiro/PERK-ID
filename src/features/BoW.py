from __future__ import annotations

import numpy as np
import joblib
from time import time
from pathlib import Path
from sklearn.cluster import KMeans


class BoWExtractor:
    
    def __init__(self, n_clusters: int = 100, **kwargs):
        self.n_clusters = n_clusters
        self.kmeans = None
    
    def fit(self, all_descriptors: np.ndarray) -> None:
        if all_descriptors.shape[0] == 0:
            raise ValueError("Vaites! no se le pasaron descriptores a BoW")
        
        print(f"[*] Ajustando KMeans con {self.n_clusters} clusters sobre {all_descriptors.shape[0]} descriptores... (el numero de clusters lo cambias en el toml)")
        inicio = time()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10, verbose=0)
        self.kmeans.fit(all_descriptors)
        print(f"[*] Ajustado Kmeans, tardó {time() - inicio:.2f}s.")
    
    def extract_histogram(self, descriptors: np.ndarray) -> np.ndarray:
        if descriptors.shape[0] == 0:
            return np.zeros(self.n_clusters, dtype=np.float32)
        
        cluster_assignments = self.kmeans.predict(descriptors)
        histogram, _ = np.histogram(cluster_assignments, bins=self.n_clusters, range=(0, self.n_clusters))
        
        #return (histogram / histogram.sum()).astype(np.float32)
        return histogram.astype(np.float32)
    
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.kmeans, str(path))
        print(f"[*] Se guardo el BoW a {path}")
    
    @staticmethod
    def load(path: Path, n_clusters: int = 100) -> BoWExtractor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"[:(] No se encontro el BoW en {path}")
        
        extractor = BoWExtractor(n_clusters=n_clusters)
        extractor.kmeans = joblib.load(str(path))
        print(f"[*] Se cargo BoW desde {path}")
        return extractor
