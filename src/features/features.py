from __future__ import annotations

from pathlib import Path

import numpy as np

from src.features.GLCM import GLCM
from src.features.HIST import HIST
from src.features.SIFT import SIFT
from src.features.read_image import read_image
from src.features.transform import transform

_FEATURE_FUNCS = (GLCM, HIST)


def features(path: Path, transformed: bool = False) -> np.ndarray | None:
    image = read_image(path)
    if image is None:
        return None
    if transformed:
        image = transform(image)

    #hist_feat = HIST(image)
    #glcm_feat = GLCM(image)
    #sift_feat = SIFT(image)
    
    feats = [f for f in _FEATURE_FUNCS]
    if not feats:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate([f(image).ravel() for f in feats]).astype(np.float32)
