from __future__ import annotations

import sys

import numpy as np

from src.features.GLCM import GLCM
from src.features.HIST import HIST
from src.features.HU_MOMENTS import HU_MOMENTS
from src.features.LBP import LBP
from src.features.ORB import ORB
from src.features.SIFT import SIFT
from src.features.read_image import read_image
from src.features.transform import transform
from src.utils.config import get_config

_FEATURE_REGISTRY = {
    "GLCM": GLCM,
    "HIST": HIST,
    "HU_MOMENTS": HU_MOMENTS,
    "LBP": LBP,
    "ORB": ORB,
    "SIFT": SIFT,
}

def _get_enabled_features():
    config = get_config()
    features_config = config.get("features", {})
    enabled_features = features_config.get("enabled", [])
    
    result = []
    for feature_name in enabled_features:
        if feature_name in _FEATURE_REGISTRY:
            feature_func = _FEATURE_REGISTRY[feature_name]
            hyperparams = features_config.get(feature_name, {})
            hyperparams = {k: v for k, v in hyperparams.items()}
            result.append((feature_func, hyperparams))
    
    return result


def _get_enabled_features_excluding(exclude_feature: str):
    enabled = _get_enabled_features()
    return [(f, params) for f, params in enabled if f.__name__ != exclude_feature]


def extract_sift_descriptors(image) -> np.ndarray:
    config = get_config()
    sift_params = config.get("features", {}).get("SIFT", {})
    descriptors = SIFT(image, **sift_params)
    return descriptors


def features(image, include_sift=False) -> np.ndarray | None:
    if include_sift:
        enabled_features = _get_enabled_features()
    else:
        enabled_features = _get_enabled_features_excluding("SIFT")
    
    if not enabled_features: 
        return np.empty((0,), dtype=np.float32)
    
    feats = [f(image, **params) for f, params in enabled_features]
    return np.concatenate([f.ravel() for f in feats]).astype(np.float32)

