import io
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.Ranker import Ranker
from src.features.BoW import BoWExtractor
from src.features.read_image import read_image
from src.utils.config import get_config

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

FEATURES_PATH = Path("data/features.npz")
BOW_PATH = Path("data/bow.pkl")
CSV_PATH = Path("data/pastillas_energycontrol.csv")
IMAGES_DIR = Path("data/imagenes_alt")
MAX_TOPK = 50

state: dict = {}


def _build_metadata_index(df: pd.DataFrame) -> dict:
    """Build filename → row lookup covering both front and rear image columns."""
    index = {}
    for _, row in df.iterrows():
        for col in ("ruta_imagen", "ruta_imagen_rear"):
            val = row.get(col)
            if pd.notna(val) and val:
                filename = Path(str(val)).name
                if filename and filename not in index:
                    index[filename] = row
    return index


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Loading feature index…")
    if not FEATURES_PATH.exists():
        raise RuntimeError(f"Feature index not found: {FEATURES_PATH}. Run src/images2index.py first.")

    data = np.load(FEATURES_PATH, allow_pickle=False)
    matrix = data["matrix"].astype(np.float32)
    paths = list(data["paths"])

    bow_extractor = None
    if BOW_PATH.exists():
        n_clusters = get_config().get("features", {}).get("bow", {}).get("n_clusters", 100)
        bow_extractor = BoWExtractor.load(BOW_PATH, n_clusters=n_clusters)

    state["ranker"] = Ranker(matrix, paths, n_neighbors=MAX_TOPK, bow_extractor=bow_extractor)

    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    state["metadata"] = _build_metadata_index(df)

    log.info(f"Ready — {len(paths)} images indexed")
    yield
    state.clear()


app = FastAPI(title="PerkID API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")

_WEB_DIR = Path("pillsearch/build/web")
if _WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_WEB_DIR), html=True), name="web")


# ── Response models ────────────────────────────────────────────────────────────

class Substance(BaseModel):
    nombre: str
    valor: float | None = None
    unidad: str | None = None


class PillResult(BaseModel):
    rank: int
    distance: float
    image_url: str
    fecha: str | None = None
    logo: str | None = None
    color: str | None = None
    procedencia: str | None = None
    peso_mg: float | None = None
    diametro_mm: float | None = None
    divisible: bool | None = None
    substances: list[Substance] = []


class QueryResponse(BaseModel):
    count: int
    results: list[PillResult]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    ranker: Ranker = state.get("ranker")
    return {
        "status": "ok",
        "index_size": len(ranker.paths) if ranker else 0,
    }


@app.post("/query", response_model=QueryResponse)
async def query_image(
    image: UploadFile = File(...),
    topk: int = Query(default=10, ge=1, le=MAX_TOPK),
):
    contents = await image.read()

    suffix = Path(image.filename).suffix if image.filename else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        img = read_image(Path(tmp_path))
    finally:
        os.unlink(tmp_path)

    if img is None:
        raise HTTPException(status_code=400, detail="Could not decode image")

    ranker: Ranker = state["ranker"]
    indices, distances, img_paths = ranker.rank(img)

    # Slice to requested topk (ranker is initialized with MAX_TOPK)
    indices = indices[:topk]
    distances = distances[:topk]
    img_paths = img_paths[:topk]

    meta = state["metadata"]
    results: list[PillResult] = []

    for rank, (dist, path) in enumerate(zip(distances, img_paths), 1):
        filename = Path(path).name
        row = meta.get(filename)

        def _get(col, default=None):
            if row is None:
                return default
            val = row.get(col, default)
            return None if pd.isna(val) else val

        substances = []
        for i in range(1, 4):
            nombre = _get(f"sustancia_{i}")
            if nombre:
                substances.append(Substance(
                    nombre=str(nombre),
                    valor=_get(f"valor_{i}_mg"),
                    unidad=_get(f"unidad_{i}"),
                ))

        divisible_raw = _get("divisible")
        divisible = bool(divisible_raw) if divisible_raw is not None else None

        fecha = _get("fecha")
        results.append(PillResult(
            rank=rank,
            distance=float(dist),
            image_url=f"/images/{filename}",
            fecha=str(fecha) if fecha is not None else None,
            logo=_get("logo"),
            color=_get("color"),
            procedencia=_get("procedencia"),
            peso_mg=_get("peso_mg"),
            diametro_mm=_get("diametro_mm"),
            divisible=divisible,
            substances=substances,
        ))

    return QueryResponse(count=len(results), results=results)
