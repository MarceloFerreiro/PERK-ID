"""
pill_indexer/extract_features.py
=================================
Pipeline de extracción de features para dataset de pastillas EnergyControl.

Entrada esperada
----------------
dataset/
  images/
    pill_001.jpg
    pill_002.jpg
    ...
  metadata.csv   ← columnas mínimas: filename, color, logo, shape  (pueden estar vacías)

Salida
------
output/
  index.npz      ← vectores + ids, listo para copiar a assets/ Android
  metadata.json  ← metadatos enriquecidos con features descriptivos
  report.txt     ← estadísticas del proceso

Uso rápido
----------
  python extract_features.py --images dataset/images --meta dataset/metadata.csv --out output/

Dependencias
------------
  pip install opencv-python-headless numpy scipy scikit-learn Pillow pandas
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONSTANTES GLOBALES
# ──────────────────────────────────────────────
CANONICAL_SIZE   = 256          # px, imagen normalizada para features
HIST_BINS_H      = 18           # matiz:   0-179 → 18 bins de 10°
HIST_BINS_S      = 8            # saturación
HIST_BINS_V      = 8            # valor/brillo
LBP_RADIUS       = 2
LBP_POINTS       = 16
PHASH_SIZE       = 8            # pHash → 8×8 = 64 bits → 8 bytes
EDGE_BINS        = 8            # ángulos de gradiente
MIN_PILL_AREA    = 0.05         # % mínimo del frame que debe ocupar la pastilla


# ──────────────────────────────────────────────
# 1. PREPROCESADO E IMAGEN CANÓNICA
# ──────────────────────────────────────────────

def load_image(path: str) -> np.ndarray:
    """Lee imagen BGR. Lanza ValueError si no puede."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo leer: {path}")
    return img


def correct_illumination(bgr: np.ndarray) -> np.ndarray:
    """CLAHE en canal L de LAB para normalizar iluminación."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def segment_pill(bgr: np.ndarray):
    """
    Segmenta la pastilla del fondo.

    Asunciones del dataset EnergyControl:
      - Pastilla centrada y única en la foto
      - Fondo mayormente uniforme pero de color variable

    Estrategia:
      1. Canny → contorno más central → bounding rect ajustado a la pastilla real
      2. GrabCut inicializado con ese rect ajustado (no fijo)
      3. Flood-fill desde las 4 esquinas para limpiar fondo uniforme residual
      4. Fallback: Canny relleno directo si GrabCut produce resultado pobre

    Devuelve (mask_bool [H,W], bgr_recortado_cuadrado [CANONICAL_SIZE, CANONICAL_SIZE, 3])
    """
    h, w = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_l = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # ── Helpers ────────────────────────────────────────────

    def _best_central_component(mask_u8):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8)
        if n <= 1:
            return mask_u8
        cx_img, cy_img = w / 2, h / 2
        best_idx, best_score = 1, -1
        for i in range(1, n):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 200:
                continue
            dist = np.hypot(centroids[i][0] - cx_img, centroids[i][1] - cy_img)
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_idx = i
        return ((labels == best_idx) * 255).astype(np.uint8)

    def _canny_rect(sigma_low=0.33):
        """Detecta el contorno más central con Canny y devuelve
        su bounding rect con padding, listo para GrabCut."""
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        median  = np.median(blurred)
        lo = max(0,   int(median * (1.0 - sigma_low)))
        hi = min(255, int(median * (1.0 + sigma_low)))
        edges = cv2.Canny(blurred, lo, hi)
        edges = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None

        # Quedarse con contorno de mayor área entre los centrales
        cx_img, cy_img = w / 2, h / 2
        best_cnt, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist  = np.hypot(cx - cx_img, cy - cy_img)
            score = area / (1.0 + dist)
            if score > best_score:
                best_score = score
                best_cnt   = cnt

        if best_cnt is None:
            return None, None

        # Bounding rect con padding del 8%
        x, y, bw, bh = cv2.boundingRect(best_cnt)
        pad_x = max(4, int(bw * 0.08))
        pad_y = max(4, int(bh * 0.08))
        x1 = max(0, x - pad_x);      y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x); y2 = min(h, y + bh + pad_y)
        rect = (x1, y1, x2 - x1, y2 - y1)
        return rect, best_cnt

    def _flood_fill_bg(mask_u8):
        """
        Flood-fill desde las 4 esquinas de la imagen sobre la máscara
        para eliminar fondo uniforme que GrabCut haya dejado como FG.
        Devuelve máscara con esos píxeles marcados como fondo.
        """
        # Invertir: fondo=255, pastilla=0 → flood fill encuentra fondo conectado
        inv = cv2.bitwise_not(mask_u8.copy())
        flood = inv.copy()
        # Añadir borde de 1px para que floodfill pueda salir por los extremos
        flood = cv2.copyMakeBorder(flood, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        mask_ff = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, mask_ff, (0, 0), 128)
        flood = flood[1:-1, 1:-1]
        # Píxeles alcanzados por flood = fondo seguro → ponerlos a 0 en mask_u8
        bg_reached = (flood == 128).astype(np.uint8) * 255
        return cv2.bitwise_and(mask_u8, cv2.bitwise_not(bg_reached))

    # ── Paso 1: Canny para obtener rect ajustado ───────────
    rect, canny_cnt = _canny_rect()

    # Si Canny no encontró nada útil, usar rect central fijo (15% margen)
    if rect is None:
        margin_x = max(2, int(w * 0.15))
        margin_y = max(2, int(h * 0.15))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    # ── Paso 2: GrabCut con el rect ajustado ───────────────
    mask_gc   = np.zeros((h, w), np.uint8)
    bgr_model = np.zeros((1, 65), np.float64)
    fg_model  = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask_gc, rect, bgr_model, fg_model,
                    5, cv2.GC_INIT_WITH_RECT)
        mask_fg = np.where(
            (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
            255, 0).astype(np.uint8)
    except Exception:
        mask_fg = np.zeros((h, w), np.uint8)

    # Cerrar huecos internos (línea divisoria, logo, grabado)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel_l, iterations=2)

    # ── Paso 3: flood-fill desde esquinas para limpiar fondo residual
    mask_fg = _flood_fill_bg(mask_fg)

    # Suavizar bordes con erosión ligera + quedarse con componente central
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)
    mask_fg = _best_central_component(mask_fg)

    pill_area_ratio = mask_fg.sum() / 255 / (h * w)

    # ── Paso 4: fallback — relleno directo del contorno Canny
    if pill_area_ratio < MIN_PILL_AREA and canny_cnt is not None:
        mask_fg = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask_fg, [canny_cnt], -1, 255, thickness=cv2.FILLED)
        mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel_l, iterations=2)
        pill_area_ratio = mask_fg.sum() / 255 / (h * w)

    # ── Paso 5: último recurso — Otsu
    if pill_area_ratio < MIN_PILL_AREA:
        _, mask_fg = cv2.threshold(gray, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        mask_fg = _best_central_component(mask_fg)

    # ── Recortar bounding box respetando proporciones
    coords = cv2.findNonZero(mask_fg)
    if coords is not None:
        x, y, bw, bh = cv2.boundingRect(coords)
        pad = int(max(bw, bh) * 0.05)
        x1 = max(0, x - pad);  y1 = max(0, y - pad)
        x2 = min(w, x + bw + pad); y2 = min(h, y + bh + pad)
        crop = bgr[y1:y2, x1:x2]
        mask_crop = mask_fg[y1:y2, x1:x2]
    else:
        crop = bgr
        mask_crop = np.ones(bgr.shape[:2], np.uint8) * 255

    # Fondo blanco sobre la máscara inversa
    crop_clean = crop.copy()
    crop_clean[mask_crop == 0] = 255

    # ── Resize manteniendo aspect ratio + padding blanco hasta CANONICAL_SIZE
    ch, cw = crop_clean.shape[:2]
    scale   = CANONICAL_SIZE / max(ch, cw)
    new_w   = int(cw * scale)
    new_h   = int(ch * scale)
    resized = cv2.resize(crop_clean, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Centrar sobre lienzo blanco cuadrado
    canonical = np.full((CANONICAL_SIZE, CANONICAL_SIZE, 3), 255, dtype=np.uint8)
    off_y = (CANONICAL_SIZE - new_h) // 2
    off_x = (CANONICAL_SIZE - new_w) // 2
    canonical[off_y:off_y + new_h, off_x:off_x + new_w] = resized
    mask_bool = mask_crop > 0

    return mask_bool, canonical


# ──────────────────────────────────────────────
# 2. EXTRACCIÓN DE FEATURES
# ──────────────────────────────────────────────

def feat_color_histogram(canonical_bgr: np.ndarray) -> np.ndarray:
    """
    Histograma HSV ponderado: solo píxeles no-blancos (la pastilla).
    Devuelve vector de 34 dims normalizado.
    """
    hsv = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2HSV)

    # Máscara: excluir fondo blanco (V>245, S<15)
    pill_mask = ~((hsv[:, :, 2] > 245) & (hsv[:, :, 1] < 15))
    pill_mask = pill_mask.astype(np.uint8) * 255

    h_hist = cv2.calcHist([hsv], [0], pill_mask, [HIST_BINS_H], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], pill_mask, [HIST_BINS_S], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], pill_mask, [HIST_BINS_V], [0, 256]).flatten()

    vec = np.concatenate([h_hist, s_hist, v_hist])
    total = vec.sum()
    if total > 0:
        vec = vec / total
    return vec.astype(np.float32)   # 34 dims


def feat_shape(canonical_bgr: np.ndarray):
    """
    Momentos de Hu + descriptores geométricos básicos.
    Devuelve (vector 12 dims, dict de metadatos de forma).
    """
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    shape_meta = {"aspect_ratio": 1.0, "extent": 0.0, "solidity": 0.0,
                  "equiv_diameter": 0.0, "hu_moments": [0.0] * 7}

    if not contours:
        return np.zeros(12, np.float32), shape_meta

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area == 0:
        return np.zeros(12, np.float32), shape_meta

    x, y, bw, bh = cv2.boundingRect(cnt)
    hull          = cv2.convexHull(cnt)
    hull_area     = cv2.contourArea(hull)

    aspect_ratio   = float(bw) / bh if bh > 0 else 1.0
    rect_area      = bw * bh
    extent         = float(area) / rect_area if rect_area > 0 else 0.0
    solidity       = float(area) / hull_area if hull_area > 0 else 0.0
    equiv_diameter = np.sqrt(4 * area / np.pi) / CANONICAL_SIZE

    moments  = cv2.moments(cnt)
    hu_raw   = cv2.HuMoments(moments).flatten()
    hu_log   = -np.sign(hu_raw) * np.log10(np.abs(hu_raw) + 1e-10)
    hu_norm  = hu_log / 10.0    # escala aproximada

    shape_meta = {
        "aspect_ratio":   round(aspect_ratio, 3),
        "extent":         round(extent, 3),
        "solidity":       round(solidity, 3),
        "equiv_diameter": round(equiv_diameter, 3),
        "hu_moments":     [round(float(v), 4) for v in hu_norm]
    }

    geo_vec = np.array([aspect_ratio, extent, solidity, equiv_diameter,
                        float(bw) / CANONICAL_SIZE], dtype=np.float32)
    feat_vec = np.concatenate([hu_norm.astype(np.float32), geo_vec])  # 12 dims
    return feat_vec, shape_meta


def _lbp_manual(gray: np.ndarray, radius: int = 2, n_points: int = 16) -> np.ndarray:
    """LBP uniforme implementado con numpy (sin skimage)."""
    h, w = gray.shape
    lbp   = np.zeros((h, w), dtype=np.uint8)
    angles = [2 * np.pi * i / n_points for i in range(n_points)]
    neighbors = np.zeros((n_points, 2), dtype=int)
    for i, a in enumerate(angles):
        neighbors[i] = [int(round(radius * np.sin(a))),
                        int(round(radius * np.cos(a)))]

    for dy, dx in neighbors:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        lbp     = (lbp << 1) | (gray >= shifted).astype(np.uint8)

    return lbp


def feat_texture_lbp(canonical_bgr: np.ndarray) -> np.ndarray:
    """
    Histograma LBP uniforme sobre zona de pastilla.
    Devuelve vector de 26 dims normalizado.
    """
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    # Excluir fondo blanco
    mask = gray < 240

    lbp = _lbp_manual(gray, LBP_RADIUS, LBP_POINTS)
    n_bins = LBP_POINTS + 2   # uniforme → LBP_POINTS+2 patrones
    hist, _ = np.histogram(lbp[mask], bins=n_bins, range=(0, 255))
    hist = hist.astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist   # 18 dims  (ajustamos a LBP_POINTS+2)


def feat_edges(canonical_bgr: np.ndarray) -> np.ndarray:
    """
    Histograma de orientaciones de borde (HOG simplificado).
    Devuelve vector de EDGE_BINS dims.
    """
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    mask = mag > 10   # solo bordes significativos
    hist, _ = np.histogram(angle[mask], bins=EDGE_BINS, range=(0, 360),
                           weights=mag[mask])
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist.astype(np.float32)   # 8 dims


def compute_phash(canonical_bgr: np.ndarray) -> str:
    """
    pHash perceptual (DCT-based) implementado con numpy.
    Devuelve string hex de 16 chars (64 bits).
    """
    gray  = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (PHASH_SIZE * 4, PHASH_SIZE * 4),
                       interpolation=cv2.INTER_AREA).astype(np.float32)

    # DCT 2D via separable 1D
    from scipy.fft import dct
    dct2 = dct(dct(small, axis=0, norm='ortho'), axis=1, norm='ortho')
    dct_low = dct2[:PHASH_SIZE, :PHASH_SIZE]

    mean = dct_low.mean()
    bits = (dct_low > mean).flatten()
    # Empaquetar 64 bits → 8 bytes → hex 16 chars
    byte_vals = np.packbits(bits)
    return byte_vals.tobytes().hex()


def phash_to_vec(phash_hex: str) -> np.ndarray:
    """Convierte pHash hex a vector float32 de 8 dims [0,1]."""
    b = bytes.fromhex(phash_hex)
    bits = np.unpackbits(np.frombuffer(b, dtype=np.uint8))
    return bits.astype(np.float32)   # 64 dims


def build_feature_vector(canonical_bgr: np.ndarray):
    """
    Concatena todos los features en un único vector descriptor.
    Dimensiones:
      color_hist:  34
      shape:       12
      texture_lbp: 18  (LBP_POINTS+2)
      edges:        8
      phash:       64
      ─────────────────
      TOTAL:      136 dims

    Devuelve (vector np.float32[136], metadata_dict)
    """
    color_vec              = feat_color_histogram(canonical_bgr)       # 34
    shape_vec, shape_meta  = feat_shape(canonical_bgr)                  # 12
    lbp_vec                = feat_texture_lbp(canonical_bgr)            # 18
    edge_vec               = feat_edges(canonical_bgr)                  # 8
    phash_hex              = compute_phash(canonical_bgr)
    phash_vec              = phash_to_vec(phash_hex)                    # 64

    feat = np.concatenate([color_vec, shape_vec, lbp_vec, edge_vec, phash_vec])

    meta = {
        "phash":       phash_hex,
        "shape_meta":  shape_meta,
        "feat_dims": {
            "color_hist": len(color_vec),
            "shape":      len(shape_vec),
            "lbp":        len(lbp_vec),
            "edges":      len(edge_vec),
            "phash":      len(phash_vec),
            "total":      len(feat)
        }
    }
    return feat, meta


# ──────────────────────────────────────────────
# 3. PESOS POR SECCIÓN (para la búsqueda KNN)
# ──────────────────────────────────────────────

# Índices de inicio de cada bloque en el vector de 136 dims
FEAT_OFFSETS = {
    "color_hist": (0,  34),
    "shape":      (34, 46),
    "lbp":        (46, 64),
    "edges":      (64, 72),
    "phash":      (72, 136),
}

# Pesos por defecto (se pueden ajustar desde Android)
DEFAULT_WEIGHTS = {
    "color_hist": 2.0,   # color es muy discriminativo en pastillas
    "shape":      1.5,
    "lbp":        1.0,
    "edges":      0.8,
    "phash":      1.0,
}

def apply_weights(feat_vec: np.ndarray,
                  weights: dict = None) -> np.ndarray:
    """Escala cada bloque del vector por su peso."""
    if weights is None:
        weights = DEFAULT_WEIGHTS
    weighted = feat_vec.copy()
    for block, (start, end) in FEAT_OFFSETS.items():
        weighted[start:end] *= weights.get(block, 1.0)
    return weighted


# ──────────────────────────────────────────────
# 4. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def process_single(image_path: str):
    """
    Procesa una imagen y devuelve (vector, extra_meta) o lanza excepción.
    """
    bgr      = load_image(image_path)
    bgr      = correct_illumination(bgr)
    _, canon = segment_pill(bgr)
    feat, meta = build_feature_vector(canon)
    return feat, meta


def run_pipeline(images_dir: str,
                 meta_csv: str,
                 output_dir: str,
                 workers: int = 4):
    """
    Procesa todas las imágenes y genera el índice.
    """
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar metadatos manuales
    if meta_csv and Path(meta_csv).exists():
        df_meta = pd.read_csv(meta_csv)
        # Normalizar nombre de columna filename
        if "filename" not in df_meta.columns:
            # Intentar detectar columna con extensión de imagen
            for col in df_meta.columns:
                if df_meta[col].astype(str).str.contains(r'\.(jpg|jpeg|png|webp)', case=False, regex=True).any():
                    df_meta = df_meta.rename(columns={col: "filename"})
                    break
        df_meta["filename"] = df_meta["filename"].astype(str).str.strip()
        meta_lookup = df_meta.set_index("filename").to_dict(orient="index")
    else:
        meta_lookup = {}
        print("[WARN] No se encontró metadata.csv, continuando sin metadatos manuales.")

    # Listar imágenes
    extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    image_paths = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in extensions])
    total = len(image_paths)
    print(f"[INFO] Encontradas {total} imágenes en {images_dir}")

    if total == 0:
        print("[ERROR] No hay imágenes. Revisa --images.")
        sys.exit(1)

    ids      = []
    vectors  = []
    records  = []
    errors   = []

    t0 = time.time()
    for i, img_path in enumerate(image_paths):
        fname = img_path.name
        pill_id = img_path.stem

        # Progreso cada 100
        if (i + 1) % 100 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            eta     = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1:5d}/{total}]  {fname:<40}  "
                  f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

        try:
            feat, feat_meta = process_single(str(img_path))

            # Metadatos manuales (color, logo, shape, etc.)
            manual_meta = meta_lookup.get(fname, {})

            record = {
                "id":       pill_id,
                "filename": fname,
                "phash":    feat_meta["phash"],
                **{k: v for k, v in manual_meta.items() if k not in ("filename",)},
                "feat_dims": feat_meta["feat_dims"],
                "shape_computed": feat_meta["shape_meta"],
            }

            ids.append(pill_id)
            vectors.append(feat)
            records.append(record)

        except Exception as e:
            errors.append({"filename": fname, "error": str(e)})
            print(f"  [ERR] {fname}: {e}")

    # ── Guardar índice
    vectors_np = np.stack(vectors, axis=0).astype(np.float32)  # [N, 136]

    index_path = output_dir / "index.npz"
    np.savez_compressed(
        str(index_path),
        ids=np.array(ids),
        vectors=vectors_np,
        feat_offsets=json.dumps(FEAT_OFFSETS),
        default_weights=json.dumps(DEFAULT_WEIGHTS),
    )
    print(f"\n[OK] Índice guardado → {index_path}  "
          f"({vectors_np.shape[0]} pastillas × {vectors_np.shape[1]} dims, "
          f"{index_path.stat().st_size / 1e6:.1f} MB)")

    # ── Guardar metadata.json
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"[OK] Metadatos    → {meta_path}")

    # ── Guardar weights_config.json (editable por el equipo)
    weights_path = output_dir / "weights_config.json"
    with open(weights_path, "w") as f:
        json.dump({
            "description": "Ajusta estos pesos para ponderar cada bloque de features en la búsqueda KNN. "
                           "Rango recomendado: 0.0 - 5.0",
            "feat_offsets": FEAT_OFFSETS,
            "weights": DEFAULT_WEIGHTS,
        }, f, indent=2)
    print(f"[OK] Pesos config → {weights_path}")

    # ── Reporte
    elapsed_total = time.time() - t0
    report_lines = [
        "=" * 60,
        "  PILL INDEXER — REPORTE",
        "=" * 60,
        f"  Imágenes procesadas : {len(vectors)}",
        f"  Errores             : {len(errors)}",
        f"  Dimensión vector    : {vectors_np.shape[1]}",
        f"  Tiempo total        : {elapsed_total:.1f}s",
        f"  Tiempo/imagen       : {elapsed_total/max(len(vectors),1)*1000:.1f}ms",
        f"  Tamaño index.npz    : {index_path.stat().st_size / 1e6:.2f} MB",
        "",
        "  Bloques de features:",
    ]
    for block, (s, e) in FEAT_OFFSETS.items():
        report_lines.append(f"    {block:<15} dims [{s:3d}:{e:3d}]  peso={DEFAULT_WEIGHTS[block]}")
    if errors:
        report_lines += ["", "  Errores:"]
        for err in errors:
            report_lines.append(f"    {err['filename']}: {err['error']}")
    report_lines.append("=" * 60)

    report_str = "\n".join(report_lines)
    print("\n" + report_str)
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write(report_str)

    return index_path, meta_path


# ──────────────────────────────────────────────
# 5. ENTRYPOINT CLI
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extrae features de un dataset de pastillas y genera index.npz")
    parser.add_argument("--images", required=True,
                        help="Directorio con las imágenes (jpg/png/webp)")
    parser.add_argument("--meta",   default="",
                        help="CSV con metadatos manuales (columnas: filename, color, logo, shape, ...)")
    parser.add_argument("--out",    default="output",
                        help="Directorio de salida (se crea si no existe)")
    args = parser.parse_args()

    run_pipeline(args.images, args.meta, args.out)