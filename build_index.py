"""
build_index.py
==============
Lee las imágenes canónicas ya generadas + el CSV original y produce:

  output/index.npz      — vectores de features (N × 136) + ids + filtros
  output/metadata.json  — toda la info por pastilla (sustancias, dosis, etc.)

Uso:
  python build_index.py \
      --canonicals  canonicals/ \
      --csv         dataset_con_imagenes.csv \
      --out         output/

Paralelizable con --workers (default: nº de cores).
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from preprocesado.extract_features import build_feature_vector, correct_illumination, FEAT_OFFSETS, DEFAULT_WEIGHTS


# ──────────────────────────────────────────────
# WORKER (proceso hijo)
# ──────────────────────────────────────────────

def _extract_one(img_path: str):
    """
    Extrae el vector de features de una canónica ya preprocesada.
    La canónica ya tiene fondo blanco y tamaño normalizado,
    así que NO aplicamos segment_pill — solo features.
    Devuelve (filename, vector, phash, shape_meta) o (filename, None, error)
    """
    try:
        bgr = cv2.imread(img_path)
        if bgr is None:
            return img_path, None, None, f"no se pudo leer"
        # Las canónicas ya están preprocesadas; correct_illumination es ligero
        # y ayuda a normalizar variaciones residuales de brillo
        bgr = correct_illumination(bgr)
        feat, meta = build_feature_vector(bgr)
        return img_path, feat, meta, None
    except Exception as e:
        return img_path, None, None, str(e)


# ──────────────────────────────────────────────
# PARSEO DE SUSTANCIAS
# ──────────────────────────────────────────────

def _parse_sustancias(row: dict) -> list:
    """
    Convierte las columnas sustancia_1..11 / valor_1_mg..11 / unidad_1..11
    en una lista limpia de dicts.
    """
    sustancias = []
    for i in range(1, 12):
        nombre = str(row.get(f"sustancia_{i}", "")).strip()
        if not nombre:
            break
        valor_raw = str(row.get(f"valor_{i}_mg", "")).strip()
        unidad    = str(row.get(f"unidad_{i}",   "")).strip()
        try:
            valor = float(valor_raw) if valor_raw else None
        except ValueError:
            valor = None
        sustancias.append({
            "nombre": nombre,
            "valor":  valor,
            "unidad": unidad or None,
        })
    return sustancias


def _safe_float(val, zero_as_none=True):
    try:
        f = float(val)
        return None if (zero_as_none and f == 0.0) else f
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def build_index(canonicals_dir: str, csv_path: str, out_dir: str, workers: int):
    canonicals_dir = Path(canonicals_dir)
    out_dir        = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Cargar CSV
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = df.columns.str.strip()
    print(f"[INFO] CSV cargado: {len(df)} filas")

    # Construir lookup por nombre de fichero (sin ruta, solo nombre)
    if "ruta_imagen" not in df.columns:
        print("[ERROR] El CSV no tiene columna 'ruta_imagen'")
        sys.exit(1)

    df["_fname"] = df["ruta_imagen"].apply(lambda x: Path(x.strip()).name)

    # Avisar de duplicados y quedarse con la primera ocurrencia
    dupes = df["_fname"][df["_fname"].duplicated()].unique()
    if len(dupes):
        print(f"[WARN] {len(dupes)} nombres de fichero duplicados en el CSV — "
              f"se usa la primera ocurrencia de cada uno")
    df = df.drop_duplicates(subset="_fname", keep="first")

    csv_lookup = df.set_index("_fname").to_dict(orient="index")

    # ── Listar canónicas
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    canon_paths = sorted([
        p for p in canonicals_dir.iterdir()
        if p.suffix.lower() in exts
    ])
    total = len(canon_paths)
    print(f"[INFO] {total} canónicas encontradas en {canonicals_dir}")
    print(f"[INFO] {workers} workers\n")

    if total == 0:
        print("[ERROR] No hay imágenes en --canonicals")
        sys.exit(1)

    # ── Extraer features en paralelo
    ids      = []
    vectors  = []
    metadata = {}
    errors   = []

    t0   = time.time()
    done = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_extract_one, str(p)): p for p in canon_paths}

        for future in as_completed(futures):
            img_path_str, feat, feat_meta, err = future.result()
            fname = Path(img_path_str).name
            pill_id = Path(img_path_str).stem
            done += 1

            if done % 200 == 0 or done == total:
                elapsed = time.time() - t0
                eta     = elapsed / done * (total - done)
                print(f"  [{done:5d}/{total}]  ok={len(ids)}  "
                      f"err={len(errors)}  elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

            if err:
                errors.append({"filename": fname, "error": err})
                continue

            # ── Metadatos del CSV
            csv_row = csv_lookup.get(fname, {})

            sustancias = _parse_sustancias(csv_row)

            record = {
                "id":          pill_id,
                "filename":    fname,
                "phash":       feat_meta["phash"],
                # Filtrables
                "color":       csv_row.get("color",       "").strip() or None,
                "logo":        csv_row.get("logo",        "").strip() or None,
                "procedencia": csv_row.get("procedencia", "").strip() or None,
                # Info para mostrar al usuario
                "fecha":       csv_row.get("fecha",       "").strip() or None,
                "peso_mg":     _safe_float(csv_row.get("peso_mg")),
                "diametro_mm": _safe_float(csv_row.get("diametro_mm")),
                "divisible":   csv_row.get("divisible",  "").strip() or None,
                "sustancias":  sustancias,
                # Diagnóstico
                "shape_computed": feat_meta["shape_meta"],
            }

            ids.append(pill_id)
            vectors.append(feat)
            metadata[pill_id] = record

    # ── Guardar index.npz
    
    vectors_np = np.stack(vectors, axis=0).astype(np.float32)

    # Extraer filtros como arrays paralelos para KNN rápido en Android
    colors = np.array([metadata[i].get("color") or "" for i in ids])
    logos  = np.array([metadata[i].get("logo")  or "" for i in ids])

    index_path = out_dir / "index.npz"
    np.savez_compressed(
        str(index_path),
        ids             = np.array(ids),
        vectors         = vectors_np,
        colors          = colors,
        logos           = logos,
        feat_offsets    = json.dumps(FEAT_OFFSETS),
        default_weights = json.dumps(DEFAULT_WEIGHTS),
    )

    # ── Guardar metadata.json
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # ── Reporte
    elapsed = time.time() - t0
    index_mb = index_path.stat().st_size / 1e6
    meta_mb  = meta_path.stat().st_size  / 1e6

    print(f"\n{'═'*55}")
    print(f"  BUILD INDEX — COMPLETADO")
    print(f"{'═'*55}")
    print(f"  Pastillas indexadas : {len(ids)}")
    print(f"  Errores             : {len(errors)}")
    print(f"  Sin CSV match       : {sum(1 for i in ids if not metadata[i].get('fecha'))}")
    print(f"  Dimensión vector    : {vectors_np.shape[1]}")
    print(f"  Tiempo total        : {elapsed:.0f}s")
    print(f"  index.npz           : {index_mb:.2f} MB  →  {index_path}")
    print(f"  metadata.json       : {meta_mb:.2f} MB  →  {meta_path}")
    print(f"{'═'*55}")

    # Stats rápidas del dataset
    print(f"\n  Top colores:")
    from collections import Counter
    color_counts = Counter(metadata[i].get("color") for i in ids if metadata[i].get("color"))
    for color, n in color_counts.most_common(8):
        print(f"    {color:<20} {n}")

    print(f"\n  Top logos:")
    logo_counts = Counter(metadata[i].get("logo") for i in ids if metadata[i].get("logo"))
    for logo, n in logo_counts.most_common(8):
        print(f"    {logo:<30} {n}")

    if errors:
        err_path = out_dir / "errors.json"
        with open(err_path, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"\n  Errores guardados → {err_path}")

    return index_path, meta_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Genera index.npz + metadata.json a partir de canónicas y CSV")
    parser.add_argument("--canonicals", required=True,
                        help="Directorio con las imágenes canónicas")
    parser.add_argument("--csv",        required=True,
                        help="dataset_con_imagenes.csv")
    parser.add_argument("--out",        default="output",
                        help="Directorio de salida (default: output/)")
    parser.add_argument("--workers",    type=int, default=os.cpu_count(),
                        help=f"Workers paralelos (default: {os.cpu_count()} cores)")
    args = parser.parse_args()

    build_index(args.canonicals, args.csv, args.out, args.workers)