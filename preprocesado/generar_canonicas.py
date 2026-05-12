"""
generate_canonicals.py
======================
Genera las imágenes canónicas (segmentadas, fondo blanco, 256×256 con
aspect ratio preservado) de todas las fotos delanteras del CSV filtrado.

Las guarda en --out con el mismo nombre de fichero que la imagen original.

Uso:
  python generate_canonicals.py \
      --csv   dataset_con_imagenes.csv \
      --base  /ruta/donde/estan/las/fotos/ \
      --out   canonicals/
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from extract_features import correct_illumination, segment_pill


# ── Worker (se ejecuta en proceso hijo) ────────────────────
# Debe ser función de módulo (no lambda ni closure) para que
# ProcessPoolExecutor pueda serializarla con pickle.

def _process_one(args):
    src_str, dst_str = args
    try:
        bgr = cv2.imread(src_str)
        if bgr is None:
            return src_str, "no se pudo leer"
        bgr = correct_illumination(bgr)
        _, canonical = segment_pill(bgr)
        cv2.imwrite(dst_str, canonical, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return src_str, None          # None = sin error
    except Exception as e:
        return src_str, str(e)


# ── Pipeline principal ──────────────────────────────────────

def generate_canonicals(csv_path: str, base_dir: str, out_dir: str, workers: int):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    base = Path(base_dir) if base_dir else Path("")

    df = pd.read_csv(csv_path, dtype=str).fillna("")
    df.columns = df.columns.str.strip()

    if "ruta_imagen" not in df.columns:
        print("[ERROR] El CSV no tiene columna 'ruta_imagen'")
        sys.exit(1)

    df = df[df["ruta_imagen"].str.strip() != ""].copy()
    total = len(df)
    print(f"[INFO] {total} imágenes delanteras · {workers} workers")

    # Construir lista de (src, dst) — solo las que aún no existen
    tasks = []
    skipped = 0
    for _, row in df.iterrows():
        ruta = row["ruta_imagen"].strip()
        src  = base / ruta if base_dir else Path(ruta)
        dst  = out_path / src.name
        if dst.exists():
            skipped += 1
            continue
        tasks.append((str(src), str(dst)))

    if skipped:
        print(f"[INFO] {skipped} ya procesadas, se omiten")

    if not tasks:
        print("[INFO] Nada que hacer.")
        return

    print(f"[INFO] {len(tasks)} imágenes a procesar\n")

    ok = errors = done = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process_one, t): t for t in tasks}
        for future in as_completed(futures):
            src_str, err = future.result()
            done += 1
            if err:
                errors += 1
                print(f"  [ERR] {Path(src_str).name}: {err}")
            else:
                ok += 1

            if done % 100 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                eta     = elapsed / done * (len(tasks) - done)
                print(f"  [{done:5d}/{len(tasks)}]  "
                      f"ok={ok}  err={errors}  "
                      f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    elapsed = time.time() - t0
    print(f"\n[OK] {ok} canónicas guardadas en {out_path}/  ({elapsed:.0f}s)")
    if errors:
        print(f"[WARN] {errors} errores")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",     required=True)
    parser.add_argument("--base",    default="",  help="Directorio base de las imágenes originales")
    parser.add_argument("--out",     required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count(),
                        help=f"Procesos paralelos (default: nº de cores = {os.cpu_count()})")
    args = parser.parse_args()

    generate_canonicals(args.csv, args.base, args.out, args.workers)