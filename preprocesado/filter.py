"""
filter.py
=====================
Lee el CSV de EnergyControl y genera uno nuevo con solo las filas
que tienen imagen local disponible (ruta_imagen no vacía y fichero existente).

Uso:
  python filter.py --csv pastillas.csv --out dataset_con_imagenes.csv

Opciones:
  --images-dir   Directorio base donde están las imágenes (si las rutas son relativas)
  --check-exists Si se pasa, verifica que el fichero realmente existe en disco
                 (si no se pasa, solo filtra por ruta no vacía)
  --also-rear    Si se pasa, exige además que imagen_rear esté presente
"""

import argparse
import os
from pathlib import Path

import pandas as pd


def filter_pills_with_images(
    csv_path: str,
    out_path: str,
    images_dir: str = "",
    check_exists: bool = True,
    also_rear: bool = False,
):
    df = pd.read_csv(csv_path, sep=",", dtype=str).fillna("")

    total = len(df)
    print(f"[INFO] Total de filas en el CSV original: {total}")

    # ── Normalizar nombres de columna (strip espacios)
    df.columns = df.columns.str.strip()

    # ── Verificar columnas necesarias
    for col in ("ruta_imagen",):
        if col not in df.columns:
            raise ValueError(
                f"Columna '{col}' no encontrada. Columnas disponibles: {list(df.columns)}"
            )

    # ── Paso 1: filtrar por ruta_imagen no vacía
    mask = df["ruta_imagen"].str.strip() != ""
    print(f"  → Con ruta_imagen rellena : {mask.sum()}")

    # ── Paso 2 (opcional): filtrar también por ruta_imagen_rear
    if also_rear and "ruta_imagen_rear" in df.columns:
        mask_rear = df["ruta_imagen_rear"].str.strip() != ""
        mask = mask & mask_rear
        print(f"  → Con ruta_imagen_rear también: {mask.sum()}")

    df_filtered = df[mask].copy()

    # ── Paso 3 (opcional): verificar que el fichero existe en disco
    if check_exists:
        base = Path(images_dir) if images_dir else Path("")

        def file_exists(ruta: str) -> bool:
            ruta = ruta.strip()
            if not ruta:
                return False
            p = base / ruta if base != Path("") else Path(ruta)
            return p.exists()

        exists_mask = df_filtered["ruta_imagen"].apply(file_exists)
        missing = (~exists_mask).sum()
        if missing:
            print(f"  [WARN] {missing} rutas no existen en disco — se excluyen")
        df_filtered = df_filtered[exists_mask].copy()

    # ── Resumen
    kept = len(df_filtered)
    dropped = total - kept
    print(f"\n[RESULTADO]")
    print(f"  Filas conservadas : {kept}  ({kept/total*100:.1f}%)")
    print(f"  Filas descartadas : {dropped}")

    # ── Guardar
    df_filtered.to_csv(out_path, index=False)
    size_kb = Path(out_path).stat().st_size / 1024
    print(f"\n[OK] CSV filtrado guardado → {out_path}  ({size_kb:.1f} KB)")

    # ── Estadísticas rápidas del subconjunto
    print("\n[STATS del subconjunto]")
    for col in ("color", "logo", "procedencia"):
        if col in df_filtered.columns:
            n_unique = df_filtered[col].replace("", pd.NA).dropna().nunique()
            top = (
                df_filtered[col]
                .replace("", pd.NA)
                .dropna()
                .value_counts()
                .head(5)
                .to_dict()
            )
            print(f"  {col:<15} {n_unique} valores únicos   top5: {top}")

    # Cuántas tienen sustancia_2 o más (pastillas con mezclas)
    if "sustancia_2" in df_filtered.columns:
        mezclas = (df_filtered["sustancia_2"].str.strip() != "").sum()
        print(f"  {'mezclas':<15} {mezclas} pastillas con ≥2 sustancias")

    # Distribución de unidades (mg vs %)
    if "unidad_1" in df_filtered.columns:
        dist = df_filtered["unidad_1"].replace("", "desconocida").value_counts().to_dict()
        print(f"  {'unidad_1':<15} {dist}")

    return df_filtered


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filtra el CSV de EnergyControl a solo pastillas con imagen"
    )
    parser.add_argument("--csv",          required=True, help="CSV original")
    parser.add_argument("--out",          required=True, help="CSV de salida filtrado")
    parser.add_argument("--images-dir",   default="",    help="Directorio base de imágenes (opcional)")
    parser.add_argument("--check-exists", action="store_true", default=False,
                        help="Verificar que el fichero existe en disco")
    parser.add_argument("--also-rear",    action="store_true", default=False,
                        help="Exigir también imagen trasera")
    args = parser.parse_args()

    filter_pills_with_images(
        csv_path     = args.csv,
        out_path     = args.out,
        images_dir   = args.images_dir,
        check_exists = args.check_exists,
        also_rear    = args.also_rear,
    )