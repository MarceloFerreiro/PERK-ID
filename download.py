"""
Descarga imágenes de pastillas en paralelo (ThreadPoolExecutor).
Genera pastillas_con_imagen.csv solo con filas que tienen imagen.

Uso:
  pip install requests pandas tqdm
  python descargar_imagenes_pastillas.py
"""

import requests
import pandas as pd
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ── Configuración ─────────────────────────────────────────────────────────────
TOKEN       = "e27AGggh5c20RXRM9BWvc4mK8eoDxLsLQenRyW5gXUU9fa0a"
CSV_ENTRADA = "pastillas_energycontrol.csv"
CSV_SALIDA  = "pastillas_con_imagen.csv"
CARPETA_IMG = Path("imagenes")
WORKERS     = 16   # hilos paralelos — sube a 24 si va bien, baja si hay errores

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Referer": "https://energycontrol.org/",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}


# ── Descarga individual ───────────────────────────────────────────────────────
def nombre_archivo(url):
    return Path(urlparse(url).path).name


def descargar(url):
    """Descarga una URL a CARPETA_IMG. Devuelve ruta local o None."""
    destino = CARPETA_IMG / nombre_archivo(url)
    if destino.exists():
        return str(destino)  # ya existe, no repetir
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.ok and "image" in r.headers.get("Content-Type", ""):
            destino.write_bytes(r.content)
            return str(destino)
        return None
    except Exception:
        return None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # 1. Cargar y filtrar CSV
    print(f"Cargando {CSV_ENTRADA}...")
    df = pd.read_csv(CSV_ENTRADA, encoding="utf-8-sig")
    df_img = df[df["imagen"].notna() & (df["imagen"] != "")].copy().reset_index(drop=True)
    print(f"  Total: {len(df)} filas → Con imagen: {len(df_img)} ({len(df)-len(df_img)} descartadas)\n")

    CARPETA_IMG.mkdir(exist_ok=True)

    # 2. Construir lista de tareas (índice, tipo, url)
    tareas = []
    for i, fila in df_img.iterrows():
        tareas.append((i, "front", fila["imagen"]))
        if pd.notna(fila.get("imagen_rear")) and fila["imagen_rear"]:
            tareas.append((i, "rear", fila["imagen_rear"]))

    print(f"Descargando {len(tareas)} imágenes con {WORKERS} hilos paralelos...\n")

    # 3. Ejecutar en paralelo
    resultados = {}  # (idx, tipo) → ruta
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(descargar, url): (idx, tipo)
                   for idx, tipo, url in tareas}
        with tqdm(total=len(tareas), unit="img") as pbar:
            for future in as_completed(futures):
                idx, tipo = futures[future]
                resultados[(idx, tipo)] = future.result()
                pbar.update(1)

    # 4. Asignar rutas al dataframe
    df_img["ruta_imagen"]      = [resultados.get((i, "front")) for i in df_img.index]
    df_img["ruta_imagen_rear"] = [resultados.get((i, "rear"))  for i in df_img.index]

    # 5. Guardar CSV
    df_img.to_csv(CSV_SALIDA, index=False, encoding="utf-8-sig")

    ok = sum(1 for v in resultados.values() if v)
    print(f"\n✅ {ok}/{len(tareas)} imágenes descargadas en '{CARPETA_IMG}/'")
    print(f"✅ CSV guardado: {CSV_SALIDA}  ({len(df_img)} filas)")


if __name__ == "__main__":
    main()