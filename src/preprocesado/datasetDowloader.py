"""
Descarga del dataset de energy control

Descarga metadatos e imágenes de pastillas de energycontrol.org y genera:
  - pastillas_energycontrol.csv   (dataset completo con rutas de imagen)

API:  https://api.energycontrol.org/pilltable/?year=YYYY&month=MM

Uso:
  pip install requests pandas tqdm
  python pipeline_energycontrol.py


"""
import os 
import requests
import pandas as pd
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Config
TOKEN        = "e27AGggh5c20RXRM9BWvc4mK8eoDxLsLQenRyW5gXUU9fa0a"
AÑOS         = list(range(2009, 2027))   # rango de años a descargar
MESES        = list(range(1, 13))        # 1-12
DELAY        = 0.4                        # segundos entre peticiones API
IMG_WORKERS  = IMG_WORKERS = os.cpu_count()                    # hilos paralelos para imágenes
CARPETA_IMG  = Path("imagenes")           # carpeta de salida de imágenes
CSV_SALIDA   = "pastillas_energycontrol.csv"
MAX_SUSTANCIAS = 11                       # columnas sustancia_N / valor_N / unidad_N

BASE_URL = "https://api.energycontrol.org/pilltable/"

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "*/*",
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://energycontrol.org",
    "Referer": "https://energycontrol.org/",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}





def descargar_mes(year: int, mes: int) -> list:
    """Descarga un mes de la API. Devuelve lista de pastillas (puede ser [])."""
    params = {"year": year, "month": mes}
    try:
        r = requests.get(BASE_URL, params=params, headers=HEADERS, timeout=15)
        if r.status_code in (200, 304):
            data = r.json()
            return data if isinstance(data, list) else []
        if r.status_code == 401:
            print("\nToken expirado (401). ")
        return []
    except Exception as e:
        print(f"\n  ⚠ Error {year}/{mes}: {e}")
        return []


def descargar_todo() -> list:
    
    print("Descarga de metadatos")
    
    todos = []
    for year in AÑOS:
        total_año = 0
        print(f"  {year}: ", end="", flush=True)
        for mes in MESES:
            resultados = descargar_mes(year, mes)
            if resultados:
                for p in resultados:
                    p["_year"] = year
                    p["_mes"]  = mes
                todos.extend(resultados)
                total_año += len(resultados)
                print(f"{mes}✓", end=" ", flush=True)
            else:
                print(f"{mes}-", end=" ", flush=True)
            time.sleep(DELAY)
        print(f"  → {total_año} pastillas")
    print(f"\n  Total descargado: {len(todos)} pastillas\n")
    return todos


def _nombre_archivo(url: str) -> str:
    return Path(urlparse(url).path).name


def _descargar_imagen(url: str) -> str | None:
    """Descarga una imagen a CARPETA_IMG. Devuelve ruta local o None."""
    destino = CARPETA_IMG / _nombre_archivo(url)
    if destino.exists():
        return str(destino)
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.ok and "image" in r.headers.get("Content-Type", ""):
            destino.write_bytes(r.content)
            return str(destino)
        return None
    except Exception:
        return None


def descargar_imagenes(df: pd.DataFrame) -> pd.DataFrame:

    print("Descarga de imágenes")


    CARPETA_IMG.mkdir(exist_ok=True)

    # Construir lista de tareas (idx, tipo, url)
    tareas = []
    for i, fila in df.iterrows():
        if pd.notna(fila.get("imagen")) and fila["imagen"]:
            tareas.append((i, "front", fila["imagen"]))
        if pd.notna(fila.get("imagen_rear")) and fila["imagen_rear"]:
            tareas.append((i, "rear", fila["imagen_rear"]))

    print(f"  {len(tareas)} imágenes a descargar ({IMG_WORKERS} workers\n")

    resultados: dict[tuple, str | None] = {}

    def _iterar(futures_map):
        for future in as_completed(futures_map):
            idx, tipo = futures_map[future]
            resultados[(idx, tipo)] = future.result()

    with ThreadPoolExecutor(max_workers=IMG_WORKERS) as executor:
        futures_map = {
            executor.submit(_descargar_imagen, url): (idx, tipo)
            for idx, tipo, url in tareas
        }
        if HAS_TQDM:
            with tqdm(total=len(tareas), unit="img") as pbar:
                for future in as_completed(futures_map):
                    idx, tipo = futures_map[future]
                    resultados[(idx, tipo)] = future.result()
                    pbar.update(1)
        else:
            _iterar(futures_map)

    df["ruta_imagen"]      = [resultados.get((i, "front")) for i in df.index]
    df["ruta_imagen_rear"] = [resultados.get((i, "rear"))  for i in df.index]

    ok = sum(1 for v in resultados.values() if v)
    print(f"\n  {ok}/{len(tareas)} imágenes descargadas en '{CARPETA_IMG}/'\n")
    return df



def aplanar(pastilla: dict) -> dict:
    """Convierte una pastilla (dict anidado) en una fila plana del CSV."""
    fila = {
        "fecha":       pastilla.get("date"),
        "year":        pastilla.get("_year"),
        "mes":         pastilla.get("_mes"),
        "logo":        pastilla.get("logo"),
        "color":       pastilla.get("color"),
        "procedencia": pastilla.get("procedence"),
        "peso_mg":     pastilla.get("pillweight"),
        "diametro_mm": pastilla.get("diameter"),
        "divisible":   pastilla.get("divisible"),
        "imagen":      pastilla.get("image"),
        "imagen_rear": pastilla.get("image_rear"),
    }
    # Expandir sustancias como columnas (hasta MAX_SUSTANCIAS)
    for i, s in enumerate(pastilla.get("substances", []), 1):
        if i > MAX_SUSTANCIAS:
            break
        fila[f"sustancia_{i}"] = s.get("substance", "")
        try:
            fila[f"valor_{i}_mg"] = float(s["value"]) if s.get("value") else None
        except (ValueError, TypeError):
            fila[f"valor_{i}_mg"] = None
        fila[f"unidad_{i}"] = s.get("unit", "")
    return fila


def construir_columnas() -> list[str]:
    """Devuelve la lista ordenada de columnas del CSV final."""
    base = [
        "fecha", "year", "mes", "logo", "color", "procedencia",
        "peso_mg", "diametro_mm", "divisible", "imagen", "imagen_rear",
    ]
    sustancias = []
    for i in range(1, MAX_SUSTANCIAS + 1):
        sustancias += [f"sustancia_{i}", f"valor_{i}_mg", f"unidad_{i}"]
    return base + sustancias + ["ruta_imagen", "ruta_imagen_rear"]


def guardar_csv(df: pd.DataFrame) -> None:

    print("Generar el CSV")


    # Asegurar que todas las columnas esperadas existen (rellena con None si faltan)
    columnas = construir_columnas()
    for col in columnas:
        if col not in df.columns:
            df[col] = None

    df = df[columnas]                                      # orden exacto
    df.sort_values("fecha", inplace=True, ignore_index=True)
    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8-sig")

    print(f"   CSV guardado: {CSV_SALIDA}")
    print(f"     {len(df)} filas × {len(df.columns)} columnas")
    print(f"\n  Primeras 3 filas:\n")
    print(df.head(3).to_string())
    print()



def main():
    print("Descarga del energy control")

    # 1. Metadatos
    todos = descargar_todo()
    if not todos:
        print(" No se descargaron datos. Comprueba el token y la conexión.")
        return

    # 2. Construir DataFrame plano
    df = pd.DataFrame([aplanar(p) for p in todos])

    # 3. Imágenes
    df = descargar_imagenes(df)

    # 4. Guardar CSV
    guardar_csv(df)

    print("Pipeline completado.\n")


if __name__ == "__main__":
    main()