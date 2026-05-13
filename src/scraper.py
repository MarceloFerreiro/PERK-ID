import os 
import sys
import json
import time
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import requests
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from src.features.read_image import read_image

load_dotenv()

# Config
TOKEN        = os.getenv('TOKEN') 
AÑOS         = list(range(2009, 2027))   # rango de años a descargar
DELAY        = 0.4                        # segundos entre peticiones API
CARPETA_IMG  = Path("data/imagenes_alt")           # carpeta de salida de imágenes
CSV_SALIDA   = "data/pastillas_energycontrol.csv"
MAX_SUSTANCIAS = 3                       # columnas sustancia_N / valor_N / unidad_N

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
    
    todos = []
    total = 0
    progreso = tqdm(AÑOS, unit='años', postfix='Descarga de Metadatos')
    for year in progreso:
        for mes in list(range(1, 13)):
            resultados = descargar_mes(year, mes)
            if resultados:
                for p in resultados:
                    p["_year"] = year
                    p["_mes"]  = mes
                todos.extend(resultados)
                total += len(resultados)
            time.sleep(DELAY)
        progreso.set_description(f"{total} pastillas")
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
            processed = read_image(destino)
            if processed is not None:
                # Save the processed numpy array back as image
                from skimage import io
                io.imsave(str(destino), processed)
                return str(destino)
            destino.unlink()  # Clean up if processing failed
        return None
    except Exception:
        return None


def descargar_imagenes(df: pd.DataFrame) -> pd.DataFrame:
    CARPETA_IMG.mkdir(exist_ok=True)

    # Construir lista de tareas (idx, tipo, url)
    tareas = []
    for i, fila in df.iterrows():
        if pd.notna(fila.get("imagen")) and fila["imagen"]:
            tareas.append((i, "front", fila["imagen"]))
        if pd.notna(fila.get("imagen_rear")) and fila["imagen_rear"]:
            tareas.append((i, "rear", fila["imagen_rear"]))

    print(f"  {len(tareas)} imágenes a descargar || {os.cpu_count()} workers\n")

    resultados: dict[tuple, str | None] = {}

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures_map = {
            executor.submit(_descargar_imagen, url): (idx, tipo)
            for idx, tipo, url in tareas
        }
        with tqdm(total=len(tareas), unit="imagenes", postfix='Descarga Imágenes') as pbar:
            for future in as_completed(futures_map):
                idx, tipo = futures_map[future]
                resultados[(idx, tipo)] = future.result()
                pbar.update(1)

    df["ruta_imagen"]      = [resultados.get((i, "front")) for i in df.index]
    df["ruta_imagen_rear"] = [resultados.get((i, "rear"))  for i in df.index]

    ok = sum(1 for v in resultados.values() if v)
    print(f"\n  {ok}/{len(tareas)} imágenes descargadas en '{CARPETA_IMG}/'\n")

    df = df[ df["ruta_imagen"].notna() | df["ruta_imagen_rear"].notna() ].copy()
        


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

def guardar_csv(df: pd.DataFrame) -> None:
    # Asegurar que todas las columnas esperadas existen (rellena con None si faltan)
    base = [
        "fecha", "year", "mes", "logo", "color", "procedencia",
        "peso_mg", "diametro_mm", "divisible", "imagen", "imagen_rear",
    ]
    sustancias = []
    for i in range(1, MAX_SUSTANCIAS + 1):
        sustancias += [f"sustancia_{i}", f"valor_{i}_mg", f"unidad_{i}"]
    columnas =  base + sustancias + ["ruta_imagen", "ruta_imagen_rear"]

    for col in columnas:
        if col not in df.columns:
            df[col] = None

    df = df[columnas]                                      # orden exacto
    df.sort_values("fecha", inplace=True, ignore_index=True)
    df.to_csv(CSV_SALIDA, index=False, encoding="utf-8-sig")

    print(df.head(10).to_string())

if __name__ == "__main__":

    todos = descargar_todo()
    if not todos:
        print(" No se descargaron datos. Comprueba el token y la conexión.")
        sys.exit()
        

    df = pd.DataFrame([aplanar(p) for p in todos])

    df = descargar_imagenes(df)

    guardar_csv(df)

    print("[--] FIN [--]")
