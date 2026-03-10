"""
Descarga el dataset completo de pastillas de energycontrol.org
API: https://api.energycontrol.org/pilltable/?year=YYYY&month=MM

Genera:
  - pastillas_energycontrol.csv   (una fila por pastilla)
  - pastillas_energycontrol.json  (datos crudos)

Uso:
  pip install requests pandas
  python descargar_pastillas_energycontrol.py
"""

import requests
import pandas as pd
import json
import time

# ── Configuración ─────────────────────────────────────────────────────────────
BASE    = "https://api.energycontrol.org/pilltable/"
TOKEN   = "e27AGggh5c20RXRM9BWvc4mK8eoDxLsLQenRyW5gXUU9fa0a"
AÑOS    = list(range(2009, 2027))
MESES   = list(range(1, 13))
DELAY   = 0.4   # segundos entre peticiones (cortesía al servidor)

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "*/*",
    "Content-Type": "application/json; charset=UTF-8",
    "Origin": "https://energycontrol.org",
    "Referer": "https://energycontrol.org/",
    "X-Requested-With": "XMLHttpRequest",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}

OUT_CSV  = "pastillas_energycontrol.csv"
OUT_JSON = "pastillas_energycontrol.json"


# ── Descarga ──────────────────────────────────────────────────────────────────
def descargar_mes(year, mes):
    params = {"year": year, "month": mes}
    try:
        r = requests.get(BASE, params=params, headers=HEADERS, timeout=15)
        if r.status_code in (200, 304):
            data = r.json()
            if isinstance(data, list):
                return data
        elif r.status_code == 401:
            print(f"\n⚠️  Token expirado (401). Actualiza TOKEN en el script.")
        return []
    except Exception as e:
        print(f"\n  Error {year}/{mes}: {e}")
        return []


def descargar_todo():
    print("=== Descargando dataset completo de Energy Control ===\n")
    todos = []

    for year in AÑOS:
        total_año = 0
        print(f"Año {year}:", end=" ", flush=True)
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

    print(f"\nTotal descargado: {len(todos)} pastillas")
    return todos


# ── Transformación ────────────────────────────────────────────────────────────
def aplanar(pastilla):
    """Una fila por pastilla; sustancias expandidas como columnas."""
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
    for i, s in enumerate(pastilla.get("substances", []), 1):
        fila[f"sustancia_{i}"] = s.get("substance", "")
        try:
            fila[f"valor_{i}_mg"] = float(s["value"]) if s.get("value") else None
        except (ValueError, TypeError):
            fila[f"valor_{i}_mg"] = None
        fila[f"unidad_{i}"] = s.get("unit", "")
    return fila


# ── Guardado ──────────────────────────────────────────────────────────────────
def guardar(todos):
    # JSON crudo
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    print(f"✅ JSON: {OUT_JSON}")

    # CSV aplanado
    df = pd.DataFrame([aplanar(p) for p in todos])
    df.sort_values("fecha", inplace=True, ignore_index=True)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"✅ CSV:  {OUT_CSV}  ({len(df)} filas × {len(df.columns)} columnas)")
    print(f"\nColumnas: {list(df.columns)}")
    print(f"\nPrimeras filas:\n{df.head(5).to_string()}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    todos = descargar_todo()
    if todos:
        guardar(todos)
    else:
        print("No se descargaron datos. Comprueba el token.")