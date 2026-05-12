"""
prueba_segmentation.py
====================
Toma N imágenes aleatorias del dataset y genera un HTML visual
con el resultado de cada paso del preprocesado:
  original → iluminación corregida → máscara → recorte canónico

Uso:
  python prueba_segmentation.py --images-dir /ruta/imagenes/ --n 20 --out test_seg.html

  # Con CSV para tomar muestras representativas por color:
  python prueba_segmentation.py --csv dataset_con_imagenes.csv --n 30 --out test_seg.html


"""

import argparse
import base64
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# ── Reutiliza las funciones del pipeline principal
sys.path.insert(0, str(Path(__file__).parent))
from extract_features import correct_illumination, segment_pill


# ──────────────────────────────────────────────
# PROCESAR UNA IMAGEN → 4 pasos visuales
# ──────────────────────────────────────────────

def process_steps(img_path: str):
    """
    Devuelve dict con las 4 imágenes de diagnóstico en base64 PNG,
    o None si falla la carga.
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None

    # Redimensionar entrada para que no sea enorme en el HTML
    h, w = bgr.shape[:2]
    max_dim = 400
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        bgr = cv2.resize(bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Paso 1: original
    step1 = bgr.copy()

    # Paso 2: corrección de iluminación
    step2 = correct_illumination(bgr)

    # Paso 3: máscara de segmentación (visualización)
    try:
        mask_bool, canonical = segment_pill(step2)
        # Colorear máscara: verde=pastilla, rojo=fondo
        mask_vis = bgr.copy()
        overlay = np.zeros_like(bgr)
        # Redimensionar mask_bool al tamaño de bgr si hace falta
        mask_resized = cv2.resize(
            mask_bool.astype(np.uint8) * 255,
            (bgr.shape[1], bgr.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        overlay[mask_resized > 0]  = [0, 180, 0]    # verde = pastilla
        overlay[mask_resized == 0] = [0, 0, 180]    # rojo  = fondo
        step3 = cv2.addWeighted(mask_vis, 0.55, overlay, 0.45, 0)

        # Paso 4: imagen canónica (256×256, fondo blanco)
        step4 = canonical

        seg_ok = True
    except Exception as e:
        step3 = np.zeros_like(bgr)
        cv2.putText(step3, f"ERROR: {str(e)[:30]}", (5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        step4 = np.zeros((256, 256, 3), dtype=np.uint8)
        seg_ok = False

    def to_b64(img):
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode()

    return {
        "path":      str(img_path),
        "name":      Path(img_path).name,
        "original":  to_b64(step1),
        "illum":     to_b64(step2),
        "mask":      to_b64(step3),
        "canonical": to_b64(step4),
        "seg_ok":    seg_ok,
        "orig_size": f"{w}×{h}",
    }


# ──────────────────────────────────────────────
# SELECCIONAR IMÁGENES DE MUESTRA
# ──────────────────────────────────────────────

def sample_images(images_dir: str = None, csv_path: str = None,
                  n: int = 20, images_base: str = "") -> list:
    """
    Devuelve lista de paths. Si hay CSV, intenta muestrear
    proporcionalmente por color para tener variedad.
    """
    paths = []

    if csv_path:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df.columns = df.columns.str.strip()

        # Detectar columna de ruta
        ruta_col = "ruta_imagen" if "ruta_imagen" in df.columns else None
        if ruta_col is None:
            print("[WARN] No se encontró columna ruta_imagen en el CSV")
        else:
            base = Path(images_base) if images_base else Path("")

            # Muestreo estratificado por color si existe la columna
            if "color" in df.columns:
                grupos = df.groupby("color")
                per_group = max(1, n // len(grupos))
                sampled = grupos.apply(
                    lambda g: g.sample(min(len(g), per_group), random_state=42)
                ).reset_index(drop=True)
                # Completar hasta n si sobra cupo
                resto = n - len(sampled)
                if resto > 0:
                    extra = df[~df.index.isin(sampled.index)].sample(
                        min(resto, len(df) - len(sampled)), random_state=42
                    )
                    sampled = pd.concat([sampled, extra])
            else:
                sampled = df.sample(min(n, len(df)), random_state=42)

            for _, row in sampled.iterrows():
                ruta = str(row[ruta_col]).strip()
                if not ruta:
                    continue
                p = base / ruta if base != Path("") else Path(ruta)
                if p.exists():
                    paths.append(str(p))

    elif images_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_imgs = [p for p in Path(images_dir).iterdir()
                    if p.suffix.lower() in exts]
        random.shuffle(all_imgs)
        paths = [str(p) for p in all_imgs[:n]]

    return paths


# ──────────────────────────────────────────────
# GENERAR HTML
# ──────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Test Segmentación — EnergyControl Pill Indexer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 6px; color: #fff; }}
  .subtitle {{ font-size: 0.85rem; color: #888; margin-bottom: 24px; }}
  .summary {{ background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px;
              padding: 14px 18px; margin-bottom: 28px; font-size: 0.85rem; }}
  .summary b {{ color: #7eb8f7; }}
  .grid {{ display: flex; flex-direction: column; gap: 20px; }}
  .card {{ background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 10px;
           overflow: hidden; }}
  .card.error {{ border-color: #7f2020; }}
  .card-header {{ padding: 10px 14px; background: #12121a;
                  display: flex; justify-content: space-between; align-items: center; }}
  .card-header .fname {{ font-size: 0.8rem; color: #aaa; font-family: monospace; }}
  .card-header .size  {{ font-size: 0.75rem; color: #666; }}
  .badge {{ font-size: 0.7rem; padding: 2px 8px; border-radius: 99px; font-weight: 600; }}
  .badge.ok  {{ background: #1a3a1a; color: #4caf50; }}
  .badge.err {{ background: #3a1a1a; color: #f44336; }}
  .steps {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
            background: #0f0f13; }}
  .step {{ padding: 8px; background: #1a1a24; }}
  .step-label {{ font-size: 0.68rem; color: #666; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 6px; }}
  .step img {{ width: 100%; border-radius: 4px; display: block; }}
  .controls {{ display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }}
  button {{ background: #2a2a3a; border: 1px solid #3a3a4a; color: #ccc;
            padding: 6px 14px; border-radius: 6px; cursor: pointer; font-size: 0.8rem; }}
  button:hover {{ background: #3a3a4a; }}
  button.active {{ background: #1a3a6a; border-color: #4a7abf; color: #7eb8f7; }}
  @media (max-width: 700px) {{
    .steps {{ grid-template-columns: repeat(2, 1fr); }}
  }}
</style>
</head>
<body>
<h1>Test de Segmentación — Pill Indexer</h1>
<p class="subtitle">Evalúa visualmente la calidad del preprocesado antes de lanzar el pipeline completo</p>

<div class="summary">
  <b>Total evaluadas:</b> {total} &nbsp;|&nbsp;
  <b>Segmentación OK:</b> {ok} ({pct_ok}%) &nbsp;|&nbsp;
  <b>Errores:</b> {errors}
</div>

<div class="controls">
  <button onclick="showAll()" class="active" id="btn-all">Todas</button>
  <button onclick="showOnly('ok')"  id="btn-ok">Solo OK</button>
  <button onclick="showOnly('err')" id="btn-err">Solo errores</button>
</div>

<div class="grid" id="grid">
{cards}
</div>

<script>
function showAll() {{
  document.querySelectorAll('.card').forEach(c => c.style.display = '');
  setActive('btn-all');
}}
function showOnly(type) {{
  document.querySelectorAll('.card').forEach(c => {{
    c.style.display = c.classList.contains(type) ? '' : 'none';
  }});
  setActive(type === 'ok' ? 'btn-ok' : 'btn-err');
}}
function setActive(id) {{
  ['btn-all','btn-ok','btn-err'].forEach(b => document.getElementById(b).classList.remove('active'));
  document.getElementById(id).classList.add('active');
}}
</script>
</body>
</html>"""

CARD_TEMPLATE = """
  <div class="card {cls}" data-status="{status}">
    <div class="card-header">
      <span class="fname">{name}</span>
      <div style="display:flex;gap:8px;align-items:center">
        <span class="size">{orig_size}</span>
        <span class="badge {badge_cls}">{badge_txt}</span>
      </div>
    </div>
    <div class="steps">
      <div class="step">
        <div class="step-label">1 · Original</div>
        <img src="data:image/jpeg;base64,{original}" alt="original">
      </div>
      <div class="step">
        <div class="step-label">2 · Iluminación</div>
        <img src="data:image/jpeg;base64,{illum}" alt="iluminación">
      </div>
      <div class="step">
        <div class="step-label">3 · Máscara</div>
        <img src="data:image/jpeg;base64,{mask}" alt="máscara">
      </div>
      <div class="step">
        <div class="step-label">4 · Canónica 256px</div>
        <img src="data:image/jpeg;base64,{canonical}" alt="canónica">
      </div>
    </div>
  </div>"""


def generate_html(results: list, out_path: str):
    ok_count  = sum(1 for r in results if r and r["seg_ok"])
    err_count = sum(1 for r in results if r and not r["seg_ok"])
    total     = len(results)
    pct_ok    = round(ok_count / total * 100) if total else 0

    cards_html = ""
    for r in results:
        if r is None:
            continue
        cls        = "ok" if r["seg_ok"] else "error err"
        status     = "ok" if r["seg_ok"] else "err"
        badge_cls  = "ok" if r["seg_ok"] else "err"
        badge_txt  = "OK" if r["seg_ok"] else "ERROR"
        cards_html += CARD_TEMPLATE.format(
            cls=cls, status=status, badge_cls=badge_cls, badge_txt=badge_txt,
            name=r["name"], orig_size=r["orig_size"],
            original=r["original"], illum=r["illum"],
            mask=r["mask"], canonical=r["canonical"],
        )

    html = HTML_TEMPLATE.format(
        total=total, ok=ok_count, pct_ok=pct_ok, errors=err_count,
        cards=cards_html,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"[OK] HTML generado → {out_path}  ({size_kb:.0f} KB)")
    print(f"     Abre en el navegador para evaluar")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test visual de segmentación de pastillas")
    parser.add_argument("--images-dir",  default="",  help="Directorio con imágenes")
    parser.add_argument("--csv",         default="",  help="CSV filtrado (para muestreo estratificado por color)")
    parser.add_argument("--images-base", default="",  help="Directorio base si las rutas del CSV son relativas")
    parser.add_argument("--n",           type=int, default=20, help="Número de imágenes a evaluar")
    parser.add_argument("--out",         default="test_segmentation.html")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if not args.images_dir and not args.csv:
        parser.error("Especifica --images-dir o --csv")

    random.seed(args.seed)

    print(f"[INFO] Seleccionando {args.n} imágenes...")
    paths = sample_images(
        images_dir  = args.images_dir or None,
        csv_path    = args.csv or None,
        n           = args.n,
        images_base = args.images_base,
    )

    if not paths:
        print("[ERROR] No se encontraron imágenes. Revisa --images-dir o --csv / --images-base")
        sys.exit(1)

    print(f"[INFO] Procesando {len(paths)} imágenes...\n")
    results = []
    for i, p in enumerate(paths):
        print(f"  [{i+1:3d}/{len(paths)}]  {Path(p).name}", end="  ")
        r = process_steps(p)
        if r:
            status = "✓" if r["seg_ok"] else "✗ ERROR"
            print(status)
        else:
            print("✗ no se pudo leer")
        results.append(r)

    generate_html([r for r in results if r is not None], args.out)