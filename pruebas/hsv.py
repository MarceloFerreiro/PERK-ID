"""
analyze_hsv.py
==============
Analiza los valores HSV de fondo (esquinas) y pastilla (centro)
de N imágenes aleatorias para calibrar el umbral de segmentación.

Genera:
  - Resumen en consola con rangos HSV de fondo vs pastilla
  - hsv_report.html  con histogramas visuales y tabla detallada

Uso:
  python hsv.py --images-dir /ruta/imagenes/ --n 100
  python hsv.py --csv dataset_con_imagenes.csv --images-base /ruta/ --n 100
"""

import argparse
import base64
import io
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def sample_paths(images_dir=None, csv_path=None, images_base="", n=100, seed=42):
    random.seed(seed)
    paths = []
    if csv_path:
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        df.columns = df.columns.str.strip()
        base = Path(images_base) if images_base else Path("")
        rutas = df["ruta_imagen"].str.strip()
        rutas = rutas[rutas != ""]
        sample = rutas.sample(min(n, len(rutas)), random_state=seed)
        for r in sample:
            p = base / r if images_base else Path(r)
            if p.exists():
                paths.append(str(p))
    elif images_dir:
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        all_imgs = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts]
        random.shuffle(all_imgs)
        paths = [str(p) for p in all_imgs[:n]]
    return paths


def analyze_image(img_path):
    """
    Devuelve dict con valores HSV de:
      - 4 esquinas + borde medio de cada lado  → fondo probable
      - parche central 20×20                   → pastilla probable
    """
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        return None

    h, w = bgr.shape[:2]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    # Puntos de fondo: esquinas y mitades de bordes (parches 5×5)
    def patch_mean(y, x, size=5):
        y1, y2 = max(0, y - size//2), min(h, y + size//2 + 1)
        x1, x2 = max(0, x - size//2), min(w, x + size//2 + 1)
        return hsv[y1:y2, x1:x2].reshape(-1, 3).mean(axis=0)

    bg_samples = [
        patch_mean(0,      0     ),  # esquina TL
        patch_mean(0,      w-1   ),  # esquina TR
        patch_mean(h-1,    0     ),  # esquina BL
        patch_mean(h-1,    w-1   ),  # esquina BR
        patch_mean(0,      w//2  ),  # borde top centro
        patch_mean(h-1,    w//2  ),  # borde bottom centro
        patch_mean(h//2,   0     ),  # borde left centro
        patch_mean(h//2,   w-1   ),  # borde right centro
    ]
    bg_arr = np.array(bg_samples)   # [8, 3]

    # Centro: parche 20×20
    cy, cx = h // 2, w // 2
    size = 20
    center_patch = hsv[cy-size//2:cy+size//2, cx-size//2:cx+size//2]
    fg_mean = center_patch.reshape(-1, 3).mean(axis=0)

    return {
        "path":    img_path,
        "name":    Path(img_path).name,
        "bg_H":    bg_arr[:, 0].tolist(),
        "bg_S":    bg_arr[:, 1].tolist(),
        "bg_V":    bg_arr[:, 2].tolist(),
        "fg_H":    float(fg_mean[0]),
        "fg_S":    float(fg_mean[1]),
        "fg_V":    float(fg_mean[2]),
        "bg_H_mean": float(bg_arr[:, 0].mean()),
        "bg_S_mean": float(bg_arr[:, 1].mean()),
        "bg_V_mean": float(bg_arr[:, 2].mean()),
        "size":    f"{w}×{h}",
    }


def compute_thresholds(records):
    """
    A partir de todos los registros, calcula los rangos HSV
    recomendados para fondo y pastilla.
    """
    bg_H = np.array([v for r in records for v in r["bg_H"]])
    bg_S = np.array([v for r in records for v in r["bg_S"]])
    bg_V = np.array([v for r in records for v in r["bg_V"]])
    fg_H = np.array([r["fg_H"] for r in records])
    fg_S = np.array([r["fg_S"] for r in records])
    fg_V = np.array([r["fg_V"] for r in records])

    def pct(arr, lo, hi):
        return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))

    return {
        "bg": {
            "H": pct(bg_H, 5, 95),
            "S": pct(bg_S, 5, 95),
            "V": pct(bg_V, 5, 95),
            "H_mean": float(bg_H.mean()),
            "S_mean": float(bg_S.mean()),
            "V_mean": float(bg_V.mean()),
        },
        "fg": {
            "H": pct(fg_H, 5, 95),
            "S": pct(fg_S, 5, 95),
            "V": pct(fg_V, 5, 95),
            "H_mean": float(fg_H.mean()),
            "S_mean": float(fg_S.mean()),
            "V_mean": float(fg_V.mean()),
        },
        "raw": {
            "bg_H": bg_H, "bg_S": bg_S, "bg_V": bg_V,
            "fg_H": fg_H, "fg_S": fg_S, "fg_V": fg_V,
        }
    }


def ascii_histogram(values, lo, hi, bins=20, width=40):
    """Mini histograma ASCII para la consola."""
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    max_c = max(counts) if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(c / max_c * width)
        label = f"{edges[i]:5.0f}-{edges[i+1]:5.0f}"
        lines.append(f"  {label} │{bar} {c}")
    return "\n".join(lines)


def svg_histogram(values, lo, hi, bins=30, color="#4a90d9", title=""):
    """SVG inline de histograma para el HTML."""
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    max_c = max(counts) if counts.max() > 0 else 1
    W, H_svg = 300, 100
    bar_w = W / bins
    bars = ""
    for i, c in enumerate(counts):
        bh = int(c / max_c * (H_svg - 20))
        x  = i * bar_w
        y  = H_svg - 20 - bh
        bars += f'<rect x="{x:.1f}" y="{y}" width="{bar_w-1:.1f}" height="{bh}" fill="{color}" opacity="0.85"/>'

    # Etiquetas eje X
    labels = ""
    for tick in [lo, (lo+hi)//2, hi]:
        x = (tick - lo) / (hi - lo) * W if hi > lo else 0
        labels += f'<text x="{x:.0f}" y="{H_svg-2}" font-size="9" fill="#888" text-anchor="middle">{tick:.0f}</text>'

    return f"""<svg viewBox="0 0 {W} {H_svg}" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:{W}px">
  <text x="2" y="12" font-size="10" fill="#aaa">{title}</text>
  {bars}{labels}
</svg>"""


def generate_html(records, thresholds, out_path):
    bg = thresholds["bg"]
    fg = thresholds["fg"]
    raw = thresholds["raw"]

    # Sugerir umbrales para segment_pill
    # Fondo: usar p10-p90 de S y V del fondo
    bg_S_lo = max(0,   int(np.percentile(raw["bg_S"], 5)  - 5))
    bg_S_hi = min(255, int(np.percentile(raw["bg_S"], 95) + 10))
    bg_V_lo = max(0,   int(np.percentile(raw["bg_V"], 5)  - 10))
    bg_V_hi = 255

    suggested = (
        f"cv2.inRange(hsv, ({int(bg['H'][0])}, {bg_S_lo}, {bg_V_lo}), "
        f"({int(bg['H'][1])}, {bg_S_hi}, {bg_V_hi}))"
    )

    # Histogramas SVG
    hist_bg_H = svg_histogram(raw["bg_H"], 0, 180, color="#e8a838", title="Fondo — H (matiz)")
    hist_bg_S = svg_histogram(raw["bg_S"], 0, 255, color="#e8a838", title="Fondo — S (saturación)")
    hist_bg_V = svg_histogram(raw["bg_V"], 0, 255, color="#e8a838", title="Fondo — V (brillo)")
    hist_fg_H = svg_histogram(raw["fg_H"], 0, 180, color="#4a90d9", title="Centro — H (matiz)")
    hist_fg_S = svg_histogram(raw["fg_S"], 0, 255, color="#4a90d9", title="Centro — S (saturación)")
    hist_fg_V = svg_histogram(raw["fg_V"], 0, 255, color="#4a90d9", title="Centro — V (brillo)")

    # Tabla de registros
    rows = ""
    for r in records:
        rows += f"""<tr>
          <td>{r['name']}</td>
          <td>{r['size']}</td>
          <td>{r['bg_H_mean']:.0f}</td>
          <td>{r['bg_S_mean']:.0f}</td>
          <td>{r['bg_V_mean']:.0f}</td>
          <td>{r['fg_H']:.0f}</td>
          <td>{r['fg_S']:.0f}</td>
          <td>{r['fg_V']:.0f}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>HSV Analysis — Pill Segmentation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f0f13; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.3rem; margin-bottom: 4px; }}
  .sub {{ color: #666; font-size: 0.82rem; margin-bottom: 28px; }}
  h2 {{ font-size: 1rem; color: #7eb8f7; margin: 28px 0 12px; }}
  .grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .box {{ background: #1a1a24; border: 1px solid #2a2a3a; border-radius: 8px; padding: 16px; }}
  .box h3 {{ font-size: 0.85rem; color: #aaa; margin-bottom: 12px; }}
  .stat-row {{ display: flex; gap: 16px; margin-bottom: 8px; }}
  .stat {{ background: #12121a; border-radius: 6px; padding: 8px 12px; flex: 1; }}
  .stat-label {{ font-size: 0.7rem; color: #666; text-transform: uppercase; }}
  .stat-val {{ font-size: 1.1rem; font-weight: 600; color: #fff; }}
  .stat-range {{ font-size: 0.72rem; color: #888; }}
  code {{ background: #12121a; border: 1px solid #2a2a3a; border-radius: 6px;
          padding: 12px 16px; display: block; font-size: 0.82rem;
          color: #7ed49a; white-space: pre-wrap; word-break: break-all; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.78rem; }}
  th {{ background: #12121a; padding: 8px; text-align: left; color: #888;
        border-bottom: 1px solid #2a2a3a; }}
  td {{ padding: 6px 8px; border-bottom: 1px solid #1a1a24; }}
  tr:hover td {{ background: #1e1e2a; }}
  .note {{ background: #1a2a1a; border: 1px solid #2a4a2a; border-radius: 8px;
           padding: 14px; font-size: 0.83rem; color: #9fd49a; margin-top: 16px; }}
</style>
</head>
<body>
<h1>Análisis HSV — Calibración de Segmentación</h1>
<p class="sub">{len(records)} imágenes analizadas · esquinas = fondo probable · centro = pastilla probable</p>

<h2>Distribuciones HSV</h2>
<div class="grid2">
  <div class="box">
    <h3>🟡 Fondo (esquinas de la imagen)</h3>
    {hist_bg_H}{hist_bg_S}{hist_bg_V}
  </div>
  <div class="box">
    <h3>🔵 Centro (pastilla probable)</h3>
    {hist_fg_H}{hist_fg_S}{hist_fg_V}
  </div>
</div>

<h2>Estadísticas p5–p95</h2>
<div class="grid2">
  <div class="box">
    <h3>Fondo</h3>
    <div class="stat-row">
      <div class="stat"><div class="stat-label">H matiz</div>
        <div class="stat-val">{bg['H_mean']:.0f}</div>
        <div class="stat-range">p5={bg['H'][0]:.0f}  p95={bg['H'][1]:.0f}</div></div>
      <div class="stat"><div class="stat-label">S saturación</div>
        <div class="stat-val">{bg['S_mean']:.0f}</div>
        <div class="stat-range">p5={bg['S'][0]:.0f}  p95={bg['S'][1]:.0f}</div></div>
      <div class="stat"><div class="stat-label">V brillo</div>
        <div class="stat-val">{bg['V_mean']:.0f}</div>
        <div class="stat-range">p5={bg['V'][0]:.0f}  p95={bg['V'][1]:.0f}</div></div>
    </div>
  </div>
  <div class="box">
    <h3>Centro (pastilla)</h3>
    <div class="stat-row">
      <div class="stat"><div class="stat-label">H matiz</div>
        <div class="stat-val">{fg['H_mean']:.0f}</div>
        <div class="stat-range">p5={fg['H'][0]:.0f}  p95={fg['H'][1]:.0f}</div></div>
      <div class="stat"><div class="stat-label">S saturación</div>
        <div class="stat-val">{fg['S_mean']:.0f}</div>
        <div class="stat-range">p5={fg['S'][0]:.0f}  p95={fg['S'][1]:.0f}</div></div>
      <div class="stat"><div class="stat-label">V brillo</div>
        <div class="stat-val">{fg['V_mean']:.0f}</div>
        <div class="stat-range">p5={fg['V'][0]:.0f}  p95={fg['V'][1]:.0f}</div></div>
    </div>
  </div>
</div>

<h2>Umbral sugerido para segment_pill()</h2>
<code>mask_bg = {suggested}</code>
<div class="note">
  ⚠️ Este umbral es un punto de partida basado en las esquinas de la imagen.<br>
  Valídalo visualmente con test_segmentation.py y ajusta si es necesario.<br>
  Si el fondo no es uniforme, puede que GrabCut sea mejor opción como paso principal.
</div>

<h2>Detalle por imagen</h2>
<table>
  <thead><tr>
    <th>Archivo</th><th>Tamaño</th>
    <th>Fondo H</th><th>Fondo S</th><th>Fondo V</th>
    <th>Centro H</th><th>Centro S</th><th>Centro V</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML → {out_path}  ({Path(out_path).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir",  default="")
    parser.add_argument("--csv",         default="")
    parser.add_argument("--images-base", default="")
    parser.add_argument("--n",           type=int, default=100)
    parser.add_argument("--out",         default="hsv_report.html")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    if not args.images_dir and not args.csv:
        parser.error("Especifica --images-dir o --csv")

    paths = sample_paths(args.images_dir or None, args.csv or None,
                         args.images_base, args.n, args.seed)
    if not paths:
        print("[ERROR] No se encontraron imágenes")
        sys.exit(1)

    print(f"[INFO] Analizando {len(paths)} imágenes...")
    records = []
    for i, p in enumerate(paths):
        r = analyze_image(p)
        if r:
            records.append(r)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(paths)}")

    thresholds = compute_thresholds(records)
    bg = thresholds["bg"]
    fg = thresholds["fg"]

    # ── Resumen consola
    print(f"\n{'═'*55}")
    print(f"  FONDO (esquinas)         p5      media     p95")
    print(f"  H (matiz 0-180)       {bg['H'][0]:6.0f}   {bg['H_mean']:6.0f}   {bg['H'][1]:6.0f}")
    print(f"  S (saturación 0-255)  {bg['S'][0]:6.0f}   {bg['S_mean']:6.0f}   {bg['S'][1]:6.0f}")
    print(f"  V (brillo 0-255)      {bg['V'][0]:6.0f}   {bg['V_mean']:6.0f}   {bg['V'][1]:6.0f}")
    print(f"\n  CENTRO (pastilla)        p5      media     p95")
    print(f"  H (matiz 0-180)       {fg['H'][0]:6.0f}   {fg['H_mean']:6.0f}   {fg['H'][1]:6.0f}")
    print(f"  S (saturación 0-255)  {fg['S'][0]:6.0f}   {fg['S_mean']:6.0f}   {fg['S'][1]:6.0f}")
    print(f"  V (brillo 0-255)      {fg['V'][0]:6.0f}   {fg['V_mean']:6.0f}   {fg['V'][1]:6.0f}")

    raw = thresholds["raw"]
    bg_S_lo = max(0,   int(np.percentile(raw["bg_S"], 5)  - 5))
    bg_S_hi = min(255, int(np.percentile(raw["bg_S"], 95) + 10))
    bg_V_lo = max(0,   int(np.percentile(raw["bg_V"], 5)  - 10))
    print(f"\n  UMBRAL SUGERIDO:")
    print(f"  cv2.inRange(hsv,")
    print(f"    ({int(bg['H'][0])}, {bg_S_lo}, {bg_V_lo}),")
    print(f"    ({int(bg['H'][1])}, {bg_S_hi}, 255))")
    print(f"{'═'*55}\n")

    generate_html(records, thresholds, args.out)