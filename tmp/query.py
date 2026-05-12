"""
query.py
========
Recibe una imagen de una pastilla, extrae sus features y muestra
las N más parecidas del índice en un HTML.

Muestra por cada resultado:
  - Imagen canónica del dataset
  - Imagen original del dataset
  - Score de similitud
  - Metadatos: color, logo, sustancias, dosis, procedencia, fecha

Uso:
  python query.py \
      --image     mi_pastilla.jpg \
      --index     output/index.npz \
      --meta      output/metadata.json \
      --canonicals canonicals/ \
      --originals  /ruta/imagenes/originales/ \
      --top        5 \
      --out        resultado.html

Filtros opcionales:
      --color  Azul
      --logo   "Chupa Chups"
"""

import argparse
import base64
import json
import sys
import webbrowser
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from preprocesado.extract_features import (
    correct_illumination, segment_pill,
    build_feature_vector, apply_weights,
    FEAT_OFFSETS, DEFAULT_WEIGHTS,
)


# ──────────────────────────────────────────────
# ÍNDICE
# ──────────────────────────────────────────────

class PillIndex:
    def __init__(self, index_path: str, meta_path: str):
        data = np.load(index_path, allow_pickle=False)
        self.ids     = data["ids"]
        self.vectors = data["vectors"].astype(np.float32)
        self.colors  = data["colors"]  if "colors" in data else np.array([""] * len(self.ids))
        self.logos   = data["logos"]   if "logos"  in data else np.array([""] * len(self.ids))
        self.N, self.D = self.vectors.shape

        with open(meta_path, encoding="utf-8") as f:
            self.meta = json.load(f)

        print(f"[Index] {self.N} pastillas × {self.D} dims")

    def search(self, query_vec, top_k=5, color=None, logo=None, weights=None):
        # Filtro por metadatos
        mask = np.ones(self.N, dtype=bool)
        if color:
            mask &= np.array([color.lower() in c.lower() for c in self.colors])
        if logo:
            mask &= np.array([logo.lower()  in l.lower() for l in self.logos])
        cand_idx = np.where(mask)[0]

        if len(cand_idx) == 0:
            return []

        # Similitud coseno ponderada
        q = apply_weights(query_vec, weights)
        db = np.array([apply_weights(self.vectors[i], weights) for i in cand_idx],
                      dtype=np.float32)

        q_norm  = q  / (np.linalg.norm(q) + 1e-10)
        db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-10)
        sims    = db_norm @ q_norm
        order   = np.argsort(-sims)[:top_k]

        results = []
        for rank, local_i in enumerate(order):
            gidx = int(cand_idx[local_i])
            pid  = str(self.ids[gidx])
            results.append({
                "rank":     rank + 1,
                "id":       pid,
                "score":    float(round(sims[local_i], 4)),
                "meta":     self.meta.get(pid, {}),
            })
        return results


# ──────────────────────────────────────────────
# UTILIDADES DE IMAGEN
# ──────────────────────────────────────────────

def img_to_b64(bgr: np.ndarray, quality=88) -> str:
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def load_thumbnail(path: str, max_dim=300) -> str | None:
    """Carga una imagen, la redimensiona y devuelve base64. None si no existe."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    bgr = cv2.imread(str(p))
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
    return img_to_b64(bgr)


def score_color(score: float) -> str:
    if score >= 0.95: return "#4caf50"
    if score >= 0.85: return "#8bc34a"
    if score >= 0.70: return "#ff9800"
    return "#f44336"


# ──────────────────────────────────────────────
# HTML
# ──────────────────────────────────────────────

def generate_html(query_b64, query_canonical_b64, results,
                  canonicals_dir, originals_dir, out_path):

    def _find_image(filename, directory) -> str | None:
        if not directory or not filename:
            return None
        p = Path(directory) / filename
        return str(p) if p.exists() else None

    # ── Construir cards
    cards_html = ""
    for r in results:
        meta     = r["meta"]
        fname    = meta.get("filename", "")
        score    = r["score"]
        pct      = int(score * 100)
        color_css = score_color(score)

        canon_b64 = load_thumbnail(_find_image(fname, canonicals_dir))
        orig_b64  = load_thumbnail(_find_image(fname, originals_dir))

        def img_tag(b64, label):
            if b64:
                return f'''<div class="img-wrap">
                    <div class="img-label">{label}</div>
                    <img src="data:image/jpeg;base64,{b64}">
                  </div>'''
            return f'<div class="img-wrap missing"><div class="img-label">{label}</div><span>no disponible</span></div>'

        # Sustancias
        sus_rows = ""
        for s in meta.get("sustancias", []):
            val = f"{s['valor']} {s['unidad']}" if s['valor'] is not None else "—"
            sus_rows += f"<tr><td>{s['nombre']}</td><td>{val}</td></tr>"
        sus_block = f"""<table class="sus-table">
            <thead><tr><th>Sustancia</th><th>Cantidad</th></tr></thead>
            <tbody>{sus_rows}</tbody>
          </table>""" if sus_rows else '<p class="no-data">Sin datos de sustancias</p>'

        def meta_item(label, value):
            if not value:
                return ""
            return f'<div class="meta-item"><span class="meta-label">{label}</span><span class="meta-val">{value}</span></div>'

        peso    = f"{meta['peso_mg']} mg"      if meta.get("peso_mg")     else None
        diam    = f"{meta['diametro_mm']} mm"  if meta.get("diametro_mm") else None

        cards_html += f"""
  <div class="card">
    <div class="card-header">
      <div class="rank">#{r['rank']}</div>
      <div class="score-wrap">
        <div class="score-bar-bg">
          <div class="score-bar" style="width:{pct}%;background:{color_css}"></div>
        </div>
        <span class="score-num" style="color:{color_css}">{pct}%</span>
      </div>
      <div class="fname">{fname}</div>
    </div>
    <div class="card-body">
      <div class="images">
        {img_tag(canon_b64, "Canónica")}
        {img_tag(orig_b64,  "Original")}
      </div>
      <div class="info">
        <div class="meta-grid">
          {meta_item("Color",       meta.get("color"))}
          {meta_item("Logo",        meta.get("logo"))}
          {meta_item("Procedencia", meta.get("procedencia"))}
          {meta_item("Fecha",       meta.get("fecha"))}
          {meta_item("Peso",        peso)}
          {meta_item("Diámetro",    diam)}
        </div>
        <div class="sus-wrap">
          <div class="sus-title">Composición analizada</div>
          {sus_block}
        </div>
      </div>
    </div>
  </div>"""

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Búsqueda de Pastillas — EnergyControl</title>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{font-family:system-ui,sans-serif;background:#0f0f13;color:#e0e0e0;padding:24px}}
  h1{{font-size:1.3rem;margin-bottom:4px;color:#fff}}
  .sub{{font-size:.82rem;color:#666;margin-bottom:28px}}

  .query-box{{background:#1a1a24;border:1px solid #2a2a3a;border-radius:10px;
              padding:16px;margin-bottom:32px;display:flex;gap:20px;align-items:flex-start}}
  .query-box .img-pair{{display:flex;gap:12px}}
  .query-box img{{width:160px;border-radius:6px;display:block}}
  .query-box .q-label{{font-size:.72rem;color:#666;text-transform:uppercase;
                        letter-spacing:.05em;margin-bottom:6px}}
  .query-box .q-info{{font-size:.82rem;color:#aaa;line-height:1.8}}
  .query-box .q-info b{{color:#7eb8f7}}

  .results-title{{font-size:.9rem;color:#888;margin-bottom:16px;
                  text-transform:uppercase;letter-spacing:.08em}}

  .card{{background:#1a1a24;border:1px solid #2a2a3a;border-radius:10px;
         margin-bottom:16px;overflow:hidden}}
  .card-header{{background:#12121a;padding:10px 16px;
                display:flex;align-items:center;gap:14px}}
  .rank{{font-size:1.1rem;font-weight:700;color:#7eb8f7;min-width:28px}}
  .score-wrap{{display:flex;align-items:center;gap:10px;flex:1}}
  .score-bar-bg{{flex:1;height:6px;background:#2a2a3a;border-radius:3px;max-width:200px}}
  .score-bar{{height:6px;border-radius:3px;transition:width .3s}}
  .score-num{{font-size:.9rem;font-weight:700;min-width:40px}}
  .fname{{font-size:.75rem;color:#555;font-family:monospace;margin-left:auto}}

  .card-body{{display:flex;gap:0;flex-wrap:wrap}}
  .images{{display:flex;gap:1px;background:#0f0f13;padding:0}}
  .img-wrap{{padding:10px;background:#1a1a24;flex:1;min-width:140px}}
  .img-wrap img{{width:100%;border-radius:4px;display:block}}
  .img-wrap.missing{{display:flex;flex-direction:column;align-items:center;
                     justify-content:center;color:#444;font-size:.75rem;min-height:100px}}
  .img-label{{font-size:.68rem;color:#555;text-transform:uppercase;
              letter-spacing:.05em;margin-bottom:6px}}

  .info{{flex:1;padding:14px 16px;min-width:260px}}
  .meta-grid{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px}}
  .meta-item{{background:#12121a;border-radius:6px;padding:6px 10px}}
  .meta-label{{font-size:.68rem;color:#555;text-transform:uppercase;
               letter-spacing:.04em;display:block}}
  .meta-val{{font-size:.85rem;color:#ddd;font-weight:500}}

  .sus-title{{font-size:.72rem;color:#666;text-transform:uppercase;
              letter-spacing:.05em;margin-bottom:8px}}
  .sus-table{{width:100%;border-collapse:collapse;font-size:.82rem}}
  .sus-table th{{color:#666;font-weight:500;text-align:left;padding:4px 8px;
                 border-bottom:1px solid #2a2a3a;font-size:.72rem;text-transform:uppercase}}
  .sus-table td{{padding:5px 8px;border-bottom:1px solid #1e1e28;color:#ccc}}
  .sus-table tr:last-child td{{border-bottom:none}}
  .no-data{{font-size:.78rem;color:#444;font-style:italic}}
</style>
</head>
<body>
<h1>Búsqueda de Pastillas Similares</h1>
<p class="sub">EnergyControl Pill Indexer — resultados por similitud de features visuales</p>

<div class="query-box">
  <div>
    <div class="q-label">Imagen de consulta</div>
    <div class="img-pair">
      <div>
        <div class="q-label">Original</div>
        <img src="data:image/jpeg;base64,{query_b64}">
      </div>
      <div>
        <div class="q-label">Canónica generada</div>
        <img src="data:image/jpeg;base64,{query_canonical_b64}">
      </div>
    </div>
  </div>
</div>

<div class="results-title">Top {len(results)} resultados más similares</div>

{cards_html}
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[OK] HTML → {out_path}")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Busca pastillas similares y genera HTML")
    parser.add_argument("--image",       required=True,  help="Imagen de consulta")
    parser.add_argument("--index",       required=True,  help="output/index.npz")
    parser.add_argument("--meta",        required=True,  help="output/metadata.json")
    parser.add_argument("--canonicals",  required=True,  help="Directorio de canónicas del dataset")
    parser.add_argument("--originals",   default="",     help="Directorio de imágenes originales del dataset")
    parser.add_argument("--top",         type=int, default=5)
    parser.add_argument("--color",       default=None,   help="Filtro por color (opcional)")
    parser.add_argument("--logo",        default=None,   help="Filtro por logo (opcional)")
    parser.add_argument("--out",         default="resultado.html")
    parser.add_argument("--no-browser",  action="store_true", help="No abrir el navegador automáticamente")
    args = parser.parse_args()

    # ── Cargar índice
    index = PillIndex(args.index, args.meta)

    # ── Procesar imagen de consulta
    print(f"[Query] Procesando {args.image} ...")
    bgr_orig = cv2.imread(args.image)
    if bgr_orig is None:
        print(f"[ERROR] No se pudo leer {args.image}")
        sys.exit(1)

    bgr_proc         = correct_illumination(bgr_orig)
    _, canonical     = segment_pill(bgr_proc)
    query_vec, qmeta = build_feature_vector(canonical)

    print(f"  pHash : {qmeta['phash']}")
    print(f"  Shape : {qmeta['shape_meta']}")

    # Thumbnails de la consulta
    query_b64           = img_to_b64(cv2.resize(bgr_orig, (300, 300) if max(bgr_orig.shape[:2]) > 300 else bgr_orig.shape[1::-1]))
    query_canonical_b64 = img_to_b64(canonical)

    # ── Buscar
    results = index.search(
        query_vec,
        top_k  = args.top,
        color  = args.color,
        logo   = args.logo,
    )

    if not results:
        print("[WARN] Sin resultados (¿filtros demasiado restrictivos?)")
        sys.exit(0)

    print(f"\n  Top {len(results)} resultados:")
    for r in results:
        m = r["meta"]
        print(f"  #{r['rank']}  {m.get('filename',''):<40}  "
              f"score={r['score']:.4f}  "
              f"color={m.get('color','?')}  logo={m.get('logo','?')}")

    # ── Generar HTML
    generate_html(
        query_b64, query_canonical_b64,
        results,
        canonicals_dir = args.canonicals,
        originals_dir  = args.originals,
        out_path       = args.out,
    )

    if not args.no_browser:
        webbrowser.open(str(Path(args.out).resolve()))