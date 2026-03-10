"""
EDA inicial del dataset de pastillas de energycontrol.org
Lee pastillas_energycontrol.csv y genera gráficas + resumen estadístico.

Uso:
  pip install pandas matplotlib seaborn
  python eda_pastillas.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
CSV      = "pastillas_energycontrol.csv"
OUT_DIR  = Path("eda_plots")
OUT_DIR.mkdir(exist_ok=True)

sns.set_theme(style="darkgrid", palette="muted")
plt.rcParams["figure.dpi"] = 130

# ── Carga ─────────────────────────────────────────────────────────────────────
df = pd.read_csv(CSV, encoding="utf-8-sig")
df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
df["year"]  = df["fecha"].dt.year.fillna(df["year"]).astype(int)

print(f"=== Dataset cargado: {len(df)} filas × {len(df.columns)} columnas ===\n")
print(df.dtypes)
print("\nNulos por columna:")
print(df.isnull().sum())


# ── 1. Pastillas por año ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
año_counts = df.groupby("year").size().reset_index(name="total")
con_img    = df[df["imagen"].notna()].groupby("year").size().reset_index(name="con_imagen")
año_merge  = año_counts.merge(con_img, on="year", how="left").fillna(0)

ax.bar(año_merge["year"], año_merge["total"],      label="Total",      color="#4C72B0", alpha=0.85)
ax.bar(año_merge["year"], año_merge["con_imagen"], label="Con imagen", color="#55A868", alpha=0.85)
ax.set_title("Pastillas analizadas por año", fontsize=14, fontweight="bold")
ax.set_xlabel("Año"); ax.set_ylabel("Nº pastillas")
ax.legend(); ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "01_pastillas_por_año.png")
plt.close()
print("\n01_pastillas_por_año.png")


# ── 2. Pastillas con imagen trasera ──────────────────────────────────────────
tiene_rear = df["imagen_rear"].notna().sum()
no_rear    = df["imagen_rear"].isna().sum()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie([tiene_rear, no_rear],
       labels=["Con imagen trasera", "Sin imagen trasera"],
       autopct="%1.1f%%", colors=["#55A868", "#C44E52"],
       startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("Cobertura de imagen trasera", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT_DIR / "02_imagen_trasera.png")
plt.close()
print("02_imagen_trasera.png")


# ── 3. Distribución de sustancias ─────────────────────────────────────────────
# Extraer todas las sustancias de las columnas sustancia_N
sust_cols = [c for c in df.columns if c.startswith("sustancia_")]
sustancias = pd.concat([df[c] for c in sust_cols]).dropna()
sustancias = sustancias[sustancias.str.strip() != ""]
top_sust = sustancias.value_counts().head(15)

fig, ax = plt.subplots(figsize=(10, 6))
top_sust.sort_values().plot(kind="barh", ax=ax, color="#4C72B0", edgecolor="white")
ax.set_title("Top 15 sustancias detectadas", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº apariciones")
plt.tight_layout()
plt.savefig(OUT_DIR / "03_top_sustancias.png")
plt.close()
print("03_top_sustancias.png")


# ── 4. Dosis de MDMA (mg) ────────────────────────────────────────────────────
# Buscar filas donde sustancia_N == "MDMA" y coger su valor
mdma_vals = []
for i in range(1, 6):
    sc = f"sustancia_{i}"
    vc = f"valor_{i}_mg"
    if sc in df.columns and vc in df.columns:
        mask = df[sc].str.upper().str.strip() == "MDMA"
        vals = pd.to_numeric(df.loc[mask, vc], errors="coerce").dropna()
        mdma_vals.extend(vals.tolist())

mdma_series = pd.Series(mdma_vals)
mdma_series = mdma_series[(mdma_series > 0) & (mdma_series < 500)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(mdma_series, bins=40, color="#C44E52", edgecolor="white", alpha=0.85)
ax.axvline(mdma_series.median(), color="gold", linestyle="--", linewidth=2,
           label=f"Mediana: {mdma_series.median():.0f} mg")
ax.axvline(mdma_series.mean(),   color="white", linestyle=":",  linewidth=2,
           label=f"Media:   {mdma_series.mean():.0f} mg")
ax.set_title("Distribución de dosis de MDMA (mg)", fontsize=14, fontweight="bold")
ax.set_xlabel("mg de MDMA"); ax.set_ylabel("Nº pastillas")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "04_dosis_mdma.png")
plt.close()
print("04_dosis_mdma.png")


# ── 5. Top colores ────────────────────────────────────────────────────────────
top_colores = df["color"].value_counts().head(12)
fig, ax = plt.subplots(figsize=(10, 5))
top_colores.sort_values().plot(kind="barh", ax=ax, color="#8172B2", edgecolor="white")
ax.set_title("Top 12 colores de pastillas", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº pastillas")
plt.tight_layout()
plt.savefig(OUT_DIR / "05_colores.png")
plt.close()
print("05_colores.png")


# ── 6. Top procedencias ───────────────────────────────────────────────────────
top_proc = df["procedencia"].value_counts().head(15)
fig, ax = plt.subplots(figsize=(10, 6))
top_proc.sort_values().plot(kind="barh", ax=ax, color="#64B5CD", edgecolor="white")
ax.set_title("Top 15 procedencias", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº pastillas")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_procedencias.png")
plt.close()
print("06_procedencias.png")


# ── 7. Evolución temporal MDMA vs 2C-B ───────────────────────────────────────
def contar_sustancia_por_año(nombre):
    rows = []
    for i in range(1, 6):
        sc = f"sustancia_{i}"
        if sc in df.columns:
            mask = df[sc].str.upper().str.strip() == nombre.upper()
            rows.append(df.loc[mask, "year"])
    if rows:
        years = pd.concat(rows)
        return years.value_counts().sort_index()
    return pd.Series()

mdma_año  = contar_sustancia_por_año("MDMA")
twocb_año = contar_sustancia_por_año("2C-B")

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(mdma_año.index,  mdma_año.values,  marker="o", label="MDMA",  color="#C44E52", linewidth=2)
ax.plot(twocb_año.index, twocb_año.values, marker="s", label="2C-B",  color="#4C72B0", linewidth=2)
ax.set_title("Evolución temporal MDMA vs 2C-B", fontsize=14, fontweight="bold")
ax.set_xlabel("Año"); ax.set_ylabel("Nº detecciones")
ax.legend(); ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "07_evolucion_mdma_2cb.png")
plt.close()
print("07_evolucion_mdma_2cb.png")


# ── 8. Peso de pastilla (mg) ─────────────────────────────────────────────────
pesos = pd.to_numeric(df["peso_mg"], errors="coerce").dropna()
pesos = pesos[(pesos > 0) & (pesos < 1000)]

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(pesos, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
ax.axvline(pesos.median(), color="gold",  linestyle="--", linewidth=2,
           label=f"Mediana: {pesos.median():.0f} mg")
ax.set_title("Distribución de peso de pastilla (mg)", fontsize=14, fontweight="bold")
ax.set_xlabel("Peso (mg)"); ax.set_ylabel("Nº pastillas")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "08_peso_pastilla.png")
plt.close()
print("08_peso_pastilla.png")
# ── 9. Top 20 logos más frecuentes ───────────────────────────────────────────
top_logos = df["logo"].value_counts().head(20)

fig, ax = plt.subplots(figsize=(10, 7))
top_logos.sort_values().plot(kind="barh", ax=ax, color="#DD8452", edgecolor="white")
ax.set_title("Top 20 logos más frecuentes", fontsize=14, fontweight="bold")
ax.set_xlabel("Nº pastillas")
plt.tight_layout()
plt.savefig(OUT_DIR / "09_top_logos.png")
plt.close()
print(" 09_top_logos.png")


# ── 10. Logos únicos por año (diversidad) ─────────────────────────────────────
logos_año = df.groupby("year")["logo"].nunique().reset_index()
logos_año.columns = ["year", "logos_unicos"]

fig, ax = plt.subplots(figsize=(13, 5))
ax.bar(logos_año["year"], logos_año["logos_unicos"], color="#DD8452", edgecolor="white", alpha=0.85)
ax.set_title("Nº de logos únicos por año", fontsize=14, fontweight="bold")
ax.set_xlabel("Año"); ax.set_ylabel("Logos distintos")
ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(OUT_DIR / "10_logos_unicos_por_año.png")
plt.close()
print(" 10_logos_unicos_por_año.png")

# ── Resumen estadístico ───────────────────────────────────────────────────────
print("\n" + "="*50)
print("RESUMEN ESTADÍSTICO")
print("="*50)
print(f"Total pastillas:          {len(df):>7,}")
print(f"Con imagen frontal:       {df['imagen'].notna().sum():>7,}")
print(f"Con imagen trasera:       {df['imagen_rear'].notna().sum():>7,}")
print(f"Años cubiertos:           {df['year'].min():.0f} – {df['year'].max():.0f}")
print(f"Sustancia más frecuente:  {sustancias.value_counts().index[0]}")
print(f"Color más frecuente:      {df['color'].value_counts().index[0]}")
print(f"Procedencia más frecuente:{df['procedencia'].value_counts().index[0]}")
if len(mdma_series) > 0:
    print(f"MDMA mediana:             {mdma_series.median():.0f} mg")
    print(f"MDMA máximo registrado:   {mdma_series.max():.0f} mg")
print(f"\nGráficas guardadas en: {OUT_DIR}/")