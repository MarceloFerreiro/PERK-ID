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
CSV      = "pastillas_con_imagen.csv"
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

df[['logo', 'ruta_imagen']].sort_values(
    by='logo',
    key=lambda col: col.map(df['logo'].value_counts()),
    ascending=False
).to_csv('salida.txt', index=False, header=False, sep='\t')
