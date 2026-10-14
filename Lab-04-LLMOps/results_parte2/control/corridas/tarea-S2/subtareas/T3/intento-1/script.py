# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Figura con UNA observación por tienda (16 tiendas) que muestra el signo y la
magnitud de difference = error_a − error_b, con una referencia visible en cero.

Salidas (rutas relativas exactas exigidas por el enunciado):
  - output/store_differences.png
  - output/store_differences.csv
Además escribe resultados.json (contrato de la subtarea).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ rutas
RUTA_DATOS = Path("data/model_errors.csv")
DIR_SALIDA = Path("output")
RUTA_PNG = DIR_SALIDA / "store_differences.png"
RUTA_CSV = DIR_SALIDA / "store_differences.csv"
RUTA_RESULTADOS = Path("resultados.json")
DIR_SALIDA.mkdir(parents=True, exist_ok=True)

COLOR_B = "#2c7fb8"    # difference > 0 -> favorece a B
COLOR_A = "#d95f02"    # difference < 0 -> favorece a A
COLOR_TIE = "#8c8c8c"  # difference = 0 -> empate
COLS = ["store_id", "error_a", "error_b", "difference", "favors"]


# ------------------------------------------------- obtención de la tabla
def _desde_csv(ruta: Path) -> pd.DataFrame:
    df = pd.read_csv(ruta)
    faltan = set(COLS) - set(df.columns)
    if faltan:
        raise ValueError(f"{ruta} carece de columnas: {sorted(faltan)}")
    return df[COLS].copy()


def _desde_resultados_t2(ruta: Path) -> pd.DataFrame:
    with open(ruta, "r", encoding="utf-8") as f:
        res = json.load(f)
    filas = res.get("tabla_pareada")
    if not isinstance(filas, list) or not filas:
        raise ValueError(f"{ruta} no contiene 'tabla_pareada'")
    df = pd.DataFrame(filas)
    faltan = set(COLS) - set(df.columns)
    if faltan:
        raise ValueError(f"{ruta} carece de columnas: {sorted(faltan)}")
    return df[COLS].copy()


def _desde_datos(ruta: Path) -> pd.DataFrame:
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró {ruta}")
    df = pd.read_csv(ruta)
    if list(df.columns) != ["store_id", "error_a", "error_b"]:
        raise ValueError(f"Columnas inesperadas en {ruta}: {list(df.columns)}")
    if df["store_id"].isna().any() or df["store_id"].duplicated().any():
        raise ValueError("store_id debe ser único y no nulo")
    for col in ("error_a", "error_b"):
        v = pd.to_numeric(df[col], errors="coerce")
        if v.isna().any() or not np.isfinite(v.to_numpy(dtype=float)).all() or (v < 0).any():
            raise ValueError(f"Valores inválidos en {col}")
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(df["difference"] > 0, "B",
                            np.where(df["difference"] < 0, "A", "tie"))
    return df[COLS].copy()


def obtener_tabla():
    """Reutiliza la tabla pareada de T2 si está disponible; si no, la recalcula."""
    for c in (Path("entrada/T2/output/store_differences.csv"),
              Path("entrada/T2/store_differences.csv")):
        if c.exists():
            return _desde_csv(c), f"tabla pareada reutilizada de {c}"
    r = Path("entrada/T2/resultados.json")
    if r.exists():
        try:
            return _desde_resultados_t2(r), f"tabla pareada reutilizada de {r}"
        except Exception:
            pass
    return _desde_datos(RUTA_DATOS), f"tabla pareada recalculada desde {RUTA_DATOS}"


# ------------------------------------------------------------------ main
tabla, origen_tabla = obtener_tabla()
tabla = tabla.sort_values("store_id").reset_index(drop=True)

if len(tabla) != 16:
    raise ValueError(f"Se esperaban 16 tiendas; la tabla tiene {len(tabla)}")
if tabla["store_id"].isna().any() or tabla["store_id"].duplicated().any():
    raise ValueError("store_id duplicado o nulo en la tabla pareada")

dif = tabla["difference"].to_numpy(dtype=float)
ids = tabla["store_id"].astype(str).tolist()
media = float(np.mean(dif))

# ------------------------------------------------------------- figura
ypos = np.arange(len(tabla))
colores = [COLOR_B if d > 0 else (COLOR_A if d < 0 else COLOR_TIE) for d in dif]

fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(ypos, dif, color=colores, edgecolor="black", linewidth=0.6, height=0.62)

# Referencia visible en cero: línea negra sólida por encima de las barras
ax.axvline(0, color="black", linewidth=1.8, zorder=3)

# Etiqueta con signo y magnitud en cada barra
paso = 0.03 * max(float(np.max(np.abs(dif))), 1e-12)
for y, d in zip(ypos, dif):
    if d >= 0:
        ax.text(d + paso, y, f"{d:+.2f}", va="center", ha="left", fontsize=9)
    else:
        ax.text(d - paso, y, f"{d:+.2f}", va="center", ha="right", fontsize=9)

ax.set_yticks(ypos)
ax.set_yticklabels(ids)
ax.invert_yaxis()  # S01 arriba, S16 abajo

lim = 1.30 * max(float(np.max(np.abs(dif))), 1e-12)
ax.set_xlim(-lim, lim)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_axisbelow(True)
ax.tick_params(axis="y", length=0)

ax.set_xlabel("difference = error_a − error_b")
ax.set_ylabel("store_id (tienda)")
ax.set_title(
    "Diferencia de error por tienda (una observación por tienda)\n"
    "difference > 0 favorece a B · difference < 0 favorece a A · "
    f"media = {media:+.4f}"
)

manejadores = [
    Patch(facecolor=COLOR_B, edgecolor="black", label="difference > 0 (favorece a B)"),
    Patch(facecolor=COLOR_A, edgecolor="black", label="difference < 0 (favorece a A)"),
    Line2D([0], [0], color="black", linewidth=1.8, label="referencia en cero"),
]
if int(np.sum(dif == 0)) > 0:
    manejadores.insert(2, Patch(facecolor=COLOR_TIE, edgecolor="black",
                                label="difference = 0 (empate)"))
ax.legend(handles=manejadores, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=2, fontsize=9, framealpha=0.95)

fig.savefig(RUTA_PNG, dpi=120, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------- CSV exigido por el enunciado
tabla[COLS].to_csv(RUTA_CSV, index=False)

# ------------------------------------------------- resultados.json (contrato)
n_A = int((tabla["favors"] == "A").sum())
n_B = int((tabla["favors"] == "B").sum())
n_tie = int((tabla["favors"] == "tie").sum())
i_max = int(np.argmax(dif))
i_min = int(np.argmin(dif))

registros = [
    {
        "store_id": str(tabla["store_id"].iloc[i]),
        "error_a": float(tabla["error_a"].iloc[i]),
        "error_b": float(tabla["error_b"].iloc[i]),
        "difference": float(dif[i]),
        "favors": str(tabla["favors"].iloc[i]),
    }
    for i in range(len(tabla))
]

resultados = {
    "subtarea": "T3_figura_diferencias_por_tienda",
    "figura": str(RUTA_PNG),
    "figura_creada": bool(RUTA_PNG.exists() and RUTA_PNG.stat().st_size > 0),
    "figura_bytes": int(RUTA_PNG.stat().st_size) if RUTA_PNG.exists() else 0,
    "csv_tabla_pareada": str(RUTA_CSV),
    "origen_tabla": origen_tabla,
    "n_tiendas": int(len(tabla)),
    "n_tiendas_esperadas_16": bool(len(tabla) == 16),
    "stores_en_figura": ids,
    "elementos_figura": {
        "una_observacion_por_tienda": True,
        "tipo": "barras horizontales: signo = dirección, longitud = |difference|, valor anotado",
        "referencia_cero": "línea vertical negra sólida en x = 0",
        "color_positivo_favorece_B": COLOR_B,
        "color_negativo_favorece_A": COLOR_A,
        "color_empate": COLOR_TIE,
    },
    "conteo_favorece": {"A": n_A, "B": n_B, "tie": n_tie},
    "tiendas_favorece_A": [ids[i] for i in range(len(ids)) if dif[i] < 0],
    "tiendas_favorece_B": [ids[i] for i in range(len(ids)) if dif[i] > 0],
    "tiendas_empate": [ids[i] for i in range(len(ids)) if dif[i] == 0],
    "mayor_diferencia_favorece_B": {
        "store_id": ids[i_max],
        "difference": float(dif[i_max]),
    },
    "mayor_diferencia_favorece_A": {
        "store_id": ids[i_min],
        "difference": float(dif[i_min]),
    },
    "media_difference": media,
    "min_difference": float(np.min(dif)),
    "max_difference": float(np.max(dif)),
    "tabla_pareada": registros,
}

with open(RUTA_RESULTADOS, "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ------------------------------------------------- resumen
print("T3 completada.")
print(f"Figura: {RUTA_PNG} ({resultados['figura_bytes']} bytes, {len(tabla)} tiendas)")
print(f"CSV:    {RUTA_CSV}")
print(f"Origen de la tabla: {origen_tabla}")
print(f"Favorece a A: {n_A} tiendas | Favorece a B: {n_B} tiendas | Empates: {n_tie}")
print(f"Mayor diferencia a favor de B: {ids[i_max]} ({dif[i_max]:+.4f})")
print(f"Mayor diferencia a favor de A: {ids[i_min]} ({dif[i_min]:+.4f})")
print(f"Media de difference: {media:+.6f}")
