# -*- coding: utf-8 -*-
"""
T3 - Identificar observaciones para revisión.

Crea una COPIA del DataFrame original (data/server_measurements.csv, timestamp con
parse dates) y le agrega:
  * load_score      -> indicador calculado en T2 (entrada/T2/load_score.npy)
  * requires_review -> bool, definido vectorizadamente:
                       (load_score > 1.5) | (temperature_c > 80)
Muestra la fila marcada para revisión, verifica que el original queda intacto y
guarda el DataFrame enriquecido en output/server_analysis.parquet.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA_C = 80.0
PESOS_DEFAULT = np.array([0.50, 0.30, 0.20])

# ------------------------------------------------------------------ 1) DataFrame original
df_original = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
columnas_originales = df_original.columns.tolist()
shape_original = [int(v) for v in df_original.shape]
snapshot_original = df_original.copy(deep=True)  # testigo para verificar integridad

# ------------------------------------------------------------------ 2) load_score proveniente de T2
res_t2 = None
if (RUTA_T2 / "resultados.json").exists():
    res_t2 = json.loads((RUTA_T2 / "resultados.json").read_text(encoding="utf-8"))

feature_names = ["gpu_utilization", "cpu_utilization", "memory_gb"]
load_score_recalc = None
if res_t2 is not None:
    feature_names = list(res_t2.get("feature_names", feature_names))
    means = np.asarray(res_t2["means"], dtype=float)
    stds = np.asarray(res_t2.get("stds_usados_en_division", res_t2.get("stds")), dtype=float)
    pesos = np.asarray(res_t2["pesos"], dtype=float)
    X = df_original[feature_names].to_numpy(dtype=float)
    zscores_recalc = (X - means) / stds
    load_score_recalc = zscores_recalc @ pesos

if (RUTA_T2 / "load_score.npy").exists():
    load_score = np.asarray(np.load(RUTA_T2 / "load_score.npy"), dtype=float).ravel()
elif load_score_recalc is not None:
    load_score = np.asarray(load_score_recalc, dtype=float).ravel()
else:
    X = df_original[feature_names].to_numpy(dtype=float)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds_seg = np.where(stds == 0, 1.0, stds)  # evitar división por cero
    load_score = ((X - means) / stds_seg) @ PESOS_DEFAULT

assert len(load_score) == len(df_original), "load_score no coincide con el nº de filas"
load_score_consistente_t2 = bool(
    load_score_recalc is not None
    and np.allclose(load_score, np.asarray(load_score_recalc, dtype=float).ravel(),
                    rtol=1e-12, atol=1e-12)
)

# ------------------------------------------------------------------ 3) Nuevo DataFrame (copia del original)
df_nuevo = df_original.copy()  # copia: el original NO se modifica
df_nuevo["load_score"] = load_score
# Definición vectorizada de requires_review (sin bucles sobre observaciones)
df_nuevo["requires_review"] = (
    (df_nuevo["load_score"] > UMBRAL_LOAD_SCORE)
    | (df_nuevo["temperature_c"] > UMBRAL_TEMPERATURA_C)
).astype(bool)

# ------------------------------------------------------------------ 4) Verificaciones
original_intacto = bool(
    df_original.columns.tolist() == columnas_originales
    and [int(v) for v in df_original.shape] == shape_original
    and df_original.equals(snapshot_original)
)
requires_review_es_bool = bool(df_nuevo["requires_review"].dtype == bool)

fila_23 = df_nuevo.loc[23]
ts_23 = pd.Timestamp(fila_23["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
fila23_coincide = bool(
    str(fila_23["server"]) == "AI-SRV-01"
    and ts_23 == "2026-08-24 09:55:00"
    and abs(float(fila_23["load_score"]) - 3.136218) < 5e-7
    and abs(float(fila_23["temperature_c"]) - 66.97) < 1e-6
    and bool(fila_23["requires_review"])
)

# ------------------------------------------------------------------ 5) Fila(s) marcadas para revisión
cols_show = ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
marcadas = df_nuevo.loc[df_nuevo["requires_review"], cols_show]


def fila_a_dict(fila) -> dict:
    return {
        "indice": int(fila.name),
        "server": str(fila["server"]),
        "timestamp": pd.Timestamp(fila["timestamp"]).strftime("%Y-%m-%d %H:%M:%S"),
        "load_score": float(fila["load_score"]),
        "temperature_c": float(fila["temperature_c"]),
        "requires_review": bool(fila["requires_review"]),
    }


filas_marcadas = [fila_a_dict(f) for _, f in marcadas.iterrows()]

mask_load = df_nuevo["load_score"] > UMBRAL_LOAD_SCORE
mask_temp = df_nuevo["temperature_c"] > UMBRAL_TEMPERATURA_C
n_marcadas = int(df_nuevo["requires_review"].sum())
n_load = int(mask_load.sum())
n_temp = int(mask_temp.sum())
n_ambas = int((mask_load & mask_temp).sum())

# ------------------------------------------------------------------ 6) Parquet exigido por el enunciado
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_nuevo.to_parquet(RUTA_PARQUET)

# ------------------------------------------------------------------ 7) resultados.json (contrato)
resultados = {
    "subtarea": "T3_copia_con_load_score_y_requires_review",
    "n_filas": int(shape_original[0]),
    "columnas_originales": columnas_originales,
    "columnas_nuevo_dataframe": df_nuevo.columns.tolist(),
    "columnas_agregadas": ["load_score", "requires_review"],
    "load_score_dtype": str(df_nuevo["load_score"].dtype),
    "requires_review_dtype": str(df_nuevo["requires_review"].dtype),
    "umbral_load_score": UMBRAL_LOAD_SCORE,
    "umbral_temperature_c": UMBRAL_TEMPERATURA_C,
    "n_requires_review_true": n_marcadas,
    "n_load_score_mayor_1_5": n_load,
    "n_temperature_mayor_80": n_temp,
    "n_ambas_condiciones": n_ambas,
    "fila_23": fila_a_dict(fila_23),
    "filas_marcadas": filas_marcadas,
    "load_score_consistente_con_T2": load_score_consistente_t2,
    "verificacion": {
        "columna_load_score_presente": bool("load_score" in df_nuevo.columns),
        "columna_requires_review_presente": bool("requires_review" in df_nuevo.columns),
        "requires_review_es_bool": requires_review_es_bool,
        "original_sin_modificar": original_intacto,
        "definicion_vectorizada_sin_bucles": True,
        "fila_23_coincide_esperado": fila23_coincide,
    },
    "archivo_parquet": str(RUTA_PARQUET),
}
RUTA_RESULTADOS.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                           encoding="utf-8")

# ------------------------------------------------------------------ 8) Resumen
print("== T3: copia del DataFrame con load_score y requires_review ==")
print(f"Filas: {len(df_nuevo)} | Columnas agregadas: load_score, requires_review "
      f"(dtype={df_nuevo['requires_review'].dtype})")
print(f"Original intacto: {original_intacto} | "
      f"load_score consistente con T2: {load_score_consistente_t2}")
print(f"Marcadas para revisión: {n_marcadas} "
      f"(load_score>1.5: {n_load} | temperature_c>80: {n_temp} | ambas: {n_ambas})")
print("\nFila(s) marcada(s) para revisión:")
print(marcadas.to_string())
print("\nFila 23 (resultado esperado por el enunciado):")
print(fila_23[cols_show].to_string())
print(f"\nVerificación fila 23: {fila23_coincide}")
print(f"Parquet escrito en: {RUTA_PARQUET}")
print(f"resultados.json escrito en: {RUTA_RESULTADOS}")
