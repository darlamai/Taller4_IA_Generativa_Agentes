from pathlib import Path
import json

import numpy as np
import pandas as pd

# --- Rutas y constantes (solo rutas relativas) ---
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20])
UMBRAL_LOAD = 1.5
UMBRAL_TEMP = 80.0

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# --- 1) DataFrame original (no se modifica en ningún momento) ---
df_original = pd.read_csv(RUTA_CSV, parse_dates=["timestamp"])
snapshot_original = df_original.copy(deep=True)
columnas_originales = list(df_original.columns)

# --- 2) load_score proveniente de T2 ---
load_score = None
fuente_load_score = None
if RUTA_T2.exists():
    with RUTA_T2.open("r", encoding="utf-8") as f:
        t2 = json.load(f)
    ls = t2.get("load_score")
    if ls is not None and len(ls) == len(df_original):
        load_score = np.asarray(ls, dtype=float)
        fuente_load_score = str(RUTA_T2)

# Respaldo: recálculo vectorizado con el mismo método de T2 (sin bucles)
X = df_original[FEATURES].to_numpy(dtype=float)
medias = X.mean(axis=0)
desvs = X.std(axis=0)
desvs_seguras = np.where(desvs == 0.0, 1.0, desvs)  # evita división por cero
zscores = (X - medias) / desvs_seguras
load_score_recalc = zscores @ PESOS
if load_score is None:
    load_score = load_score_recalc
    fuente_load_score = "recalculado con el metodo de T2"
dif_recalculo = float(np.max(np.abs(load_score - load_score_recalc)))

# --- 3) Nuevo DataFrame (copia del original) + columnas nuevas ---
df_nuevo = df_original.copy()
df_nuevo["load_score"] = load_score
# Definición vectorizada del criterio de revisión
df_nuevo["requires_review"] = (df_nuevo["load_score"] > UMBRAL_LOAD) | (
    df_nuevo["temperature_c"] > UMBRAL_TEMP
)
df_nuevo["requires_review"] = df_nuevo["requires_review"].astype(bool)

# --- 4) Verificaciones ---
original_intacto = bool(
    df_original.equals(snapshot_original) and list(df_original.columns) == columnas_originales
)
requires_review_es_bool = bool(pd.api.types.is_bool_dtype(df_nuevo["requires_review"]))
criterio_verificado = bool(
    (
        df_nuevo["requires_review"]
        == ((df_nuevo["load_score"] > UMBRAL_LOAD) | (df_nuevo["temperature_c"] > UMBRAL_TEMP))
    ).all()
)

df_revision = df_nuevo.loc[df_nuevo["requires_review"]]

# --- 5) Mostrar la fila marcada para revisión (exactamente una fila) ---
columnas_muestra = ["server", "timestamp", "load_score", "temperature_c", "requires_review"]
print("Fila(s) marcada(s) para revision (requires_review == True):")
print(df_revision[columnas_muestra])

# --- 6) Guardar el DataFrame analizado (archivo exigido por el enunciado) ---
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_nuevo.to_parquet(RUTA_PARQUET, index=False)

# --- 7) resultados.json (contrato de la subtarea) ---
if len(df_revision) > 0:
    f0 = df_revision.iloc[0]
    fila_revision = {
        "indice": int(df_revision.index[0]),
        "server": str(f0["server"]),
        "timestamp": str(f0["timestamp"]),
        "load_score": float(f0["load_score"]),
        "temperature_c": float(f0["temperature_c"]),
        "requires_review": bool(f0["requires_review"]),
    }
    coincide_enunciado = bool(
        fila_revision["indice"] == 23
        and fila_revision["server"] == "AI-SRV-01"
        and fila_revision["timestamp"] == "2026-08-24 09:55:00"
        and abs(fila_revision["load_score"] - 3.136218) < 1e-5
        and abs(fila_revision["temperature_c"] - 66.97) < 1e-9
        and fila_revision["requires_review"] is True
    )
else:
    fila_revision = None
    coincide_enunciado = False

resultados = {
    "subtarea": "T3",
    "archivo_datos": str(RUTA_CSV),
    "n_filas": int(len(df_nuevo)),
    "columnas_originales": columnas_originales,
    "columnas_dataframe_nuevo": list(df_nuevo.columns),
    "columnas_agregadas": ["load_score", "requires_review"],
    "requires_review_dtype": str(df_nuevo["requires_review"].dtype),
    "requires_review_es_bool": requires_review_es_bool,
    "original_intacto": original_intacto,
    "fuente_load_score": fuente_load_score,
    "criterio": "load_score > 1.5 o temperature_c > 80",
    "criterio_verificado": criterio_verificado,
    "n_requieren_revision": int(df_nuevo["requires_review"].sum()),
    "indices_requieren_revision": [int(i) for i in df_revision.index.tolist()],
    "fila_revision": fila_revision,
    "fila_revision_coincide_con_enunciado": coincide_enunciado,
    "load_score_resumen": {
        "min": float(load_score.min()),
        "max": float(load_score.max()),
        "media": float(load_score.mean()),
        "desviacion": float(load_score.std()),
    },
    "verificacion_recalculo_load_score_max_abs_diff": dif_recalculo,
    "parquet_escrito": str(RUTA_PARQUET),
}

with RUTA_RESULTADOS.open("w", encoding="utf-8") as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)

# --- 8) Resumen breve ---
print("-" * 70)
print(f"Filas: {len(df_nuevo)} | Columnas agregadas: load_score, requires_review")
print(f"requires_review dtype: {df_nuevo['requires_review'].dtype} | es bool: {requires_review_es_bool}")
print(f"Original intacto: {original_intacto} | Criterio verificado: {criterio_verificado}")
print(
    f"Filas que requieren revision: {int(df_nuevo['requires_review'].sum())} "
    f"| indices: {resultados['indices_requieren_revision']}"
)
print(f"load_score (T2) vs recalculo vectorizado: max|diff| = {dif_recalculo:.3e}")
print(f"Parquet escrito en: {RUTA_PARQUET} | Resultados en: {RUTA_RESULTADOS}")
