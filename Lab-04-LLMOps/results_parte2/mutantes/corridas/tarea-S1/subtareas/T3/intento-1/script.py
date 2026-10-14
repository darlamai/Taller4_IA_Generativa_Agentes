# -*- coding: utf-8 -*-
"""
Subtarea T3 — Crear una copia del DataFrame original (sin modificarlo) y agregar:
  * load_score      : vector proveniente de la subtarea T2 (zscores @ [0.50, 0.30, 0.20])
  * requires_review : bool, definida vectorizadamente como
                      (load_score > 1.5) | (temperature_c > 80)

Entradas : data/server_measurements.csv, entrada/T2/resultados.json
Salidas  : output/server_analysis.parquet (sin índice), resultados.json
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_CSV = Path("data/server_measurements.csv")
RUTA_T2 = Path("entrada/T2/resultados.json")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULT = Path("resultados.json")

PESOS = np.array([0.50, 0.30, 0.20])

# ------------------------------------------------------------------ 1) datos
df_original = pd.read_csv(RUTA_CSV)
columnas_original = list(df_original.columns)
n_filas_original = int(len(df_original))

# timestamp: leer con parse de fechas
col_time = next((c for c in df_original.columns if c.strip().lower() == "timestamp"),
                next((c for c in df_original.columns if "time" in c.lower()), None))
if col_time is not None:
    df_original[col_time] = pd.to_datetime(df_original[col_time])

col_temp = next(c for c in df_original.columns if "temperature" in c.lower())
col_server = next((c for c in df_original.columns if c.strip().lower() == "server"),
                  df_original.columns[0])

# --------------------------------------------------- 2) load_score desde T2
fuente_load_score = str(RUTA_T2)
t2_data = None
load_score = None
if RUTA_T2.exists():
    try:
        t2_data = json.loads(RUTA_T2.read_text(encoding="utf-8"))
        cand = np.asarray(t2_data.get("load_score"), dtype=float)
        if cand.ndim == 1 and cand.size == n_filas_original:
            load_score = cand
    except Exception:
        t2_data, load_score = None, None

if load_score is None:  # respaldo: recalcular según la definición de T2
    fuente_load_score = "recalculado en T3 (respaldo; salida de T2 no disponible)"
    feats = [c for c in df_original.columns
             if any(k in c.lower() for k in ("gpu", "cpu", "memory"))][:3]
    if len(feats) < 3:
        feats = [c for c in df_original.columns
                 if pd.api.types.is_numeric_dtype(df_original[c])][:3]
    X = df_original[feats].to_numpy(dtype=float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd_safe = np.where(sd == 0.0, 1.0, sd)
    load_score = ((X - mu) / sd_safe) @ PESOS
    t2_data = {"means": mu.tolist(), "stds": sd.tolist(), "pesos": PESOS.tolist()}


# verificación cruzada: recalcular load_score con medias/desv. reportadas por T2
def _recalcular_load_score(df, datos):
    try:
        means = np.asarray(datos["means"], dtype=float)
        stds = np.asarray(datos["stds"], dtype=float)
        pesos = np.asarray(datos.get("pesos", PESOS.tolist()), dtype=float)
        numericas = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        medias = {c: float(df[c].mean()) for c in numericas}
        cols = []
        for m in means:
            cand = [c for c in numericas if c not in cols and abs(medias[c] - m) < 1e-6]
            if not cand:
                return None
            cols.append(cand[0])
        X = df[cols].to_numpy(dtype=float)
        sd_safe = np.where(stds == 0.0, 1.0, stds)  # evitar división por cero
        return ((X - means) / sd_safe) @ pesos
    except Exception:
        return None


recalc = _recalcular_load_score(df_original, t2_data)
load_score_recalc_coincide = None if recalc is None else bool(
    np.allclose(recalc, load_score, rtol=0.0, atol=1e-9))

# --------------------------------------- 3) copia del original + columnas nuevas
df_new = df_original.copy()                        # el original NO se modifica
df_new["load_score"] = load_score                  # float64, proveniente de T2
df_new["requires_review"] = (df_new["load_score"] > 1.5) | (df_new[col_temp] > 80)
df_new["requires_review"] = df_new["requires_review"].astype(bool)  # tipo bool

estructura_ok = bool(
    list(df_new.columns) == columnas_original + ["load_score", "requires_review"]
    and len(df_new) == n_filas_original
)
original_sin_modificar = bool(
    list(df_original.columns) == columnas_original
    and len(df_original) == n_filas_original
    and "load_score" not in df_original.columns
    and "requires_review" not in df_original.columns
)

# ------------------------------------------------ 4) verificación de la fila 23
esperado_f23 = {
    "server": "AI-SRV-01",
    "timestamp": "2026-08-24 09:55:00",
    "load_score": 3.136218,
    "temperature_c": 66.97,
    "requires_review": True,
}
f23 = df_new.iloc[23]
fila_23 = {
    "server": str(f23[col_server]),
    "timestamp": str(f23[col_time]) if col_time is not None else None,
    "load_score": float(f23["load_score"]),
    "temperature_c": float(f23[col_temp]),
    "requires_review": bool(f23["requires_review"]),
}
fila_23_coincide = bool(
    fila_23["server"] == esperado_f23["server"]
    and fila_23["timestamp"] == esperado_f23["timestamp"]
    and abs(fila_23["load_score"] - esperado_f23["load_score"]) < 5e-7
    and abs(fila_23["temperature_c"] - esperado_f23["temperature_c"]) < 1e-6
    and fila_23["requires_review"] == esperado_f23["requires_review"]
)

# ------------------------------------------------- 5) parquet sin el índice
RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df_new.to_parquet(RUTA_PARQUET, index=False)
df_chk = pd.read_parquet(RUTA_PARQUET)
parquet_ok = bool(
    list(df_chk.columns) == list(df_new.columns)
    and len(df_chk) == len(df_new)
    and df_chk["requires_review"].dtype == bool
    and bool((df_chk["requires_review"].to_numpy()
              == df_new["requires_review"].to_numpy()).all())
    and bool(np.array_equal(df_chk["load_score"].to_numpy(),
                            df_new["load_score"].to_numpy()))
)

# --------------------------------------------------- 6) resultados.json
review_counts = df_new.groupby(col_server)["requires_review"].sum().astype(int)
resultados = {
    "subtarea": "T3",
    "descripcion": ("Copia del DataFrame original con load_score (de T2) y "
                    "requires_review (bool) = (load_score > 1.5) | (temperature_c > 80)"),
    "archivo_datos": str(RUTA_CSV),
    "load_score_fuente": fuente_load_score,
    "col_temperatura_usada": col_temp,
    "col_timestamp_usada": col_time,
    "regla_requires_review": "(load_score > 1.5) | (temperature_c > 80)",
    "n_filas_original": n_filas_original,
    "columnas_original": columnas_original,
    "n_filas_resultado": int(len(df_new)),
    "n_columnas_resultado": int(df_new.shape[1]),
    "columnas_resultado": list(df_new.columns),
    "columnas_agregadas": ["load_score", "requires_review"],
    "estructura_conserva_filas_y_columnas_originales": estructura_ok,
    "load_score_dtype": str(df_new["load_score"].dtype),
    "requires_review_dtype": str(df_new["requires_review"].dtype),
    "requires_review_es_bool": bool(df_new["requires_review"].dtype == bool),
    "timestamp_dtype": str(df_new[col_time].dtype) if col_time is not None else None,
    "original_sin_modificar": original_sin_modificar,
    "load_score_recalculado_coincide": load_score_recalc_coincide,
    "fila_23": fila_23,
    "esperado_fila_23": esperado_f23,
    "fila_23_coincide_esperado": fila_23_coincide,
    "n_requires_review_true": int(df_new["requires_review"].sum()),
    "n_requires_review_false": int((~df_new["requires_review"]).sum()),
    "n_load_score_mayor_1_5": int((df_new["load_score"] > 1.5).sum()),
    "n_temperature_mayor_80": int((df_new[col_temp] > 80).sum()),
    "review_count_por_servidor": {str(k): int(v) for k, v in review_counts.items()},
    "requires_review": [bool(x) for x in df_new["requires_review"].to_numpy()],
    "load_score": [float(x) for x in df_new["load_score"].to_numpy()],
    "parquet_escrito": str(RUTA_PARQUET),
    "parquet_index_incluido": False,
    "parquet_verificado": parquet_ok,
}
RUTA_RESULT.write_text(json.dumps(resultados, indent=2, ensure_ascii=False),
                       encoding="utf-8")

# ------------------------------------------------------------- resumen
print("=== T3: DataFrame con load_score y requires_review ===")
print(f"Filas: {len(df_new)} | Columnas: {list(df_new.columns)}")
print(f"load_score dtype: {df_new['load_score'].dtype} | "
      f"requires_review dtype: {df_new['requires_review'].dtype}")
print(f"requires_review=True: {int(df_new['requires_review'].sum())} de {len(df_new)}")
print(f"Fila 23: {fila_23}")
print(f"Fila 23 coincide con lo esperado: {fila_23_coincide}")
print(f"Original sin modificar: {original_sin_modificar} | "
      f"load_score recalculado coincide con T2: {load_score_recalc_coincide}")
print(f"Parquet escrito en {RUTA_PARQUET} (sin índice, verificado: {parquet_ok})")
print(f"review_count por servidor: {dict(review_counts)}")
