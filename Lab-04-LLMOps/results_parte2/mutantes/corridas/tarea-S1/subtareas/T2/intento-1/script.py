#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SUBTAREA T2 — Normalizar con Z-score y calcular load_score (vectorizado).

Sobre la matriz de la Tarea 1 (gpu_utilization, cpu_utilization, memory_gb):
  1) media y desviación estándar por columna,
  2) Z-score vectorizado con protección de división por cero,
  3) load_score = zscores @ [0.50, 0.30, 0.20] sin recorrer observaciones con for.

Salidas: resultados.json (contrato) y output/server_analysis.parquet.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1) Carga de datos y reconstrucción de la matriz de la Tarea 1
# ----------------------------------------------------------------------------
FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)


class ServerMeasurements:
    """Contenedor de la matriz de mediciones (misma interfaz que la Tarea 1)."""

    def __init__(self, values, feature_names):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("La matriz de valores debe ser bidimensional.")
        if values.shape[1] != len(feature_names):
            raise ValueError(
                "El número de columnas debe coincidir con la cantidad de nombres."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("Los datos deben ser numéricos.")
        self._values = values.astype(float)
        self._feature_names = tuple(feature_names)

    @property
    def values(self):
        return self._values

    @property
    def feature_names(self):
        return self._feature_names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self._values.shape}, "
            f"features={self._feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Selecciona las 3 características en orden y las convierte a matriz float."""
    X = df.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(X, FEATURES)


df = pd.read_csv("data/server_measurements.csv", parse_dates=["timestamp"])
sm = build_measurements(df)
X = sm.values  # matriz (300, 3) de la Tarea 1

# ----------------------------------------------------------------------------
# 2) Media y desviación estándar por columna (vectorizado, ddof=0)
#    ddof=0 garantiza que cada columna de zscores quede con desviación exactamente 1.
# ----------------------------------------------------------------------------
means = X.mean(axis=0)
stds = X.std(axis=0)

# ----------------------------------------------------------------------------
# 3) Z-score vectorizado con protección de división por cero (broadcasting)
# ----------------------------------------------------------------------------
safe_stds = np.where(stds == 0.0, 1.0, stds)
zscores = (X - means) / safe_stds  # sin bucles sobre observaciones

# ----------------------------------------------------------------------------
# 4) load_score = zscores @ pesos (producto punto vectorizado)
# ----------------------------------------------------------------------------
load_score = zscores @ PESOS

# ----------------------------------------------------------------------------
# 5) Verificaciones del criterio de éxito
# ----------------------------------------------------------------------------
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0)
media_cero_ok = bool(np.all(np.abs(col_means) <= 1e-10))
desv_uno_ok = bool(np.all(np.abs(col_stds - 1.0) <= 1e-10))

# comprobación independiente (también vectorizada) del producto punto
load_alt = np.sum(zscores * PESOS, axis=1)
load_coincide = bool(np.allclose(load_score, load_alt, rtol=0.0, atol=1e-12))

means_esperado = np.array([57.99313333, 50.54163333, 29.45533333])
means_ok = bool(np.allclose(means, means_esperado, rtol=0.0, atol=1e-8))

sin_nan_inf = bool(np.isfinite(zscores).all() and np.isfinite(load_score).all())

# verificación suave contra la subtarea T1 (si existe su resultados.json)
t1_shape_ok = None
t1_path = Path("entrada/T1/results.json")
if t1_path.exists():
    try:
        t1 = json.loads(t1_path.read_text(encoding="utf-8"))
        t1_shape_ok = bool(
            list(t1.get("shape", [])) == list(X.shape)
            and list(t1.get("feature_names", [])) == list(FEATURES)
        )
    except Exception:
        t1_shape_ok = None

criterio_ok = bool(
    means_ok
    and zscores.shape == (300, 3)
    and load_score.shape == (300,)
    and media_cero_ok
    and desv_uno_ok
    and load_coincide
    and sin_nan_inf
)

# ----------------------------------------------------------------------------
# 6) Parquet exigido por el enunciado: output/server_analysis.parquet
#    (datos originales + zscores + load_score; sin bucles sobre observaciones)
# ----------------------------------------------------------------------------
z_df = pd.DataFrame(
    zscores, columns=[f"zscore_{n}" for n in FEATURES], index=df.index
)
df_out = pd.concat(
    [df, z_df, pd.DataFrame({"load_score": load_score}, index=df.index)], axis=1
)

out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)
df_out.to_parquet(out_dir / "server_analysis.parquet", index=False)

# ----------------------------------------------------------------------------
# 7) resultados.json (contrato de la subtarea)
# ----------------------------------------------------------------------------
resultados = {
    "means": means.tolist(),
    "stds": stds.tolist(),
    "means_esperado": means_esperado.tolist(),
    "means_coinciden_esperado": means_ok,
    "std_ddof": 0,
    "values_dtype": str(X.dtype),
    "n_filas": int(X.shape[0]),
    "n_columnas": int(X.shape[1]),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "pesos": PESOS.tolist(),
    "load_score": load_score.tolist(),
    "zscores_primeras_3_filas": zscores[:3].tolist(),
    "zscores_medias_por_columna": col_means.tolist(),
    "zscores_desviaciones_por_columna": col_stds.tolist(),
    "media_cero_tol_1e-10": media_cero_ok,
    "desviacion_uno_tol_1e-10": desv_uno_ok,
    "load_score_igual_a_zscores_por_pesos": load_coincide,
    "load_score_min": float(load_score.min()),
    "load_score_max": float(load_score.max()),
    "load_score_media": float(load_score.mean()),
    "load_score_desv": float(load_score.std()),
    "sin_nan_inf": sin_nan_inf,
    "proteccion_division_cero": "np.where(stds == 0.0, 1.0, stds)",
    "operaciones_vectorizadas": True,
    "t1_shape_y_features_coinciden": t1_shape_ok,
    "parquet_escrito": "output/server_analysis.parquet",
    "filas_parquet": int(len(df_out)),
    "columnas_parquet": list(df_out.columns),
    "criterio_exito_cumplido": criterio_ok,
}

with open("resultados.json", "w", encoding="utf-8") as f:
    json.dump(resultados, f, ensure_ascii=False, indent=2)

# ----------------------------------------------------------------------------
# 8) Resumen
# ----------------------------------------------------------------------------
print("=== T2: Z-score y load_score ===")
print(f"means           : {means}")
print(f"stds (ddof=0)   : {stds}")
print(f"zscores.shape   : {zscores.shape}")
print(f"load_score.shape: {load_score.shape}")
print(f"medias de zscores (~0): {col_means}")
print(f"desvs de zscores  (~1): {col_stds}")
print(
    f"load_score min/max/media: "
    f"{load_score.min():.6f} / {load_score.max():.6f} / {load_score.mean():.6f}"
)
print(f"Parquet: output/server_analysis.parquet ({len(df_out)} filas)")
print(f"Criterio de éxito cumplido: {criterio_ok}")
