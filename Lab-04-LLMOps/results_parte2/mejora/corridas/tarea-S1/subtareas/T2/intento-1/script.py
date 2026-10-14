# T2: Normalizar (Z-score) la matriz de características de T1 y calcular load_score
# Operaciones 100% vectorizadas (sin bucles sobre observaciones).

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ rutas
CSV_PATH = Path("data/server_measurements.csv")
T1_RESULTS = Path("entrada/T1/resultados.json")
RESULTS_PATH = Path("resultados.json")
PARQUET_PATH = Path("output/server_analysis.parquet")

FEATURES = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20])
TOL = 1e-10

# ------------------------------------------- matriz de características de T1
df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
X = df[FEATURES].to_numpy(dtype=np.float64)  # selección y conversión vectorizadas
assert X.shape == (300, 3), f"Forma inesperada de la matriz: {X.shape}"

# Consistencia opcional con T1 (si su resultados.json está disponible)
coincide_t1 = None
if T1_RESULTS.exists():
    t1 = json.loads(T1_RESULTS.read_text(encoding="utf-8"))
    coincide_t1 = bool(
        tuple(t1.get("shape", [])) == X.shape
        and np.allclose(X[:3], np.array(t1["primeras_3_filas_values"], dtype=float))
    )

# --------------------------------- media y desviación estándar por columna
means = X.mean(axis=0)   # vector (3,)
stds = X.std(axis=0)     # vector (3,), ddof=0 (por columna)

# Z-score evitando división por cero si alguna desviación es 0 (vectorizado)
stds_safe = np.where(stds == 0.0, 1.0, stds)
zscores = (X - means) / stds_safe

# load_score = zscores @ [0.50, 0.30, 0.20]  (operación vectorizada)
load_score = zscores @ PESOS

# ------------------------------------------------------------- impresiones
print("means:", means)
print("zscores.shape:", zscores.shape)
print("load_score.shape:", load_score.shape)

# ---------------------------------------------------------- verificaciones
means_esperados = np.array([57.99313333, 50.54163333, 29.45533333])
col_means = zscores.mean(axis=0)
col_stds = zscores.std(axis=0)
load_score_ref = 0.50 * zscores[:, 0] + 0.30 * zscores[:, 1] + 0.20 * zscores[:, 2]

verif = {
    "means_coinciden_esperados": bool(np.allclose(means, means_esperados, atol=1e-6)),
    "zscores_shape_correcto": tuple(zscores.shape) == (300, 3),
    "load_score_shape_correcto": tuple(load_score.shape) == (300,),
    "zscores_media_cero_tol_1e-10": bool(np.all(np.abs(col_means) <= TOL)),
    "zscores_desv_uno_tol_1e-10": bool(np.all(np.abs(col_stds - 1.0) <= TOL)),
    "load_score_igual_a_zscores_por_pesos": bool(
        np.allclose(load_score, load_score_ref, rtol=0.0, atol=1e-12)
    ),
}
if coincide_t1 is not None:
    verif["matriz_consistente_con_T1"] = coincide_t1

for k, v in verif.items():
    print(f"  {k}: {v}")
assert all(verif.values()), "Fallo de verificación en T2"

# ------------------------------------------------------- archivos de salida
np.save("zscores.npy", zscores)
np.save("load_score.npy", load_score)

salida = df.copy()
salida["zscore_gpu_utilization"] = zscores[:, 0]
salida["zscore_cpu_utilization"] = zscores[:, 1]
salida["zscore_memory_gb"] = zscores[:, 2]
salida["load_score"] = load_score
PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)
salida.to_parquet(PARQUET_PATH, index=False)

resultados = {
    "subtarea": "T2_normalizar_y_load_score",
    "feature_names": FEATURES,
    "pesos": [0.50, 0.30, 0.20],
    "means": means.tolist(),
    "stds": stds.tolist(),
    "stds_usados_en_division": stds_safe.tolist(),
    "zscores_shape": list(zscores.shape),
    "load_score_shape": list(load_score.shape),
    "zscores_media_por_columna": col_means.tolist(),
    "zscores_desv_por_columna": col_stds.tolist(),
    "zscores_primeras_3_filas": zscores[:3].tolist(),
    "load_score_primeros_5": load_score[:5].tolist(),
    "load_score_resumen": {
        "media": float(load_score.mean()),
        "desv_est": float(load_score.std()),
        "min": float(load_score.min()),
        "max": float(load_score.max()),
    },
    "n_filas_load_score_mayor_1_5": int((load_score > 1.5).sum()),
    "verificacion": verif,
    "archivo_parquet": str(PARQUET_PATH),
    "archivos_npy": ["zscores.npy", "load_score.npy"],
}
RESULTS_PATH.write_text(
    json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
)

# ----------------------------------------------------------- resumen breve
print("-" * 60)
print("Resumen T2")
print("  means              :", means)
print("  stds               :", stds)
print("  zscores.shape      :", zscores.shape)
print("  load_score.shape   :", load_score.shape)
print("  media zscores      :", col_means, "(~0)")
print("  desv. zscores      :", col_stds, "(~1)")
print("  load_score media/desv:", float(load_score.mean()), float(load_score.std()))
print("  filas con load_score > 1.5:", int((load_score > 1.5).sum()))
print("  Parquet:", PARQUET_PATH, "| Resultados:", RESULTS_PATH)
