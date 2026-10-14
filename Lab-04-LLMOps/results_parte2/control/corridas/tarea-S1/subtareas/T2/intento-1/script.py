# -*- coding: utf-8 -*-
"""
T2 — Normalizar (Z-score) la matriz de características de T1 y calcular load_score.

Pasos:
  1. Recupera la matriz de características (300, 3) de T1
     (gpu_utilization, cpu_utilization, memory_gb, en ese orden).
  2. Calcula media y desviación estándar por columna (operaciones vectorizadas).
  3. Normaliza con Z-score evitando división por cero (std = 0 -> se usa 1.0).
  4. Calcula load_score = zscores @ [0.50, 0.30, 0.20] (vectorizado).
  5. Verifica media 0 y desviación 1 por columna (tolerancia 1e-10) y la
     igualdad load_score == zscores @ pesos.
  6. Escribe resultados.json, output/server_analysis.parquet y una figura PNG.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = Path("data/server_measurements.csv")
T1_DIR = Path("entrada/T1")
FEATURES_DEFAULT = ("gpu_utilization", "cpu_utilization", "memory_gb")
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
TOL = 1e-10


# ---------------------------------------------------------------------------
# Réplica de la clase y la función de T1 (misma lógica de selección)
# ---------------------------------------------------------------------------
class ServerMeasurements:
    """Contenedor de T1: matriz NumPy 2D flotante + nombres de características."""

    def __init__(self, values, feature_names):
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("La matriz de datos debe ser bidimensional.")
        if values.shape[1] != len(feature_names):
            raise ValueError(
                "El número de columnas debe coincidir con la cantidad de nombres."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError("Los datos deben ser numéricos.")
        self.values = values.astype(float)
        self.feature_names = tuple(feature_names)

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df, features=FEATURES_DEFAULT):
    """Selecciona las tres características (orden de T1) -> matriz NumPy flotante."""
    faltantes = [c for c in features if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en el DataFrame: {faltantes}")
    return ServerMeasurements(df[list(features)].to_numpy(dtype=float), features)


# ---------------------------------------------------------------------------
# Recuperación de la matriz de características de T1
# ---------------------------------------------------------------------------
def leer_contexto_t1():
    """Lee entrada/T1/results.json (si existe) para respetar orden y tamaño de T1."""
    info = {}
    ruta = T1_DIR / "results.json"
    if ruta.is_file():
        try:
            info = json.loads(ruta.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            info = {}
    features = tuple(info.get("feature_names") or FEATURES_DEFAULT)
    n_rows = int(info.get("n_rows", 300))
    return info, features, n_rows


def medias_referencia_t1(info, features):
    resumen = info.get("resumen_features", {}) or {}
    return [resumen.get(f, {}).get("media") for f in features]


def _valida_matriz(arr, n_rows, n_cols, referencia, hay_referencia):
    """Comprueba forma, tipo numérico, finitud y (si hay) coincidencia de medias con T1."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return None
    if arr.shape != (n_rows, n_cols) or not np.issubdtype(arr.dtype, np.number):
        return None
    arr = arr.astype(float)
    if not np.isfinite(arr).all():
        return None
    if hay_referencia and not np.allclose(arr.mean(axis=0), referencia, atol=1e-6):
        return None
    return arr


def obtener_matriz(features, n_rows, info_t1):
    """
    Devuelve (X, origen). Intenta reutilizar la matriz guardada por T1 en
    entrada/T1/ (validando forma y medias); si no está, la reconstruye desde
    el CSV con la lógica exacta de T1 (build_measurements).
    """
    referencia = medias_referencia_t1(info_t1, features)
    hay_referencia = all(m is not None for m in referencia)
    n_cols = len(features)

    if T1_DIR.is_dir():
        for ruta in sorted(T1_DIR.glob("*.npy")):
            try:
                arr = np.load(ruta, allow_pickle=False)
                arr = _valida_matriz(arr, n_rows, n_cols, referencia, hay_referencia)
            except Exception:
                arr = None
            if arr is not None:
                return arr, f"entrada/T1/{ruta.name}"

        for ruta in sorted(T1_DIR.glob("*.npz")):
            try:
                with np.load(ruta, allow_pickle=False) as z:
                    arreglos = {k: z[k] for k in z.files}
            except Exception:
                continue
            for clave, arr in arreglos.items():
                arr = _valida_matriz(arr, n_rows, n_cols, referencia, hay_referencia)
                if arr is not None:
                    return arr, f"entrada/T1/{ruta.name}[{clave}]"

        for ruta in sorted(T1_DIR.glob("*.parquet")):
            try:
                dfp = pd.read_parquet(ruta)
                if all(c in dfp.columns for c in features):
                    arr = dfp[list(features)].to_numpy(dtype=float)
                    arr = _valida_matriz(arr, n_rows, n_cols, referencia, hay_referencia)
                    if arr is not None:
                        return arr, f"entrada/T1/{ruta.name}"
            except Exception:
                continue

    # Reconstrucción determinista con la lógica exacta de T1
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    medicion = build_measurements(df, features)
    return medicion.values, str(CSV_PATH)


# ---------------------------------------------------------------------------
# Subtarea T2
# ---------------------------------------------------------------------------
def main():
    info_t1, features, n_rows = leer_contexto_t1()
    X, origen = obtener_matriz(features, n_rows, info_t1)

    if X.shape != (n_rows, len(features)):
        raise ValueError(
            f"Forma inesperada de la matriz de características: {X.shape}; "
            f"se esperaba {(n_rows, len(features))}."
        )
    if not np.isfinite(X).all():
        raise ValueError("La matriz de características contiene valores no finitos.")

    # 1) Media y desviación estándar por columna (vectorizado)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)  # desviación estándar poblacional

    # 2) Z-score vectorizado (broadcasting), evitando división por cero
    stds_cero = stds == 0
    safe_stds = np.where(stds_cero, 1.0, stds)  # si std=0 -> divisor 1.0 (columna queda en 0)
    zscores = (X - means) / safe_stds           # sin bucles sobre observaciones

    # 3) load_score vectorizado
    load_score = zscores @ PESOS

    # 4) Verificaciones (tolerancia 1e-10)
    medias_z = zscores.mean(axis=0)
    desvs_z = zscores.std(axis=0, ddof=0)
    media_cero_ok = bool(np.all(np.abs(medias_z) <= TOL))
    desv_uno_ok = bool(np.all(np.abs(desvs_z - 1.0) <= TOL))
    load_ok = bool(np.array_equal(load_score, zscores @ PESOS))

    # 5) Impresiones exigidas por la subtarea
    print("means:", means)
    print("stds :", stds)
    print("zscores.shape    ->", zscores.shape)
    print("load_score.shape ->", load_score.shape)
    print("Verificación media 0 (tol 1e-10) por columna:", medias_z, "->", media_cero_ok)
    print(
        "Verificación desviación 1 (tol 1e-10) por columna:",
        desvs_z,
        "->",
        desv_uno_ok,
    )
    print("load_score == zscores @ [0.50, 0.30, 0.20] ->", load_ok)
    print("Origen de la matriz:", origen)

    # Coincidencia con las medias reportadas por T1
    referencia = medias_referencia_t1(info_t1, features)
    medias_coinciden_t1 = None
    if all(m is not None for m in referencia):
        medias_coinciden_t1 = bool(np.allclose(means, referencia, atol=1e-9))
        print("Medias coinciden con entrada/T1/results.json:", medias_coinciden_t1)

    # 6) Archivo exigido por el enunciado: output/server_analysis.parquet
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    for i, nombre in enumerate(features):
        df[f"zscore_{nombre}"] = zscores[:, i]
    df["load_score"] = load_score
    Path("output").mkdir(parents=True, exist_ok=True)
    df.to_parquet("output/server_analysis.parquet", index=False)

    # 7) Figura de apoyo
    n_panels = len(features) + 1
    fig, ejes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 3.5))
    for ax, nombre, col in zip(ejes, features, range(len(features))):
        ax.hist(zscores[:, col], bins=25, color="#4C72B0", edgecolor="white")
        ax.set_title(f"zscore {nombre}")
        ax.set_xlabel("z")
        ax.set_ylabel("frecuencia")
    ejes[-1].hist(load_score, bins=25, color="#DD8452", edgecolor="white")
    ejes[-1].set_title("load_score")
    ejes[-1].set_xlabel("load_score")
    ejes[-1].set_ylabel("frecuencia")
    fig.suptitle("T2: Z-scores por característica y load_score")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig("t2_zscores_load_score.png", dpi=120)
    plt.close(fig)

    # 8) resultados.json (contrato de la subtarea)
    resultados = {
        "subtarea": "T2",
        "origen_matriz": origen,
        "feature_names": list(features),
        "n_observaciones": int(X.shape[0]),
        "means": means.tolist(),
        "stds": stds.tolist(),
        "stds_con_cero": stds_cero.tolist(),
        "division_segura_aplicada": bool(np.any(stds_cero)),
        "pesos": PESOS.tolist(),
        "zscores_shape": list(zscores.shape),
        "load_score_shape": list(load_score.shape),
        "zscores_medias_por_columna": medias_z.tolist(),
        "zscores_desviaciones_por_columna": desvs_z.tolist(),
        "tolerancia": TOL,
        "verificaciones": {
            "media_0_por_columna": media_cero_ok,
            "desviacion_1_por_columna": desv_uno_ok,
            "load_score_igual_a_zscores_por_pesos": load_ok,
        },
        "medias_coinciden_con_T1": medias_coinciden_t1,
        "load_score_resumen": {
            "min": float(load_score.min()),
            "max": float(load_score.max()),
            "media": float(load_score.mean()),
            "desviacion": float(load_score.std()),
        },
        "load_score": load_score.tolist(),
        "zscores": zscores.tolist(),
        "archivos_generados": {
            "parquet": "output/server_analysis.parquet",
            "figura": "t2_zscores_load_score.png",
        },
    }
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(
        "Resumen T2 -> means:",
        np.round(means, 8).tolist(),
        "| zscores:",
        zscores.shape,
        "| load_score:",
        load_score.shape,
        "| media0:",
        media_cero_ok,
        "| desv1:",
        desv_uno_ok,
        "| load_ok:",
        load_ok,
    )
    print(
        "Escritos: resultados.json, output/server_analysis.parquet, "
        "t2_zscores_load_score.png"
    )


if __name__ == "__main__":
    main()
