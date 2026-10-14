#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1 - Representar las mediciones.

Implementa la clase ServerMeasurements (matriz NumPy bidimensional `values` +
tupla `feature_names`, con validaciones ValueError) y la función
build_measurements(df), que selecciona gpu_utilization, cpu_utilization y
memory_gb en ese orden, las convierte a matriz NumPy flotante y retorna el
objeto. Ejecuta la demostración, verifica que las tres validaciones lancen
ValueError, captura el repr y escribe resultados.json junto con el parquet
exigido por el enunciado (output/server_analysis.parquet).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
RUTA_CSV = Path("data/server_measurements.csv")
RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla
    feature_names con los nombres de las características.

    Lanza ValueError si:
      1) la matriz no es bidimensional,
      2) el número de columnas no coincide con la cantidad de nombres,
      3) los datos no son numéricos.
    """

    def __init__(self, values, feature_names):
        values = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional.
        if values.ndim != 2:
            raise ValueError(
                f"La matriz debe ser bidimensional; se recibió ndim={values.ndim}."
            )

        feature_names = tuple(feature_names)

        # Validación 2: columnas == cantidad de nombres.
        if values.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({values.shape[1]}) no coincide con la "
                f"cantidad de nombres ({len(feature_names)})."
            )

        # Validación 3: los datos deben ser numéricos.
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                f"Los datos deben ser numéricos; se recibió dtype={values.dtype}."
            )

        self.values = values
        self.feature_names = feature_names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names!r})"
        )


def build_measurements(df):
    """Selecciona las tres características en el orden exigido, las convierte
    a una matriz NumPy de tipo flotante y retorna un objeto ServerMeasurements."""
    values = df.loc[:, list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(values, FEATURES)


def _df_sintetico(n_filas=300, semilla=42):
    """Respaldo local (sin red) por si el CSV no está disponible o no trae las
    columnas esperadas; permite demostrar la clase con la forma (300, 3)."""
    rng = np.random.default_rng(semilla)
    servidores = np.array([f"AI-SRV-0{i}" for i in range(1, 7)])
    gpu = np.clip(rng.normal(70, 15, n_filas), 0, 100)
    cpu = np.clip(rng.normal(55, 18, n_filas), 0, 100)
    return pd.DataFrame(
        {
            "server": servidores[rng.integers(0, 6, size=n_filas)],
            "timestamp": pd.date_range("2024-01-01", periods=n_filas, freq="h"),
            "gpu_utilization": gpu,
            "cpu_utilization": cpu,
            "memory_gb": np.clip(rng.normal(32, 8, n_filas), 4, 64),
            "power_w": np.clip(100 + 4 * gpu + 2 * cpu + rng.normal(0, 10, n_filas), 50, 900),
            "temperature_c": np.clip(30 + 0.25 * gpu + rng.normal(0, 3, n_filas), 20, 95),
        }
    )


def cargar_dataframe():
    """Lee data/server_measurements.csv (timestamp con parse de fechas); si no
    existe o faltan columnas, usa un respaldo sintético con semilla fija."""
    if RUTA_CSV.exists():
        df = pd.read_csv(RUTA_CSV)
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        if all(c in df.columns for c in FEATURES):
            return df, "csv"
        print("AVISO: el CSV no contiene las columnas esperadas; se usa respaldo sintético.")
    else:
        print("AVISO: no se encontró data/server_measurements.csv; se usa respaldo sintético.")
    return _df_sintetico(), "sintetico"


def lanza_value_error(func):
    """Retorna True si func() lanza ValueError, False en cualquier otro caso."""
    try:
        func()
    except ValueError:
        return True
    except Exception:
        return False
    return False


def main():
    df, origen = cargar_dataframe()

    # Construcción del objeto a partir del DataFrame.
    meas = build_measurements(df)
    repr_texto = repr(meas)

    # Verificación de las tres validaciones (cada una debe lanzar ValueError).
    v1 = lanza_value_error(
        lambda: ServerMeasurements(np.arange(6.0), ("a", "b", "c"))
    )  # matriz 1D -> no bidimensional
    v2 = lanza_value_error(
        lambda: ServerMeasurements(np.zeros((4, 2)), ("a", "b", "c"))
    )  # 2 columnas vs 3 nombres
    v3 = lanza_value_error(
        lambda: ServerMeasurements(
            np.array([["1.0", "2.0"], ["3.0", "4.0"]]), ("a", "b")
        )
    )  # cadenas de texto -> no numérico

    # Parquet exigido por el enunciado (serialización de las mediciones).
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df_med = pd.DataFrame(meas.values, columns=list(meas.feature_names))
    df_med.to_parquet(RUTA_PARQUET, index=False)

    # Estadísticas descriptivas por característica (informativas).
    stats = {}
    for i, nombre in enumerate(meas.feature_names):
        col = meas.values[:, i]
        stats[nombre] = {
            "media": float(np.mean(col)),
            "desv_est": float(np.std(col, ddof=1)),
            "min": float(np.min(col)),
            "max": float(np.max(col)),
        }

    resultados = {
        "subtarea": "T1_representar_mediciones",
        "origen_datos": origen,
        "repr": repr_texto,
        "shape": [int(meas.values.shape[0]), int(meas.values.shape[1])],
        "feature_names": list(meas.feature_names),
        "dtype_values": str(meas.values.dtype),
        "n_filas": int(meas.values.shape[0]),
        "n_columnas": int(meas.values.shape[1]),
        "validaciones_valueerror": {
            "matriz_no_bidimensional_lanza_value_error": bool(v1),
            "numero_columnas_no_coincide_lanza_value_error": bool(v2),
            "datos_no_numericos_lanza_value_error": bool(v3),
        },
        "primeras_3_filas_values": meas.values[:3].tolist(),
        "estadisticas_por_feature": stats,
        "archivo_parquet": str(RUTA_PARQUET),
    }
    RUTA_RESULTADOS.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 70)
    print("T1 - Representar las mediciones")
    print("=" * 70)
    print(f"Origen de datos     : {origen}")
    print(f"repr capturado      : {repr_texto}")
    print(f"Forma de values     : {meas.values.shape} | dtype: {meas.values.dtype}")
    print(f"Nombres de features : {meas.feature_names}")
    print(f"Validación 1D       -> ValueError: {v1}")
    print(f"Validación columnas -> ValueError: {v2}")
    print(f"Validación numérico -> ValueError: {v3}")
    print(f"Parquet escrito     : {RUTA_PARQUET} ({len(df_med)} filas)")
    print(f"Resultados escritos : {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
