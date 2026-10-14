# -*- coding: utf-8 -*-
"""
T1 - Representar las mediciones de servidores.

Carga data/server_measurements.csv (timestamp con parse_dates), implementa la
clase ServerMeasurements (values 2D, feature_names, validaciones ValueError,
__repr__) y build_measurements(df) que selecciona gpu_utilization,
cpu_utilization y memory_gb en ese orden y las convierte a matriz NumPy
flotante. Comprueba las tres validaciones de ValueError, verifica que el CSV
de entrada no se modifica y escribe resultados.json.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/server_measurements.csv")
RESULTADOS_PATH = Path("resultados.json")

FEATURES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


def md5_archivo(path: Path) -> str:
    """Hash MD5 del contenido de un archivo (para verificar que no cambia)."""
    return hashlib.md5(path.read_bytes()).hexdigest()


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional
        if arr.ndim != 2:
            raise ValueError("values debe ser una matriz NumPy bidimensional (2D).")

        names = tuple(feature_names)

        # Validación 2: número de columnas == cantidad de nombres
        if arr.shape[1] != len(names):
            raise ValueError(
                "El número de columnas de values no coincide con la cantidad "
                "de feature_names."
            )

        # Validación 3: los datos deben ser numéricos
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError("Los datos de values deben ser numéricos.")

        self.values = arr.astype(float, copy=False)
        self.feature_names = names

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Selecciona gpu_utilization, cpu_utilization y memory_gb (en ese orden),
    las convierte a una matriz NumPy de tipo flotante y retorna un objeto
    ServerMeasurements."""
    matriz = df[list(FEATURES)].to_numpy(dtype=float)
    return ServerMeasurements(matriz, FEATURES)


def lanza_value_error(func) -> bool:
    """Retorna True si func() lanza ValueError, False en otro caso."""
    try:
        func()
        return False
    except ValueError:
        return True
    except Exception:
        return False


def main():
    # Hash del CSV antes de cualquier operación (solo lectura: nunca se modifica)
    hash_antes = md5_archivo(CSV_PATH)

    # Carga del CSV leyendo timestamp como fecha
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])

    # Construcción del objeto principal y captura del repr
    mediciones = build_measurements(df)
    repr_capturado = repr(mediciones)
    print(repr_capturado)

    # Comprobación de las tres validaciones de ValueError
    val_no_2d = lanza_value_error(
        lambda: ServerMeasurements(np.array([1.0, 2.0, 3.0]), FEATURES)
    )
    val_columnas = lanza_value_error(
        lambda: ServerMeasurements(np.zeros((5, 4), dtype=float), FEATURES)
    )
    val_no_numerico = lanza_value_error(
        lambda: ServerMeasurements(
            np.array([["a", "b", "c"], ["d", "e", "f"]], dtype=object), FEATURES
        )
    )

    # Verificación de que el CSV de entrada no fue modificado
    hash_despues = md5_archivo(CSV_PATH)

    resultados = {
        "repr": repr_capturado,
        "repr_esperado": EXPECTED_REPR,
        "repr_coincide": repr_capturado == EXPECTED_REPR,
        "shape": [int(mediciones.values.shape[0]), int(mediciones.values.shape[1])],
        "feature_names": list(mediciones.feature_names),
        "values_dtype": str(mediciones.values.dtype),
        "n_filas": int(mediciones.values.shape[0]),
        "n_columnas": int(mediciones.values.shape[1]),
        "muestra_values_primeras_3_filas": mediciones.values[:3].tolist(),
        "validaciones": {
            "matriz_no_bidimensional_lanza_valueerror": bool(val_no_2d),
            "columnas_distintas_de_nombres_lanza_valueerror": bool(val_columnas),
            "datos_no_numericos_lanza_valueerror": bool(val_no_numerico),
        },
        "validaciones_todas_ok": bool(val_no_2d and val_columnas and val_no_numerico),
        "csv_no_modificado": hash_antes == hash_despues,
        "csv_md5": hash_antes,
        "columnas_csv": list(df.columns),
        "timestamp_dtype": str(df["timestamp"].dtype),
        "servidores": sorted(df["server"].unique().tolist()),
    }

    RESULTADOS_PATH.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Resumen breve
    print("Validaciones ValueError -> no 2D:", val_no_2d,
          "| columnas != nombres:", val_columnas,
          "| no numéricos:", val_no_numerico)
    print("repr coincide con el esperado:", resultados["repr_coincide"])
    print("CSV de entrada sin modificar:", resultados["csv_no_modificado"])
    print("Resultados escritos en:", RESULTADOS_PATH)


if __name__ == "__main__":
    main()
