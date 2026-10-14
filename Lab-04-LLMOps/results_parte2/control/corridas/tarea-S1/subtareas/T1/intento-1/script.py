# Subtarea T1: Cargar el CSV, implementar ServerMeasurements y build_measurements,
# ejecutar e imprimir el repr. El CSV de entrada no se modifica.

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CSV_PATH = Path("data/server_measurements.csv")
FEATURE_NAMES = ("gpu_utilization", "cpu_utilization", "memory_gb")
EXPECTED_REPR = (
    "ServerMeasurements(shape=(300, 3), "
    "features=('gpu_utilization', 'cpu_utilization', 'memory_gb'))"
)


class ServerMeasurements:
    """Almacena una matriz NumPy bidimensional (values) y una tupla con los
    nombres de las características (feature_names)."""

    def __init__(self, values, feature_names):
        arr = np.asarray(values)

        # Validación 1: la matriz debe ser bidimensional
        if arr.ndim != 2:
            raise ValueError(
                f"'values' debe ser una matriz bidimensional; se recibió ndim={arr.ndim}."
            )

        feature_names = tuple(feature_names)

        # Validación 2: número de columnas == cantidad de nombres
        if arr.shape[1] != len(feature_names):
            raise ValueError(
                f"El número de columnas ({arr.shape[1]}) no coincide con la "
                f"cantidad de nombres de características ({len(feature_names)})."
            )

        # Validación 3: los datos deben ser numéricos
        try:
            arr = arr.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Los datos de 'values' no son numéricos.") from exc

        self.values = arr
        self.feature_names = feature_names

    @property
    def shape(self):
        return self.values.shape

    def __repr__(self):
        return (
            f"ServerMeasurements(shape={self.values.shape}, "
            f"features={self.feature_names})"
        )


def build_measurements(df: pd.DataFrame) -> ServerMeasurements:
    """Selecciona gpu_utilization, cpu_utilization y memory_gb (en ese orden),
    las convierte a matriz NumPy flotante y retorna un ServerMeasurements."""
    missing = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes en el DataFrame: {missing}")
    matrix = df.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=float)
    return ServerMeasurements(matrix, FEATURE_NAMES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    results = {}

    # Hash del CSV antes de trabajar (para verificar que no se modifica)
    hash_before = _sha256(CSV_PATH)

    # Carga con parse_dates en timestamp
    df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"])
    print(f"CSV cargado: {df.shape[0]} filas x {df.shape[1]} columnas")
    print(f"dtypes:\n{df.dtypes.to_string()}")

    # Construcción del objeto y repr esperado
    measurements = build_measurements(df)
    print(repr(measurements))

    results["csv_path"] = str(CSV_PATH)
    results["n_rows"] = int(df.shape[0])
    results["n_columns_csv"] = int(df.shape[1])
    results["timestamp_dtype"] = str(df["timestamp"].dtype)
    results["shape"] = [int(s) for s in measurements.values.shape]
    results["feature_names"] = list(measurements.feature_names)
    results["values_dtype"] = str(measurements.values.dtype)
    results["repr"] = repr(measurements)
    results["repr_coincide_con_esperado"] = bool(repr(measurements) == EXPECTED_REPR)

    # Verificación de los tres casos inválidos -> ValueError
    checks = {}

    # Caso 1: matriz no bidimensional
    try:
        ServerMeasurements(np.array([1.0, 2.0, 3.0]), FEATURE_NAMES)
        checks["matriz_no_bidimensional_lanza_valueerror"] = False
    except ValueError:
        checks["matriz_no_bidimensional_lanza_valueerror"] = True

    # Caso 2: número de columnas distinto de la cantidad de nombres
    try:
        ServerMeasurements(np.zeros((5, 2)), FEATURE_NAMES)
        checks["columnas_distintas_de_nombres_lanza_valueerror"] = False
    except ValueError:
        checks["columnas_distintas_de_nombres_lanza_valueerror"] = True

    # Caso 3: datos no numéricos
    try:
        ServerMeasurements(np.array([["a", "b", "c"]]), FEATURE_NAMES)
        checks["datos_no_numericos_lanza_valueerror"] = False
    except ValueError:
        checks["datos_no_numericos_lanza_valueerror"] = True

    results["validaciones_valueerror"] = checks
    print(f"Validaciones ValueError (deben ser True): {checks}")

    # Verificación de que el CSV de entrada no se modificó
    hash_after = _sha256(CSV_PATH)
    results["csv_sin_modificar"] = bool(hash_before == hash_after)
    results["csv_sha256"] = hash_after
    print(f"CSV sin modificar: {results['csv_sin_modificar']}")

    # Resumen breve de las tres características cargadas
    resumen = {
        c: {
            "media": float(df[c].mean()),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }
        for c in FEATURE_NAMES
    }
    results["resumen_features"] = resumen
    for c, s in resumen.items():
        print(f"{c}: media={s['media']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")

    # Contrato de la subtarea
    with open("resultados.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("resultados.json escrito correctamente.")


if __name__ == "__main__":
    main()
