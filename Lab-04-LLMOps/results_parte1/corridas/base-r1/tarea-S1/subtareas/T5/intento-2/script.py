#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado.

1) Obtiene el DataFrame analizado de la Tarea 3 (300 filas x 9 columnas, con
   load_score y requires_review).
2) Lo guarda en output/server_analysis.parquet mediante PyArrow, SIN el índice.
3) Lo vuelve a leer y verifica el round trip:
   - filas antes == filas después
   - columnas antes == columnas después
   - esquema antes == esquema después
   - valores iguales
4) Escribe resultados.json con todas las verificaciones.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_RESULTADOS = Path("resultados.json")
RUTA_CSV = Path("data/server_measurements.csv")
DIR_T3 = Path("entrada/T3")
DIR_T2 = Path("entrada/T2")

COLUMNAS_T3 = [
    "server", "timestamp", "gpu_utilization", "cpu_utilization",
    "memory_gb", "power_w", "temperature_c", "load_score", "requires_review",
]


def _leer_json(ruta):
    try:
        with open(ruta, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _a_datetime_us(serie):
    """datetime64[us]: resolución reportada por la T3 (datos por segundo -> conversión sin pérdida)."""
    if not pd.api.types.is_datetime64_any_dtype(serie):
        serie = pd.to_datetime(serie)
    try:
        return serie.astype("datetime64[us]")
    except (TypeError, ValueError):
        return serie


def _df_desde_archivo_t3():
    """Si la T3 materializó su DataFrame (parquet/csv en entrada/T3/), se usa tal cual."""
    if not DIR_T3.is_dir():
        return None, None
    candidatos = sorted(DIR_T3.glob("*.parquet")) + sorted(DIR_T3.glob("*.csv"))
    for p in candidatos:
        try:
            d = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        except Exception:
            continue
        if "timestamp" in d.columns:
            d["timestamp"] = _a_datetime_us(d["timestamp"])
        if set(COLUMNAS_T3).issubset(d.columns):
            d = d.loc[:, COLUMNAS_T3].copy()
            d["requires_review"] = d["requires_review"].astype(bool)
            return d.reset_index(drop=True), p.as_posix()
    return None, None


def _listas_previas():
    """Recupera load_score y requires_review de los resultados publicados (T3; respaldo T2)."""
    load_score, req_review, fuentes = None, None, {}
    for etiqueta, ruta in (("T3", DIR_T3 / "resultados.json"), ("T2", DIR_T2 / "resultados.json")):
        data = _leer_json(ruta)
        if not isinstance(data, dict):
            continue
        if load_score is None and isinstance(data.get("load_score"), list):
            load_score = data["load_score"]
            fuentes["load_score"] = ruta.as_posix() + "#load_score"
        if req_review is None and etiqueta == "T3" and isinstance(data.get("requires_review"), list):
            req_review = data["requires_review"]
            fuentes["requires_review"] = ruta.as_posix() + "#requires_review"
    return load_score, req_review, fuentes


def construir_df_t3():
    """Obtiene el DataFrame analizado de la Tarea 3 (300 x 9)."""
    df_t3, fuente = _df_desde_archivo_t3()
    if df_t3 is not None:
        return df_t3, fuente

    if not RUTA_CSV.exists():
        raise FileNotFoundError("No se encontró data/server_measurements.csv ni archivos en entrada/T3/")
    df = pd.read_csv(RUTA_CSV)
    df["timestamp"] = _a_datetime_us(df["timestamp"])
    n = len(df)

    load_score, req_review, fuentes = _listas_previas()

    if load_score is not None and len(load_score) == n:
        load_score = np.asarray(load_score, dtype="float64")
    else:
        # Reserva (no debería ejecutarse: T3/T2 publican la lista completa en resultados.json)
        cols = ["gpu_utilization", "cpu_utilization", "memory_gb"]
        z = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
        load_score = z.mean(axis=1).to_numpy(dtype="float64")
        fuentes["load_score"] = "recalculo_local_de_reserva"

    if req_review is not None and len(req_review) == n:
        req_review = np.asarray(req_review, dtype=bool)
    else:
        req_review = (load_score > 1.5) | (df["temperature_c"].to_numpy() > 80)
        fuentes["requires_review"] = "regla: (load_score > 1.5) | (temperature_c > 80)"

    df_t3 = df.copy()
    df_t3["load_score"] = load_score
    df_t3["requires_review"] = req_review
    df_t3 = df_t3.loc[:, COLUMNAS_T3]
    fuente = "data/server_measurements.csv + " + "; ".join(sorted(fuentes.values()))
    return df_t3.reset_index(drop=True), fuente


def main():
    df_t3, fuente = construir_df_t3()

    # --- Guardar con PyArrow, sin el índice ---
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_t3, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # --- Volver a leer (round trip) ---
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_leido = tabla_despues.to_pandas()

    # --- Verificaciones mínimas exigidas ---
    filas_ok = len(df_t3) == len(df_leido)
    columnas_ok = (df_t3.shape[1] == df_leido.shape[1]
                   and list(df_t3.columns) == list(df_leido.columns))
    esquema_ok = bool(tabla_antes.schema.equals(tabla_despues.schema))
    dtypes_iguales = all(str(ta) == str(tb) for ta, tb in zip(df_t3.dtypes, df_leido.dtypes))
    esquema_ok = bool(esquema_ok and dtypes_iguales)

    antes = df_t3.reset_index(drop=True)
    despues = df_leido.reset_index(drop=True)
    valores_ok = bool(antes.equals(despues))
    valores_por_columna = {}
    for c in antes.columns:
        va, vb = antes[c].to_numpy(), despues[c].to_numpy()
        if pd.api.types.is_float_dtype(antes[c]):
            valores_por_columna[c] = bool(np.array_equal(va, vb, equal_nan=True))
        else:
            valores_por_columna[c] = bool((va == vb).all())
    valores_ok = bool(valores_ok and all(valores_por_columna.values()))

    todo_ok = bool(filas_ok and columnas_ok and esquema_ok and valores_ok)

    # --- Comprobación de coherencia con la T3 (fila 23) ---
    fila23_leida, fila23_esperado, fila23_ok = None, None, False
    if len(despues) > 23:
        f23 = despues.iloc[23]
        fila23_leida = {
            "server": str(f23["server"]),
            "timestamp": str(f23["timestamp"]),
            "load_score": float(f23["load_score"]),
            "temperature_c": float(f23["temperature_c"]),
            "requires_review": bool(f23["requires_review"]),
        }
        fila23_esperado = {
            "server": "AI-SRV-01",
            "timestamp": "2026-08-24 09:55:00",
            "load_score": 3.136217861183744,
            "temperature_c": 66.97,
            "requires_review": True,
        }
        fila23_ok = bool(
            fila23_leida["server"] == fila23_esperado["server"]
            and pd.Timestamp(fila23_leida["timestamp"]) == pd.Timestamp(fila23_esperado["timestamp"])
            and abs(fila23_leida["load_score"] - fila23_esperado["load_score"]) <= 1e-9
            and abs(fila23_leida["temperature_c"] - fila23_esperado["temperature_c"]) <= 1e-9
            and fila23_leida["requires_review"] == fila23_esperado["requires_review"]
        )

    # --- Salida exigida por el enunciado ---
    if todo_ok:
        print(f"Round trip verificado: {RUTA_PARQUET.as_posix()}")
    else:
        print(f"Round trip NO verificado: {RUTA_PARQUET.as_posix()}")
    print(f"Filas: {filas_ok}")
    print(f"Columnas: {columnas_ok}")
    print(f"Esquema: {esquema_ok}")
    print(f"Valores: {valores_ok}")

    # --- Resumen breve ---
    print("-" * 64)
    print(f"DataFrame T3: {df_t3.shape[0]} filas x {df_t3.shape[1]} columnas | fuente: {fuente}")
    print(f"Parquet escrito con PyArrow {pa.__version__} (sin índice): "
          f"{RUTA_PARQUET.as_posix()} ({RUTA_PARQUET.stat().st_size} bytes)")
    print(f"requires_review=True tras releer: {int(despues['requires_review'].sum())} | "
          f"fila 23 coincide con T3: {fila23_ok}")

    # --- resultados.json (contrato de la subtarea) ---
    esquema_antes = {f.name: str(f.type) for f in tabla_antes.schema}
    esquema_despues = {f.name: str(f.type) for f in tabla_despues.schema}
    dtypes_antes = {c: str(t) for c, t in df_t3.dtypes.items()}
    dtypes_despues = {c: str(t) for c, t in df_leido.dtypes.items()}

    resultados = {
        "subtarea": "T5",
        "descripcion": ("Guardar el DataFrame analizado de la T3 en output/server_analysis.parquet "
                        "con PyArrow sin el índice, releerlo y verificar el round trip: "
                        "filas, columnas, esquema y valores."),
        "archivo_parquet": RUTA_PARQUET.as_posix(),
        "archivo_existe": bool(RUTA_PARQUET.exists()),
        "motor": "pyarrow",
        "version_pyarrow": pa.__version__,
        "version_pandas": pd.__version__,
        "version_numpy": np.__version__,
        "fuente_dataframe_t3": fuente,
        "indice_incluido": False,
        "indice_no_incluido_en_archivo": bool("__index_level_0__" not in tabla_antes.column_names
                                              and isinstance(despues.index, pd.RangeIndex)),
        "n_filas_antes": int(len(df_t3)),
        "n_filas_despues": int(len(df_leido)),
        "n_columnas_antes": int(df_t3.shape[1]),
        "n_columnas_despues": int(df_leido.shape[1]),
        "columnas_antes": list(df_t3.columns),
        "columnas_despues": list(df_leido.columns),
        "esquema_antes": esquema_antes,
        "esquema_despues": esquema_despues,
        "esquema_antes_repr": str(tabla_antes.schema),
        "esquema_despues_repr": str(tabla_despues.schema),
        "esquema_metadata_igual": bool(tabla_antes.schema.equals(tabla_despues.schema, check_metadata=True)),
        "dtypes_antes": dtypes_antes,
        "dtypes_despues": dtypes_despues,
        "dtypes_iguales": bool(dtypes_iguales),
        "verificacion_filas": bool(filas_ok),
        "verificacion_columnas": bool(columnas_ok),
        "verificacion_esquema": bool(esquema_ok),
        "verificacion_valores": bool(valores_ok),
        "valores_por_columna": valores_por_columna,
        "round_trip_ok": todo_ok,
        "tamano_archivo_bytes": int(RUTA_PARQUET.stat().st_size),
        "n_requires_review_true_leido": int(despues["requires_review"].sum()),
        "fila_23_leida": fila23_leida,
        "fila_23_esperado_t3": fila23_esperado,
        "fila_23_coincide_con_t3": fila23_ok,
    }
    RUTA_RESULTADOS.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
