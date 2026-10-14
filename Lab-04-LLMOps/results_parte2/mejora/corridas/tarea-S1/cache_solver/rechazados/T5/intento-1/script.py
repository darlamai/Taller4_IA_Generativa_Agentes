# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado.

1) Obtiene el DataFrame analizado de la Tarea 3 (7 columnas originales +
   load_score + requires_review).
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Vuelve a leer el archivo y verifica el round trip:
   - filas antes == filas después
   - columnas antes == columnas después
   - esquema antes == esquema después
   - valores iguales
Imprime: 'Round trip verificado: output/server_analysis.parquet' con las
cuatro verificaciones (Filas, Columnas, Esquema, Valores).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

T3_DIR = Path("entrada") / "T3"
CSV_PATH = Path("data") / "server_measurements.csv"
OUT_DIR = Path("output")
OUT_PATH = OUT_DIR / "server_analysis.parquet"

EXPECTED_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
                 "memory_gb", "power_w", "temperature_c", "load_score",
                 "requires_review"]
FLOAT_COLS = ["gpu_utilization", "cpu_utilization", "memory_gb", "power_w",
              "temperature_c", "load_score"]


def normalizar_dataframe(df):
    """Ajusta columnas y dtypes al formato del DataFrame analizado de la Tarea 3."""
    df = df.copy()
    for c in ("Unnamed: 0", "index", "__index_level_0__"):
        if c in df.columns:
            df = df.drop(columns=c)
    faltantes = [c for c in EXPECTED_COLS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas del DataFrame analizado: {faltantes}")
    df = df.loc[:, EXPECTED_COLS].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    for c in FLOAT_COLS:
        df[c] = pd.to_numeric(df[c]).astype("float64")
    r = df["requires_review"]
    if pd.api.types.is_bool_dtype(r):
        df["requires_review"] = r.astype("bool")
    else:
        df["requires_review"] = (r.astype(str).str.strip().str.lower()
                                 .isin(["true", "1", "t", "yes"]).astype("bool"))
    return df.reset_index(drop=True)


def leer_resultados_t3():
    """Lee entrada/T3/results.json: n_filas, columnas y load_score conocidos."""
    n_filas, cols, known = None, None, {}
    f = T3_DIR / "results.json"
    if f.exists():
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            res = {}
        if isinstance(res.get("n_filas"), (int, float)):
            n_filas = int(res["n_filas"])
        if isinstance(res.get("columnas_nuevo_dataframe"), list):
            cols = [str(c) for c in res["columnas_nuevo_dataframe"]]
        registros = []
        if isinstance(res.get("fila_23"), dict):
            registros.append(res["fila_23"])
        if isinstance(res.get("filas_marcadas"), list):
            registros += [r for r in res["filas_marcadas"] if isinstance(r, dict)]
        for r in registros:
            try:
                known[int(r["indice"])] = float(r["load_score"])
            except (KeyError, TypeError, ValueError):
                pass
    return n_filas, cols, known


def cargar_desde_t3(n_filas, cols_esperadas):
    """Carga el DataFrame analizado desde los archivos que dejó la Tarea 3."""
    if not T3_DIR.exists():
        return None
    puntuados = []
    orden = 0
    for patron in ("*.parquet", "*.pq", "*.feather", "*.csv"):
        for f in sorted(T3_DIR.glob(patron)):
            try:
                if f.suffix in (".parquet", ".pq"):
                    dfc = pd.read_parquet(f)
                elif f.suffix == ".feather":
                    dfc = pd.read_feather(f)
                else:
                    dfc = pd.read_csv(f, encoding="utf-8-sig")
            except Exception:
                continue
            if not {"load_score", "requires_review"}.issubset(dfc.columns):
                continue  # no es el DataFrame analizado
            score = 1
            if n_filas is not None and len(dfc) == n_filas:
                score += 4
            if cols_esperadas and list(dfc.columns) == list(cols_esperadas):
                score += 2
            if "timestamp" in dfc.columns:
                score += 1
            if f.suffix in (".parquet", ".pq", ".feather"):
                score += 1
            puntuados.append((score, orden, dfc))
            orden += 1
    puntuados.sort(key=lambda t: (-t[0], -len(t[2]), t[1]))
    for _, _, dfc in puntuados:
        if n_filas is not None and len(dfc) != n_filas:
            continue
        try:
            return normalizar_dataframe(dfc)
        except ValueError:
            continue
    return None


def reconstruir_desde_csv(known_scores):
    """Plan B: reconstruye el DataFrame analizado (T3) desde el CSV original."""
    df = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    def z(s, ddof):
        s = s.astype(float)
        return (s - s.mean()) / s.std(ddof=ddof)

    g = df["gpu_utilization"]
    c = df["cpu_utilization"]
    m = df["memory_gb"]
    candidatos = {}
    for ddof in (0, 1):
        zg, zc, zm = z(g, ddof), z(c, ddof), z(m, ddof)
        candidatos[f"z_gpu_ddof{ddof}"] = zg
        candidatos[f"z_cpu_ddof{ddof}"] = zc
        candidatos[f"z_mem_ddof{ddof}"] = zm
        candidatos[f"z_power_ddof{ddof}"] = z(df["power_w"], ddof)
        candidatos[f"z_gpu_mas_cpu_ddof{ddof}"] = z(g + c, ddof)
        candidatos[f"z_gpu_mas_cpu_mas_mem_ddof{ddof}"] = z(g + c + m, ddof)
        candidatos[f"media_z_gpu_cpu_mem_ddof{ddof}"] = (zg + zc + zm) / 3.0
        candidatos[f"max_z_gpu_cpu_mem_ddof{ddof}"] = pd.concat(
            [zg, zc, zm], axis=1).max(axis=1)
    candidatos["gpu/100+cpu/100+mem/64"] = g / 100.0 + c / 100.0 + m / 64.0
    candidatos["gpu/100+cpu/100+mem/32"] = g / 100.0 + c / 100.0 + m / 32.0
    candidatos["(gpu+cpu+mem)/100"] = (g + c + m) / 100.0

    formula = None
    if known_scores:
        idx = np.array(sorted(known_scores), dtype=int)
        objetivo = np.array([known_scores[i] for i in idx], dtype=float)
        for nombre, cand in candidatos.items():
            vals = np.asarray(cand, dtype=float)[idx]
            if np.allclose(vals, objetivo, rtol=0.0, atol=1e-6):
                df["load_score"] = cand.astype("float64")
                formula = nombre
                break
    if formula is None:
        df["load_score"] = (g / 100.0 + c / 100.0 + m / 64.0).astype("float64")
        formula = "gpu/100 + cpu/100 + mem/64 (defecto)"
    df["requires_review"] = ((df["load_score"] > 1.5)
                             | (df["temperature_c"] > 80)).astype("bool")
    return normalizar_dataframe(df), formula


def main():
    n_filas_t3, cols_t3, known_scores = leer_resultados_t3()

    df = cargar_desde_t3(n_filas_t3, cols_t3)
    if df is not None:
        fuente = "entrada/T3 (DataFrame analizado de la Tarea 3)"
    else:
        df, formula = reconstruir_desde_csv(known_scores)
        fuente = f"reconstruido desde {CSV_PATH.as_posix()} (load_score: {formula})"

    # Coherencia informativa con los load_score reportados por la Tarea 3
    coherencia_t3 = None
    if known_scores:
        pares = [(i, v) for i, v in known_scores.items() if i in df.index]
        if pares:
            coherencia_t3 = all(abs(float(df.loc[i, "load_score"]) - v) <= 1e-6
                                for i, v in pares)

    # ---------- Guardar con PyArrow, sin el índice ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pandas(df, preserve_index=False)  # sin el índice
    schema_antes = tabla.schema
    pq.write_table(tabla, str(OUT_PATH))

    # ---------- Volver a leer el archivo ----------
    tabla2 = pq.read_table(str(OUT_PATH))
    schema_leido = tabla2.schema
    df2 = tabla2.to_pandas()
    schema_despues = pa.Table.from_pandas(df2, preserve_index=False).schema

    # ---------- Verificaciones del round trip ----------
    filas_antes, filas_despues = len(df), len(df2)
    cols_antes, cols_despues = list(df.columns), list(df2.columns)

    ok_filas = bool(filas_antes == filas_despues)
    ok_columnas = bool(cols_antes == cols_despues)

    esquema_pandas_ok = (cols_antes == cols_despues and all(
        str(df[c].dtype) == str(df2[c].dtype) for c in cols_antes))
    esquema_arrow_ok = (schema_antes.equals(schema_leido, check_metadata=False)
                        and schema_antes.equals(schema_despues,
                                                check_metadata=False))
    ok_esquema = bool(esquema_pandas_ok and esquema_arrow_ok)

    ok_valores = bool(df.reset_index(drop=True).equals(
        df2.reset_index(drop=True)))

    max_diff = 0.0
    for c in cols_antes:
        if pd.api.types.is_float_dtype(df[c]):
            d = np.abs(df[c].to_numpy(dtype=float) - df2[c].to_numpy(dtype=float))
            if d.size:
                max_diff = max(max_diff, float(np.nanmax(d)))

    # ---------- resultados.json (contrato de la subtarea) ----------
    resultados = {
        "subtarea": "T5_guardar_y_verificar_parquet",
        "archivo_parquet": OUT_PATH.as_posix(),
        "metodo_guardado": "pyarrow.parquet.write_table, preserve_index=False (sin indice)",
        "fuente_dataframe": fuente,
        "filas_antes": int(filas_antes),
        "filas_despues": int(filas_despues),
        "verificacion_filas": ok_filas,
        "columnas_antes": cols_antes,
        "columnas_despues": cols_despues,
        "n_columnas_antes": int(len(cols_antes)),
        "n_columnas_despues": int(len(cols_despues)),
        "verificacion_columnas": ok_columnas,
        "esquema_antes": {c: str(df[c].dtype) for c in cols_antes},
        "esquema_despues": {c: str(df2[c].dtype) for c in cols_despues},
        "esquema_arrow_antes": str(schema_antes),
        "esquema_arrow_leido": str(schema_leido),
        "verificacion_esquema": ok_esquema,
        "verificacion_valores": ok_valores,
        "max_diferencia_absoluta_floats": float(max_diff),
        "n_requires_review_true": int(df["requires_review"].sum()),
        "fila_23": ({
            "load_score": float(df.loc[23, "load_score"]),
            "temperature_c": float(df.loc[23, "temperature_c"]),
            "requires_review": bool(df.loc[23, "requires_review"]),
        } if 23 in df.index else None),
        "coherencia_load_score_con_T3": coherencia_t3,
        "round_trip_ok": bool(ok_filas and ok_columnas and ok_esquema
                              and ok_valores),
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Resumen ----------
    print("T5 - Guardar y verificar el resultado (Parquet con PyArrow)")
    print(f"Fuente del DataFrame analizado: {fuente}")
    print(f"Filas antes: {filas_antes} | Filas despues: {filas_despues}")
    print(f"Columnas antes: {len(cols_antes)} | "
          f"Columnas despues: {len(cols_despues)}")
    print(f"Archivo escrito: {OUT_PATH.as_posix()} "
          f"({OUT_PATH.stat().st_size} bytes)")
    print("------------------------------------------------------------")
    print(f"Round trip verificado: {OUT_PATH.as_posix()}")
    print(f"Filas: {ok_filas}")
    print(f"Columnas: {ok_columnas}")
    print(f"Esquema: {ok_esquema}")
    print(f"Valores: {ok_valores}")
    print("resultados.json escrito con las verificaciones del round trip.")


if __name__ == "__main__":
    main()
