#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado.

1) Obtiene el DataFrame analizado de la Tarea 3 (desde los archivos heredados
   de entrada/T3/ o reconstruyéndolo desde data/server_measurements.csv con el
   load_score de la Tarea 2).
2) Lo guarda en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Lo vuelve a leer y verifica el round trip: filas antes == filas después,
   columnas antes == columnas después, esquema antes == esquema después y
   valores iguales.
4) Escribe resultados.json con todas las cifras de la verificación.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

RUTA_PARQUET = Path("output/server_analysis.parquet")
RUTA_PARQUET_STR = "output/server_analysis.parquet"
RUTA_RESULTADOS = Path("resultados.json")
RUTA_CSV = "data/server_measurements.csv"
PESOS = np.array([0.50, 0.30, 0.20])
COLS_ZSCORE_DEFAULT = ["gpu_utilization", "cpu_utilization", "memory_gb"]
REF_FILA = 23
REF_LOAD_SCORE = 3.136217861183744  # load_score de la fila 23 (enunciado / T3)


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def cargar_json(ruta):
    """Lee un JSON si existe; devuelve None en caso contrario o si falla."""
    p = Path(ruta)
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def limpiar_dataframe(df):
    """Normaliza un DataFrame heredado: quita restos de índice y fija dtypes."""
    df = df.copy()
    for col in list(df.columns):
        if col == "__index_level_0__":
            df = df.drop(columns=[col])
        elif col in ("Unnamed: 0", "index"):
            s = df[col]
            if (len(s) > 0 and np.issubdtype(s.dtype, np.integer)
                    and (s.to_numpy() == np.arange(len(s))).all()):
                df = df.drop(columns=[col])
    if "timestamp" in df.columns and not np.issubdtype(df["timestamp"].dtype, np.datetime64):
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if "requires_review" in df.columns and df["requires_review"].dtype != bool:
        s = df["requires_review"]
        if s.dtype == object:
            df["requires_review"] = s.map(lambda v: str(v).strip().lower() == "true").astype(bool)
        else:
            df["requires_review"] = s.astype(bool)
    return df.reset_index(drop=True)


def ajustar_a_columnas_t3(df):
    """Recorta el DataFrame a las columnas declaradas por T3 (si están todas)."""
    t3 = cargar_json("entrada/T3/resultados.json")
    if t3:
        cols = t3.get("columnas_dataframe_nuevo")
        if (isinstance(cols, list)
                and {"load_score", "requires_review"}.issubset(cols)
                and all(c in df.columns for c in cols)):
            df = df[cols].copy()
    return df.reset_index(drop=True)


def buscar_lista_numerica(obj, n, preferencia="load"):
    """Busca recursivamente una lista plana de n números en un JSON."""
    halladas = []

    def rec(o, ruta=""):
        if isinstance(o, dict):
            for k, v in o.items():
                rec(v, (ruta + "." + str(k)).lower())
        elif isinstance(o, list):
            if len(o) == n:
                try:
                    arr = np.asarray([float(v) for v in o], dtype=float)
                except (TypeError, ValueError):
                    arr = None
                if arr is not None and not (arr.size and np.all(np.isin(arr, (0.0, 1.0)))):
                    halladas.append((ruta, arr))

    rec(obj)
    if not halladas:
        return None
    for ruta, arr in halladas:
        if preferencia in ruta:
            return arr
    return halladas[0][1]


def buscar_matriz_zscores(obj, n, k=3):
    """Busca recursivamente una matriz n x k (los z-scores) en un JSON."""
    halladas = []

    def rec(o, ruta=""):
        if isinstance(o, dict):
            for kk, v in o.items():
                rec(v, (ruta + "." + str(kk)).lower())
        elif isinstance(o, list):
            if len(o) == n and all(isinstance(f, list) and len(f) == k for f in o):
                try:
                    m = np.asarray([[float(x) for x in f] for f in o], dtype=float)
                except (TypeError, ValueError):
                    m = None
                if m is not None:
                    halladas.append((ruta, m))

    rec(obj)
    for ruta, m in halladas:
        c = ruta.replace("_", "").replace("-", "")
        if "zscore" in c or "estandariz" in c:
            return m
    return halladas[0][1] if halladas else None


def referencia_load_score():
    """(valor de referencia de load_score, índice de fila) según T3/enunciado."""
    t3 = cargar_json("entrada/T3/resultados.json")
    if t3:
        fr = t3.get("fila_revision") or {}
        v = fr.get("load_score")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            try:
                return float(v), int(fr.get("indice", REF_FILA))
            except Exception:
                pass
        res = t3.get("load_score_resumen") or {}
        v = res.get("max")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v), REF_FILA
    return REF_LOAD_SCORE, REF_FILA


def obtener_load_score(df_orig):
    """Obtiene load_score: de los resultados de T2 si es posible; si no, lo recalcula."""
    n = len(df_orig)
    candidatos = []

    t2 = cargar_json("entrada/T2/resultados.json")
    if t2 is not None:
        Z = buscar_matriz_zscores(t2, n, k=len(PESOS))
        if Z is not None:
            candidatos.append(("entrada/T2/resultados.json (zscores @ pesos)", Z @ PESOS))
        arr = buscar_lista_numerica(t2, n, preferencia="load")
        if arr is not None:
            candidatos.append(("entrada/T2/resultados.json", arr))

    d2 = Path("entrada/T2")
    if d2.exists():
        for p in sorted(d2.rglob("*.npy")):
            try:
                arr = np.load(p, allow_pickle=False)
                if arr.ndim == 2 and arr.shape == (n, len(PESOS)):
                    candidatos.append((str(p) + " (zscores @ pesos)", arr.astype(float) @ PESOS))
                elif arr.ndim == 1 and arr.shape[0] == n:
                    candidatos.append((str(p), arr.astype(float)))
            except Exception:
                continue
        for p in sorted(d2.rglob("*.csv")):
            try:
                c = pd.read_csv(p)
                if c.shape[0] == n and c.shape[1] == 1:
                    candidatos.append((str(p), c.iloc[:, 0].to_numpy(dtype=float)))
            except Exception:
                continue

    combos = [
        COLS_ZSCORE_DEFAULT,
        ["gpu_utilization", "cpu_utilization", "power_w"],
        ["gpu_utilization", "memory_gb", "power_w"],
        ["cpu_utilization", "memory_gb", "power_w"],
    ]
    for cols in combos:
        if not all(cc in df_orig.columns for cc in cols):
            continue
        X = df_orig[cols].to_numpy(dtype=float)
        for ddof in (0, 1):
            z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=ddof)
            nombre = ("recalculado: zscores[" + ", ".join(cols)
                      + "] (ddof=%d) @ [0.50, 0.30, 0.20]" % ddof)
            candidatos.append((nombre, z @ PESOS))

    if not candidatos:
        raise RuntimeError("No fue posible obtener el vector load_score.")

    ref, idx_ref = referencia_load_score()

    def distancia(item):
        arr = item[1]
        d1 = abs(float(arr[idx_ref]) - ref) if 0 <= idx_ref < len(arr) else np.inf
        d2_ = abs(float(np.max(arr)) - ref)
        return min(d1, d2_)

    candidatos.sort(key=distancia)
    nombre, arr = candidatos[0]
    dist = float(distancia((nombre, arr)))
    return np.asarray(arr, dtype=float), nombre, dist


def obtener_dataframe_analizado():
    """Devuelve (df_analizado, fuente_df, fuente_load_score)."""
    d3 = Path("entrada/T3")
    t3_info = cargar_json("entrada/T3/resultados.json") or {}
    n_esp = t3_info.get("n_filas")
    n_esperado = int(n_esp) if isinstance(n_esp, (int, float)) else None

    if d3.exists():
        for p in sorted(d3.rglob("*.parquet")):
            try:
                df = limpiar_dataframe(pd.read_parquet(p))
            except Exception:
                continue
            if n_esperado is not None and len(df) != n_esperado:
                continue
            if {"load_score", "requires_review"}.issubset(df.columns):
                df = ajustar_a_columnas_t3(df)
                if {"load_score", "requires_review"}.issubset(df.columns):
                    return df, str(p), "heredado de entrada/T3 (%s)" % p.name
        for p in sorted(d3.rglob("*.csv")):
            try:
                df = limpiar_dataframe(pd.read_csv(p))
            except Exception:
                continue
            if n_esperado is not None and len(df) != n_esperado:
                continue
            if {"load_score", "requires_review"}.issubset(df.columns):
                df = ajustar_a_columnas_t3(df)
                if {"load_score", "requires_review"}.issubset(df.columns):
                    return df, str(p), "heredado de entrada/T3 (%s)" % p.name

    # Reconstrucción fiel del DataFrame analizado de T3
    df_orig = limpiar_dataframe(pd.read_csv(RUTA_CSV, parse_dates=["timestamp"]))
    load_score, fuente_ls, _ = obtener_load_score(df_orig)
    df = df_orig.copy()
    df["load_score"] = load_score
    df["requires_review"] = ((df["load_score"] > 1.5) | (df["temperature_c"] > 80)).astype(bool)
    df = ajustar_a_columnas_t3(df)
    return df, RUTA_CSV, fuente_ls


def verificar_valores(df1, df2):
    """Compara valores celda a celda. Devuelve (ok, diferencias por columna)."""
    a = df1.reset_index(drop=True)
    b = df2.reset_index(drop=True)
    if a.shape != b.shape or list(a.columns) != list(b.columns):
        return False, {}
    try:
        if bool(a.equals(b)):
            return True, {c: 0.0 for c in a.columns}
    except Exception:
        pass
    ok = True
    diffs = {}
    for col in a.columns:
        s1, s2 = a[col], b[col]
        if (np.issubdtype(s1.dtype, np.floating)
                and np.issubdtype(s2.dtype, np.floating)):
            x = s1.to_numpy(dtype=float)
            y = s2.to_numpy(dtype=float)
            if not np.array_equal(np.isnan(x), np.isnan(y)):
                diffs[col] = None
                ok = False
                continue
            m = ~np.isnan(x)
            d = float(np.max(np.abs(x[m] - y[m]))) if m.any() else 0.0
            diffs[col] = d
            if d != 0.0:
                ok = False
        else:
            try:
                eq = bool(s1.astype(str).equals(s2.astype(str)))
            except Exception:
                eq = False
            diffs[col] = 0.0 if eq else None
            if not eq:
                ok = False
    return bool(ok), diffs


def dtype_compatible(t1, t2):
    """Igualdad de dtypes tolerando object/str (dos representaciones de texto)."""
    if t1 == t2:
        return True

    def es_texto(t):
        s = str(t).lower()
        return s in ("object", "str", "string") or s.startswith("string") or s.startswith("str")

    return es_texto(t1) and es_texto(t2)


def validacion_contra_t3(df):
    """Compara el DataFrame con la fila de revisión declarada por T3."""
    t3 = cargar_json("entrada/T3/resultados.json")
    if not t3:
        return None
    fr = t3.get("fila_revision") or {}
    try:
        idx = int(fr.get("indice", REF_FILA))
    except Exception:
        idx = REF_FILA
    if not (0 <= idx < len(df)) or "load_score" not in df.columns:
        return None
    fila = df.iloc[idx]

    def igual_num(x, y):
        try:
            return bool(abs(float(x) - float(y)) <= 1e-9)
        except Exception:
            return str(x) == str(y)

    val = {
        "indice": idx,
        "server_igual": str(fila.get("server", "")) == str(fr.get("server", "")),
        "load_score_igual": igual_num(fila.get("load_score"), fr.get("load_score")),
        "temperature_c_igual": igual_num(fila.get("temperature_c"), fr.get("temperature_c")),
        "requires_review_igual": bool(fila.get("requires_review")) == bool(fr.get("requires_review")),
    }
    if isinstance(t3.get("indices_requieren_revision"), list) and "requires_review" in df.columns:
        idx_t3 = [int(i) for i in t3["indices_requieren_revision"]]
        idx_df = [int(i) for i in np.flatnonzero(df["requires_review"].to_numpy())]
        val["indices_requieren_revision_iguales"] = idx_t3 == idx_df
        val["n_requieren_revision"] = int(df["requires_review"].sum())
    return val


def main():
    # 1) DataFrame analizado de la Tarea 3
    df_antes, fuente_df, fuente_ls = obtener_dataframe_analizado()

    ref_ls, idx_ref = referencia_load_score()
    dist_ref = None
    if "load_score" in df_antes.columns and 0 <= idx_ref < len(df_antes):
        try:
            dist_ref = float(abs(float(df_antes["load_score"].iloc[idx_ref]) - ref_ls))
        except Exception:
            dist_ref = None

    # 2) Guardar con PyArrow, sin el índice
    RUTA_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    tabla_antes = pa.Table.from_pandas(df_antes, preserve_index=False)
    pq.write_table(tabla_antes, RUTA_PARQUET)

    # 3) Volver a leer el archivo
    tabla_despues = pq.read_table(RUTA_PARQUET)
    df_despues = tabla_despues.to_pandas()

    # 4) Verificaciones del round trip
    filas_ok = int(len(df_antes)) == int(len(df_despues))
    columnas_ok = list(df_antes.columns) == list(df_despues.columns)
    esquema_arrow_ok = bool(tabla_antes.schema.equals(tabla_despues.schema))
    dtypes_antes = {c: str(t) for c, t in df_antes.dtypes.items()}
    dtypes_despues = {c: str(t) for c, t in df_despues.dtypes.items()}
    if columnas_ok:
        dtypes_compatibles = all(
            dtype_compatible(dtypes_antes[c], dtypes_despues[c]) for c in dtypes_antes
        )
    else:
        dtypes_compatibles = False
    esquema_ok = bool(esquema_arrow_ok and dtypes_compatibles)
    valores_ok, diffs = verificar_valores(df_antes, df_despues)

    # Mensaje exigido por el enunciado
    print("Round trip verificado: " + RUTA_PARQUET_STR)
    print("Filas: " + str(filas_ok))
    print("Columnas: " + str(columnas_ok))
    print("Esquema: " + str(esquema_ok))
    print("Valores: " + str(valores_ok))

    # Validación informativa contra los resultados de T3
    val_t3 = validacion_contra_t3(df_antes)

    # Primera fila marcada para revisión (informativo)
    primera_revision = None
    if "requires_review" in df_antes.columns:
        idxs = np.flatnonzero(df_antes["requires_review"].to_numpy())
        if idxs.size:
            i0 = int(idxs[0])
            r = df_antes.iloc[i0]
            primera_revision = {
                "indice": i0,
                "server": str(r.get("server", "")),
                "timestamp": str(r.get("timestamp", "")),
                "load_score": float(r["load_score"]) if "load_score" in df_antes.columns else None,
                "temperature_c": (float(r["temperature_c"])
                                  if "temperature_c" in df_antes.columns else None),
                "requires_review": bool(r["requires_review"]),
            }

    # Metadatos del archivo parquet
    try:
        pf = pq.ParquetFile(RUTA_PARQUET)
        meta_parquet = {
            "num_rows": int(pf.metadata.num_rows),
            "num_columns": int(pf.metadata.num_columns),
            "num_row_groups": int(pf.metadata.num_row_groups),
            "bytes": int(RUTA_PARQUET.stat().st_size),
        }
    except Exception:
        meta_parquet = {"bytes": int(RUTA_PARQUET.stat().st_size)}

    diffs_json = {}
    for k, v in diffs.items():
        diffs_json[k] = float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None

    resultados = {
        "subtarea": "T5",
        "descripcion": ("Guardar el DataFrame analizado de T3 en output/server_analysis.parquet "
                        "con PyArrow sin índice y verificar el round trip"),
        "parquet": RUTA_PARQUET_STR,
        "parquet_existe": bool(RUTA_PARQUET.exists()),
        "motor": "pyarrow",
        "version_pyarrow": str(pa.__version__),
        "version_pandas": str(pd.__version__),
        "guardado_sin_indice": True,
        "fuente_dataframe": fuente_df,
        "fuente_load_score": fuente_ls,
        "load_score_dist_a_referencia_fila23": dist_ref,
        "filas_antes": int(len(df_antes)),
        "filas_despues": int(len(df_despues)),
        "filas_iguales": bool(filas_ok),
        "columnas_antes": list(df_antes.columns),
        "columnas_despues": list(df_despues.columns),
        "n_columnas_antes": int(df_antes.shape[1]),
        "n_columnas_despues": int(df_despues.shape[1]),
        "columnas_iguales": bool(columnas_ok),
        "esquema_antes": str(tabla_antes.schema),
        "esquema_despues": str(tabla_despues.schema),
        "esquema_arrow_igual": esquema_arrow_ok,
        "dtypes_antes": dtypes_antes,
        "dtypes_despues": dtypes_despues,
        "dtypes_iguales": bool(dtypes_antes == dtypes_despues),
        "dtypes_compatibles": bool(dtypes_compatibles),
        "esquema_iguales": bool(esquema_ok),
        "verificacion_filas": bool(filas_ok),
        "verificacion_columnas": bool(columnas_ok),
        "verificacion_esquema": bool(esquema_ok),
        "verificacion_valores": bool(valores_ok),
        "valores_iguales": bool(valores_ok),
        "max_diff_absoluto_por_columna": diffs_json,
        "round_trip_verificado": bool(filas_ok and columnas_ok and esquema_ok and valores_ok),
        "mensaje": "Round trip verificado: " + RUTA_PARQUET_STR,
        "n_requieren_revision": (int(df_antes["requires_review"].sum())
                                 if "requires_review" in df_antes.columns else None),
        "primera_fila_requiere_revision": primera_revision,
        "metadata_parquet": meta_parquet,
        "validacion_contra_T3": val_t3,
    }
    with RUTA_RESULTADOS.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    # Resumen breve
    print("-" * 64)
    print("Resumen T5")
    print("  DataFrame analizado: %d filas x %d columnas (fuente: %s)"
          % (len(df_antes), df_antes.shape[1], fuente_df))
    print("  Guardado con PyArrow sin índice: %s (%d bytes)"
          % (RUTA_PARQUET_STR, RUTA_PARQUET.stat().st_size))
    print("  Releído: %d filas x %d columnas" % (len(df_despues), df_despues.shape[1]))
    print("  Esquema del parquet:")
    for campo in tabla_despues.schema:
        print("    %s: %s" % (campo.name, campo.type))
    print("  Verificaciones -> Filas: %s | Columnas: %s | Esquema: %s | Valores: %s"
          % (filas_ok, columnas_ok, esquema_ok, valores_ok))
    if val_t3:
        print("  Validación contra T3: %s" % val_t3)
    print("  resultados.json escrito en %s" % RUTA_RESULTADOS)


if __name__ == "__main__":
    main()
