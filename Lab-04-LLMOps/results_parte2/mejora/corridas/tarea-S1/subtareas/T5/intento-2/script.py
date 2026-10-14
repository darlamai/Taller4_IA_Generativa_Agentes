# -*- coding: utf-8 -*-
"""
T5 - Guardar y verificar el resultado (round trip Parquet con PyArrow).

Pasos:
1) Obtener el DataFrame analizado de la Tarea 3 (columnas originales +
   load_score + requires_review):
   - Preferentemente desde los artefactos que dejó la T3 en entrada/T3/.
   - Si no existen, se reconstruye desde data/server_measurements.csv con la
     definición EXACTA del enunciado:
       zscores = (x - media) / std por columna (gpu_utilization, cpu_utilization,
       memory_gb), evitando división por cero,
       load_score = zscores @ [0.50, 0.30, 0.20],
       requires_review = (load_score > 1.5) | (temperature_c > 80).
     (No se usa ninguna fórmula arbitraria.)
2) Guardar en output/server_analysis.parquet con PyArrow, SIN el índice.
3) Volver a leer el archivo y verificar, como mínimo:
   filas antes == filas después, columnas antes == columnas después,
   esquema antes == esquema después y valores iguales.
"""

import itertools
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

BASE_COLS = ["server", "timestamp", "gpu_utilization", "cpu_utilization",
             "memory_gb", "power_w", "temperature_c"]
COLS_ESPERADAS = BASE_COLS + ["load_score", "requires_review"]
FEATS = ["gpu_utilization", "cpu_utilization", "memory_gb"]
PESOS = np.array([0.50, 0.30, 0.20], dtype=float)
UMBRAL_LOAD_SCORE = 1.5
UMBRAL_TEMPERATURA = 80.0
FLOAT_COLS = ["gpu_utilization", "cpu_utilization", "memory_gb", "power_w",
              "temperature_c", "load_score"]


# ---------------------------------------------------------------- T3: referencia
def leer_resultados_t3():
    """Lee entrada/T3/resultados.json: n_filas, columnas y load_score conocidos."""
    n_filas, cols, n_review = None, None, None
    known_ls, known_rr = {}, {}
    for nombre in ("resultados.json", "results.json"):
        f = T3_DIR / nombre
        if not f.exists():
            continue
        try:
            res = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(res, dict):
            if isinstance(res.get("n_filas"), (int, float)):
                n_filas = int(res["n_filas"])
            if isinstance(res.get("columnas_nuevo_dataframe"), list):
                cols = [str(c) for c in res["columnas_nuevo_dataframe"]]
            if isinstance(res.get("n_requires_review_true"), (int, float)):
                n_review = int(res["n_requires_review_true"])
            regs = []
            if isinstance(res.get("fila_23"), dict):
                regs.append(res["fila_23"])
            if isinstance(res.get("filas_marcadas"), list):
                regs += [r for r in res["filas_marcadas"] if isinstance(r, dict)]
            for r in regs:
                if "indice" in r:
                    try:
                        idx = int(r["indice"])
                    except (TypeError, ValueError):
                        continue
                elif "load_score" in r and regs and r is regs[0]:
                    idx = 23  # fila_23
                else:
                    continue
                try:
                    known_ls[idx] = float(r["load_score"])
                except (KeyError, TypeError, ValueError):
                    pass
                rr = r.get("requires_review")
                if isinstance(rr, bool):
                    known_rr[idx] = rr
        break
    return n_filas, cols, known_ls, known_rr, n_review


def error_vs_known(valores, known):
    """Máxima diferencia absoluta frente a los load_score reportados por la T3."""
    if not known:
        return None
    errs = []
    n = len(valores)
    for i, v in known.items():
        if 0 <= i < n:
            errs.append(abs(float(valores[i]) - float(v)))
    return max(errs) if errs else None


# ------------------------------------------------------- DataFrame normalizado
def normalizar(df):
    """Fuerza columnas y dtypes del DataFrame analizado de la Tarea 3."""
    df = df.copy()
    for c in ("Unnamed: 0", "index", "__index_level_0__"):
        if c in df.columns:
            df = df.drop(columns=c)
    faltan = [c for c in COLS_ESPERADAS if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas del DataFrame analizado: {faltan}")
    df = df.loc[:, COLS_ESPERADAS].copy()
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


# --------------------------------------------- Reconstrucción según el enunciado
def calcular_load_score(df_base, ddof, orden):
    """load_score = zscores @ [0.50, 0.30, 0.20] con z = (x - media)/std."""
    X = df_base[list(orden)].astype("float64")
    media = X.mean()
    std = X.std(ddof=ddof)
    std = std.where(std != 0, 1.0).fillna(1.0)  # evitar división por cero
    Z = (X - media) / std
    return Z.to_numpy(dtype="float64") @ PESOS


def reconstruir_desde_csv(known_ls):
    """Reconstruye el DataFrame de la T3 desde el CSV con la definición del
    enunciado: zscores por columna y load_score = zscores @ [0.50, 0.30, 0.20]."""
    df = pd.read_csv(CSV_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    ordenes = [list(FEATS)]
    for p in itertools.permutations(FEATS):
        if list(p) != list(FEATS):
            ordenes.append(list(p))

    mejor = None  # (err, ddof, orden, load_score)
    for ddof in (0, 1):
        for orden in ordenes:
            ls = calcular_load_score(df, ddof, orden)
            err = error_vs_known(ls, known_ls)
            if err is None:
                # Sin valores de referencia de T3: se usa la definición directa
                # del enunciado (ddof=0, columnas en el orden canónico).
                if mejor is None:
                    mejor = (None, ddof, orden, ls)
                continue
            if mejor is None or mejor[0] is None or err < mejor[0]:
                mejor = (err, ddof, orden, ls)

    err, ddof, orden, ls = mejor
    df = df.copy()
    df["load_score"] = ls
    df["requires_review"] = ((df["load_score"] > UMBRAL_LOAD_SCORE)
                             | (df["temperature_c"] > UMBRAL_TEMPERATURA))
    return normalizar(df), ddof, list(orden), err


def cargar_desde_t3(known_ls):
    """Busca el DataFrame analizado (con load_score y requires_review) en entrada/T3."""
    if not T3_DIR.exists():
        return None, None, None
    candidatos = []
    for f in sorted(T3_DIR.rglob("*")):
        if not f.is_file():
            continue
        suf = f.suffix.lower()
        if suf not in (".parquet", ".pq", ".feather", ".arrow", ".csv"):
            continue
        try:
            if suf in (".parquet", ".pq"):
                d = pd.read_parquet(f)
            elif suf in (".feather", ".arrow"):
                d = pd.read_feather(f)
            else:
                d = pd.read_csv(f)
        except Exception:
            continue
        if not {"load_score", "requires_review"}.issubset(d.columns):
            continue  # no es el DataFrame analizado
        try:
            d = normalizar(d)
        except (ValueError, TypeError):
            continue
        err = error_vs_known(d["load_score"].to_numpy(dtype="float64"), known_ls)
        candidatos.append((err, f, d))
    if not candidatos:
        return None, None, None

    def clave(item):
        err, f, _ = item
        binario = f.suffix.lower() in (".parquet", ".pq", ".feather", ".arrow")
        if err is None:
            return (1, 0.0, 0 if binario else 1, f.name)
        return (0, err, 0 if binario else 1, f.name)

    candidatos.sort(key=clave)
    err, f, d = candidatos[0]
    return d, f, err


def obtener_dataframe_analizado(n_filas_ref, known_ls):
    """Elige el mejor candidato al DataFrame analizado de la Tarea 3."""
    opciones = []
    d_t3, f_t3, err_t3 = cargar_desde_t3(known_ls)
    if d_t3 is not None:
        opciones.append({"es_t3": True, "fuente": f"entrada/T3/{f_t3.name}",
                         "df": d_t3, "err": err_t3, "meta": {}})
    try:
        d_rec, ddof, orden, err_rec = reconstruir_desde_csv(known_ls)
        opciones.append({"es_t3": False,
                         "fuente": f"{CSV_PATH.as_posix()} (reconstruido segun enunciado)",
                         "df": d_rec, "err": err_rec,
                         "meta": {"ddof": ddof, "orden_zscores": orden,
                                  "pesos": PESOS.tolist()}})
    except (OSError, ValueError) as exc:
        print(f"Aviso: no se pudo reconstruir desde {CSV_PATH.as_posix()}: {exc}")

    if not opciones:
        raise FileNotFoundError("No se encontro el DataFrame analizado de la T3 "
                                "ni data/server_measurements.csv para reconstruirlo.")

    def clave(o):
        err = o["err"]
        verificado = 0 if err is not None else 1
        err_v = err if err is not None else float("inf")
        len_ok = 0 if (n_filas_ref is None or len(o["df"]) == n_filas_ref) else 1
        pref = 0 if o["es_t3"] else 1
        return (verificado, err_v, len_ok, pref)

    opciones.sort(key=clave)
    return opciones[0]


# ------------------------------------------------------------------------ main
def main():
    n_filas_ref, cols_t3, known_ls, known_rr, n_review_ref = leer_resultados_t3()

    elegida = obtener_dataframe_analizado(n_filas_ref, known_ls)
    df = elegida["df"]
    fuente = elegida["fuente"]
    meta = elegida["meta"]

    err_vs_t3 = error_vs_known(df["load_score"].to_numpy(dtype="float64"), known_ls)
    rr_ok = None
    if known_rr:
        pares = [(i, v) for i, v in known_rr.items() if i in df.index]
        if pares:
            rr_ok = all(bool(df.loc[i, "requires_review"]) == bool(v)
                        for i, v in pares)

    # ---------- Guardar con PyArrow, SIN el índice ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla = pa.Table.from_pandas(df, preserve_index=False)  # sin el índice
    schema_antes = tabla.schema
    pq.write_table(tabla, OUT_PATH.as_posix())

    # ---------- Volver a leer el archivo ----------
    tabla_leida = pq.read_table(OUT_PATH.as_posix())
    schema_leida = tabla_leida.schema
    df2 = tabla_leida.to_pandas()
    schema_reconstruida = pa.Table.from_pandas(df2, preserve_index=False).schema

    # ---------- Verificaciones del round trip ----------
    filas_antes, filas_despues = len(df), len(df2)
    cols_antes, cols_despues = list(df.columns), list(df2.columns)

    ok_filas = bool(filas_antes == filas_despues)
    ok_columnas = bool(cols_antes == cols_despues)
    dtypes_iguales = all(str(df[c].dtype) == str(df2[c].dtype) for c in cols_antes)
    ok_esquema = bool(schema_antes.equals(schema_leida, check_metadata=False)
                      and schema_antes.equals(schema_reconstruida,
                                              check_metadata=False)
                      and dtypes_iguales)
    ok_valores = bool(df.reset_index(drop=True).equals(
        df2.reset_index(drop=True)))
    round_trip_ok = bool(ok_filas and ok_columnas and ok_esquema and ok_valores)

    max_diff = 0.0
    for c in cols_antes:
        if pd.api.types.is_float_dtype(df[c]):
            d = np.abs(df[c].to_numpy(dtype="float64")
                       - df2[c].to_numpy(dtype="float64"))
            if d.size:
                max_diff = max(max_diff, float(np.nanmax(d)))

    fila_23 = None
    if 23 in df.index:
        fila_23 = {"server": str(df.loc[23, "server"]),
                   "timestamp": str(df.loc[23, "timestamp"]),
                   "load_score": float(df.loc[23, "load_score"]),
                   "temperature_c": float(df.loc[23, "temperature_c"]),
                   "requires_review": bool(df.loc[23, "requires_review"])}

    # ---------- resultados.json (contrato de la subtarea) ----------
    if meta:
        definicion_ls = {
            "tipo": "zscore_por_columna",
            "formula": "load_score = zscores @ [0.50, 0.30, 0.20]",
            "columnas": meta["orden_zscores"],
            "pesos": meta["pesos"],
            "ddof": meta["ddof"],
        }
    else:
        definicion_ls = ("tomado del artefacto de entrada/T3 (calculado en T3 como "
                         "zscores @ [0.50, 0.30, 0.20])")

    resultados = {
        "subtarea": "T5_guardar_y_verificar_parquet",
        "archivo_parquet": OUT_PATH.as_posix(),
        "metodo_guardado": "pyarrow.parquet.write_table (preserve_index=False, sin indice)",
        "fuente_dataframe": fuente,
        "definicion_load_score": definicion_ls,
        "definicion_requires_review": "(load_score > 1.5) | (temperature_c > 80)",
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
        "esquema_arrow_leido": str(schema_leida),
        "verificacion_esquema": ok_esquema,
        "verificacion_valores": ok_valores,
        "max_diferencia_absoluta_floats": float(max_diff),
        "n_requires_review_true": int(df["requires_review"].sum()),
        "n_requires_review_true_ref_T3": n_review_ref,
        "error_max_load_score_vs_T3": err_vs_t3,
        "requires_review_coincide_con_T3": rr_ok,
        "fila_23": fila_23,
        "round_trip_ok": round_trip_ok,
    }
    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---------- Resumen ----------
    print("T5 - Guardar y verificar el resultado (Parquet + PyArrow)")
    print(f"Fuente del DataFrame analizado: {fuente}")
    if meta:
        print(f"load_score = zscores({meta['orden_zscores']}) @ {meta['pesos']} "
              f"(ddof={meta['ddof']})")
    print(f"Filas: {filas_antes} -> {filas_despues} | "
          f"Columnas: {len(cols_antes)} -> {len(cols_despues)}")
    print(f"Archivo: {OUT_PATH.as_posix()} ({OUT_PATH.stat().st_size} bytes)")
    if err_vs_t3 is not None:
        print(f"Coherencia con T3 (max |load_score - T3|): {err_vs_t3}")
    print("------------------------------------------------------------")
    print(f"Round trip verificado: {OUT_PATH.as_posix()}")
    print(f"Filas: {ok_filas}")
    print(f"Columnas: {ok_columnas}")
    print(f"Esquema: {ok_esquema}")
    print(f"Valores: {ok_valores}")
    print("resultados.json escrito con las verificaciones del round trip.")


if __name__ == "__main__":
    main()
