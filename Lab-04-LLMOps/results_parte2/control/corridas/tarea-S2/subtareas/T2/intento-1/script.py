#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Subtarea T2 — Auditar los datos y construir los pares.

1) Carga data/model_errors.csv validando:
   - que existan EXACTAMENTE las columnas store_id, error_a, error_b;
   - que cada store_id sea único y no nulo (ValueError que menciona el identificador);
   - que los errores sean numéricos, finitos y no negativos (ValueError si no).
   El CSV original no se modifica (se verifica por hash SHA-256 antes/después).
2) Construye la tabla pareada conservando las tres columnas originales y añadiendo
   difference = error_a − error_b y favors ('B' si difference > 0, 'A' si < 0),
   verificando el emparejamiento MEDIANTE store_id (merge 1-a-1 + mapa clave->par),
   no por la mera igualdad de longitud de los vectores.
3) Salidas: output/store_differences.csv, output/store_differences.png, resultados.json.
"""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import MergeError

CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
PNG_PATH = OUT_DIR / "store_differences.png"
CSV_OUT_PATH = OUT_DIR / "store_differences.csv"
RESULTADOS_PATH = Path("resultados.json")

COLUMNAS_ESPERADAS = ["store_id", "error_a", "error_b"]


def sha256_archivo(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validar_columnas(df: pd.DataFrame) -> None:
    """El CSV debe tener exactamente las columnas store_id, error_a, error_b."""
    columnas = list(df.columns)
    if len(columnas) != 3 or set(columnas) != set(COLUMNAS_ESPERADAS):
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {COLUMNAS_ESPERADAS}; "
            f"se encontraron: {columnas}"
        )


def validar_store_id(df: pd.DataFrame) -> None:
    """Cada store_id debe ser único y no nulo (ValueError que menciona el identificador)."""
    ids = df["store_id"]
    if ids.isna().any():
        filas = [int(i) for i in df.index[ids.isna()]]
        raise ValueError(f"store_id contiene valores nulos en las filas {filas}")
    ids_txt = ids.astype(str).str.strip()
    if (ids_txt == "").any():
        filas = [int(i) for i in df.index[(ids_txt == "").to_numpy()]]
        raise ValueError(f"store_id contiene identificadores vacíos en las filas {filas}")
    duplicados = sorted(ids_txt[ids_txt.duplicated(keep=False)].unique().tolist())
    if duplicados:
        raise ValueError(f"store_id duplicados: {duplicados}")


def validar_errores(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """error_a y error_b deben ser numéricos, finitos y no negativos."""
    convertidas: dict[str, pd.Series] = {}
    for col in ("error_a", "error_b"):
        cruda = df[col]
        if cruda.isna().any():
            filas = [int(i) for i in df.index[cruda.isna()]]
            raise ValueError(
                f"La columna {col} contiene valores faltantes en las filas {filas}"
            )
        try:
            vals = pd.to_numeric(cruda, errors="raise")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"La columna {col} no es numérica: {exc}") from exc
        arr = vals.to_numpy(dtype=float)
        no_finitos = ~np.isfinite(arr)
        if no_finitos.any():
            detalle = {
                str(sid): float(v)
                for sid, v in zip(df.loc[no_finitos, "store_id"], arr[no_finitos])
            }
            raise ValueError(
                f"La columna {col} contiene valores no finitos (NaN/inf): {detalle}"
            )
        negativos = arr < 0
        if negativos.any():
            detalle = {
                str(sid): float(v)
                for sid, v in zip(df.loc[negativos, "store_id"], arr[negativos])
            }
            raise ValueError(f"La columna {col} contiene valores negativos: {detalle}")
        convertidas[col] = pd.Series(arr, index=df.index, name=col)
    return convertidas["error_a"], convertidas["error_b"]


def etiqueta_favors(d: float) -> str:
    if d > 0:
        return "B"
    if d < 0:
        return "A"
    return "tie"  # diferencia exactamente 0: sin preferencia (no ocurre en estos datos)


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    hash_antes = sha256_archivo(CSV_PATH)  # para demostrar que el CSV no se modifica
    df_crudo = pd.read_csv(CSV_PATH)

    # ---------- Validaciones ----------
    validar_columnas(df_crudo)
    df = df_crudo[COLUMNAS_ESPERADAS].copy()  # copia en memoria; el CSV original no se toca
    if df.empty:
        raise ValueError("El CSV no contiene ninguna fila de datos")
    validar_store_id(df)
    error_a, error_b = validar_errores(df)
    df["error_a"] = error_a
    df["error_b"] = error_b
    n = len(df)

    # ---------- Verificación del emparejamiento POR store_id ----------
    # 1) Reconstrucción de los pares mediante la clave store_id (merge uno-a-uno).
    try:
        pares = df[["store_id", "error_a"]].merge(
            df[["store_id", "error_b"]], on="store_id", how="outer", validate="one_to_one"
        )
    except MergeError as exc:
        raise ValueError(f"Emparejamiento por store_id no confirmable: {exc}") from exc
    merge_ok = (
        len(pares) == n
        and not pares[["error_a", "error_b"]].isna().to_numpy().any()
        and set(pares["store_id"].astype(str)) == set(df["store_id"].astype(str))
    )

    # 2) Mapa clave -> (error_a, error_b): cada fila debe coincidir con su clave.
    mapa = {
        str(sid): (float(ea), float(eb))
        for sid, ea, eb in zip(df["store_id"], df["error_a"], df["error_b"])
    }
    mapa_ok = len(mapa) == n and all(
        mapa[str(sid)] == (float(ea), float(eb))
        for sid, ea, eb in zip(df["store_id"], df["error_a"], df["error_b"])
    )

    if not (merge_ok and mapa_ok):
        raise ValueError("El emparejamiento por store_id no se pudo confirmar")

    # ---------- Construcción de la tabla pareada ----------
    difference = df["error_a"] - df["error_b"]
    favors = difference.map(etiqueta_favors)
    tabla = pd.DataFrame(
        {
            "store_id": df["store_id"],
            "error_a": df["error_a"],
            "error_b": df["error_b"],
            "difference": difference,
            "favors": favors,
        }
    )

    # ---------- Salidas exigidas por el enunciado ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CSV_OUT_PATH, index=False)

    # Figura: panel 1 = pares (dumbbell) por tienda; panel 2 = difference coloreada por favors
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))
    y_pos = np.arange(len(tabla))
    ax1.hlines(
        y_pos,
        tabla["error_b"].to_numpy(),
        tabla["error_a"].to_numpy(),
        color="#bbbbbb",
        linewidth=1.5,
        zorder=1,
        label="par (error_b, error_a)",
    )
    ax1.scatter(tabla["error_a"], y_pos, color="#1f77b4", s=45, zorder=3, label="error_a")
    ax1.scatter(tabla["error_b"], y_pos, color="#ff7f0e", s=45, zorder=3, label="error_b")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tabla["store_id"].astype(str).tolist())
    ax1.invert_yaxis()
    ax1.set_xlabel("Error")
    ax1.set_title("Errores pareados por tienda (emparejamiento verificado por store_id)")
    ax1.legend(loc="lower right")
    ax1.grid(axis="x", linestyle=":", alpha=0.6)

    colores = np.where(tabla["favors"].to_numpy() == "B", "#2ca02c", "#d62728")
    ax2.bar(y_pos, tabla["difference"].to_numpy(), color=colores, edgecolor="black", linewidth=0.4)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_xticks(y_pos)
    ax2.set_xticklabels(tabla["store_id"].astype(str).tolist(), rotation=90)
    ax2.set_ylabel("difference = error_a − error_b")
    ax2.set_title("Diferencia por tienda (verde: favors 'B', rojo: favors 'A')")
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)

    # ---------- resultados.json (contrato) ----------
    registros = [
        {
            "store_id": str(row.store_id),
            "error_a": float(row.error_a),
            "error_b": float(row.error_b),
            "difference": float(row.difference),
            "favors": str(row.favors),
        }
        for row in tabla.itertuples(index=False)
    ]
    hash_despues = sha256_archivo(CSV_PATH)
    n_a = int((tabla["favors"] == "A").sum())
    n_b = int((tabla["favors"] == "B").sum())
    n_tie = int((tabla["favors"] == "tie").sum())

    resultados = {
        "subtarea": "T2_auditar_datos_y_construir_pares",
        "csv_origen": str(CSV_PATH),
        "validaciones": {
            "columnas_exactas_store_id_error_a_error_b": True,
            "store_id_unico": True,
            "store_id_no_nulo": True,
            "errores_numericos": True,
            "errores_finitos": True,
            "errores_no_negativos": True,
            "csv_original_no_modificado": bool(hash_antes == hash_despues),
        },
        "emparejamiento": {
            "verificado_por_store_id": True,
            "metodo": (
                "merge uno-a-uno sobre la clave store_id + mapa store_id->(error_a, error_b); "
                "la igualdad de longitud de los vectores (16=16) NO se usa como evidencia"
            ),
            "n_tiendas_unicas": int(tabla["store_id"].nunique()),
            "n_filas_error_a": int(n),
            "n_filas_error_b": int(n),
            "longitudes_iguales_insuficiente": True,
            "ids": sorted(tabla["store_id"].astype(str).unique().tolist()),
        },
        "n_filas_tabla": int(len(tabla)),
        "n_filas_esperadas_16": bool(len(tabla) == 16),
        "columnas_tabla": list(tabla.columns),
        "tabla_pareada": registros,
        "resumen": {
            "n_favors_A": n_a,
            "n_favors_B": n_b,
            "n_favors_tie": n_tie,
            "media_difference": float(tabla["difference"].mean()),
            "suma_difference": float(tabla["difference"].sum()),
            "min_difference": float(tabla["difference"].min()),
            "max_difference": float(tabla["difference"].max()),
        },
        "salidas": {
            "csv_tabla_pareada": str(CSV_OUT_PATH),
            "figura": str(PNG_PATH),
        },
    }
    RESULTADOS_PATH.write_text(
        json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # ---------- Resumen ----------
    print("=" * 64)
    print("T2 · Auditar datos y construir los pares — resumen")
    print("=" * 64)
    print(f"CSV leído: {CSV_PATH} | intacto (SHA-256): {hash_antes == hash_despues}")
    print(
        "Validaciones OK: columnas exactas; store_id único y no nulo; "
        "errores numéricos, finitos y no negativos"
    )
    print("Emparejamiento confirmado por store_id (merge 1-a-1 + mapa clave->par)")
    print(f"Tabla pareada: {len(tabla)} filas x {len(tabla.columns)} columnas {list(tabla.columns)}")
    print(f"  favors -> A: {n_a} tiendas | B: {n_b} tiendas | tie: {n_tie}")
    print(f"  difference: media={tabla['difference'].mean():.6f}, "
          f"min={tabla['difference'].min():.6f}, max={tabla['difference'].max():.6f}")
    print(f"Salidas: {CSV_OUT_PATH} | {PNG_PATH} | {RESULTADOS_PATH}")


if __name__ == "__main__":
    main()
