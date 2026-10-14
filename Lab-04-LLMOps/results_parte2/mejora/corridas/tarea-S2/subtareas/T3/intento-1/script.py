#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T3 — Hacer visible la evidencia por tienda.

Genera una figura de barras divergentes (UNA observación por tienda) que muestra
el signo y la magnitud de difference = error_a − error_b respecto a una
referencia visible en cero, rotulada con store_id, y la guarda en:
  - output/store_differences.png
También escribe la tabla pareada en output/store_differences.csv y el contrato
de cifras en resultados.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COLUMNAS_ESPERADAS = ["store_id", "error_a", "error_b"]
COLOR_POS = "#1f77b4"    # difference > 0 -> favorece a B
COLOR_NEG = "#d62728"    # difference < 0 -> favorece a A
COLOR_EMPATE = "#7f7f7f"  # difference == 0


def cargar_tabla_pareada():
    """Devuelve (tabla, fuente). Tabla: store_id, error_a, error_b, difference, favors."""
    csv_path = Path("data/model_errors.csv")
    t2_json = Path("entrada/T2/resultados.json")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if list(df.columns) != COLUMNAS_ESPERADAS:
            raise ValueError(
                f"Columnas esperadas exactas {COLUMNAS_ESPERADAS}, "
                f"encontradas {list(df.columns)}"
            )
        # Identificador único y no nulo
        if df["store_id"].isna().any():
            nulos = int(df["store_id"].isna().sum())
            raise ValueError(f"El identificador store_id contiene {nulos} valores nulos")
        dup = df.loc[df["store_id"].duplicated(), "store_id"].tolist()
        if dup:
            raise ValueError(f"El identificador store_id está duplicado: {dup}")
        # Errores numéricos, finitos y no negativos
        for col in ("error_a", "error_b"):
            if not np.issubdtype(df[col].dtype, np.number):
                raise ValueError(f"La columna {col} no es numérica (dtype={df[col].dtype})")
            vals = df[col].to_numpy(dtype=float)
            if not np.all(np.isfinite(vals)):
                raise ValueError(f"La columna {col} contiene valores no finitos")
            if np.any(vals < 0):
                raise ValueError(f"La columna {col} contiene valores negativos")
        # Emparejamiento verificado por la clave store_id (unión 1:1, no por posición)
        n = len(df)
        base = df[["store_id", "error_a"]].merge(
            df[["store_id", "error_b"]], on="store_id", how="inner", validate="one_to_one"
        )
        if len(base) != n:
            raise ValueError("La unión 1:1 por store_id no recuperó todos los pares")
        fuente = "data/model_errors.csv"
    elif t2_json.exists():
        t2 = json.loads(t2_json.read_text(encoding="utf-8"))
        base = pd.DataFrame(t2["tabla_pareada"])[COLUMNAS_ESPERADAS].copy()
        fuente = "entrada/T2/resultados.json (tabla_pareada de T2)"
    else:
        raise FileNotFoundError(
            "No se encontró data/model_errors.csv ni entrada/T2/resultados.json"
        )

    base = base.copy()
    base["difference"] = base["error_a"].astype(float) - base["error_b"].astype(float)
    base["favors"] = np.where(
        base["difference"] > 0, "B", np.where(base["difference"] < 0, "A", "EMPATE")
    )
    base = base.sort_values("store_id", kind="mergesort").reset_index(drop=True)
    return base, fuente


def verificar_contra_t2(tabla):
    """Comprueba que la tabla reconstruida coincida con la validada en T2."""
    t2_json = Path("entrada/T2/resultados.json")
    if not t2_json.exists():
        return {"verificado": False, "motivo": "entrada/T2/resultados.json no disponible"}
    try:
        t2 = json.loads(t2_json.read_text(encoding="utf-8"))
        filas = t2.get("tabla_pareada", [])
        if not filas:
            return {"verificado": False, "motivo": "sin tabla_pareada en T2"}
        ref = {str(r["store_id"]): float(r["difference"]) for r in filas}
        mios = {str(s): float(d) for s, d in zip(tabla["store_id"], tabla["difference"])}
        if set(ref) != set(mios):
            return {"verificado": False, "motivo": "los store_ids difieren de T2"}
        max_abs = float(max(abs(ref[s] - mios[s]) for s in ref))
        return {
            "verificado": True,
            "n_tiendas": len(ref),
            "max_dif_abs_vs_T2": max_abs,
            "coincide": bool(max_abs < 1e-9),
        }
    except Exception as exc:  # noqa: BLE001
        return {"verificado": False, "motivo": f"error al comparar con T2: {exc}"}


def crear_figura(orden, ruta_png):
    """Barras divergentes horizontales: una por tienda, referencia visible en cero."""
    colores = {"B": COLOR_POS, "A": COLOR_NEG, "EMPATE": COLOR_EMPATE}
    bar_colors = [colores[f] for f in orden["favors"]]
    y = np.arange(len(orden))
    diffs = orden["difference"].to_numpy(dtype=float)
    xmax = float(np.max(np.abs(diffs))) if diffs.size else 1.0
    pad = 0.05 * xmax if xmax > 0 else 0.05

    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    ax.barh(y, diffs, color=bar_colors, edgecolor="black", linewidth=0.6,
            height=0.68, zorder=2)

    # Referencia visible en cero
    ax.axvline(0.0, color="black", linewidth=1.8, zorder=4)

    # Etiqueta numérica con signo al final de cada barra (magnitud legible)
    i_max_pos = int(np.argmax(diffs))
    i_max_neg = int(np.argmin(diffs))
    for yi, d, i in zip(y, diffs, range(diffs.size)):
        if d > 0:
            txt, xpos, ha = f"+{d:.2f}", d + pad, "left"
        elif d < 0:
            txt, xpos, ha = f"{d:.2f}", d - pad, "right"
        else:
            txt, xpos, ha = "0.00", pad, "left"
        peso = "bold" if i in (i_max_pos, i_max_neg) else "normal"
        ax.text(xpos, yi, txt, va="center", ha=ha, fontsize=9,
                fontweight=peso, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(orden["store_id"].tolist(), fontsize=10)
    ax.invert_yaxis()  # la mayor diferencia positiva queda arriba
    ax.set_xlim(float(diffs.min()) - 8 * pad, float(diffs.max()) + 8 * pad)
    ax.set_xlabel("difference = error_a − error_b", fontsize=11)
    ax.set_title(
        "Diferencia de error por tienda: modelo A vs modelo B\n"
        f"Barras divergentes respecto a la referencia en cero · {len(orden)} tiendas",
        fontsize=12,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=9)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)

    handles = [
        Patch(facecolor=COLOR_POS, edgecolor="black",
              label="difference > 0 → favorece a B"),
        Patch(facecolor=COLOR_NEG, edgecolor="black",
              label="difference < 0 → favorece a A"),
        Line2D([0], [0], color="black", linewidth=1.8,
               label="Referencia en cero (difference = 0)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9, framealpha=0.95)

    nA = int((orden["favors"] == "A").sum())
    nB = int((orden["favors"] == "B").sum())
    nE = int((orden["favors"] == "EMPATE").sum())
    ax.text(0.99, 0.01,
            f"{len(orden)} tiendas · favorecen a B: {nB} · favorecen a A: {nA} · empates: {nE}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.7", alpha=0.9))

    fig.tight_layout()
    ruta_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(ruta_png), dpi=120)
    plt.close(fig)


def escribir_csv(tabla, ruta_csv):
    """Escribe la tabla pareada; si T2 dejó su CSV, se conserva tal cual."""
    ruta_csv.parent.mkdir(parents=True, exist_ok=True)
    candidatos = [
        Path("entrada/T2/store_differences.csv"),
        Path("entrada/T2/output/store_differences.csv"),
    ]
    t2_csv = next((p for p in candidatos if p.exists()), None)
    if t2_csv is not None:
        ruta_csv.write_bytes(t2_csv.read_bytes())
    else:
        tabla[["store_id", "error_a", "error_b", "difference", "favors"]].to_csv(
            ruta_csv, index=False
        )


def main():
    tabla, fuente = cargar_tabla_pareada()
    consistencia = verificar_contra_t2(tabla)

    # Orden para la figura: de mayor diferencia positiva a mayor negativa
    orden = tabla.sort_values("difference", ascending=False,
                              kind="mergesort").reset_index(drop=True)

    ruta_png = Path("output/store_differences.png")
    ruta_csv = Path("output/store_differences.csv")
    crear_figura(orden, ruta_png)
    escribir_csv(tabla, ruta_csv)

    diffs = tabla["difference"].astype(float)
    i_pos = int(np.argmax(diffs.to_numpy()))
    i_neg = int(np.argmin(diffs.to_numpy()))
    fila_pos = tabla.iloc[i_pos]
    fila_neg = tabla.iloc[i_neg]

    ids_A = tabla.loc[tabla["favors"] == "A", "store_id"].tolist()
    ids_B = tabla.loc[tabla["favors"] == "B", "store_id"].tolist()
    ids_E = tabla.loc[tabla["favors"] == "EMPATE", "store_id"].tolist()
    nA, nB, nE = len(ids_A), len(ids_B), len(ids_E)

    png_ok = ruta_png.exists() and ruta_png.stat().st_size > 0
    if not png_ok:
        raise RuntimeError("No se pudo generar output/store_differences.png")

    por_tienda = []
    for r in tabla.sort_values("store_id", kind="mergesort").itertuples(index=False):
        por_tienda.append({
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        })

    resultados = {
        "subtarea": "T3",
        "descripcion": (
            "Figura con una observación por tienda: signo y magnitud de difference "
            "respecto a una referencia visible en cero (barras divergentes rotuladas "
            "con store_id)."
        ),
        "fuente_datos": fuente,
        "n_tiendas": int(len(tabla)),
        "store_ids": tabla["store_id"].tolist(),
        "conteos": {
            "favorece_A": nA,
            "favorece_B": nB,
            "empates": nE,
            "suma_conteos": int(nA + nB + nE),
        },
        "tiendas_favorecen_A": ids_A,
        "tiendas_favorecen_B": ids_B,
        "tiendas_empate": ids_E,
        "mayor_diferencia_positiva": {
            "store_id": str(fila_pos["store_id"]),
            "difference": float(fila_pos["difference"]),
            "favors": str(fila_pos["favors"]),
        },
        "mayor_diferencia_negativa": {
            "store_id": str(fila_neg["store_id"]),
            "difference": float(fila_neg["difference"]),
            "favors": str(fila_neg["favors"]),
        },
        "diferencias_por_tienda": por_tienda,
        "rango_difference": {"min": float(diffs.min()), "max": float(diffs.max())},
        "figura": {
            "ruta": "output/store_differences.png",
            "tipo": "barras horizontales divergentes, una por tienda, ordenadas por difference",
            "n_observaciones": int(len(orden)),
            "store_ids_en_figura": orden["store_id"].tolist(),
            "referencia_cero_visible": True,
            "codificacion_signo": (
                "lado respecto de 0 y color (azul: difference>0 favorece a B; "
                "rojo: difference<0 favorece a A)"
            ),
            "codificacion_magnitud": (
                "longitud de la barra y etiqueta numérica con signo al final de cada "
                "barra (2 decimales; en negrita las de mayor magnitud en cada dirección)"
            ),
        },
        "archivos_generados": {
            "figura_png": "output/store_differences.png",
            "tabla_csv": "output/store_differences.csv",
        },
        "verificacion": {
            "png_existe": bool(png_ok),
            "png_bytes": int(ruta_png.stat().st_size),
            "csv_existe": bool(ruta_csv.exists()),
            "una_observacion_por_tienda": bool(tabla["store_id"].is_unique),
            "todas_las_tiendas_en_figura": bool(
                len(orden) == len(tabla)
                and set(orden["store_id"]) == set(tabla["store_id"])
            ),
            "consistencia_con_T2": consistencia,
        },
    }

    Path("resultados.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 64)
    print("T3 — Figura de diferencias por tienda (barras divergentes)")
    print("=" * 64)
    print(f"Fuente de datos       : {fuente}")
    print(f"Tiendas representadas : {len(tabla)} (una observación por tienda)")
    print(f"Favorecen a B (d>0)   : {nB} -> {', '.join(ids_B)}")
    print(f"Favorecen a A (d<0)   : {nA} -> {', '.join(ids_A)}")
    print(f"Empates               : {nE}")
    print(f"Mayor diferencia +    : {fila_pos['store_id']} "
          f"({float(fila_pos['difference']):+.2f})")
    print(f"Mayor diferencia -    : {fila_neg['store_id']} "
          f"({float(fila_neg['difference']):+.2f})")
    print(f"Figura generada       : {ruta_png} ({ruta_png.stat().st_size} bytes)")
    print(f"Tabla pareada         : {ruta_csv}")
    print(f"Contrato de cifras    : resultados.json")
    print(f"Consistencia con T2   : {consistencia}")


if __name__ == "__main__":
    main()
