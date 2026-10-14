#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T2 - Auditar los datos y construir los pares.

Carga data/model_errors.csv con validación estricta:
  * columnas exactas: store_id, error_a, error_b
  * store_id único y no nulo (ValueError que menciona el identificador)
  * errores numéricos, finitos y no negativos (ValueError)
  * el CSV original NO se modifica (se verifica por bytes)

Construye la tabla pareada conservando las tres columnas originales y añadiendo
difference = error_a - error_b y favors ('B' si difference > 0, 'A' si < 0),
verificando el emparejamiento mediante la clave store_id (unión 1:1 por clave,
no por posición). Cuenta tiendas que favorecen a A y a B, detecta empates e
identifica las diferencias de mayor magnitud en cada dirección.

Salidas:
  - output/store_differences.csv  (tabla pareada)
  - output/store_differences.png  (figura de diferencias por tienda)
  - resultados.json               (contrato de la subtarea)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

CSV_PATH = Path("data/model_errors.csv")
OUT_DIR = Path("output")
PNG_PATH = OUT_DIR / "store_differences.png"
TABLA_PATH = OUT_DIR / "store_differences.csv"
RESULTS_PATH = Path("resultados.json")

EXPECTED_COLS = ["store_id", "error_a", "error_b"]
COLOR_MAP = {"A": "#d62728", "B": "#1f77b4", "empate": "#7f7f7f"}


def auditar_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """Lee el CSV y valida columnas, identificadores y errores (solo lectura)."""
    df_raw = pd.read_csv(path)

    # 1) Deben existir EXACTAMENTE las columnas store_id, error_a, error_b
    cols = list(df_raw.columns)
    if len(cols) != len(EXPECTED_COLS) or set(cols) != set(EXPECTED_COLS):
        raise ValueError(
            f"Columnas incorrectas: se esperaban exactamente {EXPECTED_COLS}, "
            f"se encontraron {cols}"
        )
    df = df_raw[EXPECTED_COLS].copy()  # copia en memoria; el archivo no se toca

    # 2) store_id único y no nulo (ValueError que mencione el identificador)
    problemas: list[str] = []
    nulos = df["store_id"].isna()
    if nulos.any():
        filas = df.index[nulos].tolist()
        problemas.append(f"store_id contiene valores nulos en las filas {filas}")
    duplicados = (
        df.loc[df["store_id"].duplicated(keep=False), "store_id"]
        .dropna()
        .unique()
        .tolist()
    )
    if duplicados:
        problemas.append(
            f"store_id duplicado(s): {sorted(str(d) for d in duplicados)}"
        )

    # 3) Errores numéricos, finitos y no negativos
    for col in ("error_a", "error_b"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            try:
                df[col] = pd.to_numeric(df[col], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"La columna '{col}' no es numérica "
                    f"(contiene valores no convertibles): {exc}"
                ) from exc
        vals = df[col].to_numpy(dtype=float)
        no_finitos = ~np.isfinite(vals)
        if no_finitos.any():
            ids = df.loc[no_finitos, "store_id"].astype(str).tolist()
            problemas.append(
                f"La columna '{col}' tiene valores no finitos (NaN/inf) en tiendas {ids}"
            )
        negativos = vals < 0
        if negativos.any():
            ids = df.loc[negativos, "store_id"].astype(str).tolist()
            problemas.append(
                f"La columna '{col}' tiene valores negativos en tiendas {ids}"
            )

    if problemas:
        raise ValueError("Auditoría del CSV fallida -> " + " | ".join(problemas))

    return df, cols


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {CSV_PATH}")

    bytes_antes = CSV_PATH.read_bytes()  # para verificar al final que queda intacto
    df, cols_encontradas = auditar_csv(CSV_PATH)

    # ---------- Emparejamiento verificado por store_id (NO por posición) ----------
    # Comprobar que dos vectores tienen la misma longitud no demuestra que la
    # fila i de uno corresponda a la fila i del otro: la alineación se hace por
    # la clave única store_id mediante una unión 1 a 1.
    serie_a = df.set_index("store_id")["error_a"]
    serie_b = df.set_index("store_id")["error_b"]

    if not (serie_a.index.is_unique and serie_b.index.is_unique):
        raise ValueError("store_id debe ser único para emparejar 1 a 1")
    if set(serie_a.index) != set(serie_b.index):
        faltan = set(serie_a.index) ^ set(serie_b.index)
        raise ValueError(
            "Los store_id de error_a y error_b no coinciden: "
            f"{sorted(str(s) for s in faltan)}"
        )

    paired = pd.merge(
        serie_a.rename("error_a"),
        serie_b.rename("error_b"),
        left_index=True,
        right_index=True,
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(df):
        raise ValueError(
            f"Emparejamiento incompleto: {len(paired)} pares para {len(df)} tiendas"
        )

    # ---------- Tabla pareada: difference y favors ----------
    paired["difference"] = paired["error_a"] - paired["error_b"]
    paired["favors"] = np.select(
        [paired["difference"] > 0, paired["difference"] < 0],
        ["B", "A"],
        default="empate",
    )
    paired = paired.reset_index()[["store_id", "error_a", "error_b", "difference", "favors"]]

    # ---------- Conteos por modelo y empates ----------
    mask_a = paired["favors"] == "A"
    mask_b = paired["favors"] == "B"
    mask_emp = paired["favors"] == "empate"
    n_a, n_b, n_tie = int(mask_a.sum()), int(mask_b.sum()), int(mask_emp.sum())
    tiendas_a = paired.loc[mask_a, "store_id"].astype(str).tolist()
    tiendas_b = paired.loc[mask_b, "store_id"].astype(str).tolist()
    tiendas_empate = paired.loc[mask_emp, "store_id"].astype(str).tolist()

    # ---------- Diferencias de mayor magnitud en cada dirección ----------
    def registro_extremo(filas: pd.DataFrame, valor: float) -> dict:
        return {
            "store_id": [str(s) for s in filas["store_id"]],
            "difference": float(valor),
            "favors": str(filas["favors"].iloc[0]),
            "error_a": float(filas["error_a"].iloc[0]),
            "error_b": float(filas["error_b"].iloc[0]),
        }

    dmax = float(paired["difference"].max())
    dmin = float(paired["difference"].min())
    mayor_pos = registro_extremo(paired[paired["difference"] == dmax], dmax) if dmax > 0 else None
    mayor_neg = registro_extremo(paired[paired["difference"] == dmin], dmin) if dmin < 0 else None

    # ---------- Verificar que el CSV original permanece intacto ----------
    csv_intacto = CSV_PATH.read_bytes() == bytes_antes
    if not csv_intacto:
        raise RuntimeError("El CSV original fue modificado durante la ejecución")

    # ---------- Salidas exigidas por el enunciado ----------
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paired.to_csv(TABLA_PATH, index=False)

    # Figura: diferencias por tienda, coloreadas según el modelo favorecido
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colores = [COLOR_MAP[f] for f in paired["favors"]]
    bars = ax.bar(
        paired["store_id"], paired["difference"], color=colores,
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    for extremo, etiqueta, desplaz in (
        (mayor_pos, "max +", 16),
        (mayor_neg, "max -", -18),
    ):
        if extremo is None:
            continue
        for sid in extremo["store_id"]:
            pos = int(paired.index[paired["store_id"] == sid][0])
            bars[pos].set_edgecolor("#b8860b")
            bars[pos].set_linewidth(2.2)
            ax.annotate(
                etiqueta, (pos, extremo["difference"]),
                textcoords="offset points", xytext=(0, desplaz),
                ha="center", fontsize=8, fontweight="bold", color="#8a6d0b",
            )
    for bar, diff in zip(bars, paired["difference"]):
        if diff >= 0:
            ax.text(bar.get_x() + bar.get_width() / 2, diff + 0.05, f"{diff:+.2f}",
                    ha="center", va="bottom", fontsize=7.5)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, diff - 0.05, f"{diff:+.2f}",
                    ha="center", va="top", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.9, zorder=2)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a - error_b")
    ax.set_title("Diferencia de error por tienda (pares verificados por store_id)")
    ax.set_ylim(paired["difference"].min() - 0.7, paired["difference"].max() + 0.7)
    leyenda = [
        Patch(facecolor=COLOR_MAP["B"], edgecolor="black", label=f"Favorece a B (n={n_b})"),
        Patch(facecolor=COLOR_MAP["A"], edgecolor="black", label=f"Favorece a A (n={n_a})"),
    ]
    if n_tie:
        leyenda.append(Patch(facecolor=COLOR_MAP["empate"], edgecolor="black",
                             label=f"Empate (n={n_tie})"))
    ax.legend(handles=leyenda, loc="lower left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(PNG_PATH, dpi=120)
    plt.close(fig)

    # ---------- resultados.json (contrato de la subtarea) ----------
    tabla_registros = [
        {
            "store_id": str(r.store_id),
            "error_a": float(r.error_a),
            "error_b": float(r.error_b),
            "difference": float(r.difference),
            "favors": str(r.favors),
        }
        for r in paired.itertuples(index=False)
    ]
    resultados = {
        "subtarea": "T2",
        "descripcion": "Auditoría del CSV y construcción de la tabla pareada por store_id",
        "archivo_entrada": str(CSV_PATH),
        "validacion": {
            "columnas_esperadas": EXPECTED_COLS,
            "columnas_encontradas": cols_encontradas,
            "columnas_exactas": True,
            "n_filas": int(len(df)),
            "store_id_nulos": int(df["store_id"].isna().sum()),
            "store_id_unicos": bool(df["store_id"].is_unique),
            "errores_numericos": True,
            "errores_finitos": True,
            "errores_no_negativos": True,
            "emparejamiento_verificado_por_store_id": True,
            "metodo_emparejamiento": "unión 1:1 por la clave store_id (no por posición)",
            "csv_original_intacto": bool(csv_intacto),
        },
        "n_pares_validados": int(len(paired)),
        "store_ids": [str(s) for s in paired["store_id"]],
        "tabla_pareada": tabla_registros,
        "conteos": {
            "favorece_A": n_a,
            "favorece_B": n_b,
            "empates": n_tie,
            "suma_conteos": int(n_a + n_b + n_tie),
            "tiendas_favorecen_A": tiendas_a,
            "tiendas_favorecen_B": tiendas_b,
            "tiendas_empate": tiendas_empate,
        },
        "mayor_diferencia_positiva": mayor_pos,
        "mayor_diferencia_negativa": mayor_neg,
        "archivos_generados": {
            "tabla_csv": str(TABLA_PATH),
            "figura_png": str(PNG_PATH),
        },
        "nota_longitud_insuficiente": (
            "Comprobar solo que error_a y error_b tengan la misma longitud verifica el "
            "tamaño de los vectores, no la correspondencia entre filas: un "
            "reordenamiento, un id duplicado o un id faltante emparejaría tiendas "
            "distintas sin detectarlo. Aquí los errores se alinean uniendo por la "
            "clave única store_id."
        ),
    }
    RESULTS_PATH.write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---------- Resumen ----------
    print("=" * 64)
    print("T2 - Auditoría y tabla pareada (data/model_errors.csv)")
    print("=" * 64)
    print(f"Columnas encontradas: {cols_encontradas} (exactas: True)")
    print(f"Filas leídas: {len(df)} | pares validados por store_id: {len(paired)}")
    print("store_id únicos y no nulos: True | errores numéricos, finitos y >= 0: True")
    print(f"CSV original intacto: {csv_intacto}")
    print(f"Favorece A: {n_a} tiendas -> {tiendas_a}")
    print(f"Favorece B: {n_b} tiendas -> {tiendas_b}")
    print(f"Empates: {n_tie} -> {tiendas_empate} (suma de conteos = {n_a + n_b + n_tie})")
    if mayor_pos:
        print(f"Mayor diferencia positiva (favorece B): {mayor_pos['store_id']} "
              f"con difference = {mayor_pos['difference']:+.4f}")
    else:
        print("No hay diferencias positivas.")
    if mayor_neg:
        print(f"Mayor diferencia negativa (favorece A): {mayor_neg['store_id']} "
              f"con difference = {mayor_neg['difference']:+.4f}")
    else:
        print("No hay diferencias negativas.")
    print(f"Tabla pareada -> {TABLA_PATH}")
    print(f"Figura        -> {PNG_PATH}")
    print(f"Resultados    -> {RESULTS_PATH}")


if __name__ == "__main__":
    main()
