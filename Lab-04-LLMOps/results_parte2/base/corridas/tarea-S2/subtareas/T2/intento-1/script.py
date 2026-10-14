# -*- coding: utf-8 -*-
"""
Subtarea T2 — Auditar los datos y construir los pares.

- Carga data/model_errors.csv (solo lectura; el CSV original NO se modifica)
  validando: columnas exactas (store_id, error_a, error_b), store_id único y
  no nulo, errores numéricos, finitos y no negativos (ValueError si falla).
- Construye la tabla pareada conservando las tres columnas originales y
  añadiendo difference = error_a - error_b y favors ('B' si difference > 0,
  'A' si difference < 0, 'Empate' si difference == 0).
- Verifica el emparejamiento mediante store_id (join uno-a-uno por clave y
  comparación de valores), NO por la longitud de los vectores.
- Cuenta tiendas que favorecen a A y a B, empates, y las diferencias de mayor
  magnitud en cada dirección.
- Salidas: output/store_differences.csv, output/store_differences.png y
  resultados.json (contrato de la subtarea).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUTA_CSV = Path("data") / "model_errors.csv"
DIR_SALIDA = Path("output")
RUTA_TABLA_CSV = DIR_SALIDA / "store_differences.csv"
RUTA_FIGURA = DIR_SALIDA / "store_differences.png"
RUTA_RESULTADOS = Path("resultados.json")

COLUMNAS_ESPERADAS = ["store_id", "error_a", "error_b"]
N_PARES_ESPERADOS = 16
COLORES = {"A": "#d62728", "B": "#1f77b4", "Empate": "#7f7f7f"}


def cargar_y_validar_csv(ruta: Path) -> pd.DataFrame:
    """Carga el CSV (solo lectura) y aplica todas las validaciones de la Tarea 2."""
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de datos: {ruta}")

    df = pd.read_csv(ruta)  # solo lectura: el CSV original nunca se modifica

    # 1) Deben existir EXACTAMENTE las columnas store_id, error_a, error_b.
    columnas = [str(c) for c in df.columns]
    if len(columnas) != len(COLUMNAS_ESPERADAS) or set(columnas) != set(COLUMNAS_ESPERADAS):
        raise ValueError(
            f"El CSV debe tener exactamente las columnas {COLUMNAS_ESPERADAS}; "
            f"se encontraron {columnas}"
        )

    # 2) store_id: no nulo y único (ValueError que mencione el identificador).
    ids = df["store_id"]
    if ids.isna().any():
        filas = df.index[ids.isna().to_numpy()].tolist()
        raise ValueError(f"store_id es nulo en las filas con índice {filas}")
    ids_txt = ids.astype(str)
    vacias = (ids_txt.str.strip() == "").to_numpy()
    if vacias.any():
        filas = df.index[vacias].tolist()
        raise ValueError(f"store_id está vacío en las filas con índice {filas}")
    duplicados = ids[ids.duplicated(keep=False)].unique().tolist()
    if duplicados:
        raise ValueError(
            f"store_id duplicado: {[str(d) for d in duplicados]} — "
            "cada identificador debe ser único"
        )

    # 3) error_a y error_b: numéricos, finitos y no negativos.
    for col in ("error_a", "error_b"):
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(
                f"La columna {col} debe ser numérica; dtype encontrado: {df[col].dtype}"
            )
        valores = df[col].to_numpy(dtype=float)
        no_finitos = ~np.isfinite(valores)
        if no_finitos.any():
            malos = df.loc[no_finitos, "store_id"].astype(str).tolist()
            raise ValueError(
                f"La columna {col} contiene valores no finitos (NaN/inf) para: {malos}"
            )
        negativos = valores < 0
        if negativos.any():
            malos = df.loc[negativos, "store_id"].astype(str).tolist()
            raise ValueError(f"La columna {col} contiene valores negativos para: {malos}")

    return df


def construir_tabla_pareada(df: pd.DataFrame) -> pd.DataFrame:
    """Conserva las tres columnas originales y añade difference y favors."""
    tabla = df[COLUMNAS_ESPERADAS].copy()
    tabla["difference"] = tabla["error_a"] - tabla["error_b"]
    tabla["favors"] = np.where(
        tabla["difference"] > 0, "B",
        np.where(tabla["difference"] < 0, "A", "Empate"),
    )
    return tabla[["store_id", "error_a", "error_b", "difference", "favors"]]


def verificar_emparejamiento_por_store_id(df_csv: pd.DataFrame, tabla: pd.DataFrame) -> int:
    """
    Comprueba los pares por la clave store_id (no por la longitud de los vectores):
    mismos conjuntos de claves, correspondencia uno a uno y valores idénticos a los
    del CSV original tras el join por store_id.
    """
    claves_csv = set(df_csv["store_id"])
    claves_tabla = set(tabla["store_id"])
    if claves_csv != claves_tabla:
        raise ValueError(
            "El emparejamiento por store_id falla: "
            f"claves solo en el CSV={sorted(map(str, claves_csv - claves_tabla))}, "
            f"claves solo en la tabla={sorted(map(str, claves_tabla - claves_csv))}"
        )

    ctrl = tabla.merge(
        df_csv[COLUMNAS_ESPERADAS],
        on="store_id",
        how="inner",
        suffixes=("", "_csv"),
        validate="one_to_one",
    )
    if len(ctrl) != len(df_csv):
        raise ValueError(
            "El emparejamiento por store_id no es uno a uno para todos los pares "
            f"({len(ctrl)} de {len(df_csv)})"
        )
    coincide = np.isclose(ctrl["error_a"], ctrl["error_a_csv"]) & np.isclose(
        ctrl["error_b"], ctrl["error_b_csv"]
    )
    if not bool(coincide.all()):
        malos = ctrl.loc[~coincide, "store_id"].astype(str).tolist()
        raise ValueError(f"Pares mal emparejados según store_id: {malos}")
    return int(len(ctrl))


def extremo_por_direccion(tabla: pd.DataFrame, direccion: str):
    """Diferencia de mayor magnitud en una dirección ('B' positiva, 'A' negativa)."""
    sub = tabla[tabla["favors"] == direccion]
    if sub.empty:
        return None
    if direccion == "B":
        fila = sub.loc[sub["difference"].idxmax()]
    else:
        fila = sub.loc[sub["difference"].idxmin()]
    return {
        "store_id": str(fila["store_id"]),
        "difference": float(fila["difference"]),
        "error_a": float(fila["error_a"]),
        "error_b": float(fila["error_b"]),
    }


def ranking_diferencias(tabla: pd.DataFrame, direccion: str, k: int = 3):
    """Top-k diferencias por dirección (mayor magnitud primero)."""
    sub = tabla[tabla["favors"] == direccion]
    if sub.empty:
        return []
    orden = sub.sort_values("difference", ascending=(direccion == "A"))
    return [
        {"store_id": str(r["store_id"]), "difference": float(r["difference"])}
        for _, r in orden.head(k).iterrows()
    ]


def graficar_diferencias(tabla: pd.DataFrame, ruta: Path) -> None:
    """Figura de barras de difference por tienda, coloreada según favors."""
    etiquetas = tabla["store_id"].astype(str)
    diferencias = tabla["difference"].to_numpy(dtype=float)
    colores = [COLORES[f] for f in tabla["favors"]]

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    barras = ax.bar(etiquetas, diferencias, color=colores, edgecolor="black", linewidth=0.4)

    if len(diferencias):
        max_abs = float(np.abs(diferencias).max())
        margen = 0.12 * max(max_abs, 1e-9)
        for barra, valor in zip(barras, diferencias):
            if valor >= 0:
                ax.text(barra.get_x() + barra.get_width() / 2, valor + margen * 0.35,
                        f"{valor:+.1f}", ha="center", va="bottom", fontsize=8)
            else:
                ax.text(barra.get_x() + barra.get_width() / 2, valor - margen * 0.35,
                        f"{valor:+.1f}", ha="center", va="top", fontsize=8)
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylim(diferencias.min() - 3 * margen, diferencias.max() + 3 * margen)

    ax.set_xlabel("store_id")
    ax.set_ylabel("difference = error_a − error_b")
    ax.set_title("T2 · Diferencia de error por tienda (pareo verificado por store_id)")
    ax.yaxis.grid(True, linestyle=":", alpha=0.6)
    ax.set_axisbelow(True)

    manejadores = [
        Patch(facecolor=COLORES["B"], edgecolor="black", linewidth=0.4,
              label="favorece a B (difference > 0)"),
        Patch(facecolor=COLORES["A"], edgecolor="black", linewidth=0.4,
              label="favorece a A (difference < 0)"),
    ]
    ax.legend(handles=manejadores, loc="lower right", fontsize=8)

    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(ruta, dpi=120)
    plt.close(fig)


def main() -> None:
    # 1) Carga y auditoría del CSV (sin modificarlo).
    df_csv = cargar_y_validar_csv(RUTA_CSV)

    # 2) Tabla pareada con difference y favors.
    tabla = construir_tabla_pareada(df_csv)

    # 3) Verificación del emparejamiento POR store_id (no por longitud).
    n_claves = verificar_emparejamiento_por_store_id(df_csv, tabla)

    # 4) Conteos y extremos por dirección.
    n_A = int((tabla["favors"] == "A").sum())
    n_B = int((tabla["favors"] == "B").sum())
    n_empates = int((tabla["favors"] == "Empate").sum())
    ext_B = extremo_por_direccion(tabla, "B")
    ext_A = extremo_por_direccion(tabla, "A")

    # 5) Salidas exigidas por el enunciado.
    DIR_SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(RUTA_TABLA_CSV, index=False)
    graficar_diferencias(tabla, RUTA_FIGURA)

    registros = [
        {
            "store_id": str(r["store_id"]),
            "error_a": float(r["error_a"]),
            "error_b": float(r["error_b"]),
            "difference": float(r["difference"]),
            "favors": str(r["favors"]),
        }
        for _, r in tabla.iterrows()
    ]

    resultados = {
        "subtarea": "T2: auditar los datos y construir los pares",
        "archivo_csv": str(RUTA_CSV),
        "columnas_exigidas": COLUMNAS_ESPERADAS,
        "columnas_encontradas": [str(c) for c in df_csv.columns],
        "csv_original_no_modificado": True,
        "n_filas_csv": int(len(df_csv)),
        "n_pares_tabla_pareada": int(len(tabla)),
        "n_pares_esperados": N_PARES_ESPERADOS,
        "n_pares_coincide_con_esperado": int(len(tabla)) == N_PARES_ESPERADOS,
        "emparejamiento_verificado_por_store_id": True,
        "n_claves_emparejadas_uno_a_uno": n_claves,
        "metodo_verificacion_emparejamiento": (
            "join uno a uno por store_id con comprobación de que error_a y error_b "
            "coinciden con los del CSV; la igualdad de longitud de dos vectores no "
            "demuestra que las filas correspondan a las mismas tiendas"
        ),
        "tiendas_favorecen_A": n_A,
        "tiendas_favorecen_B": n_B,
        "empates": n_empates,
        "stores_favorecen_A": tabla.loc[tabla["favors"] == "A", "store_id"].astype(str).tolist(),
        "stores_favorecen_B": tabla.loc[tabla["favors"] == "B", "store_id"].astype(str).tolist(),
        "stores_en_empate": tabla.loc[tabla["favors"] == "Empate", "store_id"].astype(str).tolist(),
        "mayor_diferencia_favor_B": ext_B,
        "mayor_diferencia_favor_A": ext_A,
        "top3_diferencias_favor_B": ranking_diferencias(tabla, "B"),
        "top3_diferencias_favor_A": ranking_diferencias(tabla, "A"),
        "tabla_pareada": registros,
        "salidas": {
            "tabla_csv": str(RUTA_TABLA_CSV),
            "figura_png": str(RUTA_FIGURA),
        },
    }

    with RUTA_RESULTADOS.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)

    # 6) Resumen breve.
    print("=== T2: auditoría y construcción de los pares ===")
    print(f"CSV validado: {RUTA_CSV} ({len(df_csv)} filas, columnas {list(df_csv.columns)}); original sin modificar.")
    print(f"Emparejamiento verificado por store_id (join 1:1): {n_claves} pares.")
    print(f"Tiendas que favorecen a A: {n_A} | a B: {n_B} | empates: {n_empates}")
    if ext_B is not None:
        print(f"Mayor diferencia a favor de B: {ext_B['store_id']} (difference={ext_B['difference']:+.4f})")
    if ext_A is not None:
        print(f"Mayor diferencia a favor de A: {ext_A['store_id']} (difference={ext_A['difference']:+.4f})")
    print(f"Tabla pareada guardada en: {RUTA_TABLA_CSV}")
    print(f"Figura guardada en: {RUTA_FIGURA}")
    print(f"Resultados guardados en: {RUTA_RESULTADOS}")


if __name__ == "__main__":
    main()
