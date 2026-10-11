# T3: Figura con una observación por tienda (signo y magnitud de difference) y
# referencia visible en cero. Salidas exigidas: output/store_differences.png
# y output/store_differences.csv. Cifras del informe en resultados.json.

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# --------------------------------------------------------------------------- rutas
RUTA_RESULTADOS_T2 = Path("entrada/T2/resultados.json")
RUTA_CSV_DATOS = Path("data/model_errors.csv")
CARPETA_SALIDA = Path("output")
RUTA_PNG = CARPETA_SALIDA / "store_differences.png"
RUTA_CSV_PAREADA = CARPETA_SALIDA / "store_differences.csv"
RUTA_RESULTADOS_T3 = Path("resultados.json")

COLOR_FAV_B = "#d62728"   # difference > 0 -> favorece al modelo B
COLOR_FAV_A = "#1f77b4"   # difference < 0 -> favorece al modelo A
COLOR_EMPATE = "#7f7f7f"  # difference = 0 -> empate


def cargar_tabla_pareada() -> pd.DataFrame:
    """Obtiene la tabla pareada de la subtarea previa T2; si no está disponible,
    la reconstruye desde data/model_errors.csv con la misma validación de T2."""
    if RUTA_RESULTADOS_T2.exists():
        with RUTA_RESULTADOS_T2.open("r", encoding="utf-8") as f:
            t2 = json.load(f)
        tabla = t2.get("tabla_pareada")
        if tabla:
            df = pd.DataFrame(tabla)
            columnas = ["store_id", "error_a", "error_b", "difference", "favors"]
            if all(c in df.columns for c in columnas):
                df = df[columnas].copy()
                df["store_id"] = df["store_id"].astype(str)
                for c in ["error_a", "error_b", "difference"]:
                    df[c] = pd.to_numeric(df[c])
                return df.sort_values("store_id").reset_index(drop=True)

    # Respaldo: reconstruir desde el CSV original (validaciones de T2)
    if not RUTA_CSV_DATOS.exists():
        raise FileNotFoundError(
            f"No se encontró ni {RUTA_RESULTADOS_T2} ni {RUTA_CSV_DATOS}"
        )
    crudo = pd.read_csv(RUTA_CSV_DATOS)
    exigidas = ["store_id", "error_a", "error_b"]
    if [c.strip() for c in crudo.columns] != exigidas:
        raise ValueError(
            f"Columnas exigidas {exigidas}, encontradas {list(crudo.columns)}"
        )
    if crudo["store_id"].isna().any() or crudo["store_id"].duplicated().any():
        raise ValueError("El identificador store_id debe ser único y no nulo")
    for c in ["error_a", "error_b"]:
        vals = pd.to_numeric(crudo[c], errors="coerce")
        if vals.isna().any() or not np.isfinite(vals.to_numpy()).all() or (vals < 0).any():
            raise ValueError(f"La columna {c} debe ser numérica, finita y no negativa")
    df = crudo.copy()
    df["difference"] = df["error_a"] - df["error_b"]
    df["favors"] = np.where(
        df["difference"] > 0, "B", np.where(df["difference"] < 0, "A", "empate")
    )
    return df.sort_values("store_id").reset_index(drop=True)


def crear_figura(df: pd.DataFrame) -> None:
    """Barras: una por tienda, altura = difference (signo y magnitud),
    con línea de referencia visible en cero."""
    df = df.sort_values("store_id").reset_index(drop=True)
    n = len(df)
    x = np.arange(n)
    diff = df["difference"].to_numpy(dtype=float)

    colores = np.where(
        diff > 0, COLOR_FAV_B, np.where(diff < 0, COLOR_FAV_A, COLOR_EMPATE)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x, diff, color=colores, edgecolor="black", linewidth=0.6, zorder=3)

    # Referencia visible en cero
    ax.axhline(0, color="black", linewidth=1.8, zorder=4)
    ax.text(-0.6, -0.08, "referencia en cero (difference = 0)", ha="left", va="top",
            fontsize=8, color="black", zorder=5)

    # Signo y magnitud anotados en cada barra
    for xi, d in zip(x, diff):
        etiqueta = "0.00" if d == 0 else f"{d:+.2f}"
        if d >= 0:
            ax.annotate(etiqueta, (xi, d), textcoords="offset points", xytext=(0, 4),
                        ha="center", va="bottom", fontsize=8, zorder=5)
        else:
            ax.annotate(etiqueta, (xi, d), textcoords="offset points", xytext=(0, -4),
                        ha="center", va="top", fontsize=8, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["store_id"].tolist())
    for tick, fav in zip(ax.get_xticklabels(), df["favors"]):
        tick.set_color(COLOR_FAV_B if fav == "B" else
                       COLOR_FAV_A if fav == "A" else "black")
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_ylim(float(min(diff)) - 0.9, float(max(diff)) + 1.1)
    ax.set_xlabel("Tienda (store_id) — una observación por tienda")
    ax.set_ylabel("difference = error_a − error_b")
    ax.set_title("Diferencia de error pareada por tienda (16 tiendas): "
                 "signo y magnitud de difference")
    ax.grid(axis="y", linestyle=":", alpha=0.5, zorder=0)

    # Conteo por preferencia dentro del gráfico
    n_a = int((df["favors"] == "A").sum())
    n_b = int((df["favors"] == "B").sum())
    n_e = int((df["favors"] == "empate").sum())
    ax.text(0.01, 0.98,
            f"Favorecen a A: {n_a}   ·   Favorecen a B: {n_b}   ·   Empates: {n_e}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="0.6", alpha=0.9), zorder=6)

    manejadores = [
        Patch(facecolor=COLOR_FAV_B, edgecolor="black",
              label="difference > 0 → favorece al modelo B"),
        Patch(facecolor=COLOR_FAV_A, edgecolor="black",
              label="difference < 0 → favorece al modelo A"),
    ]
    if n_e:
        manejadores.append(Patch(facecolor=COLOR_EMPATE, edgecolor="black",
                                 label="difference = 0 → empate"))
    ax.legend(handles=manejadores, loc="lower left", fontsize=9, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(RUTA_PNG, dpi=120)
    plt.close(fig)


def main() -> None:
    CARPETA_SALIDA.mkdir(parents=True, exist_ok=True)

    df = cargar_tabla_pareada()
    if len(df) == 0:
        raise ValueError("La tabla pareada está vacía: no hay tiendas que mostrar")
    if len(df) != 16:
        print(f"Advertencia: se esperaban 16 tiendas y hay {len(df)}")
    df = df.sort_values("store_id").reset_index(drop=True)

    # CSV exigido por el enunciado (tabla pareada que la figura visualiza)
    df.to_csv(RUTA_CSV_PAREADA, index=False)

    # Figura exigida por el enunciado
    crear_figura(df)

    diff = df["difference"].astype(float)
    fav_b = df[df["favors"] == "B"].sort_values("difference", ascending=False)
    fav_a = df[df["favors"] == "A"].sort_values("difference")

    def top3(sub: pd.DataFrame) -> list:
        return [{"store_id": str(r["store_id"]), "difference": float(r["difference"])}
                for _, r in sub.head(3).iterrows()]

    resultados = {
        "subtarea": "T3: figura de diferencias por tienda (signo, magnitud y referencia en cero)",
        "figura_png": str(RUTA_PNG),
        "figura_creada": RUTA_PNG.exists(),
        "tabla_csv": str(RUTA_CSV_PAREADA),
        "fuente_datos": ("entrada/T2/resultados.json (tabla_pareada de la subtarea previa)"
                         if RUTA_RESULTADOS_T2.exists() else str(RUTA_CSV_DATOS)),
        "convencion_signo": ("difference = error_a - error_b; positivo favorece al modelo B, "
                             "negativo favorece al modelo A"),
        "referencia_cero_incluida": True,
        "n_tiendas_mostradas": int(len(df)),
        "stores_mostradas": df["store_id"].tolist(),
        "tiendas_favorecen_A": int((df["favors"] == "A").sum()),
        "tiendas_favorecen_B": int((df["favors"] == "B").sum()),
        "empates": int((df["favors"] == "empate").sum()),
        "stores_favorecen_A": df.loc[df["favors"] == "A", "store_id"].tolist(),
        "stores_favorecen_B": df.loc[df["favors"] == "B", "store_id"].tolist(),
        "stores_en_empate": df.loc[df["favors"] == "empate", "store_id"].tolist(),
        "mayor_diferencia_favor_B": ({"store_id": str(fav_b.iloc[0]["store_id"]),
                                      "difference": float(fav_b.iloc[0]["difference"])}
                                     if len(fav_b) else None),
        "mayor_diferencia_favor_A": ({"store_id": str(fav_a.iloc[0]["store_id"]),
                                      "difference": float(fav_a.iloc[0]["difference"])}
                                     if len(fav_a) else None),
        "top3_diferencias_favor_B": top3(fav_b),
        "top3_diferencias_favor_A": top3(fav_a),
        "diferencia_minima": float(diff.min()),
        "diferencia_maxima": float(diff.max()),
        "diferencias_por_tienda": [
            {"store_id": str(r["store_id"]), "error_a": float(r["error_a"]),
             "error_b": float(r["error_b"]), "difference": float(r["difference"]),
             "favors": str(r["favors"])}
            for _, r in df.iterrows()
        ],
    }

    with RUTA_RESULTADOS_T3.open("w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print("T3 completada.")
    print(f"Figura: {RUTA_PNG} "
          f"({resultados['n_tiendas_mostradas']} tiendas, referencia en cero incluida)")
    print(f"Tabla:  {RUTA_CSV_PAREADA}")
    print(f"Favorecen a A: {resultados['tiendas_favorecen_A']} -> "
          f"{', '.join(resultados['stores_favorecen_A'])}")
    print(f"Favorecen a B: {resultados['tiendas_favorecen_B']} -> "
          f"{', '.join(resultados['stores_favorecen_B'])}")
    print(f"Empates: {resultados['empates']}")
    if resultados["mayor_diferencia_favor_B"]:
        m = resultados["mayor_diferencia_favor_B"]
        print(f"Mayor diferencia a favor de B: {m['store_id']} "
              f"(difference = {m['difference']:+.2f})")
    if resultados["mayor_diferencia_favor_A"]:
        m = resultados["mayor_diferencia_favor_A"]
        print(f"Mayor diferencia a favor de A: {m['store_id']} "
              f"(difference = {m['difference']:+.2f})")


if __name__ == "__main__":
    main()
